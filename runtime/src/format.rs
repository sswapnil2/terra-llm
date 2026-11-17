//! Strict parser, validator, 4KB-aligned buffer primitives, and synthetic fixture
//! builders for `core.bin` (`TRAC`) and `experts.bin` (`TRAE`) per `schemas/terra_format.md`.

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::ptr::NonNull;

use half::f16;
use thiserror::Error;

pub const CORE_MAGIC: [u8; 4] = *b"TRAC";
pub const EXPERTS_MAGIC: [u8; 4] = *b"TRAE";
pub const FORMAT_VERSION: u32 = 1;

pub const CORE_HEADER_BYTES: usize = 256;
pub const CORE_TENSOR_ENTRY_BYTES: usize = 128;
pub const CORE_DATA_ALIGN: usize = 64;

pub const PAGE_ALIGN: usize = 4096;
pub const EXPERTS_HEADER_BYTES: usize = 4096;
pub const EXPERT_INDEX_ENTRY_BYTES: usize = 40;
pub const EXPERT_FLAG_TERNARY_SWIGLU: u32 = 1;

pub const FLAG_BITLINEAR_RMS_PRESCALE: u32 = 1 << 0;
pub const FLAG_NORM_TOPK_PROB: u32 = 1 << 1;
pub const FLAG_TIE_WORD_EMBEDDINGS: u32 = 1 << 2;
pub const FLAG_HAS_HINT_EMBEDDING: u32 = 1 << 3;

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid magic header: expected {expected:?}, got {actual:?}")]
    InvalidMagic { expected: [u8; 4], actual: [u8; 4] },

    #[error("unsupported format version {0} (expected {FORMAT_VERSION})")]
    UnsupportedVersion(u32),

    #[error("header CRC32 mismatch: expected 0x{expected:08x}, actual 0x{actual:08x}")]
    HeaderCrcMismatch { expected: u32, actual: u32 },

    #[error("index CRC32 mismatch: expected 0x{expected:08x}, actual 0x{actual:08x}")]
    IndexCrcMismatch { expected: u32, actual: u32 },

    #[error("tensor `{name}` CRC32 mismatch ({kind}): expected 0x{expected:08x}, actual 0x{actual:08x}")]
    TensorCrcMismatch {
        name: String,
        kind: &'static str,
        expected: u32,
        actual: u32,
    },

    #[error("expert (layer={layer_id}, expert={expert_id}) CRC32 mismatch: expected 0x{expected:08x}, actual 0x{actual:08x}")]
    ExpertCrcMismatch {
        layer_id: u16,
        expert_id: u16,
        expected: u32,
        actual: u32,
    },

    #[error("alignment violation in {context}: value {value} is not aligned to {align} bytes")]
    AlignmentViolation {
        context: String,
        value: u64,
        align: usize,
    },

    #[error("invalid ternary crumb 0b11 at byte {byte_index}, lane {lane} in {context}")]
    InvalidTernaryCrumb {
        context: String,
        byte_index: usize,
        lane: usize,
    },

    #[error("invalid header or tensor metadata: {0}")]
    InvalidMetadata(String),

    #[error("missing required tensor `{0}` in core.bin")]
    MissingTensor(String),

    #[error("dtype mismatch for tensor `{name}`: expected {expected:?}, got {actual:?}")]
    DtypeMismatch {
        name: String,
        expected: TensorDtype,
        actual: TensorDtype,
    },

    #[error("invalid allocation layout for size={size}, align={align}")]
    InvalidLayout { size: usize, align: usize },
}

#[inline]
pub const fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

#[inline]
pub const fn align_up_u64(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

#[inline]
pub const fn expected_expert_payload_bytes(hidden_size: usize, intermediate_size: usize) -> usize {
    (3 * hidden_size * intermediate_size) / 4 + 4 * (2 * intermediate_size + hidden_size)
}

/// Validates that no 2-bit crumb in `packed` carries the reserved `0b11` bit pattern.
pub fn validate_no_invalid_crumbs(packed: &[u8], context: &str) -> Result<(), FormatError> {
    let chunks = packed.chunks_exact(8);
    let rem = chunks.remainder();
    for (chunk_idx, chunk) in chunks.enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        let bad = (word & (word >> 1)) & 0x5555_5555_5555_5555u64;
        if bad != 0 {
            let base = chunk_idx * 8;
            for (i, &b) in chunk.iter().enumerate() {
                for lane in 0..4 {
                    if ((b >> (lane * 2)) & 0x03) == 0x03 {
                        return Err(FormatError::InvalidTernaryCrumb {
                            context: context.to_string(),
                            byte_index: base + i,
                            lane,
                        });
                    }
                }
            }
        }
    }
    let base = (packed.len() / 8) * 8;
    for (i, &b) in rem.iter().enumerate() {
        for lane in 0..4 {
            if ((b >> (lane * 2)) & 0x03) == 0x03 {
                return Err(FormatError::InvalidTernaryCrumb {
                    context: context.to_string(),
                    byte_index: base + i,
                    lane,
                });
            }
        }
    }
    Ok(())
}

/// Packs floating-point weights `[rows, cols]` into 2-bit ternary crumbs + `f32` per-row scales
/// using Section 1.1 & 1.2 `absmean` quantization (`00 -> 0`, `01 -> +1`, `10 -> -1`).
pub fn quantize_pack_ternary_absmean(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<f32>), FormatError> {
    if cols == 0 || cols % 4 != 0 {
        return Err(FormatError::InvalidMetadata(format!(
            "ternary cols ({cols}) must be non-zero and divisible by 4"
        )));
    }
    if weights.len() != rows * cols {
        return Err(FormatError::InvalidMetadata(format!(
            "weights.len() ({}) != rows * cols ({})",
            weights.len(),
            rows * cols
        )));
    }

    let row_packed_bytes = cols / 4;
    let mut packed = vec![0u8; rows * row_packed_bytes];
    let mut scales = vec![0.0f32; rows];

    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let abs_sum: f32 = row.iter().map(|v| v.abs()).sum();
        let gamma = (abs_sum / (cols as f32)).max(1e-8);
        scales[r] = gamma;

        let out_row = &mut packed[r * row_packed_bytes..(r + 1) * row_packed_bytes];
        for b in 0..row_packed_bytes {
            let mut byte = 0u8;
            for lane in 0..4 {
                let w = row[4 * b + lane];
                let q = (w / gamma).clamp(-1.0, 1.0).round_ties_even() as i32;
                let code: u8 = match q {
                    0 => 0b00,
                    1 => 0b01,
                    -1 => 0b10,
                    other => {
                        return Err(FormatError::InvalidMetadata(format!(
                            "unexpected quantized ternary value {other}"
                        )))
                    }
                };
                byte |= code << (lane * 2);
            }
            out_row[b] = byte;
        }
    }

    Ok((packed, scales))
}

/// Unpacks a single packed ternary row into `{-1, 0, +1}` `i8` weights, rejecting `0b11`.
pub fn unpack_ternary_row(
    packed_row: &[u8],
    cols: usize,
    context: &str,
) -> Result<Vec<i8>, FormatError> {
    if cols % 4 != 0 || packed_row.len() * 4 != cols {
        return Err(FormatError::InvalidMetadata(format!(
            "packed_row.len() ({}) * 4 != cols ({cols})",
            packed_row.len()
        )));
    }
    validate_no_invalid_crumbs(packed_row, context)?;
    let mut out = Vec::with_capacity(cols);
    for &b in packed_row {
        for lane in 0..4 {
            let code = (b >> (lane * 2)) & 0x03;
            let val = match code {
                0b00 => 0i8,
                0b01 => 1i8,
                0b10 => -1i8,
                _ => unreachable!("validated above"),
            };
            out.push(val);
        }
    }
    Ok(out)
}

/// Safe RAII heap buffer guaranteed to start on a 4096-byte page boundary
/// with length padded to a multiple of 4096 bytes for `O_DIRECT` / `F_NOCACHE` DMA.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` uniquely owns its heap allocation like `Box<[u8]>`.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocates a zero-initialized buffer aligned to 4096 bytes, rounding `min_len` up to 4096.
    pub fn new_zeroed(min_len: usize) -> Result<Self, FormatError> {
        if min_len == 0 {
            return Err(FormatError::InvalidLayout {
                size: 0,
                align: PAGE_ALIGN,
            });
        }
        let padded_len = align_up(min_len, PAGE_ALIGN);
        Self::new_exact(padded_len)
    }

    /// Allocates a zero-initialized buffer of exact `padded_len` bytes (`padded_len % 4096 == 0`).
    pub fn new_exact(padded_len: usize) -> Result<Self, FormatError> {
        if padded_len == 0 || padded_len % PAGE_ALIGN != 0 {
            return Err(FormatError::InvalidLayout {
                size: padded_len,
                align: PAGE_ALIGN,
            });
        }
        let layout = Layout::from_size_align(padded_len, PAGE_ALIGN).map_err(|_| {
            FormatError::InvalidLayout {
                size: padded_len,
                align: PAGE_ALIGN,
            }
        })?;
        // SAFETY: `padded_len > 0` and `PAGE_ALIGN` is a power of two.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = match NonNull::new(raw) {
            Some(p) => p,
            None => handle_alloc_error(layout),
        };
        assert_eq!(
            (ptr.as_ptr() as usize) % PAGE_ALIGN,
            0,
            "allocator violated 4096-byte alignment"
        );
        Ok(Self {
            ptr,
            len: padded_len,
            layout,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn is_page_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % PAGE_ALIGN == 0 && self.len % PAGE_ALIGN == 0
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is valid and initialized for `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is uniquely owned and valid for `len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut copy = Self::new_exact(self.len).expect("AlignedBuffer clone layout valid");
        copy.copy_from_slice(self);
        copy
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("addr", &self.ptr.as_ptr())
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated with `self.layout` via `alloc_zeroed`.
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerKind {
    AnchorGqa = 0,
    GatedLinearRecurrence = 1,
}

impl LayerKind {
    pub fn from_u8(v: u8) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::AnchorGqa),
            1 => Ok(Self::GatedLinearRecurrence),
            other => Err(FormatError::InvalidMetadata(format!(
                "invalid layer_kind byte {other} (expected 0=AnchorGQA or 1=GatedLinearRecurrence)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDtype {
    Ternary2Bit = 0,
    F32 = 1,
    F16 = 2,
    I8 = 3,
}

impl TensorDtype {
    pub fn from_u8(v: u8) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::Ternary2Bit),
            1 => Ok(Self::F32),
            2 => Ok(Self::F16),
            3 => Ok(Self::I8),
            other => Err(FormatError::InvalidMetadata(format!(
                "unknown tensor dtype code {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoreHeader {
    pub version: u32,
    pub flags: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub max_position_embeddings: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub lookahead_offset: u32,
    pub layer_kinds: Vec<LayerKind>,
    pub raw_layer_kinds: [u8; 64],
    pub tensor_count: u32,
    pub data_section_offset: u64,
    pub header_crc32: u32,
}

impl CoreHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() < CORE_HEADER_BYTES {
            return Err(FormatError::InvalidMetadata(format!(
                "core.bin too short ({} bytes < {CORE_HEADER_BYTES})",
                bytes.len()
            )));
        }
        let header_slice = &bytes[..CORE_HEADER_BYTES];
        let magic: [u8; 4] = header_slice[0..4].try_into().unwrap();
        if magic != CORE_MAGIC {
            return Err(FormatError::InvalidMagic {
                expected: CORE_MAGIC,
                actual: magic,
            });
        }
        let expected_crc = read_u32_le(header_slice, 252);
        let actual_crc = crc32fast::hash(&header_slice[0..252]);
        if expected_crc != actual_crc {
            return Err(FormatError::HeaderCrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        let version = read_u32_le(header_slice, 4);
        if version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }

        let flags = read_u32_le(header_slice, 8);
        let vocab_size = read_u32_le(header_slice, 12);
        let hidden_size = read_u32_le(header_slice, 16);
        let intermediate_size = read_u32_le(header_slice, 20);
        let num_layers = read_u32_le(header_slice, 24);
        let num_attention_heads = read_u32_le(header_slice, 28);
        let num_kv_heads = read_u32_le(header_slice, 32);
        let head_dim = read_u32_le(header_slice, 36);
        let num_experts = read_u32_le(header_slice, 40);
        let top_k = read_u32_le(header_slice, 44);
        let max_position_embeddings = read_u32_le(header_slice, 48);
        let rope_theta = read_f32_le(header_slice, 52);
        let rms_norm_eps = read_f32_le(header_slice, 56);
        let lookahead_offset = read_u32_le(header_slice, 60);

        if vocab_size == 0 {
            return Err(FormatError::InvalidMetadata("vocab_size must be > 0".into()));
        }
        if hidden_size == 0 || hidden_size % 4 != 0 {
            return Err(FormatError::InvalidMetadata(format!(
                "hidden_size ({hidden_size}) must be > 0 and divisible by 4"
            )));
        }
        if intermediate_size == 0 || intermediate_size % 4 != 0 {
            return Err(FormatError::InvalidMetadata(format!(
                "intermediate_size ({intermediate_size}) must be > 0 and divisible by 4"
            )));
        }
        if num_layers == 0 || num_layers > 64 {
            return Err(FormatError::InvalidMetadata(format!(
                "num_layers ({num_layers}) must be in 1..=64"
            )));
        }
        if num_attention_heads == 0 || num_kv_heads == 0 || num_attention_heads % num_kv_heads != 0 {
            return Err(FormatError::InvalidMetadata(format!(
                "invalid head counts H={num_attention_heads}, H_kv={num_kv_heads}"
            )));
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(FormatError::InvalidMetadata(format!(
                "head_dim ({head_dim}) must be > 0 and even for RoPE"
            )));
        }
        if num_experts == 0 || top_k == 0 || top_k > num_experts {
            return Err(FormatError::InvalidMetadata(format!(
                "invalid MoE counts num_experts={num_experts}, top_k={top_k}"
            )));
        }

        let mut raw_layer_kinds = [0u8; 64];
        raw_layer_kinds.copy_from_slice(&header_slice[64..128]);
        let mut layer_kinds = Vec::with_capacity(num_layers as usize);
        for l in 0..(num_layers as usize) {
            layer_kinds.push(LayerKind::from_u8(raw_layer_kinds[l])?);
        }

        let tensor_count = read_u32_le(header_slice, 128);
        let data_section_offset = read_u64_le(header_slice, 132);
        if data_section_offset % (CORE_DATA_ALIGN as u64) != 0 {
            return Err(FormatError::AlignmentViolation {
                context: "core.bin data_section_offset".into(),
                value: data_section_offset,
                align: CORE_DATA_ALIGN,
            });
        }

        let min_data_offset =
            (CORE_HEADER_BYTES as u64) + (tensor_count as u64) * (CORE_TENSOR_ENTRY_BYTES as u64);
        if data_section_offset < min_data_offset {
            return Err(FormatError::InvalidMetadata(format!(
                "data_section_offset ({data_section_offset}) overlaps tensor directory end ({min_data_offset})"
            )));
        }

        Ok(Self {
            version,
            flags,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            num_experts,
            top_k,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            lookahead_offset,
            layer_kinds,
            raw_layer_kinds,
            tensor_count,
            data_section_offset,
            header_crc32: expected_crc,
        })
    }

    #[inline]
    pub fn has_rms_prescale(&self) -> bool {
        (self.flags & FLAG_BITLINEAR_RMS_PRESCALE) != 0
    }

    #[inline]
    pub fn norm_topk_prob(&self) -> bool {
        (self.flags & FLAG_NORM_TOPK_PROB) != 0
    }

    #[inline]
    pub fn tie_word_embeddings(&self) -> bool {
        (self.flags & FLAG_TIE_WORD_EMBEDDINGS) != 0
    }

    #[inline]
    pub fn has_hint_embedding(&self) -> bool {
        (self.flags & FLAG_HAS_HINT_EMBEDDING) != 0
    }
}

#[derive(Debug, Clone)]
pub struct CoreTensorEntry {
    pub name: String,
    pub dtype: TensorDtype,
    pub ndim: u8,
    pub flags: u16,
    pub shape: [u32; 4],
    pub data_offset: u64,
    pub data_bytes: u64,
    pub scale_offset: u64,
    pub scale_bytes: u64,
    pub data_crc32: u32,
    pub scale_crc32: u32,
}

impl CoreTensorEntry {
    pub fn parse(entry_bytes: &[u8]) -> Result<Self, FormatError> {
        debug_assert_eq!(entry_bytes.len(), CORE_TENSOR_ENTRY_BYTES);
        let raw_name = &entry_bytes[0..64];
        let nul_pos = raw_name.iter().position(|&b| b == 0).unwrap_or(64);
        let name_str = std::str::from_utf8(&raw_name[..nul_pos]).map_err(|e| {
            FormatError::InvalidMetadata(format!("invalid UTF-8 tensor name: {e}"))
        })?;
        if name_str.is_empty() {
            return Err(FormatError::InvalidMetadata("empty tensor name".into()));
        }

        let dtype = TensorDtype::from_u8(entry_bytes[64])?;
        let ndim = entry_bytes[65];
        if !(1..=4).contains(&ndim) {
            return Err(FormatError::InvalidMetadata(format!(
                "tensor `{name_str}` has invalid ndim={ndim}"
            )));
        }
        let flags = u16::from_le_bytes(entry_bytes[66..68].try_into().unwrap());
        let shape = [
            read_u32_le(entry_bytes, 68),
            read_u32_le(entry_bytes, 72),
            read_u32_le(entry_bytes, 76),
            read_u32_le(entry_bytes, 80),
        ];
        let data_offset = read_u64_le(entry_bytes, 84);
        let data_bytes = read_u64_le(entry_bytes, 92);
        let scale_offset = read_u64_le(entry_bytes, 100);
        let scale_bytes = read_u64_le(entry_bytes, 108);
        let data_crc32 = read_u32_le(entry_bytes, 116);
        let scale_crc32 = read_u32_le(entry_bytes, 120);

        Ok(Self {
            name: name_str.to_string(),
            dtype,
            ndim,
            flags,
            shape,
            data_offset,
            data_bytes,
            scale_offset,
            scale_bytes,
            data_crc32,
            scale_crc32,
        })
    }

    pub fn dims(&self) -> Vec<usize> {
        self.shape[..(self.ndim as usize)]
            .iter()
            .map(|&d| d as usize)
            .collect()
    }

    pub fn numel(&self) -> usize {
        self.dims().into_iter().product()
    }
}

#[derive(Debug, Clone)]
pub struct TernaryTensorView<'a> {
    pub rows: usize,
    pub cols: usize,
    pub packed: &'a [u8],
    pub scales: Cow<'a, [f32]>,
}

/// Validated resident `core.bin` image in RAM.
#[derive(Debug)]
pub struct CoreFile {
    pub header: CoreHeader,
    pub entries: Vec<CoreTensorEntry>,
    index_by_name: HashMap<String, usize>,
    bytes: Vec<u8>,
}

impl CoreFile {
    pub fn open(path: &Path) -> Result<Self, FormatError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, FormatError> {
        let header = CoreHeader::parse(&bytes)?;
        let dir_end =
            CORE_HEADER_BYTES + (header.tensor_count as usize) * CORE_TENSOR_ENTRY_BYTES;
        if bytes.len() < dir_end || (bytes.len() as u64) < header.data_section_offset {
            return Err(FormatError::InvalidMetadata(format!(
                "core.bin truncated ({} bytes)",
                bytes.len()
            )));
        }

        let mut entries = Vec::with_capacity(header.tensor_count as usize);
        let mut index_by_name = HashMap::with_capacity(header.tensor_count as usize);

        for idx in 0..(header.tensor_count as usize) {
            let start = CORE_HEADER_BYTES + idx * CORE_TENSOR_ENTRY_BYTES;
            let entry = CoreTensorEntry::parse(&bytes[start..start + CORE_TENSOR_ENTRY_BYTES])?;
            if index_by_name.contains_key(&entry.name) {
                return Err(FormatError::InvalidMetadata(format!(
                    "duplicate tensor name `{}` in core.bin",
                    entry.name
                )));
            }

            if entry.data_offset < header.data_section_offset {
                return Err(FormatError::InvalidMetadata(format!(
                    "tensor `{}` data_offset ({}) < data_section_offset ({})",
                    entry.name, entry.data_offset, header.data_section_offset
                )));
            }
            if entry.data_offset % (CORE_DATA_ALIGN as u64) != 0 {
                return Err(FormatError::AlignmentViolation {
                    context: format!("core.bin tensor `{}` data_offset", entry.name),
                    value: entry.data_offset,
                    align: CORE_DATA_ALIGN,
                });
            }

            let data_end = entry
                .data_offset
                .checked_add(entry.data_bytes)
                .ok_or_else(|| FormatError::InvalidMetadata("data_offset overflow".into()))?;
            if data_end > bytes.len() as u64 {
                return Err(FormatError::InvalidMetadata(format!(
                    "tensor `{}` data range {}..{} exceeds file size {}",
                    entry.name,
                    entry.data_offset,
                    data_end,
                    bytes.len()
                )));
            }

            let data_slice = &bytes[entry.data_offset as usize..data_end as usize];
            let actual_data_crc = crc32fast::hash(data_slice);
            if actual_data_crc != entry.data_crc32 {
                return Err(FormatError::TensorCrcMismatch {
                    name: entry.name,
                    kind: "data",
                    expected: entry.data_crc32,
                    actual: actual_data_crc,
                });
            }

            match entry.dtype {
                TensorDtype::Ternary2Bit => {
                    if entry.ndim != 2 {
                        return Err(FormatError::InvalidMetadata(format!(
                            "ternary tensor `{}` must have ndim=2, got {}",
                            entry.name, entry.ndim
                        )));
                    }
                    let rows = entry.shape[0] as usize;
                    let cols = entry.shape[1] as usize;
                    if rows == 0 || cols == 0 || cols % 4 != 0 {
                        return Err(FormatError::InvalidMetadata(format!(
                            "ternary tensor `{}` has invalid shape [{rows}, {cols}]",
                            entry.name
                        )));
                    }
                    let expected_data_bytes = (rows * (cols / 4)) as u64;
                    let expected_scale_bytes = (rows * 4) as u64;
                    if entry.data_bytes != expected_data_bytes {
                        return Err(FormatError::InvalidMetadata(format!(
                            "ternary tensor `{}` data_bytes={} != expected {}",
                            entry.name, entry.data_bytes, expected_data_bytes
                        )));
                    }
                    if entry.scale_bytes != expected_scale_bytes || entry.scale_offset == 0 {
                        return Err(FormatError::InvalidMetadata(format!(
                            "ternary tensor `{}` invalid scale metadata offset={} bytes={}",
                            entry.name, entry.scale_offset, entry.scale_bytes
                        )));
                    }
                    let scale_end = entry
                        .scale_offset
                        .checked_add(entry.scale_bytes)
                        .ok_or_else(|| {
                            FormatError::InvalidMetadata("scale_offset overflow".into())
                        })?;
                    if entry.scale_offset < header.data_section_offset
                        || scale_end > bytes.len() as u64
                    {
                        return Err(FormatError::InvalidMetadata(format!(
                            "ternary tensor `{}` scale range out of bounds",
                            entry.name
                        )));
                    }
                    let scale_slice = &bytes[entry.scale_offset as usize..scale_end as usize];
                    let actual_scale_crc = crc32fast::hash(scale_slice);
                    if actual_scale_crc != entry.scale_crc32 {
                        return Err(FormatError::TensorCrcMismatch {
                            name: entry.name,
                            kind: "scale",
                            expected: entry.scale_crc32,
                            actual: actual_scale_crc,
                        });
                    }
                    validate_no_invalid_crumbs(data_slice, &entry.name)?;
                }
                TensorDtype::F32 => {
                    let expected = (entry.numel() * 4) as u64;
                    if entry.data_bytes != expected {
                        return Err(FormatError::InvalidMetadata(format!(
                            "F32 tensor `{}` data_bytes={} != expected {}",
                            entry.name, entry.data_bytes, expected
                        )));
                    }
                }
                TensorDtype::F16 => {
                    let expected = (entry.numel() * 2) as u64;
                    if entry.data_bytes != expected {
                        return Err(FormatError::InvalidMetadata(format!(
                            "F16 tensor `{}` data_bytes={} != expected {}",
                            entry.name, entry.data_bytes, expected
                        )));
                    }
                }
                TensorDtype::I8 => {
                    let expected = entry.numel() as u64;
                    if entry.data_bytes != expected {
                        return Err(FormatError::InvalidMetadata(format!(
                            "I8 tensor `{}` data_bytes={} != expected {}",
                            entry.name, entry.data_bytes, expected
                        )));
                    }
                }
            }

            index_by_name.insert(entry.name.clone(), idx);
            entries.push(entry);
        }

        Ok(Self {
            header,
            entries,
            index_by_name,
            bytes,
        })
    }

    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.bytes.len()
            + self.entries.capacity() * std::mem::size_of::<CoreTensorEntry>()
            + self.index_by_name.capacity() * 48
    }

    pub fn entry(&self, name: &str) -> Option<&CoreTensorEntry> {
        self.index_by_name.get(name).map(|&i| &self.entries[i])
    }

    pub fn entry_index(&self, name: &str) -> Option<usize> {
        self.index_by_name.get(name).copied()
    }

    pub fn require_entry_index(&self, name: &str) -> Result<usize, FormatError> {
        self.entry_index(name)
            .ok_or_else(|| FormatError::MissingTensor(name.to_string()))
    }

    pub fn ternary_by_index(&self, idx: usize) -> TernaryTensorView<'_> {
        let e = &self.entries[idx];
        debug_assert_eq!(e.dtype, TensorDtype::Ternary2Bit);
        let rows = e.shape[0] as usize;
        let cols = e.shape[1] as usize;
        let packed =
            &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        let scale_bytes =
            &self.bytes[e.scale_offset as usize..(e.scale_offset + e.scale_bytes) as usize];
        TernaryTensorView {
            rows,
            cols,
            packed,
            scales: bytes_as_f32_cow(scale_bytes),
        }
    }

    pub fn f32_by_index(&self, idx: usize) -> Cow<'_, [f32]> {
        let e = &self.entries[idx];
        debug_assert_eq!(e.dtype, TensorDtype::F32);
        let raw = &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        bytes_as_f32_cow(raw)
    }

    pub fn f16_bits_by_index(&self, idx: usize) -> Cow<'_, [u16]> {
        let e = &self.entries[idx];
        debug_assert_eq!(e.dtype, TensorDtype::F16);
        let raw = &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        bytes_as_u16_cow(raw)
    }

    pub fn require_entry(&self, name: &str) -> Result<&CoreTensorEntry, FormatError> {
        self.entry(name)
            .ok_or_else(|| FormatError::MissingTensor(name.to_string()))
    }

    pub fn ternary_tensor(&self, name: &str) -> Result<TernaryTensorView<'_>, FormatError> {
        let e = self.require_entry(name)?;
        if e.dtype != TensorDtype::Ternary2Bit {
            return Err(FormatError::DtypeMismatch {
                name: name.to_string(),
                expected: TensorDtype::Ternary2Bit,
                actual: e.dtype,
            });
        }
        let rows = e.shape[0] as usize;
        let cols = e.shape[1] as usize;
        let packed =
            &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        let scale_bytes =
            &self.bytes[e.scale_offset as usize..(e.scale_offset + e.scale_bytes) as usize];
        let scales = bytes_as_f32_cow(scale_bytes);
        Ok(TernaryTensorView {
            rows,
            cols,
            packed,
            scales,
        })
    }

    pub fn f32_tensor(&self, name: &str) -> Result<(Vec<usize>, Cow<'_, [f32]>), FormatError> {
        let e = self.require_entry(name)?;
        if e.dtype != TensorDtype::F32 {
            return Err(FormatError::DtypeMismatch {
                name: name.to_string(),
                expected: TensorDtype::F32,
                actual: e.dtype,
            });
        }
        let raw = &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        Ok((e.dims(), bytes_as_f32_cow(raw)))
    }

    pub fn f16_tensor_bits(&self, name: &str) -> Result<(Vec<usize>, Cow<'_, [u16]>), FormatError> {
        let e = self.require_entry(name)?;
        if e.dtype != TensorDtype::F16 {
            return Err(FormatError::DtypeMismatch {
                name: name.to_string(),
                expected: TensorDtype::F16,
                actual: e.dtype,
            });
        }
        let raw = &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        Ok((e.dims(), bytes_as_u16_cow(raw)))
    }

    pub fn f16_tensor_as_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), FormatError> {
        let (dims, bits) = self.f16_tensor_bits(name)?;
        let vals = bits
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect();
        Ok((dims, vals))
    }

    pub fn i8_tensor(&self, name: &str) -> Result<(Vec<usize>, &[i8]), FormatError> {
        let e = self.require_entry(name)?;
        if e.dtype != TensorDtype::I8 {
            return Err(FormatError::DtypeMismatch {
                name: name.to_string(),
                expected: TensorDtype::I8,
                actual: e.dtype,
            });
        }
        let raw = &self.bytes[e.data_offset as usize..(e.data_offset + e.data_bytes) as usize];
        // SAFETY: `u8` and `i8` have identical alignment (1) and size (1).
        let slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
        Ok((e.dims(), slice))
    }
}

/// Page-0 header of `experts.bin` (4096 bytes).
#[derive(Debug, Clone)]
pub struct ExpertsHeader {
    pub version: u32,
    pub num_layers: u32,
    pub num_experts: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub index_offset: u64,
    pub index_bytes: u64,
    pub data_start_offset: u64,
    pub index_crc32: u32,
    pub expert_payload_bytes: u64,
    pub expert_padded_bytes: u64,
    pub header_crc32: u32,
}

impl ExpertsHeader {
    pub fn parse(page0: &[u8]) -> Result<Self, FormatError> {
        if page0.len() < EXPERTS_HEADER_BYTES {
            return Err(FormatError::InvalidMetadata(format!(
                "experts.bin header too short ({} < {EXPERTS_HEADER_BYTES})",
                page0.len()
            )));
        }
        let slice = &page0[..EXPERTS_HEADER_BYTES];
        let magic: [u8; 4] = slice[0..4].try_into().unwrap();
        if magic != EXPERTS_MAGIC {
            return Err(FormatError::InvalidMagic {
                expected: EXPERTS_MAGIC,
                actual: magic,
            });
        }
        let expected_crc = read_u32_le(slice, 4092);
        let actual_crc = crc32fast::hash(&slice[0..4092]);
        if expected_crc != actual_crc {
            return Err(FormatError::HeaderCrcMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        let version = read_u32_le(slice, 4);
        if version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }

        let num_layers = read_u32_le(slice, 8);
        let num_experts = read_u32_le(slice, 12);
        let hidden_size = read_u32_le(slice, 16);
        let intermediate_size = read_u32_le(slice, 20);
        let index_offset = read_u64_le(slice, 24);
        let index_bytes = read_u64_le(slice, 32);
        let data_start_offset = read_u64_le(slice, 40);
        let index_crc32 = read_u32_le(slice, 48);
        let expert_payload_bytes = read_u64_le(slice, 52);
        let expert_padded_bytes = read_u64_le(slice, 60);

        if num_layers == 0 || num_experts == 0 {
            return Err(FormatError::InvalidMetadata(
                "experts.bin num_layers and num_experts must be > 0".into(),
            ));
        }
        if hidden_size == 0 || hidden_size % 4 != 0 || intermediate_size == 0 || intermediate_size % 4 != 0 {
            return Err(FormatError::InvalidMetadata(format!(
                "experts.bin invalid dimensions D={hidden_size}, I={intermediate_size}"
            )));
        }
        if index_offset != PAGE_ALIGN as u64 {
            return Err(FormatError::AlignmentViolation {
                context: "experts.bin index_offset".into(),
                value: index_offset,
                align: PAGE_ALIGN,
            });
        }
        let expected_index_bytes =
            (num_layers as u64) * (num_experts as u64) * (EXPERT_INDEX_ENTRY_BYTES as u64);
        if index_bytes != expected_index_bytes {
            return Err(FormatError::InvalidMetadata(format!(
                "experts.bin index_bytes ({index_bytes}) != expected ({expected_index_bytes})"
            )));
        }
        let expected_data_start = align_up_u64(index_offset + index_bytes, PAGE_ALIGN as u64);
        if data_start_offset != expected_data_start || data_start_offset % (PAGE_ALIGN as u64) != 0 {
            return Err(FormatError::AlignmentViolation {
                context: "experts.bin data_start_offset".into(),
                value: data_start_offset,
                align: PAGE_ALIGN,
            });
        }

        let expected_payload =
            expected_expert_payload_bytes(hidden_size as usize, intermediate_size as usize) as u64;
        if expert_payload_bytes != expected_payload {
            return Err(FormatError::InvalidMetadata(format!(
                "experts.bin expert_payload_bytes ({expert_payload_bytes}) != expected ({expected_payload})"
            )));
        }
        let expected_padded = align_up_u64(expert_payload_bytes, PAGE_ALIGN as u64);
        if expert_padded_bytes != expected_padded || expert_padded_bytes % (PAGE_ALIGN as u64) != 0 {
            return Err(FormatError::AlignmentViolation {
                context: "experts.bin expert_padded_bytes".into(),
                value: expert_padded_bytes,
                align: PAGE_ALIGN,
            });
        }

        Ok(Self {
            version,
            num_layers,
            num_experts,
            hidden_size,
            intermediate_size,
            index_offset,
            index_bytes,
            data_start_offset,
            index_crc32,
            expert_payload_bytes,
            expert_padded_bytes,
            header_crc32: expected_crc,
        })
    }
}

/// 40-byte `ExpertIndexEntry` in `experts.bin`.
#[derive(Debug, Clone, Copy)]
pub struct ExpertIndexEntry {
    pub layer_id: u16,
    pub expert_id: u16,
    pub flags: u32,
    pub offset: u64,
    pub payload_bytes: u64,
    pub padded_bytes: u64,
    pub crc32: u32,
}

impl ExpertIndexEntry {
    pub fn parse(raw: &[u8]) -> Self {
        debug_assert_eq!(raw.len(), EXPERT_INDEX_ENTRY_BYTES);
        Self {
            layer_id: u16::from_le_bytes(raw[0..2].try_into().unwrap()),
            expert_id: u16::from_le_bytes(raw[2..4].try_into().unwrap()),
            flags: read_u32_le(raw, 4),
            offset: read_u64_le(raw, 8),
            payload_bytes: read_u64_le(raw, 16),
            padded_bytes: read_u64_le(raw, 24),
            crc32: read_u32_le(raw, 32),
        }
    }
}

/// Validated metadata & directory of `experts.bin` (reads only header + index pages into RAM).
#[derive(Debug, Clone)]
pub struct ExpertsDirectory {
    pub header: ExpertsHeader,
    pub entries: Vec<ExpertIndexEntry>,
    pub file_len: u64,
}

impl ExpertsDirectory {
    /// Opens `experts.bin` and validates Page 0 header, index CRC32, zero-padding,
    /// and 4096-byte page alignment of every expert record without loading expert blocks into RAM.
    pub fn open(path: &Path) -> Result<Self, FormatError> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < EXPERTS_HEADER_BYTES as u64 {
            return Err(FormatError::InvalidMetadata(format!(
                "experts.bin too small ({file_len} bytes)"
            )));
        }

        let mut page0 = vec![0u8; EXPERTS_HEADER_BYTES];
        file.read_exact(&mut page0)?;
        let header = ExpertsHeader::parse(&page0)?;

        if file_len < header.data_start_offset {
            return Err(FormatError::InvalidMetadata(format!(
                "experts.bin truncated before data_start_offset ({file_len} < {})",
                header.data_start_offset
            )));
        }

        let index_and_pad_len = (header.data_start_offset - header.index_offset) as usize;
        let mut index_region = vec![0u8; index_and_pad_len];
        file.seek(SeekFrom::Start(header.index_offset))?;
        file.read_exact(&mut index_region)?;

        Self::from_header_and_index_region(header, &index_region, file_len)
    }

    /// Parses and validates an in-memory `experts.bin` buffer, including all expert payload CRC32s.
    pub fn validate_full_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        if bytes.len() < EXPERTS_HEADER_BYTES {
            return Err(FormatError::InvalidMetadata("experts.bin too short".into()));
        }
        let header = ExpertsHeader::parse(&bytes[..EXPERTS_HEADER_BYTES])?;
        if (bytes.len() as u64) < header.data_start_offset {
            return Err(FormatError::InvalidMetadata(
                "experts.bin truncated before data_start_offset".into(),
            ));
        }
        let index_region =
            &bytes[header.index_offset as usize..header.data_start_offset as usize];
        let dir = Self::from_header_and_index_region(header, index_region, bytes.len() as u64)?;
        let packed_ternary_len =
            (3 * (dir.header.hidden_size as usize) * (dir.header.intermediate_size as usize)) / 4;

        for entry in &dir.entries {
            let start = entry.offset as usize;
            let payload_end = start + entry.payload_bytes as usize;
            let padded_end = start + entry.padded_bytes as usize;
            let actual_crc = crc32fast::hash(&bytes[start..payload_end]);
            if actual_crc != entry.crc32 {
                return Err(FormatError::ExpertCrcMismatch {
                    layer_id: entry.layer_id,
                    expert_id: entry.expert_id,
                    expected: entry.crc32,
                    actual: actual_crc,
                });
            }
            validate_no_invalid_crumbs(
                &bytes[start..start + packed_ternary_len],
                &format!("expert({},{})", entry.layer_id, entry.expert_id),
            )?;
            if bytes[payload_end..padded_end].iter().any(|&b| b != 0) {
                return Err(FormatError::InvalidMetadata(format!(
                    "non-zero tail padding in expert ({}, {})",
                    entry.layer_id, entry.expert_id
                )));
            }
        }
        Ok(dir)
    }

    fn from_header_and_index_region(
        header: ExpertsHeader,
        index_region: &[u8],
        file_len: u64,
    ) -> Result<Self, FormatError> {
        let index_bytes = header.index_bytes as usize;
        let index_slice = &index_region[..index_bytes];
        let pad_slice = &index_region[index_bytes..];

        let actual_index_crc = crc32fast::hash(index_slice);
        if actual_index_crc != header.index_crc32 {
            return Err(FormatError::IndexCrcMismatch {
                expected: header.index_crc32,
                actual: actual_index_crc,
            });
        }

        if pad_slice.iter().any(|&b| b != 0) {
            return Err(FormatError::InvalidMetadata(
                "non-zero bytes between index end and data_start_offset in experts.bin".into(),
            ));
        }

        let total = (header.num_layers as usize) * (header.num_experts as usize);
        let mut entries = Vec::with_capacity(total);
        for flat_idx in 0..total {
            let exp_layer = (flat_idx / (header.num_experts as usize)) as u16;
            let exp_expert = (flat_idx % (header.num_experts as usize)) as u16;
            let raw = &index_slice
                [flat_idx * EXPERT_INDEX_ENTRY_BYTES..(flat_idx + 1) * EXPERT_INDEX_ENTRY_BYTES];
            let entry = ExpertIndexEntry::parse(raw);

            if entry.layer_id != exp_layer || entry.expert_id != exp_expert {
                return Err(FormatError::InvalidMetadata(format!(
                    "expert directory out of row-major order at {flat_idx}: expected ({exp_layer}, {exp_expert}), got ({}, {})",
                    entry.layer_id, entry.expert_id
                )));
            }
            if entry.flags != EXPERT_FLAG_TERNARY_SWIGLU {
                return Err(FormatError::InvalidMetadata(format!(
                    "unsupported expert flags {} at ({exp_layer}, {exp_expert})",
                    entry.flags
                )));
            }
            if entry.offset < header.data_start_offset || entry.offset % (PAGE_ALIGN as u64) != 0 {
                return Err(FormatError::AlignmentViolation {
                    context: format!("expert({exp_layer},{exp_expert}) offset"),
                    value: entry.offset,
                    align: PAGE_ALIGN,
                });
            }
            if entry.payload_bytes != header.expert_payload_bytes {
                return Err(FormatError::InvalidMetadata(format!(
                    "expert({exp_layer},{exp_expert}) payload_bytes={} != header {}",
                    entry.payload_bytes, header.expert_payload_bytes
                )));
            }
            if entry.padded_bytes != header.expert_padded_bytes
                || entry.padded_bytes % (PAGE_ALIGN as u64) != 0
                || entry.padded_bytes != align_up_u64(entry.payload_bytes, PAGE_ALIGN as u64)
            {
                return Err(FormatError::AlignmentViolation {
                    context: format!("expert({exp_layer},{exp_expert}) padded_bytes"),
                    value: entry.padded_bytes,
                    align: PAGE_ALIGN,
                });
            }
            let end = entry
                .offset
                .checked_add(entry.padded_bytes)
                .ok_or_else(|| FormatError::InvalidMetadata("expert offset overflow".into()))?;
            if end > file_len {
                return Err(FormatError::InvalidMetadata(format!(
                    "expert({exp_layer},{exp_expert}) range {}..{end} exceeds file size {file_len}",
                    entry.offset
                )));
            }
            entries.push(entry);
        }

        Ok(Self {
            header,
            entries,
            file_len,
        })
    }

    #[inline]
    pub fn entry(&self, layer_id: usize, expert_id: usize) -> Result<&ExpertIndexEntry, FormatError> {
        if layer_id >= self.header.num_layers as usize
            || expert_id >= self.header.num_experts as usize
        {
            return Err(FormatError::InvalidMetadata(format!(
                "expert coordinate ({layer_id}, {expert_id}) out of bounds"
            )));
        }
        let flat = layer_id * (self.header.num_experts as usize) + expert_id;
        Ok(&self.entries[flat])
    }

    /// Validates that `core.bin` and `experts.bin` dimensions match.
    pub fn validate_matches_core(&self, core: &CoreHeader) -> Result<(), FormatError> {
        if self.header.num_layers != core.num_layers
            || self.header.num_experts != core.num_experts
            || self.header.hidden_size != core.hidden_size
            || self.header.intermediate_size != core.intermediate_size
        {
            return Err(FormatError::InvalidMetadata(format!(
                "core.bin / experts.bin shape mismatch: core(L={}, E={}, D={}, I={}) vs experts(L={}, E={}, D={}, I={})",
                core.num_layers,
                core.num_experts,
                core.hidden_size,
                core.intermediate_size,
                self.header.num_layers,
                self.header.num_experts,
                self.header.hidden_size,
                self.header.intermediate_size
            )));
        }
        Ok(())
    }
}

/// Zero-copy view into a 4KB-aligned `AlignedBuffer` holding one expert block
/// laid out per Section 3.3 of `schemas/terra_format.md`.
#[derive(Debug, Clone, Copy)]
pub struct ExpertBlockView<'a> {
    pub gate_packed: &'a [u8],
    pub up_packed: &'a [u8],
    pub down_packed: &'a [u8],
    pub gate_scales: &'a [f32],
    pub up_scales: &'a [f32],
    pub down_scales: &'a [f32],
}

impl<'a> ExpertBlockView<'a> {
    pub fn from_aligned(
        buf: &'a AlignedBuffer,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, FormatError> {
        Self::from_aligned_slice(buf, hidden_size, intermediate_size)
    }

    pub fn from_aligned_slice(
        bytes: &'a [u8],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, FormatError> {
        if hidden_size == 0 || hidden_size % 4 != 0 || intermediate_size == 0 || intermediate_size % 4 != 0 {
            return Err(FormatError::InvalidMetadata(format!(
                "invalid expert dims D={hidden_size}, I={intermediate_size}"
            )));
        }
        let payload_len = expected_expert_payload_bytes(hidden_size, intermediate_size);
        if bytes.len() < payload_len {
            return Err(FormatError::InvalidMetadata(format!(
                "expert buffer too small ({} < {payload_len})",
                bytes.len()
            )));
        }
        if (bytes.as_ptr() as usize) % 4 != 0 {
            return Err(FormatError::AlignmentViolation {
                context: "ExpertBlockView base pointer".into(),
                value: bytes.as_ptr() as u64,
                align: 4,
            });
        }

        let gate_bytes = intermediate_size * (hidden_size / 4);
        let up_bytes = intermediate_size * (hidden_size / 4);
        let down_bytes = hidden_size * (intermediate_size / 4);

        let off0 = 0;
        let off1 = off0 + gate_bytes;
        let off2 = off1 + up_bytes;
        let off3 = off2 + down_bytes;
        let off4 = off3 + intermediate_size * 4;
        let off5 = off4 + intermediate_size * 4;
        let off6 = off5 + hidden_size * 4;
        debug_assert_eq!(off6, payload_len);
        debug_assert_eq!(off3 % 4, 0);

        let gate_packed = &bytes[off0..off1];
        let up_packed = &bytes[off1..off2];
        let down_packed = &bytes[off2..off3];

        // SAFETY: `bytes.as_ptr()` is at least 4-byte aligned (4096-byte aligned for `AlignedBuffer`)
        // and `off3`, `off4`, `off5` are multiples of 4 (`D % 4 == 0 && I % 4 == 0 => 3*D*I/4 % 4 == 0`).
        let gate_scales = unsafe {
            std::slice::from_raw_parts(bytes[off3..off4].as_ptr() as *const f32, intermediate_size)
        };
        let up_scales = unsafe {
            std::slice::from_raw_parts(bytes[off4..off5].as_ptr() as *const f32, intermediate_size)
        };
        let down_scales = unsafe {
            std::slice::from_raw_parts(bytes[off5..off6].as_ptr() as *const f32, hidden_size)
        };

        Ok(Self {
            gate_packed,
            up_packed,
            down_packed,
            gate_scales,
            up_scales,
            down_scales,
        })
    }
}

#[inline]
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

#[inline]
fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}

#[inline]
fn read_f32_le(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

#[inline]
fn bytes_as_f32_cow(raw: &[u8]) -> Cow<'_, [f32]> {
    debug_assert_eq!(raw.len() % 4, 0);
    let count = raw.len() / 4;
    if (raw.as_ptr() as usize) % std::mem::align_of::<f32>() == 0 && cfg!(target_endian = "little")
    {
        // SAFETY: pointer is 4-byte aligned, little-endian, and valid for `count * 4` bytes.
        let slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, count) };
        Cow::Borrowed(slice)
    } else {
        let vec = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Cow::Owned(vec)
    }
}

#[inline]
fn bytes_as_u16_cow(raw: &[u8]) -> Cow<'_, [u16]> {
    debug_assert_eq!(raw.len() % 2, 0);
    let count = raw.len() / 2;
    if (raw.as_ptr() as usize) % std::mem::align_of::<u16>() == 0 && cfg!(target_endian = "little")
    {
        // SAFETY: pointer is 2-byte aligned, little-endian, and valid for `count * 2` bytes.
        let slice = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, count) };
        Cow::Borrowed(slice)
    } else {
        let vec = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Cow::Owned(vec)
    }
}

/// Pure-Rust deterministic synthetic `core.bin` + `experts.bin` builder for standalone testing.
pub mod fixtures {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct SyntheticModelConfig {
        pub vocab_size: usize,
        pub hidden_size: usize,
        pub intermediate_size: usize,
        pub num_layers: usize,
        pub num_attention_heads: usize,
        pub num_kv_heads: usize,
        pub head_dim: usize,
        pub num_experts: usize,
        pub top_k: usize,
        pub max_position_embeddings: usize,
        pub rope_theta: f32,
        pub rms_norm_eps: f32,
        pub lookahead_offset: usize,
        pub flags: u32,
        pub layer_kinds: Vec<LayerKind>,
        pub include_lookahead_routers: bool,
        pub hint_buckets: usize,
        pub hint_dim: usize,
        pub seed: u64,
    }

    impl Default for SyntheticModelConfig {
        fn default() -> Self {
            let num_layers = 8;
            let mut layer_kinds = vec![LayerKind::GatedLinearRecurrence; num_layers];
            layer_kinds[num_layers - 1] = LayerKind::AnchorGqa;
            Self {
                vocab_size: 64,
                hidden_size: 64,
                intermediate_size: 128,
                num_layers,
                num_attention_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
                num_experts: 8,
                top_k: 2,
                max_position_embeddings: 2048,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-5,
                lookahead_offset: 3,
                flags: FLAG_BITLINEAR_RMS_PRESCALE | FLAG_NORM_TOPK_PROB,
                layer_kinds,
                include_lookahead_routers: true,
                hint_buckets: 32,
                hint_dim: 8,
                seed: 0x5EED_CAFE_1234_5678,
            }
        }
    }

    struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
            }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_f32(&mut self, scale: f32) -> f32 {
            let u = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
            (u * 2.0 - 1.0) * scale
        }

        fn vec_f32(&mut self, len: usize, scale: f32) -> Vec<f32> {
            (0..len).map(|_| self.next_f32(scale)).collect()
        }
    }

    enum PendingPayload {
        F32 { shape: [u32; 4], ndim: u8, vals: Vec<f32> },
        F16 { shape: [u32; 4], ndim: u8, vals: Vec<f32> },
        Ternary { rows: usize, cols: usize, packed: Vec<u8>, scales: Vec<f32> },
    }

    /// Serializes a complete synthetic `(core_bytes, experts_bytes)` pair matching `schemas/terra_format.md`.
    pub fn build_synthetic_model_bytes(
        cfg: &SyntheticModelConfig,
    ) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
        let mut rng = XorShift64::new(cfg.seed);
        let d = cfg.hidden_size;
        let h = cfg.num_attention_heads;
        let h_kv = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        assert_eq!(h * hd, d, "H * head_dim must equal hidden_size");
        assert_eq!(cfg.layer_kinds.len(), cfg.num_layers);

        let mut tensors: Vec<(String, PendingPayload)> = Vec::new();

        // Embeddings (`F16` to exercise F16 decoding path too)
        let embed_vals = rng.vec_f32(cfg.vocab_size * d, 0.25);
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            PendingPayload::F16 {
                shape: [cfg.vocab_size as u32, d as u32, 0, 0],
                ndim: 2,
                vals: embed_vals,
            },
        ));

        let norm_vals: Vec<f32> = (0..d).map(|_| 1.0 + rng.next_f32(0.1)).collect();
        tensors.push((
            "model.norm.weight".to_string(),
            PendingPayload::F32 {
                shape: [d as u32, 0, 0, 0],
                ndim: 1,
                vals: norm_vals,
            },
        ));

        if (cfg.flags & FLAG_TIE_WORD_EMBEDDINGS) == 0 {
            let lm_head_w = rng.vec_f32(cfg.vocab_size * d, 0.2);
            let (packed, scales) = quantize_pack_ternary_absmean(&lm_head_w, cfg.vocab_size, d)?;
            tensors.push((
                "lm_head.weight".to_string(),
                PendingPayload::Ternary {
                    rows: cfg.vocab_size,
                    cols: d,
                    packed,
                    scales,
                },
            ));
        }

        if (cfg.flags & FLAG_HAS_HINT_EMBEDDING) != 0 {
            tensors.push((
                "model.hint_embedding.table.weight".to_string(),
                PendingPayload::F32 {
                    shape: [cfg.hint_buckets as u32, cfg.hint_dim as u32, 0, 0],
                    ndim: 2,
                    vals: rng.vec_f32(cfg.hint_buckets * cfg.hint_dim, 0.15),
                },
            ));
            tensors.push((
                "model.hint_embedding.proj.weight".to_string(),
                PendingPayload::F32 {
                    shape: [d as u32, (4 * cfg.hint_dim) as u32, 0, 0],
                    ndim: 2,
                    vals: rng.vec_f32(d * 4 * cfg.hint_dim, 0.15),
                },
            ));
        }

        // Pre-generate MoE gate weights for all layers so lookahead_router[l] can correlate
        // strongly with mlp.gate[l + lookahead_offset] to exercise realistic high-recall prefetching!
        let mut gate_weights_per_layer: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_layers);
        for _l in 0..cfg.num_layers {
            gate_weights_per_layer.push(rng.vec_f32(cfg.num_experts * d, 0.5));
        }

        for l in 0..cfg.num_layers {
            let in_ln: Vec<f32> = (0..d).map(|_| 1.0 + rng.next_f32(0.08)).collect();
            let post_ln: Vec<f32> = (0..d).map(|_| 1.0 + rng.next_f32(0.08)).collect();
            tensors.push((
                format!("layers.{l}.input_layernorm.weight"),
                PendingPayload::F32 {
                    shape: [d as u32, 0, 0, 0],
                    ndim: 1,
                    vals: in_ln,
                },
            ));
            tensors.push((
                format!("layers.{l}.post_attention_layernorm.weight"),
                PendingPayload::F32 {
                    shape: [d as u32, 0, 0, 0],
                    ndim: 1,
                    vals: post_ln,
                },
            ));

            for (proj_name, rows, cols) in [
                ("q_proj", h * hd, d),
                ("k_proj", h_kv * hd, d),
                ("v_proj", h_kv * hd, d),
                ("o_proj", d, h * hd),
            ] {
                let w = rng.vec_f32(rows * cols, 0.2);
                let (packed, scales) = quantize_pack_ternary_absmean(&w, rows, cols)?;
                tensors.push((
                    format!("layers.{l}.self_attn.{proj_name}.weight"),
                    PendingPayload::Ternary {
                        rows,
                        cols,
                        packed,
                        scales,
                    },
                ));
            }

            let q_norm: Vec<f32> = (0..(h * hd)).map(|_| 1.0 + rng.next_f32(0.05)).collect();
            let k_norm: Vec<f32> = (0..(h_kv * hd)).map(|_| 1.0 + rng.next_f32(0.05)).collect();
            tensors.push((
                format!("layers.{l}.self_attn.q_norm.weight"),
                PendingPayload::F32 {
                    shape: [(h * hd) as u32, 0, 0, 0],
                    ndim: 1,
                    vals: q_norm,
                },
            ));
            tensors.push((
                format!("layers.{l}.self_attn.k_norm.weight"),
                PendingPayload::F32 {
                    shape: [(h_kv * hd) as u32, 0, 0, 0],
                    ndim: 1,
                    vals: k_norm,
                },
            ));

            if cfg.layer_kinds[l] == LayerKind::GatedLinearRecurrence {
                tensors.push((
                    format!("layers.{l}.self_attn.decay_proj.weight"),
                    PendingPayload::F32 {
                        shape: [h as u32, d as u32, 0, 0],
                        ndim: 2,
                        vals: rng.vec_f32(h * d, 0.15),
                    },
                ));
                tensors.push((
                    format!("layers.{l}.self_attn.decay_proj.bias"),
                    PendingPayload::F32 {
                        shape: [h as u32, 0, 0, 0],
                        ndim: 1,
                        vals: vec![1.0f32; h],
                    },
                ));
            }

            tensors.push((
                format!("layers.{l}.mlp.gate.weight"),
                PendingPayload::F32 {
                    shape: [cfg.num_experts as u32, d as u32, 0, 0],
                    ndim: 2,
                    vals: gate_weights_per_layer[l].clone(),
                },
            ));

            if cfg.include_lookahead_routers && l + cfg.lookahead_offset < cfg.num_layers {
                let target_gate = &gate_weights_per_layer[l + cfg.lookahead_offset];
                let lookahead_vals: Vec<f32> = target_gate
                    .iter()
                    .map(|&v| v + rng.next_f32(0.02))
                    .collect();
                tensors.push((
                    format!("layers.{l}.lookahead_router.weight"),
                    PendingPayload::F32 {
                        shape: [cfg.num_experts as u32, d as u32, 0, 0],
                        ndim: 2,
                        vals: lookahead_vals,
                    },
                ));
            }
        }

        let core_bytes = serialize_core_file(cfg, &tensors)?;
        let experts_bytes = serialize_experts_file(cfg, &mut rng)?;
        Ok((core_bytes, experts_bytes))
    }

    pub fn write_synthetic_model_dir(
        dir: &Path,
        cfg: &SyntheticModelConfig,
    ) -> Result<( std::path::PathBuf, std::path::PathBuf ), FormatError> {
        std::fs::create_dir_all(dir)?;
        let (core_bytes, experts_bytes) = build_synthetic_model_bytes(cfg)?;
        let core_path = dir.join("core.bin");
        let experts_path = dir.join("experts.bin");
        std::fs::write(&core_path, core_bytes)?;
        std::fs::write(&experts_path, experts_bytes)?;
        Ok((core_path, experts_path))
    }

    fn serialize_core_file(
        cfg: &SyntheticModelConfig,
        tensors: &[(String, PendingPayload)],
    ) -> Result<Vec<u8>, FormatError> {
        let tensor_count = tensors.len();
        let dir_end = CORE_HEADER_BYTES + tensor_count * CORE_TENSOR_ENTRY_BYTES;
        let data_section_offset = align_up(dir_end, CORE_DATA_ALIGN);

        let mut data_buf: Vec<u8> = Vec::new();
        let mut entries_buf = vec![0u8; tensor_count * CORE_TENSOR_ENTRY_BYTES];

        for (idx, (name, payload)) in tensors.iter().enumerate() {
            let pad = align_up(data_section_offset + data_buf.len(), CORE_DATA_ALIGN)
                - (data_section_offset + data_buf.len());
            data_buf.resize(data_buf.len() + pad, 0);
            let data_offset = (data_section_offset + data_buf.len()) as u64;

            let (dtype, ndim, shape, data_bytes, scale_offset, scale_bytes, data_crc, scale_crc) =
                match payload {
                    PendingPayload::F32 { shape, ndim, vals } => {
                        let start = data_buf.len();
                        for &v in vals {
                            data_buf.extend_from_slice(&v.to_le_bytes());
                        }
                        let slice = &data_buf[start..];
                        let crc = crc32fast::hash(slice);
                        (
                            TensorDtype::F32,
                            *ndim,
                            *shape,
                            slice.len() as u64,
                            0u64,
                            0u64,
                            crc,
                            0u32,
                        )
                    }
                    PendingPayload::F16 { shape, ndim, vals } => {
                        let start = data_buf.len();
                        for &v in vals {
                            let bits = f16::from_f32(v).to_bits();
                            data_buf.extend_from_slice(&bits.to_le_bytes());
                        }
                        let slice = &data_buf[start..];
                        let crc = crc32fast::hash(slice);
                        (
                            TensorDtype::F16,
                            *ndim,
                            *shape,
                            slice.len() as u64,
                            0u64,
                            0u64,
                            crc,
                            0u32,
                        )
                    }
                    PendingPayload::Ternary {
                        rows,
                        cols,
                        packed,
                        scales,
                    } => {
                        let start = data_buf.len();
                        data_buf.extend_from_slice(packed);
                        let data_len = packed.len() as u64;
                        let data_crc = crc32fast::hash(&data_buf[start..]);

                        let scale_pad =
                            align_up(data_section_offset + data_buf.len(), CORE_DATA_ALIGN)
                                - (data_section_offset + data_buf.len());
                        data_buf.resize(data_buf.len() + scale_pad, 0);
                        let s_offset = (data_section_offset + data_buf.len()) as u64;
                        let s_start = data_buf.len();
                        for &s in scales {
                            data_buf.extend_from_slice(&s.to_le_bytes());
                        }
                        let s_slice = &data_buf[s_start..];
                        let s_crc = crc32fast::hash(s_slice);

                        (
                            TensorDtype::Ternary2Bit,
                            2u8,
                            [*rows as u32, *cols as u32, 0, 0],
                            data_len,
                            s_offset,
                            s_slice.len() as u64,
                            data_crc,
                            s_crc,
                        )
                    }
                };

            let rec = &mut entries_buf
                [idx * CORE_TENSOR_ENTRY_BYTES..(idx + 1) * CORE_TENSOR_ENTRY_BYTES];
            let name_bytes = name.as_bytes();
            if name_bytes.len() > 64 {
                return Err(FormatError::InvalidMetadata(format!(
                    "tensor name `{name}` exceeds 64 bytes"
                )));
            }
            rec[..name_bytes.len()].copy_from_slice(name_bytes);
            rec[64] = dtype as u8;
            rec[65] = ndim;
            rec[66..68].copy_from_slice(&0u16.to_le_bytes());
            for s_i in 0..4 {
                rec[68 + s_i * 4..72 + s_i * 4].copy_from_slice(&shape[s_i].to_le_bytes());
            }
            rec[84..92].copy_from_slice(&data_offset.to_le_bytes());
            rec[92..100].copy_from_slice(&data_bytes.to_le_bytes());
            rec[100..108].copy_from_slice(&scale_offset.to_le_bytes());
            rec[108..116].copy_from_slice(&scale_bytes.to_le_bytes());
            rec[116..120].copy_from_slice(&data_crc.to_le_bytes());
            rec[120..124].copy_from_slice(&scale_crc.to_le_bytes());
            rec[124..128].copy_from_slice(&0u32.to_le_bytes());
        }

        let mut header = vec![0u8; CORE_HEADER_BYTES];
        header[0..4].copy_from_slice(&CORE_MAGIC);
        header[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[8..12].copy_from_slice(&cfg.flags.to_le_bytes());
        header[12..16].copy_from_slice(&(cfg.vocab_size as u32).to_le_bytes());
        header[16..20].copy_from_slice(&(cfg.hidden_size as u32).to_le_bytes());
        header[20..24].copy_from_slice(&(cfg.intermediate_size as u32).to_le_bytes());
        header[24..28].copy_from_slice(&(cfg.num_layers as u32).to_le_bytes());
        header[28..32].copy_from_slice(&(cfg.num_attention_heads as u32).to_le_bytes());
        header[32..36].copy_from_slice(&(cfg.num_kv_heads as u32).to_le_bytes());
        header[36..40].copy_from_slice(&(cfg.head_dim as u32).to_le_bytes());
        header[40..44].copy_from_slice(&(cfg.num_experts as u32).to_le_bytes());
        header[44..48].copy_from_slice(&(cfg.top_k as u32).to_le_bytes());
        header[48..52].copy_from_slice(&(cfg.max_position_embeddings as u32).to_le_bytes());
        header[52..56].copy_from_slice(&cfg.rope_theta.to_le_bytes());
        header[56..60].copy_from_slice(&cfg.rms_norm_eps.to_le_bytes());
        header[60..64].copy_from_slice(&(cfg.lookahead_offset as u32).to_le_bytes());
        for (l, &kind) in cfg.layer_kinds.iter().enumerate() {
            header[64 + l] = kind as u8;
        }
        header[128..132].copy_from_slice(&(tensor_count as u32).to_le_bytes());
        header[132..140].copy_from_slice(&(data_section_offset as u64).to_le_bytes());
        let header_crc = crc32fast::hash(&header[0..252]);
        header[252..256].copy_from_slice(&header_crc.to_le_bytes());

        let mut file_bytes = Vec::with_capacity(data_section_offset + data_buf.len());
        file_bytes.extend_from_slice(&header);
        file_bytes.extend_from_slice(&entries_buf);
        file_bytes.resize(data_section_offset, 0);
        file_bytes.extend_from_slice(&data_buf);
        Ok(file_bytes)
    }

    fn serialize_experts_file(
        cfg: &SyntheticModelConfig,
        rng: &mut XorShift64,
    ) -> Result<Vec<u8>, FormatError> {
        let d = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let total_experts = cfg.num_layers * cfg.num_experts;
        let index_offset = PAGE_ALIGN;
        let index_bytes = total_experts * EXPERT_INDEX_ENTRY_BYTES;
        let data_start_offset = align_up(index_offset + index_bytes, PAGE_ALIGN);
        let payload_bytes = expected_expert_payload_bytes(d, inter);
        let padded_bytes = align_up(payload_bytes, PAGE_ALIGN);

        let total_file_bytes = data_start_offset + total_experts * padded_bytes;
        let mut out = vec![0u8; total_file_bytes];

        let mut index_buf = vec![0u8; index_bytes];

        for l in 0..cfg.num_layers {
            for e in 0..cfg.num_experts {
                let flat = l * cfg.num_experts + e;
                let offset = data_start_offset + flat * padded_bytes;

                let gate_w = rng.vec_f32(inter * d, 0.25);
                let up_w = rng.vec_f32(inter * d, 0.25);
                let down_w = rng.vec_f32(d * inter, 0.2);

                let (gate_packed, gate_scales) = quantize_pack_ternary_absmean(&gate_w, inter, d)?;
                let (up_packed, up_scales) = quantize_pack_ternary_absmean(&up_w, inter, d)?;
                let (down_packed, down_scales) = quantize_pack_ternary_absmean(&down_w, d, inter)?;

                let block = &mut out[offset..offset + payload_bytes];
                let mut cursor = 0usize;
                for slice in [&gate_packed[..], &up_packed[..], &down_packed[..]] {
                    block[cursor..cursor + slice.len()].copy_from_slice(slice);
                    cursor += slice.len();
                }
                for scales in [&gate_scales[..], &up_scales[..], &down_scales[..]] {
                    for &s in scales {
                        block[cursor..cursor + 4].copy_from_slice(&s.to_le_bytes());
                        cursor += 4;
                    }
                }
                debug_assert_eq!(cursor, payload_bytes);
                let crc = crc32fast::hash(block);

                let rec = &mut index_buf
                    [flat * EXPERT_INDEX_ENTRY_BYTES..(flat + 1) * EXPERT_INDEX_ENTRY_BYTES];
                rec[0..2].copy_from_slice(&(l as u16).to_le_bytes());
                rec[2..4].copy_from_slice(&(e as u16).to_le_bytes());
                rec[4..8].copy_from_slice(&EXPERT_FLAG_TERNARY_SWIGLU.to_le_bytes());
                rec[8..16].copy_from_slice(&(offset as u64).to_le_bytes());
                rec[16..24].copy_from_slice(&(payload_bytes as u64).to_le_bytes());
                rec[24..32].copy_from_slice(&(padded_bytes as u64).to_le_bytes());
                rec[32..36].copy_from_slice(&crc.to_le_bytes());
                rec[36..40].copy_from_slice(&0u32.to_le_bytes());
            }
        }

        let index_crc = crc32fast::hash(&index_buf);
        out[index_offset..index_offset + index_bytes].copy_from_slice(&index_buf);

        let page0 = &mut out[0..EXPERTS_HEADER_BYTES];
        page0[0..4].copy_from_slice(&EXPERTS_MAGIC);
        page0[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        page0[8..12].copy_from_slice(&(cfg.num_layers as u32).to_le_bytes());
        page0[12..16].copy_from_slice(&(cfg.num_experts as u32).to_le_bytes());
        page0[16..20].copy_from_slice(&(d as u32).to_le_bytes());
        page0[20..24].copy_from_slice(&(inter as u32).to_le_bytes());
        page0[24..32].copy_from_slice(&(index_offset as u64).to_le_bytes());
        page0[32..40].copy_from_slice(&(index_bytes as u64).to_le_bytes());
        page0[40..48].copy_from_slice(&(data_start_offset as u64).to_le_bytes());
        page0[48..52].copy_from_slice(&index_crc.to_le_bytes());
        page0[52..60].copy_from_slice(&(payload_bytes as u64).to_le_bytes());
        page0[60..68].copy_from_slice(&(padded_bytes as u64).to_le_bytes());
        let header_crc = crc32fast::hash(&page0[0..4092]);
        page0[4092..4096].copy_from_slice(&header_crc.to_le_bytes());

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_ieee_vector() {
        assert_eq!(crc32fast::hash(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_aligned_buffer_4096_invariant() {
        for req in [1usize, 128, 4096, 4097, 1_589_248] {
            let mut buf = AlignedBuffer::new_zeroed(req).unwrap();
            assert_eq!((buf.as_ptr() as usize) % PAGE_ALIGN, 0);
            assert_eq!((buf.as_mut_ptr() as usize) % PAGE_ALIGN, 0);
            assert_eq!(buf.len() % PAGE_ALIGN, 0);
            assert!(buf.len() >= req);
            assert!(buf.is_page_aligned());
            buf[0] = 0xAB;
            let last = buf.len() - 1;
            buf[last] = 0xCD;
            let cloned = buf.clone();
            assert!(cloned.is_page_aligned());
            assert_eq!(cloned[0], 0xAB);
            assert_eq!(cloned[cloned.len() - 1], 0xCD);
        }
        assert!(AlignedBuffer::new_exact(0).is_err());
        assert!(AlignedBuffer::new_exact(4095).is_err());
    }

    #[test]
    fn test_rejects_invalid_0b11_crumb_and_corrupted_crc_and_alignment() {
        let cfg = fixtures::SyntheticModelConfig::default();
        let (core_bytes, experts_bytes) = fixtures::build_synthetic_model_bytes(&cfg).unwrap();

        let core = CoreFile::from_bytes(core_bytes.clone()).unwrap();
        let dir = ExpertsDirectory::validate_full_bytes(&experts_bytes).unwrap();
        dir.validate_matches_core(&core.header).unwrap();

        // 1. Corrupt a ternary tensor with an invalid 0b11 crumb and update its CRC so crumb check triggers
        let q_entry = core.require_entry("layers.0.self_attn.q_proj.weight").unwrap();
        let mut bad_crumb_core = core_bytes.clone();
        bad_crumb_core[q_entry.data_offset as usize] |= 0b0000_0011;
        let new_data_crc = crc32fast::hash(
            &bad_crumb_core
                [q_entry.data_offset as usize..(q_entry.data_offset + q_entry.data_bytes) as usize],
        );
        // Find entry 3 in directory and patch CRC so it reaches `validate_no_invalid_crumbs`
        for idx in 0..(core.header.tensor_count as usize) {
            let start = CORE_HEADER_BYTES + idx * CORE_TENSOR_ENTRY_BYTES;
            let parsed =
                CoreTensorEntry::parse(&bad_crumb_core[start..start + CORE_TENSOR_ENTRY_BYTES])
                    .unwrap();
            if parsed.name == "layers.0.self_attn.q_proj.weight" {
                bad_crumb_core[start + 116..start + 120]
                    .copy_from_slice(&new_data_crc.to_le_bytes());
                break;
            }
        }
        let err = CoreFile::from_bytes(bad_crumb_core).unwrap_err();
        assert!(matches!(err, FormatError::InvalidTernaryCrumb { .. }), "got {err:?}");

        // 2. Corrupt expert block payload -> must reject with ExpertCrcMismatch
        let mut bad_crc_experts = experts_bytes.clone();
        let e0_off = dir.entries[0].offset as usize;
        bad_crc_experts[e0_off] ^= 0x01;
        let err = ExpertsDirectory::validate_full_bytes(&bad_crc_experts).unwrap_err();
        assert!(matches!(err, FormatError::ExpertCrcMismatch { .. }), "got {err:?}");

        // 3. Unaligned expert offset in directory -> must reject with AlignmentViolation
        let mut unaligned_experts = experts_bytes.clone();
        let unaligned_off = dir.entries[0].offset + 64; // not multiple of 4096!
        unaligned_experts[PAGE_ALIGN + 8..PAGE_ALIGN + 16]
            .copy_from_slice(&unaligned_off.to_le_bytes());
        let new_idx_crc = crc32fast::hash(
            &unaligned_experts[PAGE_ALIGN..PAGE_ALIGN + dir.header.index_bytes as usize],
        );
        unaligned_experts[48..52].copy_from_slice(&new_idx_crc.to_le_bytes());
        let new_hdr_crc = crc32fast::hash(&unaligned_experts[0..4092]);
        unaligned_experts[4092..4096].copy_from_slice(&new_hdr_crc.to_le_bytes());
        let err = ExpertsDirectory::validate_full_bytes(&unaligned_experts).unwrap_err();
        assert!(matches!(err, FormatError::AlignmentViolation { .. }), "got {err:?}");
    }
}
