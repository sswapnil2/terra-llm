//! Async Direct-DMA NVMe Expert Reader (`io_uring` + `O_DIRECT` on Linux, `F_NOCACHE` + `pread`
//! SQE/CQE worker pool on macOS) and RAM-Budgeted `ExpertLruPool` (`schemas/terra_format.md` Section 3).

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::format::{
    validate_no_invalid_crumbs, AlignedBuffer, ExpertBlockView, ExpertsDirectory, FormatError,
    PAGE_ALIGN,
};

#[derive(Debug, Error)]
pub enum NvmeError {
    #[error("NVMe I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("format/validation error: {0}")]
    Format(#[from] FormatError),

    #[error("short DMA read on req_id={req_id}: expected {expected} bytes, got {actual}")]
    ShortRead {
        req_id: u64,
        expected: usize,
        actual: usize,
    },

    #[error("insufficient RAM budget ({budget_bytes} B): core+state require {fixed_bytes} B, need at least {min_expert_bytes} B for top_k={top_k} experts")]
    InsufficientRamBudget {
        budget_bytes: usize,
        fixed_bytes: usize,
        min_expert_bytes: usize,
        top_k: usize,
    },

    #[error("invalid pool capacity {capacity} (must be >= top_k={top_k})")]
    InvalidPoolCapacity { capacity: usize, top_k: usize },

    #[error("expert ({layer_id}, {expert_id}) is not resident after acquire")]
    ExpertNotResident { layer_id: usize, expert_id: usize },
}

pub struct CompletedRead {
    pub req_id: u64,
    pub buf: AlignedBuffer,
    pub result: Result<usize, std::io::Error>,
}

/// Asynchronous 4KB-aligned block reader abstraction over Linux `io_uring` (`O_DIRECT`)
/// and macOS `F_NOCACHE` + `pread` SQE/CQE worker queues.
pub trait AsyncBlockReader: Send {
    /// Submits an aligned read. If submission fails before transfer begins, returns `(err, buf)`
    /// so the caller reclaims the page-aligned buffer without leaking pool capacity.
    fn submit_read(
        &mut self,
        req_id: u64,
        offset: u64,
        padded_len: usize,
        buf: AlignedBuffer,
    ) -> Result<(), (NvmeError, AlignedBuffer)>;

    fn reap_completions(&mut self, blocking_min: usize) -> Result<Vec<CompletedRead>, NvmeError>;

    fn in_flight_count(&self) -> usize;

    fn direct_io_active(&self) -> bool;

    fn backend_name(&self) -> &'static str;
}

// ============================================================================
// Linux `io_uring` + `O_DIRECT` Implementation
// ============================================================================

#[cfg(target_os = "linux")]
pub struct IoUringDirectReader {
    file: File,
    ring: io_uring::IoUring,
    in_flight: HashMap<u64, (AlignedBuffer, usize)>,
    direct_io: bool,
}

#[cfg(target_os = "linux")]
impl IoUringDirectReader {
    pub fn open(path: &Path, queue_depth: usize) -> Result<Self, NvmeError> {
        use std::os::unix::fs::OpenOptionsExt;

        let depth = queue_depth.max(8).next_power_of_two() as u32;
        // Open with O_DIRECT for unbuffered page-aligned NVMe DMA.
        // Fall back only if tmpfs in CI explicitly rejects EINVAL on O_DIRECT?
        // Per guardrail #1/#2, open directly with O_DIRECT on real filesystems; if tmpfs returns EINVAL,
        // wait: let's try O_DIRECT first and report direct_io = true!
        let (file, direct_io) = match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(f) => (f, true),
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                // Note: Linux tmpfs (`/tmp` in memory-only containers) returns EINVAL on O_DIRECT;
                // real ext4/xfs/nvme filesystems succeed with `direct_io = true`.
                let f = std::fs::OpenOptions::new().read(true).open(path)?;
                (f, false)
            }
            Err(e) => return Err(NvmeError::Io(e)),
        };

        let ring = io_uring::IoUring::new(depth)?;
        Ok(Self {
            file,
            ring,
            in_flight: HashMap::with_capacity(depth as usize),
            direct_io,
        })
    }
}

#[cfg(target_os = "linux")]
impl IoUringDirectReader {
    fn drain_ready_cqes(&mut self, out: &mut Vec<CompletedRead>) {
        let cq = self.ring.completion();
        for cqe in cq {
            let req_id = cqe.user_data();
            let res = cqe.result();
            if let Some((buf, expected_len)) = self.in_flight.remove(&req_id) {
                let result = if res < 0 {
                    Err(std::io::Error::from_raw_os_error(-res))
                } else if (res as usize) != expected_len {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "short io_uring read on req_id={req_id}: expected {expected_len} B, got {res} B"
                        ),
                    ))
                } else {
                    Ok(res as usize)
                };
                out.push(CompletedRead {
                    req_id,
                    buf,
                    result,
                });
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl AsyncBlockReader for IoUringDirectReader {
    fn submit_read(
        &mut self,
        req_id: u64,
        offset: u64,
        padded_len: usize,
        mut buf: AlignedBuffer,
    ) -> Result<(), (NvmeError, AlignedBuffer)> {
        if offset % (PAGE_ALIGN as u64) != 0 {
            return Err((
                NvmeError::Format(FormatError::AlignmentViolation {
                    context: "IoUringDirectReader submit offset".into(),
                    value: offset,
                    align: PAGE_ALIGN,
                }),
                buf,
            ));
        }
        if padded_len == 0
            || padded_len % PAGE_ALIGN != 0
            || !buf.is_page_aligned()
            || buf.len() < padded_len
        {
            return Err((
                NvmeError::Format(FormatError::AlignmentViolation {
                    context: "IoUringDirectReader buffer/length".into(),
                    value: padded_len as u64,
                    align: PAGE_ALIGN,
                }),
                buf,
            ));
        }

        let read_op = io_uring::opcode::Read::new(
            io_uring::types::Fd(self.file.as_raw_fd()),
            buf.as_mut_ptr(),
            padded_len as u32,
        )
        .offset(offset)
        .build()
        .user_data(req_id);

        // SAFETY: `buf` is heap-allocated at a stable 4KB-aligned address. Once pushed onto the SQ ring,
        // `buf` is moved into `self.in_flight` and NEVER freed until its CQE is reaped (or intentionally
        // leaked via `mem::forget` if the kernel ring encounters an unrecoverable teardown error).
        unsafe {
            if self.ring.submission().push(&read_op).is_err() {
                loop {
                    match self.ring.submit() {
                        Ok(_) => break,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => return Err((NvmeError::Io(e), buf)),
                    }
                }
                if self.ring.submission().push(&read_op).is_err() {
                    return Err((
                        NvmeError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "io_uring SQ full",
                        )),
                        buf,
                    ));
                }
            }
        }

        self.in_flight.insert(req_id, (buf, padded_len));

        loop {
            match self.ring.submit() {
                Ok(_) => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // Note: SQE is already queued in kernel shared memory (`self.in_flight` retains `buf`
                // so `Drop` will safely await or `mem::forget` it rather than freeing under DMA).
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn reap_completions(&mut self, blocking_min: usize) -> Result<Vec<CompletedRead>, NvmeError> {
        let mut out = Vec::new();
        self.drain_ready_cqes(&mut out);

        while out.len() < blocking_min && !self.in_flight.is_empty() {
            let need = (blocking_min - out.len()).min(self.in_flight.len());
            match self.ring.submit_and_wait(need) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    if out.is_empty() {
                        return Err(NvmeError::Io(e));
                    }
                    break;
                }
            }
            self.drain_ready_cqes(&mut out);
        }

        Ok(out)
    }

    fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    fn direct_io_active(&self) -> bool {
        self.direct_io
    }

    fn backend_name(&self) -> &'static str {
        "io_uring+O_DIRECT"
    }
}

#[cfg(target_os = "linux")]
impl Drop for IoUringDirectReader {
    fn drop(&mut self) {
        let mut drained = Vec::new();
        self.drain_ready_cqes(&mut drained);
        drop(drained);

        // Wait for all in-flight kernel DMAs to finish before freeing `AlignedBuffer` pages.
        while !self.in_flight.is_empty() {
            match self.ring.submit_and_wait(1) {
                Ok(_) => {
                    let mut batch = Vec::new();
                    self.drain_ready_cqes(&mut batch);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    // Kernel ring entered an unrecoverable state while SQEs may still reference
                    // these pages: leak the remaining buffers intentionally to prevent DMA use-after-free.
                    for (_, (buf, _)) in self.in_flight.drain() {
                        std::mem::forget(buf);
                    }
                    break;
                }
            }
        }
    }
}

// ============================================================================
// macOS `F_NOCACHE` + Portable Bounded SQE/CQE `pread` Worker Pool
// ============================================================================

struct SqeReadTask {
    req_id: u64,
    offset: u64,
    padded_len: usize,
    buf: AlignedBuffer,
}

struct SharedPreadQueues {
    sq: Mutex<(VecDeque<SqeReadTask>, bool)>, // (queue, shutdown)
    sq_nonempty: Condvar,
    sq_notfull: Condvar,
    cq: Mutex<VecDeque<CompletedRead>>,
    cq_ready: Condvar,
    sq_capacity: usize,
}

pub struct PreadDirectReader {
    shared: Arc<SharedPreadQueues>,
    workers: Vec<JoinHandle<()>>,
    in_flight: usize,
    nocache_enabled: bool,
    _file: Arc<File>,
}

impl PreadDirectReader {
    pub fn open(path: &Path, queue_depth: usize, num_workers: usize) -> Result<Self, NvmeError> {
        #[cfg(target_os = "macos")]
        let (file, nocache_enabled) = {
            let f = File::open(path)?;
            // Bypass Darwin unified buffer cache via `F_NOCACHE`.
            // SAFETY: `f.as_raw_fd()` is a valid open file descriptor.
            let rc = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
            if rc == -1 {
                return Err(NvmeError::Io(std::io::Error::last_os_error()));
            }
            (f, true)
        };

        #[cfg(target_os = "linux")]
        let (file, nocache_enabled) = {
            use std::os::unix::fs::OpenOptionsExt;
            match std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
            {
                Ok(f) => (f, true),
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                    let f = File::open(path)?;
                    (f, false)
                }
                Err(e) => return Err(NvmeError::Io(e)),
            }
        };

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        let (file, nocache_enabled) = (File::open(path)?, false);

        let file_arc = Arc::new(file);
        let sq_capacity = queue_depth.max(8);
        let worker_count = num_workers.max(1);

        let shared = Arc::new(SharedPreadQueues {
            sq: Mutex::new((VecDeque::with_capacity(sq_capacity), false)),
            sq_nonempty: Condvar::new(),
            sq_notfull: Condvar::new(),
            cq: Mutex::new(VecDeque::with_capacity(sq_capacity)),
            cq_ready: Condvar::new(),
            sq_capacity,
        });

        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let q = Arc::clone(&shared);
            let f = Arc::clone(&file_arc);
            workers.push(thread::spawn(move || pread_worker_loop(f, q)));
        }

        Ok(Self {
            shared,
            workers,
            in_flight: 0,
            nocache_enabled,
            _file: file_arc,
        })
    }
}

fn pread_worker_loop(file: Arc<File>, shared: Arc<SharedPreadQueues>) {
    let fd = file.as_raw_fd();
    loop {
        let task = {
            let mut guard = shared.sq.lock().unwrap();
            while guard.0.is_empty() && !guard.1 {
                guard = shared.sq_nonempty.wait(guard).unwrap();
            }
            if let Some(t) = guard.0.pop_front() {
                shared.sq_notfull.notify_one();
                t
            } else {
                // Shutdown requested and queue drained.
                return;
            }
        };

        let SqeReadTask {
            req_id,
            offset,
            padded_len,
            mut buf,
        } = task;

        let result = pread_exact_into_aligned(fd, &mut buf[..padded_len], offset);
        {
            let mut cq_guard = shared.cq.lock().unwrap();
            cq_guard.push_back(CompletedRead {
                req_id,
                buf,
                result,
            });
            shared.cq_ready.notify_all();
        }
    }
}

fn pread_exact_into_aligned(
    fd: libc::c_int,
    dst: &mut [u8],
    offset: u64,
) -> Result<usize, std::io::Error> {
    let mut total = 0usize;
    while total < dst.len() {
        let rem = dst.len() - total;
        let cur_off = (offset + total as u64) as libc::off_t;
        // SAFETY: `fd` is valid via `Arc<File>`, `dst[total..]` is valid mutable memory of length `rem`.
        let n = unsafe {
            libc::pread(
                fd,
                dst.as_mut_ptr().add(total) as *mut libc::c_void,
                rem,
                cur_off,
            )
        };
        if n < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "EOF during aligned pread",
            ));
        }
        total += n as usize;
    }
    Ok(total)
}

impl AsyncBlockReader for PreadDirectReader {
    fn submit_read(
        &mut self,
        req_id: u64,
        offset: u64,
        padded_len: usize,
        buf: AlignedBuffer,
    ) -> Result<(), (NvmeError, AlignedBuffer)> {
        if offset % (PAGE_ALIGN as u64) != 0 {
            return Err((
                NvmeError::Format(FormatError::AlignmentViolation {
                    context: "PreadDirectReader submit offset".into(),
                    value: offset,
                    align: PAGE_ALIGN,
                }),
                buf,
            ));
        }
        if padded_len == 0
            || padded_len % PAGE_ALIGN != 0
            || !buf.is_page_aligned()
            || buf.len() < padded_len
        {
            return Err((
                NvmeError::Format(FormatError::AlignmentViolation {
                    context: "PreadDirectReader buffer/length".into(),
                    value: padded_len as u64,
                    align: PAGE_ALIGN,
                }),
                buf,
            ));
        }

        let mut guard = self.shared.sq.lock().unwrap();
        while guard.0.len() >= self.shared.sq_capacity && !guard.1 {
            guard = self.shared.sq_notfull.wait(guard).unwrap();
        }
        if guard.1 {
            return Err((
                NvmeError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "PreadDirectReader worker pool shut down",
                )),
                buf,
            ));
        }
        guard.0.push_back(SqeReadTask {
            req_id,
            offset,
            padded_len,
            buf,
        });
        self.in_flight += 1;
        self.shared.sq_nonempty.notify_one();
        Ok(())
    }

    fn reap_completions(&mut self, blocking_min: usize) -> Result<Vec<CompletedRead>, NvmeError> {
        let target = blocking_min.min(self.in_flight);
        let mut cq_guard = self.shared.cq.lock().unwrap();
        while cq_guard.len() < target {
            cq_guard = self.shared.cq_ready.wait(cq_guard).unwrap();
        }
        let mut out = Vec::with_capacity(cq_guard.len());
        while let Some(c) = cq_guard.pop_front() {
            self.in_flight = self.in_flight.saturating_sub(1);
            out.push(c);
        }
        Ok(out)
    }

    fn in_flight_count(&self) -> usize {
        self.in_flight
    }

    fn direct_io_active(&self) -> bool {
        self.nocache_enabled
    }

    fn backend_name(&self) -> &'static str {
        #[cfg(target_os = "macos")]
        {
            "F_NOCACHE+pread_pool"
        }
        #[cfg(target_os = "linux")]
        {
            if self.nocache_enabled {
                "O_DIRECT+pread_pool"
            } else {
                "pread_pool"
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            "pread_pool"
        }
    }
}

impl Drop for PreadDirectReader {
    fn drop(&mut self) {
        {
            let mut guard = self.shared.sq.lock().unwrap();
            guard.1 = true;
            self.shared.sq_nonempty.notify_all();
            self.shared.sq_notfull.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Opens the platform-native `AsyncBlockReader` (`IoUringDirectReader` on Linux with graceful
/// fallback to `PreadDirectReader` when `io_uring` is blocked by seccomp/sysctl, and `PreadDirectReader` on macOS).
pub fn open_default_reader(
    path: &Path,
    queue_depth: usize,
) -> Result<Box<dyn AsyncBlockReader>, NvmeError> {
    #[cfg(target_os = "linux")]
    {
        match IoUringDirectReader::open(path, queue_depth) {
            Ok(r) => return Ok(Box::new(r)),
            Err(_) => {
                let fallback = PreadDirectReader::open(path, queue_depth, 4)?;
                return Ok(Box::new(fallback));
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let r = PreadDirectReader::open(path, queue_depth, 4)?;
        Ok(Box::new(r))
    }
}

// ============================================================================
// Fixed-Capacity RAM-Budgeted `ExpertLruPool`
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefetchMetrics {
    pub cache_hits: u64,
    pub prefetched_hits: u64,
    pub cold_misses: u64,
    pub prefetch_requests: u64,
    pub evictions: u64,
    pub prefetch_stall_us: u64,
    pub bytes_read_nvme: u64,
}

impl PrefetchMetrics {
    pub fn total_acquisitions(&self) -> u64 {
        self.cache_hits + self.prefetched_hits + self.cold_misses
    }

    pub fn effective_hit_rate(&self) -> f64 {
        let total = self.total_acquisitions();
        if total == 0 {
            1.0
        } else {
            ((self.cache_hits + self.prefetched_hits) as f64) / (total as f64)
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PoolSizingConfig {
    pub ram_budget_bytes: Option<usize>,
    pub max_cached_experts: Option<usize>,
    pub core_resident_bytes: usize,
    pub state_reserved_bytes: usize,
}

enum SlotState {
    Evicted,
    InFlight,
    Resident {
        buf: AlignedBuffer,
        last_used_tick: u64,
        pinned: bool,
        prefetched_unused: bool,
    },
}

struct InFlightMeta {
    flat_idx: usize,
    layer_id: usize,
    expert_id: usize,
    from_prefetch: bool,
    queued_tick: u64,
}

pub struct ExpertLruPool {
    directory: ExpertsDirectory,
    reader: Box<dyn AsyncBlockReader>,
    top_k: usize,
    max_cached_experts: usize,
    slots: Vec<SlotState>,
    free_buffers: Vec<AlignedBuffer>,
    in_flight_meta: HashMap<u64, InFlightMeta>,
    next_req_id: u64,
    tick: u64,
    metrics: PrefetchMetrics,
}

impl ExpertLruPool {
    pub fn open(
        experts_path: &Path,
        top_k: usize,
        sizing: PoolSizingConfig,
    ) -> Result<Self, NvmeError> {
        let directory = ExpertsDirectory::open(experts_path)?;
        let total_experts =
            (directory.header.num_layers as usize) * (directory.header.num_experts as usize);
        let padded_bytes = directory.header.expert_padded_bytes as usize;

        let max_cached_experts = Self::compute_max_slots(
            total_experts,
            padded_bytes,
            top_k,
            sizing,
        )?;
        let queue_depth = (max_cached_experts * 2).max(16);
        let reader = open_default_reader(experts_path, queue_depth)?;
        Self::with_reader(directory, reader, top_k, max_cached_experts)
    }

    pub fn with_reader(
        directory: ExpertsDirectory,
        reader: Box<dyn AsyncBlockReader>,
        top_k: usize,
        max_cached_experts: usize,
    ) -> Result<Self, NvmeError> {
        if max_cached_experts < top_k || top_k == 0 {
            return Err(NvmeError::InvalidPoolCapacity {
                capacity: max_cached_experts,
                top_k,
            });
        }
        let total_experts =
            (directory.header.num_layers as usize) * (directory.header.num_experts as usize);
        let cap = max_cached_experts.min(total_experts);
        let padded_bytes = directory.header.expert_padded_bytes as usize;

        let mut free_buffers = Vec::with_capacity(cap);
        for _ in 0..cap {
            let buf = AlignedBuffer::new_exact(padded_bytes)?;
            debug_assert!(buf.is_page_aligned());
            free_buffers.push(buf);
        }

        let mut slots = Vec::with_capacity(total_experts);
        for _ in 0..total_experts {
            slots.push(SlotState::Evicted);
        }

        Ok(Self {
            directory,
            reader,
            top_k,
            max_cached_experts: cap,
            slots,
            free_buffers,
            in_flight_meta: HashMap::with_capacity(cap),
            next_req_id: 1,
            tick: 1,
            metrics: PrefetchMetrics::default(),
        })
    }

    fn compute_max_slots(
        total_experts: usize,
        padded_bytes: usize,
        top_k: usize,
        sizing: PoolSizingConfig,
    ) -> Result<usize, NvmeError> {
        if let Some(explicit_cap) = sizing.max_cached_experts {
            if explicit_cap < top_k {
                return Err(NvmeError::InvalidPoolCapacity {
                    capacity: explicit_cap,
                    top_k,
                });
            }
            return Ok(explicit_cap.min(total_experts));
        }

        if let Some(budget_bytes) = sizing.ram_budget_bytes {
            let fixed_bytes = sizing
                .core_resident_bytes
                .saturating_add(sizing.state_reserved_bytes);
            let min_expert_bytes = top_k * padded_bytes;
            let remaining = budget_bytes.checked_sub(fixed_bytes).unwrap_or(0);
            let slots = remaining / padded_bytes;
            if slots < top_k {
                return Err(NvmeError::InsufficientRamBudget {
                    budget_bytes,
                    fixed_bytes,
                    min_expert_bytes,
                    top_k,
                });
            }
            return Ok(slots.min(total_experts));
        }

        Ok(total_experts)
    }

    #[inline]
    pub fn max_cached_experts(&self) -> usize {
        self.max_cached_experts
    }

    #[inline]
    pub fn pool_buffer_bytes(&self) -> usize {
        self.max_cached_experts * (self.directory.header.expert_padded_bytes as usize)
    }

    #[inline]
    pub fn direct_io_active(&self) -> bool {
        self.reader.direct_io_active()
    }

    #[inline]
    pub fn backend_name(&self) -> &'static str {
        self.reader.backend_name()
    }

    #[inline]
    pub fn metrics(&self) -> &PrefetchMetrics {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics = PrefetchMetrics::default();
    }

    pub fn resident_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, SlotState::Resident { .. }))
            .count()
    }

    pub fn in_flight_count(&self) -> usize {
        self.in_flight_meta.len()
    }

    #[inline]
    fn flat_index(&self, layer_id: usize, expert_id: usize) -> usize {
        layer_id * (self.directory.header.num_experts as usize) + expert_id
    }

    #[inline]
    fn bump_tick(&mut self) -> u64 {
        let t = self.tick;
        self.tick += 1;
        t
    }

    /// Non-blockingly submits read SQEs for cold experts predicted for `target_layer` (e.g., `L + 3`).
    pub fn prefetch_layer_experts(
        &mut self,
        target_layer: usize,
        predicted_experts: &[usize],
    ) -> Result<(), NvmeError> {
        if target_layer >= self.directory.header.num_layers as usize {
            return Ok(());
        }

        // Harvest any completed reads first without blocking; never fail caller on background prefetch errors.
        let _ = self.poll_completions(0);

        // Reserve `top_k` capacity so speculative in-flight DMA never starves the active layer.
        let max_speculative_inflight = self.max_cached_experts.saturating_sub(self.top_k);
        if max_speculative_inflight == 0 {
            return Ok(());
        }

        for &expert_id in predicted_experts {
            if expert_id >= self.directory.header.num_experts as usize {
                continue;
            }
            let flat = self.flat_index(target_layer, expert_id);
            let tick = self.bump_tick();

            match &mut self.slots[flat] {
                SlotState::Resident {
                    last_used_tick, ..
                } => {
                    *last_used_tick = tick;
                    continue;
                }
                SlotState::InFlight => {
                    continue;
                }
                SlotState::Evicted => {}
            }

            if self.in_flight_meta.len() >= max_speculative_inflight {
                break;
            }

            let Some(buf) = self.try_acquire_buffer_for_prefetch() else {
                break;
            };

            if self
                .submit_slot_read(target_layer, expert_id, flat, buf, true, tick)
                .is_ok()
            {
                self.metrics.prefetch_requests += 1;
            }
        }

        Ok(())
    }

    /// Ensures all `required_experts` for `layer` are resident, CRC32-verified, and pinned in RAM.
    pub fn acquire_layer_experts(
        &mut self,
        layer: usize,
        required_experts: &[usize],
    ) -> Result<(), NvmeError> {
        // Harvest any background prefetches that already completed before measuring stalls!
        self.poll_completions(0)?;

        let mut need_wait = false;
        let mut misses_to_submit: Vec<(usize, usize)> = Vec::new();
        let mut seen = Vec::with_capacity(required_experts.len());

        for &expert_id in required_experts {
            if seen.contains(&expert_id) {
                continue;
            }
            seen.push(expert_id);

            let flat = self.flat_index(layer, expert_id);
            let tick = self.bump_tick();

            match &mut self.slots[flat] {
                SlotState::Resident {
                    last_used_tick,
                    pinned,
                    prefetched_unused,
                    ..
                } => {
                    if *prefetched_unused {
                        self.metrics.prefetched_hits += 1;
                        *prefetched_unused = false;
                    } else {
                        self.metrics.cache_hits += 1;
                    }
                    *pinned = true;
                    *last_used_tick = tick;
                }
                SlotState::InFlight => {
                    need_wait = true;
                }
                SlotState::Evicted => {
                    self.metrics.cold_misses += 1;
                    misses_to_submit.push((expert_id, flat));
                    need_wait = true;
                }
            }
        }

        if !need_wait {
            return Ok(());
        }

        let stall_start = Instant::now();

        let wait_res: Result<(), NvmeError> = (|| {
            for (expert_id, flat) in misses_to_submit {
                let buf = self.acquire_buffer_blocking()?;
                let tick = self.bump_tick();
                self.submit_slot_read(layer, expert_id, flat, buf, false, tick)?;
            }

            loop {
                let mut all_ready = true;
                for &expert_id in &seen {
                    let flat = self.flat_index(layer, expert_id);
                    match self.slots[flat] {
                        SlotState::Resident { .. } => {}
                        SlotState::InFlight => {
                            all_ready = false;
                        }
                        SlotState::Evicted => {
                            // Speculative prefetch failed in background and reverted to Evicted; re-read on demand.
                            all_ready = false;
                            self.metrics.cold_misses += 1;
                            let buf = self.acquire_buffer_blocking()?;
                            let tick = self.bump_tick();
                            self.submit_slot_read(layer, expert_id, flat, buf, false, tick)?;
                        }
                    }
                }
                if all_ready {
                    break;
                }
                self.poll_completions(1)?;
            }
            Ok(())
        })();

        if let Err(err) = wait_res {
            // Unpin any already-resident experts pinned earlier in this call so pool slots never leak!
            self.release_layer_experts(layer, &seen);
            return Err(err);
        }

        self.metrics.prefetch_stall_us += stall_start.elapsed().as_micros() as u64;

        for &expert_id in &seen {
            let flat = self.flat_index(layer, expert_id);
            let tick = self.bump_tick();
            match &mut self.slots[flat] {
                SlotState::Resident {
                    last_used_tick,
                    pinned,
                    prefetched_unused,
                    ..
                } => {
                    if *prefetched_unused {
                        self.metrics.prefetched_hits += 1;
                        *prefetched_unused = false;
                    }
                    *pinned = true;
                    *last_used_tick = tick;
                }
                _ => {
                    return Err(NvmeError::ExpertNotResident {
                        layer_id: layer,
                        expert_id,
                    });
                }
            }
        }

        Ok(())
    }

    /// Returns a zero-copy `ExpertBlockView` for a currently pinned/resident `(layer, expert_id)`.
    pub fn expert_view(
        &self,
        layer: usize,
        expert_id: usize,
    ) -> Result<ExpertBlockView<'_>, NvmeError> {
        let flat = self.flat_index(layer, expert_id);
        match &self.slots[flat] {
            SlotState::Resident { buf, .. } => Ok(ExpertBlockView::from_aligned(
                buf,
                self.directory.header.hidden_size as usize,
                self.directory.header.intermediate_size as usize,
            )?),
            _ => Err(NvmeError::ExpertNotResident {
                layer_id: layer,
                expert_id,
            }),
        }
    }

    /// Unpins `experts` after Layer `layer`'s SwiGLU computation finishes so LRU can recycle them later.
    pub fn release_layer_experts(&mut self, layer: usize, experts: &[usize]) {
        for &expert_id in experts {
            let flat = self.flat_index(layer, expert_id);
            if let SlotState::Resident { pinned, .. } = &mut self.slots[flat] {
                *pinned = false;
            }
        }
    }

    fn try_acquire_buffer_for_prefetch(&mut self) -> Option<AlignedBuffer> {
        if let Some(buf) = self.free_buffers.pop() {
            return Some(buf);
        }
        // For speculative prefetch, only evict unpinned Resident slots whose prefetch was already consumed
        // (`!prefetched_unused`), picking the least-recently-used (`last_used_tick` min).
        let victim_idx = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(idx, state)| match state {
                SlotState::Resident {
                    last_used_tick,
                    pinned: false,
                    prefetched_unused: false,
                    ..
                } => Some((idx, *last_used_tick)),
                _ => None,
            })
            .min_by_key(|&(_, tick)| tick)
            .map(|(idx, _)| idx)?;

        self.evict_slot(victim_idx)
    }

    fn acquire_buffer_blocking(&mut self) -> Result<AlignedBuffer, NvmeError> {
        loop {
            if let Some(buf) = self.free_buffers.pop() {
                return Ok(buf);
            }
            // Prefer evicting unpinned already-consumed experts (`prefetched_unused == false`),
            // falling back to unpinned `prefetched_unused == true` only if the pool is completely tight.
            let victim_idx = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(idx, state)| match state {
                    SlotState::Resident {
                        last_used_tick,
                        pinned: false,
                        prefetched_unused,
                        ..
                    } => Some((idx, *prefetched_unused, *last_used_tick)),
                    _ => None,
                })
                .min_by_key(|&(_, prefetched_unused, tick)| (prefetched_unused, tick))
                .map(|(idx, _, _)| idx);

            if let Some(idx) = victim_idx {
                if let Some(buf) = self.evict_slot(idx) {
                    return Ok(buf);
                }
            }

            // All non-pinned buffers are currently `InFlight`; wait for 1 CQE so a slot becomes `Resident`.
            if self.in_flight_meta.is_empty() {
                return Err(NvmeError::InvalidPoolCapacity {
                    capacity: self.max_cached_experts,
                    top_k: self.top_k,
                });
            }
            self.poll_completions(1)?;
        }
    }

    fn evict_slot(&mut self, flat_idx: usize) -> Option<AlignedBuffer> {
        let prev = std::mem::replace(&mut self.slots[flat_idx], SlotState::Evicted);
        match prev {
            SlotState::Resident { buf, pinned, .. } => {
                debug_assert!(!pinned, "attempted to evict a pinned expert!");
                self.metrics.evictions += 1;
                Some(buf)
            }
            other => {
                self.slots[flat_idx] = other;
                None
            }
        }
    }

    fn submit_slot_read(
        &mut self,
        layer_id: usize,
        expert_id: usize,
        flat_idx: usize,
        buf: AlignedBuffer,
        from_prefetch: bool,
        queued_tick: u64,
    ) -> Result<(), NvmeError> {
        let entry = match self.directory.entry(layer_id, expert_id) {
            Ok(&e) => e,
            Err(err) => {
                self.free_buffers.push(buf);
                return Err(NvmeError::Format(err));
            }
        };
        let req_id = self.next_req_id;
        self.next_req_id += 1;

        if let Err((e, returned_buf)) =
            self.reader
                .submit_read(req_id, entry.offset, entry.padded_bytes as usize, buf)
        {
            self.free_buffers.push(returned_buf);
            return Err(e);
        }

        self.slots[flat_idx] = SlotState::InFlight;
        self.in_flight_meta.insert(
            req_id,
            InFlightMeta {
                flat_idx,
                layer_id,
                expert_id,
                from_prefetch,
                queued_tick,
            },
        );
        Ok(())
    }

    fn poll_completions(&mut self, blocking_min: usize) -> Result<(), NvmeError> {
        if self.in_flight_meta.is_empty() {
            return Ok(());
        }
        let completed = self.reader.reap_completions(blocking_min)?;
        let d = self.directory.header.hidden_size as usize;
        let inter = self.directory.header.intermediate_size as usize;
        let packed_ternary_len = (3 * d * inter) / 4;

        let mut first_err: Option<NvmeError> = None;

        for item in completed {
            let Some(meta) = self.in_flight_meta.remove(&item.req_id) else {
                self.free_buffers.push(item.buf);
                continue;
            };
            let entry = match self.directory.entry(meta.layer_id, meta.expert_id) {
                Ok(&e) => e,
                Err(fmt_err) => {
                    self.slots[meta.flat_idx] = SlotState::Evicted;
                    self.free_buffers.push(item.buf);
                    if !meta.from_prefetch {
                        first_err.get_or_insert(NvmeError::Format(fmt_err));
                    }
                    continue;
                }
            };

            let actual_bytes = match item.result {
                Ok(n) => n,
                Err(io_err) => {
                    self.slots[meta.flat_idx] = SlotState::Evicted;
                    self.free_buffers.push(item.buf);
                    if !meta.from_prefetch {
                        first_err.get_or_insert(NvmeError::Io(io_err));
                    }
                    continue;
                }
            };

            if actual_bytes != entry.padded_bytes as usize {
                self.slots[meta.flat_idx] = SlotState::Evicted;
                self.free_buffers.push(item.buf);
                if !meta.from_prefetch {
                    first_err.get_or_insert(NvmeError::ShortRead {
                        req_id: item.req_id,
                        expected: entry.padded_bytes as usize,
                        actual: actual_bytes,
                    });
                }
                continue;
            }

            self.metrics.bytes_read_nvme += actual_bytes as u64;

            let actual_crc = crc32fast::hash(&item.buf[..entry.payload_bytes as usize]);
            if actual_crc != entry.crc32 {
                self.slots[meta.flat_idx] = SlotState::Evicted;
                self.free_buffers.push(item.buf);
                if !meta.from_prefetch {
                    first_err.get_or_insert(NvmeError::Format(FormatError::ExpertCrcMismatch {
                        layer_id: entry.layer_id,
                        expert_id: entry.expert_id,
                        expected: entry.crc32,
                        actual: actual_crc,
                    }));
                }
                continue;
            }

            if let Err(crumb_err) = validate_no_invalid_crumbs(
                &item.buf[..packed_ternary_len],
                &format!("expert({},{})", meta.layer_id, meta.expert_id),
            ) {
                self.slots[meta.flat_idx] = SlotState::Evicted;
                self.free_buffers.push(item.buf);
                if !meta.from_prefetch {
                    first_err.get_or_insert(NvmeError::Format(crumb_err));
                }
                continue;
            }

            debug_assert!(item.buf.is_page_aligned());
            self.slots[meta.flat_idx] = SlotState::Resident {
                buf: item.buf,
                last_used_tick: meta.queued_tick,
                pinned: false,
                prefetched_unused: meta.from_prefetch,
            };
        }

        // Strict conservation check in ALL builds: zero buffer leaks across both ok and error paths!
        assert_eq!(
            self.free_buffers.len() + self.in_flight_meta.len() + self.resident_count(),
            self.max_cached_experts,
            "ExpertLruPool buffer conservation invariant violated"
        );

        if let Some(err) = first_err {
            return Err(err);
        }
        Ok(())
    }
}
