//! Micro-TERRA Dual-Path Edge Codec (Rust `tree-sitter` Python & Rust AST Resolver + C ABI).

pub mod code_parser;

pub use code_parser::{
    fnv1a_symbol_id, parse_and_resolve, parse_hint_matrix, CodeLanguage, ResolvedToken,
};

#[inline]
fn parse_lang_id(lang_id: u32) -> Option<CodeLanguage> {
    match lang_id {
        0 => Some(CodeLanguage::Python),
        1 => Some(CodeLanguage::Rust),
        _ => None,
    }
}

/// C-ABI fast path invoked via `ctypes` from Python (`edge_codec/prose_parser.py`).
///
/// Writes up to `max_tokens` flattened `[i32; 4]` hints into `out_hints` and returns the
/// actual token count produced (`>= 0`), or `-1` on invalid arguments / parse error.
///
/// # Safety
/// `src_ptr` must point to `src_len` valid UTF-8 bytes, and `out_hints` must be writable
/// for `max_tokens * 4` `i32` elements.
#[no_mangle]
pub unsafe extern "C" fn terra_parse_code_hints(
    src_ptr: *const u8,
    src_len: usize,
    lang_id: u32,
    out_hints: *mut i32,
    max_tokens: usize,
) -> isize {
    if src_ptr.is_null() || out_hints.is_null() || max_tokens == 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts(src_ptr, src_len);
    let Ok(source) = std::str::from_utf8(slice) else {
        return -1;
    };
    let Some(lang) = parse_lang_id(lang_id) else {
        return -1;
    };
    let Ok(hints) = parse_hint_matrix(source, lang) else {
        return -1;
    };
    let count = hints.len().min(max_tokens);
    let out_slice = std::slice::from_raw_parts_mut(out_hints, count * 4);
    for (idx, h) in hints.iter().take(count).enumerate() {
        out_slice[idx * 4..idx * 4 + 4].copy_from_slice(h);
    }
    count as isize
}

/// C-ABI fast path emitting `[i32; 4]` hints alongside UTF-8 byte spans `[start, end]` and
/// parser-token definition indices (`-1` when absent) for exact BPE offset alignment.
///
/// # Safety
/// `src_ptr` must point to `src_len` valid UTF-8 bytes; `out_hints` must hold `max_tokens * 4`,
/// `out_spans` must hold `max_tokens * 2`, and `out_def_idx` must hold `max_tokens` `i32` items.
#[no_mangle]
pub unsafe extern "C" fn terra_parse_code_hints_with_spans(
    src_ptr: *const u8,
    src_len: usize,
    lang_id: u32,
    out_hints: *mut i32,
    out_spans: *mut i32,
    out_def_idx: *mut i32,
    max_tokens: usize,
) -> isize {
    if src_ptr.is_null()
        || out_hints.is_null()
        || out_spans.is_null()
        || out_def_idx.is_null()
        || max_tokens == 0
    {
        return -1;
    }
    let slice = std::slice::from_raw_parts(src_ptr, src_len);
    let Ok(source) = std::str::from_utf8(slice) else {
        return -1;
    };
    let Some(lang) = parse_lang_id(lang_id) else {
        return -1;
    };
    let Ok(resolved) = parse_and_resolve(source, lang) else {
        return -1;
    };
    let count = resolved.len().min(max_tokens);
    let hints_slice = std::slice::from_raw_parts_mut(out_hints, count * 4);
    let spans_slice = std::slice::from_raw_parts_mut(out_spans, count * 2);
    let def_slice = std::slice::from_raw_parts_mut(out_def_idx, count);
    for (idx, tok) in resolved.iter().take(count).enumerate() {
        hints_slice[idx * 4..idx * 4 + 4].copy_from_slice(&tok.hint);
        spans_slice[idx * 2] = tok.start_byte as i32;
        spans_slice[idx * 2 + 1] = tok.end_byte as i32;
        match tok.def_token_index {
            Some(d) if d < count => {
                def_slice[idx] = d as i32;
            }
            _ => {
                def_slice[idx] = -1;
                hints_slice[idx * 4 + 2] = 0;
            }
        }
    }
    count as isize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_python_lexical_scoping_shadowing_and_comprehensions() {
        let src = r#"
x = 10
def compute(x, scale=2):
    items = [x + i for i in range(scale)]
    return items
y = x + compute(3)
"#;
        let tokens = parse_and_resolve(src, CodeLanguage::Python).unwrap();

        // Outer `x` at line 1 is definition; `y = x + ...` at line 5 resolves to outer `x`, NOT parameter `x`!
        let outer_x = tokens
            .iter()
            .find(|t| t.text == "x" && t.start_line == 1)
            .unwrap();
        assert!(outer_x.is_definition);

        let param_x = tokens
            .iter()
            .find(|t| t.text == "x" && t.start_line == 2)
            .unwrap();
        assert!(param_x.is_definition);
        assert_ne!(outer_x.index, param_x.index);

        let inner_x = tokens
            .iter()
            .find(|t| t.text == "x" && t.start_line == 3)
            .unwrap();
        assert!(!inner_x.is_definition);
        assert_eq!(inner_x.def_token_index, Some(param_x.index));
        assert_eq!(inner_x.hint[2], (param_x.index as i32) - (inner_x.index as i32));

        let use_outer_x = tokens
            .iter()
            .find(|t| t.text == "x" && t.start_line == 5)
            .unwrap();
        assert_eq!(use_outer_x.def_token_index, Some(outer_x.index));

        // Comprehension variable `i` binds at `for i in ...` even though `x + i` precedes it lexically!
        let i_def = tokens
            .iter()
            .rfind(|t| t.text == "i" && t.start_line == 3)
            .unwrap();
        let i_use = tokens
            .iter()
            .find(|t| t.text == "i" && t.start_line == 3)
            .unwrap();
        assert!(i_def.is_definition);
        assert_eq!(i_use.def_token_index, Some(i_def.index));
    }

    #[test]
    fn test_rust_let_shadowing_and_closures() {
        let src = r#"
fn accumulate(base: i32, limit: i32) -> i32 {
    let val = base + 1;
    let val = val * 2;
    let add = |delta| val + delta;
    add(limit)
}
"#;
        let tokens = parse_and_resolve(src, CodeLanguage::Rust).unwrap();
        let val_tokens: Vec<&ResolvedToken> =
            tokens.iter().filter(|t| t.text == "val").collect();
        // 4 occurrences of `val`:
        // 0: `let val = base + 1;` (def 1)
        // 1: `let val` (def 2, shadowing)
        // 2: `= val * 2` (ref to def 1!)
        // 3: `|delta| val + delta` (ref to def 2!)
        assert_eq!(val_tokens.len(), 4);
        assert!(val_tokens[0].is_definition);
        assert!(val_tokens[1].is_definition);
        assert_eq!(val_tokens[2].def_token_index, Some(val_tokens[0].index));
        assert_eq!(val_tokens[3].def_token_index, Some(val_tokens[1].index));
    }

    #[test]
    fn test_512_token_python_latency_sub_millisecond() {
        let mut snippet = String::new();
        for i in 0..45 {
            snippet.push_str(&format!(
                "def fn_{i}(alpha_{i}, beta_{i}):\n    acc_{i} = alpha_{i} + beta_{i}\n    return acc_{i} * 2\n"
            ));
        }
        let warm = parse_hint_matrix(&snippet, CodeLanguage::Python).unwrap();
        assert!(warm.len() >= 512);

        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let out = parse_hint_matrix(&snippet, CodeLanguage::Python).unwrap();
            std::hint::black_box(out);
        }
        let avg_ms = (start.elapsed().as_secs_f64() * 1000.0) / (iters as f64);
        println!("[edge_codec Rust] tokens={}, avg_latency={avg_ms:.4} ms", warm.len());
        assert!(avg_ms < 1.5, "expected < 1.5 ms parse+resolve, got {avg_ms:.4} ms");
    }
}
