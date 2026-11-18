//! Writes raw little-endian f32 logits for a fixed token sequence: `dump_logits <model_dir> <out_file> <n_tokens>`.
use std::io::Write;
use terra_runtime::{EngineConfig, TerraEngine};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: u32 = a[3].parse().unwrap();
    let mut eng = TerraEngine::open_dir(
        std::path::Path::new(&a[1]),
        EngineConfig { ram_budget_mb: None, max_cached_experts: None, enable_lookahead_prefetch: false, kv_reserve_tokens: 512 },
    )
    .unwrap();
    let mut f = std::io::BufWriter::new(std::fs::File::create(&a[2]).unwrap());
    for t in 0..n {
        // Fixed, state-independent token ids so both builds see the same inputs.
        let logits = eng.forward_token((t * 7919 + 13) % 50304).unwrap();
        for v in logits {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}
