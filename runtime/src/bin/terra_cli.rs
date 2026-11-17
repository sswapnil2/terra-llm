//! `terra-cli`: Command-line runner, golden parity verifier, and JSON benchmark harness
//! for the Micro-TERRA bare-metal Rust inference engine.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use terra_runtime::{verify_golden_dir_with_config, EngineConfig, TerraEngine};

#[derive(Debug, Parser)]
#[command(
    name = "terra-cli",
    about = "Micro-TERRA 1.58-Bit Ternary SIMD + io_uring / F_NOCACHE Inference Engine"
)]
struct CliArgs {
    /// Directory containing `core.bin`, `experts.bin` (and `golden_ref.json` for `--verify-golden`).
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Explicit path to `core.bin` (alternative to `--model-dir`).
    #[arg(long)]
    core: Option<PathBuf>,

    /// Explicit path to `experts.bin` (alternative to `--model-dir`).
    #[arg(long)]
    experts: Option<PathBuf>,

    /// Strict total RAM budget in megabytes (enforces cold-expert NVMe streaming).
    #[arg(long)]
    ram_budget_mb: Option<usize>,

    /// Explicit cap on resident 4KB-aligned expert buffers in the LRU pool.
    #[arg(long)]
    max_cached_experts: Option<usize>,

    /// Maximum sequence length reserved for anchor KV caches (`0` = full `header.max_position_embeddings`).
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,

    /// Disable 3-layer lookahead router prefetching (demand-only NVMe baseline).
    #[arg(long, default_value_t = false)]
    no_lookahead: bool,

    /// Verify numerical parity against `golden_ref.json` in `--model-dir`.
    #[arg(long, default_value_t = false)]
    verify_golden: bool,

    /// Maximum allowed absolute difference for `--verify-golden`.
    #[arg(long, default_value_t = 1e-2)]
    golden_tol: f32,

    /// Comma-separated input prompt token IDs (e.g. `1,42,17,99`).
    #[arg(long, default_value = "1,42,17,99")]
    prompt_tokens: String,

    /// Number of new tokens to decode greedily.
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,

    /// Emit structured JSON benchmark report (pass flag alone for stdout, or `--benchmark-json <FILE>`).
    #[arg(long, num_args = 0..=1, default_missing_value = "-")]
    benchmark_json: Option<String>,
}

fn parse_prompt_tokens(raw: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let tok: u32 = trimmed
            .parse()
            .map_err(|e| format!("invalid token ID `{trimmed}`: {e}"))?;
        out.push(tok);
    }
    if out.is_empty() {
        return Err("prompt_tokens cannot be empty".into());
    }
    Ok(out)
}

fn resolve_paths(args: &CliArgs) -> Result<(PathBuf, PathBuf, Option<PathBuf>), String> {
    match (&args.model_dir, &args.core, &args.experts) {
        (Some(dir), None, None) => Ok((
            dir.join("core.bin"),
            dir.join("experts.bin"),
            Some(dir.clone()),
        )),
        (_, Some(c), Some(e)) => Ok((c.clone(), e.clone(), args.model_dir.clone())),
        _ => Err("specify either `--model-dir <DIR>` or both `--core <FILE>` and `--experts <FILE>`".into()),
    }
}

fn run(args: CliArgs) -> Result<(), Box<dyn std::error::Error>> {
    let (core_path, experts_path, model_dir) = resolve_paths(&args)?;
    let config = EngineConfig {
        ram_budget_mb: args.ram_budget_mb,
        max_cached_experts: args.max_cached_experts,
        enable_lookahead_prefetch: !args.no_lookahead,
        kv_reserve_tokens: args.max_seq_len,
    };

    if args.verify_golden {
        let dir = model_dir.ok_or("`--verify-golden` requires `--model-dir <DIR>`")?;
        let max_diff = verify_golden_dir_with_config(&dir, args.golden_tol, config.clone())?;
        if args.benchmark_json.is_none() {
            println!(
                "PASS golden parity check on {}: max_abs_diff = {:.6e} <= tol = {:.6e}",
                dir.display(),
                max_diff,
                args.golden_tol
            );
        }
        if args.benchmark_json.is_none() {
            return Ok(());
        }
    }

    let prompt = parse_prompt_tokens(&args.prompt_tokens)?;
    let mut engine = TerraEngine::open(&core_path, &experts_path, config)?;
    let report = engine.generate_greedy(&prompt, args.max_new_tokens)?;

    if let Some(target) = args.benchmark_json {
        let payload = serde_json::to_string_pretty(&report)?;
        if target == "-" {
            println!("{payload}");
        } else {
            std::fs::write(&target, &payload)?;
            println!("{payload}");
        }
    } else {
        println!(
            "Micro-TERRA Inference Complete | SIMD={} | IO={} (direct={}) | lookahead={}",
            report.simd_backend,
            report.io_backend,
            report.direct_io_active,
            report.lookahead_enabled
        );
        println!(
            "Prompt tokens: {} | Generated tokens: {:?} | TTFT: {:.2} ms | Decode: {:.2} tok/s",
            report.prompt_tokens,
            report.output_token_ids,
            report.ttft_ms,
            report.decode_tokens_per_sec
        );
        println!(
            "Cache hit rate: {:.2}% (hits={}, prefetched={}, cold_misses={}, evictions={}, stall={} us)",
            report.cache_hit_rate_pct,
            report.prefetch_metrics.cache_hits,
            report.prefetch_metrics.prefetched_hits,
            report.prefetch_metrics.cold_misses,
            report.prefetch_metrics.evictions,
            report.prefetch_metrics.prefetch_stall_us
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    let args = CliArgs::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
