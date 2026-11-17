//! Micro-TERRA Bare-Metal 1.58-Bit Ternary SIMD + `io_uring` / `F_NOCACHE` Inference Engine.

pub mod engine;
pub mod format;
pub mod kernels;
pub mod nvme_pool;
pub mod state;

use thiserror::Error;

pub use engine::{
    verify_golden_dir, verify_golden_dir_with_config, BenchmarkReport, EngineConfig,
    ForwardStepOutput, GoldenReference, MemoryBreakdown, TerraEngine,
};
pub use format::{AlignedBuffer, CoreFile, ExpertBlockView, ExpertsDirectory, FormatError};
pub use nvme_pool::{AsyncBlockReader, ExpertLruPool, NvmeError, PoolSizingConfig, PrefetchMetrics};
pub use state::HybridStateManager;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("format error: {0}")]
    Format(#[from] FormatError),

    #[error("NVMe pool error: {0}")]
    Nvme(#[from] NvmeError),

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("token ID {token_id} out of range for vocab_size={vocab_size}")]
    TokenOutOfRange { token_id: u32, vocab_size: u32 },

    #[error("prompt_tokens must not be empty")]
    EmptyPrompt,

    #[error("sequence length {seq_len} exceeds reserved max_seq_len={max_seq_len}")]
    ContextLengthExceeded { seq_len: usize, max_seq_len: usize },

    #[error("golden_ref.json contains no reference logits (`step_logits` or `prefill_last_logits`) to verify")]
    MissingGoldenTargets,

    #[error("golden reference shape mismatch on `{field}`: expected {expected}, got {actual}")]
    GoldenShapeMismatch {
        field: String,
        expected: usize,
        actual: usize,
    },

    #[error("golden parity tolerance exceeded: max_abs_diff={max_abs_diff:.6e} > tol={tolerance:.6e} at {detail}")]
    GoldenParityExceeded {
        max_abs_diff: f32,
        tolerance: f32,
        detail: String,
    },
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::engine::GoldenLayer0;
    use crate::format::fixtures::{write_synthetic_model_dir, SyntheticModelConfig};
    use crate::format::{
        LayerKind, FLAG_BITLINEAR_RMS_PRESCALE, FLAG_HAS_HINT_EMBEDDING, FLAG_NORM_TOPK_PROB,
        FLAG_TIE_WORD_EMBEDDINGS,
    };
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "terra_runtime_test_{label}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_constrained_ram_lru_prefetch_stress_bit_identical_logits() {
        // 8 layers x 8 experts = 64 experts on disk; cap cache strictly to 12 experts!
        let dir = unique_temp_dir("lru_stress_64_to_12");
        let cfg = SyntheticModelConfig {
            num_layers: 8,
            num_experts: 8,
            top_k: 2,
            lookahead_offset: 3,
            hidden_size: 128,
            intermediate_size: 256,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir, &cfg).unwrap();

        let prompt: Vec<u32> = vec![3, 17, 42, 9, 61, 12, 5, 28];

        // 1. Unconstrained full-RAM engine (all 64 experts fit in pool)
        let mut full_engine = TerraEngine::open_dir(
            &dir,
            EngineConfig {
                ram_budget_mb: None,
                max_cached_experts: None,
                enable_lookahead_prefetch: false,
                kv_reserve_tokens: 64,
            },
        )
        .unwrap();
        assert_eq!(full_engine.nvme_pool().max_cached_experts(), 64);

        // 2. Constrained streaming engine (only 12 experts cached out of 64, 3-layer lookahead active!)
        let mut constrained_engine = TerraEngine::open_dir(
            &dir,
            EngineConfig {
                ram_budget_mb: None,
                max_cached_experts: Some(12),
                enable_lookahead_prefetch: true,
                kv_reserve_tokens: 64,
            },
        )
        .unwrap();
        assert_eq!(constrained_engine.nvme_pool().max_cached_experts(), 12);

        #[cfg(target_os = "macos")]
        assert!(
            constrained_engine.nvme_pool().direct_io_active(),
            "F_NOCACHE must be active on macOS"
        );

        for (step_idx, &tok) in prompt.iter().enumerate() {
            let full_logits = full_engine.forward_token(tok).unwrap();
            let stream_logits = constrained_engine.forward_token(tok).unwrap();
            assert_eq!(full_logits.len(), stream_logits.len());
            for (v_idx, (&a, &b)) in full_logits.iter().zip(stream_logits.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "bit mismatch at token step {step_idx}, vocab {v_idx}: full={a} vs constrained={b}"
                );
            }
        }

        let metrics = constrained_engine.nvme_pool().metrics();
        println!(
            "[LRU 12/64 Stress] cache_hits={}, prefetched_hits={}, cold_misses={}, evictions={}, stall_us={}, nvme_bytes={}",
            metrics.cache_hits,
            metrics.prefetched_hits,
            metrics.cold_misses,
            metrics.evictions,
            metrics.prefetch_stall_us,
            metrics.bytes_read_nvme
        );

        assert!(
            metrics.prefetched_hits > 0,
            "expected lookahead prefetched_hits > 0, got {metrics:?}"
        );
        assert!(
            metrics.evictions > 0,
            "expected LRU evictions > 0 with 12/64 cache cap, got {metrics:?}"
        );
        assert!(metrics.bytes_read_nvme > 0);
        assert!(constrained_engine.nvme_pool().resident_count() <= 12);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_ram_budget_mb_enforcement_and_zipfian_cache_hits() {
        // 8 layers x 8 experts = 64 experts; with 32-slot cache (`1 MB` budget), hot experts stay cached across tokens!
        let dir = unique_temp_dir("ram_budget_mb");
        let cfg = SyntheticModelConfig {
            num_layers: 8,
            num_experts: 8,
            top_k: 2,
            lookahead_offset: 3,
            hidden_size: 128,
            intermediate_size: 256,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir, &cfg).unwrap();

        // 1. Budget of 0 MB (smaller than core + state + top_k) must deterministically fail with InsufficientRamBudget
        let err = match TerraEngine::open_dir(
            &dir,
            EngineConfig {
                ram_budget_mb: Some(0),
                max_cached_experts: None,
                enable_lookahead_prefetch: true,
                kv_reserve_tokens: 64,
            },
        ) {
            Ok(_) => panic!("expected 0 MB budget to fail"),
            Err(e) => e,
        };
        assert!(
            matches!(err, RuntimeError::Nvme(NvmeError::InsufficientRamBudget { .. })),
            "got {err:?}"
        );

        // 2. Budget of 1 MB (~1,048,576 B): after subtracting resident core.bin (~270 KB) + state (~90 KB),
        //    leaves ~680 KB -> caps pool to < 64 experts (around 27 slots of 28,672 B each), forcing both
        //    Zipfian cross-token `cache_hits > 0`, `prefetched_hits > 0`, and `evictions > 0`!
        let mut budgeted_engine = TerraEngine::open_dir(
            &dir,
            EngineConfig {
                ram_budget_mb: Some(1),
                max_cached_experts: None,
                enable_lookahead_prefetch: true,
                kv_reserve_tokens: 64,
            },
        )
        .unwrap();
        let cap = budgeted_engine.nvme_pool().max_cached_experts();
        assert!(cap >= 2 && cap < 64, "expected 1MB budget to cap slots in 2..64, got {cap}");
        assert!(budgeted_engine.memory_breakdown().total_resident_bytes <= 1024 * 1024);

        let report = budgeted_engine
            .generate_greedy(&[5, 5, 12, 5, 12, 19], 4)
            .unwrap();
        assert_eq!(report.generated_tokens, 4);
        assert!(report.prefetch_metrics.cache_hits > 0, "got {:?}", report.prefetch_metrics);
        assert!(report.prefetch_metrics.prefetched_hits > 0, "got {:?}", report.prefetch_metrics);
        assert!(report.prefetch_metrics.evictions > 0, "got {:?}", report.prefetch_metrics);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_hint_embedding_tied_weights_and_golden_verifier() {
        let dir = unique_temp_dir("golden_and_hints");
        let mut kinds = vec![LayerKind::GatedLinearRecurrence; 4];
        kinds[3] = LayerKind::AnchorGqa;

        let cfg = SyntheticModelConfig {
            vocab_size: 48,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 4,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            num_experts: 6,
            top_k: 2,
            lookahead_offset: 2,
            flags: FLAG_BITLINEAR_RMS_PRESCALE
                | FLAG_NORM_TOPK_PROB
                | FLAG_TIE_WORD_EMBEDDINGS
                | FLAG_HAS_HINT_EMBEDDING,
            layer_kinds: kinds,
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir, &cfg).unwrap();

        let prompt = vec![2u32, 11, 29];
        let hints = vec![[1, 5, -2, 0], [0, 9, 3, 2], [1, 5, -4, 1]];

        let mut engine = TerraEngine::open_dir(&dir, EngineConfig::default()).unwrap();
        let mut step_logits = Vec::new();
        let mut layer0_steps = Vec::new();
        for (i, &tok) in prompt.iter().enumerate() {
            let out = engine
                .forward_token_with_hints(tok, Some(hints[i]), true)
                .unwrap();
            step_logits.push(out.logits);
            layer0_steps.push(out.layer0_post_moe.unwrap());
        }

        let golden = GoldenReference {
            prompt_token_ids: prompt,
            hint_matrix: Some(hints),
            prefill_last_logits: step_logits.last().cloned(),
            layer0_post_moe: Some(GoldenLayer0::LastStep(layer0_steps.last().unwrap().clone())),
            layer0_post_moe_steps: Some(layer0_steps),
            step_logits: Some(step_logits),
        };
        std::fs::write(
            dir.join("golden_ref.json"),
            serde_json::to_string_pretty(&golden).unwrap(),
        )
        .unwrap();

        let max_diff = verify_golden_dir(&dir, 1e-5).unwrap();
        assert_eq!(max_diff, 0.0);

        // Corrupting golden_ref.json beyond tolerance must return GoldenParityExceeded
        let mut corrupted = golden;
        corrupted.prefill_last_logits.as_mut().unwrap()[0] += 0.5;
        std::fs::write(
            dir.join("golden_ref.json"),
            serde_json::to_string_pretty(&corrupted).unwrap(),
        )
        .unwrap();
        let err = verify_golden_dir(&dir, 1e-2).unwrap_err();
        assert!(matches!(err, RuntimeError::GoldenParityExceeded { .. }));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pytorch_exporter_golden_parity_within_1e_2() {
        // If Python `.venv/bin/python` and `exporter/export_olmoe.py` exist in workspace,
        // run a real cross-language PyTorch -> Rust golden parity verification!
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let venv_python = workspace_root.join(".venv/bin/python");
        let exporter_script = workspace_root.join("exporter/export_olmoe.py");
        if !venv_python.exists() || !exporter_script.exists() {
            return;
        }

        let dir = unique_temp_dir("pytorch_cross_lang_parity");
        let status = std::process::Command::new(&venv_python)
            .current_dir(&workspace_root)
            .args([
                "-m",
                "exporter.export_olmoe",
                "--synthetic-tiny",
                "--out-dir",
                dir.to_str().unwrap(),
            ])
            .status()
            .expect("failed to spawn python exporter");
        assert!(status.success(), "python exporter --synthetic-tiny failed");

        let max_diff = verify_golden_dir(&dir, 1e-2).unwrap();
        println!(
            "[PyTorch <-> Rust Golden Parity] max_abs_diff = {:.6e} (tol = 1.0e-2)",
            max_diff
        );
        assert!(
            max_diff <= 1e-2,
            "PyTorch <-> Rust max_abs_diff {max_diff} exceeded 1e-2"
        );

        // Also verify under constrained LRU cache (12 of 64 experts) against the exact same PyTorch golden_ref!
        let max_diff_constrained = verify_golden_dir_with_config(
            &dir,
            1e-2,
            EngineConfig {
                ram_budget_mb: None,
                max_cached_experts: Some(12),
                enable_lookahead_prefetch: true,
                kv_reserve_tokens: 64,
            },
        )
        .unwrap();
        assert_eq!(
            max_diff.to_bits(),
            max_diff_constrained.to_bits(),
            "constrained-cache parity differed from full-RAM parity"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_poll_completions_error_batch_conserves_buffers_and_never_hangs() {
        use std::io::{Seek, SeekFrom, Write};

        let dir = unique_temp_dir("poll_error_conservation");
        let cfg = SyntheticModelConfig {
            num_layers: 4,
            num_experts: 6,
            top_k: 2,
            lookahead_offset: 2,
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            layer_kinds: vec![
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::AnchorGqa,
            ],
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir, &cfg).unwrap();

        let experts_path = dir.join("experts.bin");
        let experts_dir = ExpertsDirectory::open(&experts_path).unwrap();
        let bad_entry = *experts_dir.entry(2, 1).unwrap();

        // Corrupt 1 byte inside expert (2, 1)'s payload on disk (after directory validation passes)
        let mut pool = ExpertLruPool::open(
            &experts_path,
            2,
            PoolSizingConfig {
                ram_budget_bytes: None,
                max_cached_experts: Some(6),
                core_resident_bytes: 0,
                state_reserved_bytes: 0,
            },
        )
        .unwrap();

        {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&experts_path)
                .unwrap();
            f.seek(SeekFrom::Start(bad_entry.offset + 16)).unwrap();
            f.write_all(&[0x55]).unwrap(); // flips payload bytes -> CRC mismatch!
            f.sync_all().unwrap();
        }

        // Submit concurrent prefetches for BOTH corrupted expert (2, 1) and valid experts (2, 0), (2, 2), (2, 3)
        pool.prefetch_layer_experts(2, &[0, 1, 2, 3]).unwrap();

        // Acquiring layer 2 experts `[0, 1]` must surface ExpertCrcMismatch without leaking buffers or stranding slots!
        let err = pool.acquire_layer_experts(2, &[0, 1]).unwrap_err();
        assert!(
            matches!(
                err,
                NvmeError::Format(FormatError::ExpertCrcMismatch {
                    layer_id: 2,
                    expert_id: 1,
                    ..
                })
            ),
            "expected ExpertCrcMismatch on (2,1), got {err:?}"
        );

        // Drain all 3 valid prefetched experts `(2, 0), (2, 2), (2, 3)`; verify 0 stuck InFlight slots and full usability!
        pool.acquire_layer_experts(2, &[0, 2, 3]).unwrap();
        assert_eq!(pool.in_flight_count(), 0);
        pool.release_layer_experts(2, &[0, 2, 3]);
        pool.acquire_layer_experts(1, &[3, 4]).unwrap();
        pool.release_layer_experts(1, &[3, 4]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_ram_budget_strict_ceiling_and_decode_metrics_and_empty_golden() {
        let dir = unique_temp_dir("strict_budget_and_metrics");
        let cfg = SyntheticModelConfig {
            num_layers: 8,
            num_experts: 8,
            top_k: 2,
            hidden_size: 128,
            intermediate_size: 256,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir, &cfg).unwrap();

        let budget_mb = 1usize;
        let budget_bytes = budget_mb * 1024 * 1024;
        let max_seq = 12usize;
        let mut engine = TerraEngine::open_dir(
            &dir,
            EngineConfig {
                ram_budget_mb: Some(budget_mb),
                max_cached_experts: None,
                enable_lookahead_prefetch: true,
                kv_reserve_tokens: max_seq,
            },
        )
        .unwrap();

        // 1. Check `max_new_tokens = 1` does NOT inflate decode_tokens_per_sec (`decode_steps == 0`)
        let r1 = engine.generate_greedy(&[1, 2, 3], 1).unwrap();
        assert_eq!(r1.generated_tokens, 1);
        assert_eq!(r1.decode_steps, 0);
        assert_eq!(r1.decode_tokens_per_sec, 0.0);
        assert!(engine.memory_breakdown().total_resident_bytes <= budget_bytes);

        // 2. Step all the way to `max_seq` (12 tokens total; 3 already stepped -> 9 more)
        let r2 = engine.generate_greedy(&[4, 5], 8).unwrap(); // 2 prefill + 7 decode forward passes = 9 steps -> total 12!
        assert_eq!(r2.generated_tokens, 8);
        assert_eq!(r2.decode_steps, 7);
        assert!(r2.decode_tokens_per_sec > 0.0);
        assert_eq!(engine.state().seq_len(), max_seq);
        assert!(engine.memory_breakdown().total_resident_bytes <= budget_bytes);

        // 3. Stepping token 13 (`> max_seq`) must return ContextLengthExceeded instead of reallocating Vecs past budget!
        let overflow_err = engine.forward_token(7).unwrap_err();
        assert!(
            matches!(
                overflow_err,
                RuntimeError::ContextLengthExceeded {
                    seq_len: 13,
                    max_seq_len: 12
                }
            ),
            "got {overflow_err:?}"
        );

        // 4. Empty optional fields in `golden_ref.json` must return `MissingGoldenTargets` instead of `Ok(0.0)`
        std::fs::write(
            dir.join("golden_ref.json"),
            r#"{"prompt_token_ids": [1, 2, 3]}"#,
        )
        .unwrap();
        let golden_err = verify_golden_dir(&dir, 1e-2).unwrap_err();
        assert!(
            matches!(golden_err, RuntimeError::MissingGoldenTargets),
            "got {golden_err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_failed_token_rolls_back_hybrid_state_without_panic_or_drift() {
        use crate::state::LayerState;
        use std::io::{Read, Seek, SeekFrom, Write};

        let dir_clean = unique_temp_dir("rollback_clean");
        let dir_faulty = unique_temp_dir("rollback_faulty");

        // Layer 0 = Recurrent, Layer 1 = AnchorGqa, Layer 2 = Recurrent, Layer 3 = Recurrent!
        // Thus when Layer 3 fails mid-token, BOTH Recurrent (layers 0, 2) and AnchorGqa (layer 1)
        // have already mutated state for the uncommitted token.
        let cfg = SyntheticModelConfig {
            vocab_size: 64,
            num_layers: 8,
            num_experts: 6,
            top_k: 2,
            lookahead_offset: 3,
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            layer_kinds: vec![
                LayerKind::GatedLinearRecurrence,
                LayerKind::AnchorGqa,
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::AnchorGqa,
            ],
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir_clean, &cfg).unwrap();
        write_synthetic_model_dir(&dir_faulty, &cfg).unwrap();

        let pool_cfg = EngineConfig {
            ram_budget_mb: None,
            // Small pool (4 slots across 48 experts) ensures Layer 3 experts from earlier tokens are evicted
            // before token 19 executes, guaranteeing Layer 3 performs a real NVMe read on token 19.
            max_cached_experts: Some(4),
            enable_lookahead_prefetch: true,
            kv_reserve_tokens: 16,
        };

        let mut clean_engine = TerraEngine::open_dir(&dir_clean, pool_cfg.clone()).unwrap();
        let mut faulty_engine = TerraEngine::open_dir(&dir_faulty, pool_cfg).unwrap();

        for &tok in &[5u32, 12] {
            let a = clean_engine.forward_token(tok).unwrap();
            let b = faulty_engine.forward_token(tok).unwrap();
            assert_eq!(a.len(), b.len());
        }
        assert_eq!(clean_engine.state().seq_len(), 2);
        assert_eq!(faulty_engine.state().seq_len(), 2);

        let rec0_before = match faulty_engine.state().layer_state(0) {
            LayerState::Recurrent(r) => r.state.clone(),
            _ => panic!("expected recurrent layer 0"),
        };
        let kv1_before = match faulty_engine.state().layer_state(1) {
            LayerState::AnchorGqa(kv) => (kv.k_cache.clone(), kv.v_cache.clone(), kv.seq_len),
            _ => panic!("expected anchor layer 1"),
        };

        // Temporarily corrupt 1 byte in every Layer 3 expert on disk (saving originals to restore later).
        let experts_path = dir_faulty.join("experts.bin");
        let experts_dir = ExpertsDirectory::open(&experts_path).unwrap();
        let mut saved_bytes = Vec::new();
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&experts_path)
                .unwrap();
            for e in 0..cfg.num_experts {
                let off = experts_dir.entry(3, e).unwrap().offset + 16;
                let mut orig = [0u8; 1];
                f.seek(SeekFrom::Start(off)).unwrap();
                f.read_exact(&mut orig).unwrap();
                saved_bytes.push((off, orig[0]));
                f.seek(SeekFrom::Start(off)).unwrap();
                f.write_all(&[orig[0] ^ 0xFF]).unwrap();
            }
            f.sync_all().unwrap();
        }

        // Token 19 must fail at Layer 3 with ExpertCrcMismatch AND leave hybrid state byte-for-byte untouched!
        let err = faulty_engine.forward_token(19).unwrap_err();
        assert!(
            matches!(
                err,
                RuntimeError::Nvme(NvmeError::Format(FormatError::ExpertCrcMismatch {
                    layer_id: 3,
                    ..
                }))
            ),
            "expected Layer 3 ExpertCrcMismatch, got {err:?}"
        );
        assert_eq!(faulty_engine.state().seq_len(), 2);

        match faulty_engine.state().layer_state(0) {
            LayerState::Recurrent(r) => assert_eq!(r.state, rec0_before),
            _ => panic!("expected recurrent layer 0"),
        }
        match faulty_engine.state().layer_state(1) {
            LayerState::AnchorGqa(kv) => {
                assert_eq!(kv.seq_len, kv1_before.2);
                assert_eq!(kv.k_cache, kv1_before.0);
                assert_eq!(kv.v_cache, kv1_before.1);
            }
            _ => panic!("expected anchor layer 1"),
        }

        // Restore original bytes on disk and resume without `reset_state()`.
        {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&experts_path)
                .unwrap();
            for (off, orig) in saved_bytes {
                f.seek(SeekFrom::Start(off)).unwrap();
                f.write_all(&[orig]).unwrap();
            }
            f.sync_all().unwrap();
        }

        for &tok in &[19u32, 27] {
            let clean_logits = clean_engine.forward_token(tok).unwrap();
            let recovered_logits = faulty_engine.forward_token(tok).unwrap();
            for (v, (&a, &b)) in clean_logits.iter().zip(recovered_logits.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "state drift after rollback on tok={tok} vocab={v}: clean={a} vs recovered={b}"
                );
            }
        }
        assert_eq!(faulty_engine.state().seq_len(), 4);

        let _ = std::fs::remove_dir_all(&dir_clean);
        let _ = std::fs::remove_dir_all(&dir_faulty);
    }

    #[test]
    fn test_failed_speculative_prefetch_dropped_when_unused_and_reread_when_needed() {
        use std::io::{Read, Seek, SeekFrom, Write};

        let dir = unique_temp_dir("speculative_prefetch_resilience");
        let cfg = SyntheticModelConfig {
            vocab_size: 256,
            num_layers: 4,
            num_experts: 6,
            top_k: 2,
            lookahead_offset: 3, // Layer 0 -> Layer 3 is the ONLY lookahead prefetch pair
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            layer_kinds: vec![
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::GatedLinearRecurrence,
                LayerKind::AnchorGqa,
            ],
            ..SyntheticModelConfig::default()
        };
        write_synthetic_model_dir(&dir, &cfg).unwrap();

        let full_cfg = EngineConfig {
            ram_budget_mb: None,
            max_cached_experts: None,
            enable_lookahead_prefetch: true,
            kv_reserve_tokens: 16,
        };
        let demand_cfg = EngineConfig {
            enable_lookahead_prefetch: false,
            ..full_cfg.clone()
        };

        // Find (1) a token where Layer 0 speculatively prefetches an expert NOT used by Layer 3, and
        // (2) a token where Layer 0 speculatively prefetches ALL experts actually used by Layer 3.
        let mut tok_unused_case: Option<(u32, usize, Vec<f32>)> = None;
        let mut tok_overlap_case: Option<(u32, usize, Vec<f32>)> = None;

        for tok in 0..cfg.vocab_size as u32 {
            let mut d_eng = TerraEngine::open_dir(&dir, demand_cfg.clone()).unwrap();
            d_eng.forward_token(tok).unwrap();
            let needed_l3: Vec<usize> = (0..cfg.num_experts)
                .filter(|&e| d_eng.nvme_pool().expert_view(3, e).is_ok())
                .collect();

            let mut l_eng = TerraEngine::open_dir(&dir, full_cfg.clone()).unwrap();
            let clean_logits = l_eng.forward_token(tok).unwrap();
            let touched_l3: Vec<usize> = (0..cfg.num_experts)
                .filter(|&e| l_eng.nvme_pool().expert_view(3, e).is_ok())
                .collect();

            if tok_unused_case.is_none() {
                if let Some(&unused_e) = touched_l3.iter().find(|e| !needed_l3.contains(e)) {
                    tok_unused_case = Some((tok, unused_e, clean_logits.clone()));
                }
            }
            if tok_overlap_case.is_none()
                && touched_l3 == needed_l3
                && l_eng.nvme_pool().metrics().prefetched_hits == cfg.top_k as u64
            {
                tok_overlap_case = Some((tok, needed_l3[0], clean_logits));
            }
            if tok_unused_case.is_some() && tok_overlap_case.is_some() {
                break;
            }
        }

        let (tok_unused, unused_expert, clean_unused_logits) =
            tok_unused_case.expect("expected at least one speculative-only expert in vocab");
        let (tok_overlap, overlap_expert, clean_overlap_logits) =
            tok_overlap_case.expect("expected at least one exact top-k prefetch match in vocab");

        let experts_path = dir.join("experts.bin");
        let experts_dir = ExpertsDirectory::open(&experts_path).unwrap();

        let flip_byte = |offset: u64| -> u8 {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&experts_path)
                .unwrap();
            let mut orig = [0u8; 1];
            f.seek(SeekFrom::Start(offset)).unwrap();
            f.read_exact(&mut orig).unwrap();
            f.seek(SeekFrom::Start(offset)).unwrap();
            f.write_all(&[orig[0] ^ 0xFF]).unwrap();
            f.sync_all().unwrap();
            orig[0]
        };
        let restore_byte = |offset: u64, orig: u8| {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&experts_path)
                .unwrap();
            f.seek(SeekFrom::Start(offset)).unwrap();
            f.write_all(&[orig]).unwrap();
            f.sync_all().unwrap();
        };

        // Case A: Corrupt an expert `(3, unused_expert)` that Layer 0 speculatively prefetches but Layer 3 never needs.
        // The token MUST succeed with bit-identical logits, dropping the prefetch error silently!
        let unused_off = experts_dir.entry(3, unused_expert).unwrap().offset + 16;
        let orig_unused = flip_byte(unused_off);
        let mut eng_a = TerraEngine::open_dir(&dir, full_cfg.clone()).unwrap();
        let logits_a = eng_a.forward_token(tok_unused).unwrap();
        assert_eq!(logits_a.len(), clean_unused_logits.len());
        for (&a, &b) in logits_a.iter().zip(clean_unused_logits.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert!(eng_a.nvme_pool().metrics().prefetch_requests > 0);
        assert!(eng_a.nvme_pool().expert_view(3, unused_expert).is_err());
        restore_byte(unused_off, orig_unused);

        // Case B: Corrupt `(3, overlap_expert)` which IS speculatively prefetched at Layer 0 AND needed at Layer 3.
        // Layer 0 drops the prefetch failure, Layer 3 re-reads on demand (`cold_misses >= 1`), surfaces
        // ExpertCrcMismatch on Layer 3, and rolls back state so restoring disk bytes recovers seamlessly!
        let overlap_off = experts_dir.entry(3, overlap_expert).unwrap().offset + 16;
        let orig_overlap = flip_byte(overlap_off);
        let mut eng_b = TerraEngine::open_dir(&dir, full_cfg).unwrap();
        let err_b = eng_b.forward_token(tok_overlap).unwrap_err();
        assert!(
            matches!(
                err_b,
                RuntimeError::Nvme(NvmeError::Format(FormatError::ExpertCrcMismatch {
                    layer_id: 3,
                    expert_id,
                    ..
                })) if expert_id == overlap_expert as u16
            ),
            "expected Layer 3 expert {overlap_expert} ExpertCrcMismatch, got {err_b:?}"
        );
        assert_eq!(eng_b.state().seq_len(), 0);
        assert!(eng_b.nvme_pool().metrics().prefetch_requests > 0);
        assert!(eng_b.nvme_pool().metrics().cold_misses >= 1);

        restore_byte(overlap_off, orig_overlap);
        let recovered_b = eng_b.forward_token(tok_overlap).unwrap();
        for (&a, &b) in recovered_b.iter().zip(clean_overlap_logits.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(eng_b.state().seq_len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
