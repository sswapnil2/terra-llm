"""Verification Gate Suite for Milestone 3 (`train/`):
- Step 3A: 1.58-Bit Ternary `BitLinear` & `BitLinearExperts` STE gradients & integer GEMV parity
- Step 3B: 7:1 Hybrid Gated Linear Recurrence (`0..6, 8..14` recurrent + `7, 15` GQA anchors),
           parallel-vs-recurrent $O(1)$-state equivalence, and differentiable `HintEmbedding`
- Step 3C: 3-Layer Lookahead Router achieving >= 80% Top-8 Recall (`E = 64, k = 8`) on held-out batches
- End-to-End Distillation (`KL + Hidden MSE + Lookahead CE + Load-Balance`) & Offline Teacher Caching
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OlmoeConfig, OlmoeForCausalLM

from train.bitlinear import (
    BitLinear,
    BitLinearExperts,
    absmean_quantize_ste,
    apply_l2_optimal_row_prescale_,
    convert_model_to_bitlinear,
    set_model_quant_lambda,
)
from train.distill import (
    DistillLossWeights,
    build_student_from_teacher,
    cache_teacher_batches,
    distill_step,
    evaluate_perplexity,
    export_distilled_student,
)
from train.hybrid_ssm import (
    GatedLinearRecurrenceAttention,
    HintEmbedding,
    convert_olmoe_to_hybrid,
)
from train.lookahead_router import (
    LookaheadRouterBank,
    compute_lookahead_loss,
    compute_top_k_recall,
)


def _tiny_olmoe_config(num_layers: int = 16, num_experts: int = 8, top_k: int = 2) -> OlmoeConfig:
    return OlmoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        eos_token_id=1,
        pad_token_id=0,
    )


def test_bitlinear_ste_gradients_and_exact_integer_parity() -> None:
    torch.manual_seed(7)
    layer = BitLinear(in_features=64, out_features=48, bias=False, use_rms_prescale=True)
    x = torch.randn(3, 10, 64, requires_grad=True)

    out_qat = layer(x)
    out_int = layer.forward_exact_integer(x.detach())
    assert torch.allclose(out_qat.detach(), out_int, atol=1e-5, rtol=1e-4)

    w_ternary, scales = layer.export_quantized()
    assert set(w_ternary.unique().tolist()).issubset({-1, 0, 1})
    assert scales.shape == (48,)
    assert torch.all(scales > 0)

    loss = out_qat.pow(2).mean()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0.0
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum().item() > 0.0


def test_7_to_1_hybrid_layer_schedule_and_parallel_recurrent_equivalence() -> None:
    torch.manual_seed(11)
    # Use H = 8, H_kv = 2 (GQA group = 4) to exercise `_repeat_kv` on recurrent & anchor layers!
    cfg = OlmoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = OlmoeForCausalLM(cfg)
    layer_kinds = convert_olmoe_to_hybrid(model)

    # Verify exact 7:1 hybrid layout: layers 0..6 and 8..14 are recurrent (1), 7 and 15 are GQA anchors (0)
    expected_kinds = [1] * 7 + [0] + [1] * 7 + [0]
    assert layer_kinds == expected_kinds
    for idx in range(16):
        attn_mod = model.model.layers[idx].self_attn
        if idx in (7, 15):
            assert not isinstance(attn_mod, GatedLinearRecurrenceAttention)
        else:
            assert isinstance(attn_mod, GatedLinearRecurrenceAttention)

    # Set NON-ZERO dynamic `decay_proj.weight` + varied per-head biases so alpha_{t, h} varies per token & head!
    rec_block: GatedLinearRecurrenceAttention = model.model.layers[0].self_attn
    with torch.no_grad():
        nn.init.normal_(rec_block.decay_proj.weight, mean=0.0, std=0.6)
        rec_block.decay_proj.bias.copy_(torch.linspace(-1.5, 2.5, steps=8))
    rec_block.eval()

    x = torch.randn(2, 50, 64)
    with torch.no_grad():
        y_recurrent, full_state = rec_block.forward_recurrent(x)
        # Test multiple chunk sizes (7, 16, 128) across T = 50 (exercising multi-chunk + remainder tail!)
        for c_size in (7, 16, 128):
            y_chunk_parallel = rec_block.forward_parallel(x, chunk_size=c_size)
            assert torch.allclose(y_chunk_parallel, y_recurrent, atol=1e-5, rtol=1e-4), (
                f"Chunkwise parallel (chunk_size={c_size}) diverged from recurrent!"
            )

        y_chunk1, state1 = rec_block.forward_recurrent(x[:, :19, :])
        y_chunk2, state2 = rec_block.forward_recurrent(x[:, 19:, :], state=state1)
        y_chunked = torch.cat([y_chunk1, y_chunk2], dim=1)

    assert full_state.shape == (2, 8, 8, 8)  # Strict O(1) [B, H, d, d] memory independent of T!
    assert torch.allclose(y_recurrent, y_chunked, atol=1e-5, rtol=1e-4)
    assert torch.allclose(full_state, state2, atol=1e-5, rtol=1e-4)


def test_fp16_frozen_teacher_produces_fp32_fully_trainable_student_and_accelerator_step() -> None:
    """Verify High #1 (device alignment), High #2 (FP32 master weights), and Medium #8 (all params trainable)."""
    torch.manual_seed(55)
    cfg = _tiny_olmoe_config(num_layers=8, num_experts=8, top_k=2)
    teacher = OlmoeForCausalLM(cfg).half()
    with torch.no_grad():
        for layer in teacher.model.layers:
            nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=0.04)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student, lookahead_bank, _ = build_student_from_teacher(teacher, enable_hint_embedding=True)

    # Every single student parameter MUST be float32 and requires_grad == True!
    for name, p in student.named_parameters():
        assert p.dtype == torch.float32, f"{name} has dtype {p.dtype}, expected float32"
        assert p.requires_grad, f"{name} still has requires_grad=False!"

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    teacher_dev = teacher.to(dev)
    student_dev = student.to(dev)
    ids = torch.randint(2, cfg.vocab_size, (2, 12), device=dev)

    # Snapshot previously-frozen parameter groups before step
    emb_before = student_dev.backbone.model.embed_tokens.weight.detach().clone()
    ln_before = student_dev.backbone.model.layers[0].input_layernorm.weight.detach().clone()
    gate_before = student_dev.backbone.model.layers[0].mlp.gate.weight.detach().clone()
    lm_before = student_dev.backbone.lm_head.weight.detach().clone()

    opt = torch.optim.AdamW(student_dev.parameters(), lr=1e-3)
    with torch.no_grad():
        t_out = teacher_dev(input_ids=ids, output_hidden_states=True, use_cache=False)

    opt.zero_grad()
    loss, _ = distill_step(student_dev, lookahead_bank, ids, t_out.logits, t_out.hidden_states)
    loss.backward()
    opt.step()

    assert not torch.equal(emb_before, student_dev.backbone.model.embed_tokens.weight)
    assert not torch.equal(ln_before, student_dev.backbone.model.layers[0].input_layernorm.weight)
    assert not torch.equal(gate_before, student_dev.backbone.model.layers[0].mlp.gate.weight)
    assert not torch.equal(lm_before, student_dev.backbone.lm_head.weight)


def test_differentiable_hint_embedding_end_to_end() -> None:
    torch.manual_seed(19)
    cfg = _tiny_olmoe_config(num_layers=8)
    teacher = OlmoeForCausalLM(cfg)
    student, lookahead_bank, _ = build_student_from_teacher(teacher, enable_hint_embedding=True)

    ids = torch.randint(2, cfg.vocab_size, (2, 16))
    # [modality_flag, entity_or_var_id, relative_def_offset, entropy_bucket]
    hints = torch.stack(
        [
            torch.randint(0, 3, (2, 16)),
            torch.randint(0, 500, (2, 16)),
            torch.randint(-32, 32, (2, 16)),
            torch.randint(0, 8, (2, 16)),
        ],
        dim=-1,
    )

    out = student(input_ids=ids, hints=hints, output_hidden_states=True)
    look_loss, _ = lookahead_bank.forward_from_captured()
    loss = out.logits.pow(2).mean() + look_loss
    loss.backward()

    assert student.hint_embedding is not None
    assert student.hint_embedding.proj.weight.grad is not None
    assert student.hint_embedding.proj.weight.grad.abs().sum().item() > 0.0
    assert student.hint_embedding.table.weight.grad is not None
    assert student.hint_embedding.table.weight.grad.abs().sum().item() > 0.0
    # Verify BitLinearExperts gradients also flowed cleanly
    expert_mod = student.backbone.model.layers[0].mlp.experts
    assert isinstance(expert_mod, BitLinearExperts)
    assert expert_mod.gate_up_proj.grad is not None
    assert expert_mod.gate_up_proj.grad.abs().sum().item() > 0.0


def test_lookahead_router_achieves_ge_80_percent_top8_recall_on_64_experts() -> None:
    """Milestone 3 Gate: Lookahead router achieves >= 80% top-8 recall at L -> L+3 on held-out validation."""
    torch.manual_seed(123)
    hidden_size = 128
    num_experts = 64
    top_k = 8

    # Ground-truth frozen 3-layer residual + non-linear FFN dynamics from Layer L -> Layer L+3
    res_blocks = nn.ModuleList(
        [
            nn.Sequential(
                nn.RMSNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=False),
            )
            for _ in range(3)
        ]
    )
    for block in res_blocks:
        for p in block.parameters():
            nn.init.normal_(p, mean=0.0, std=0.04)
            p.requires_grad_(False)

    gt_post_ln = nn.RMSNorm(hidden_size)
    gt_router = nn.Linear(hidden_size, num_experts, bias=False)
    nn.init.normal_(gt_router.weight, mean=0.0, std=0.08)
    for p in gt_post_ln.parameters():
        p.requires_grad_(False)
    gt_router.weight.requires_grad_(False)

    def _forward_l_to_l3(h_l: torch.Tensor) -> torch.Tensor:
        h = h_l
        for blk in res_blocks:
            h = h + 0.25 * blk(h)
        return gt_router(gt_post_ln(h))

    # Auxiliary linear lookahead head at Layer L
    lookahead_head = nn.Linear(hidden_size, num_experts, bias=False)
    optimizer = torch.optim.AdamW(lookahead_head.parameters(), lr=1.5e-2, weight_decay=1e-4)

    # Check untrained recall ~ random chance (8 / 64 = 12.5%)
    with torch.no_grad():
        val_h = torch.randn(32, 16, hidden_size)
        val_gt_logits = _forward_l_to_l3(val_h)
        val_gt_topk = torch.topk(val_gt_logits, k=top_k, dim=-1).indices
        init_recall = compute_top_k_recall(lookahead_head(val_h), val_gt_topk, k=top_k)
    assert init_recall < 0.30

    # Train on independent random draws (never touching `val_h`!)
    for _ in range(180):
        h_batch = torch.randn(16, 16, hidden_size)
        with torch.no_grad():
            target_logits = _forward_l_to_l3(h_batch)
        pred_logits = lookahead_head(h_batch)
        loss, _ = compute_lookahead_loss(
            pred_logits,
            target_logits,
            k=top_k,
            topk_uniform_weight=0.75,
            load_balance_coef=0.01,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on held-out validation batch `val_h`
    with torch.no_grad():
        final_pred = lookahead_head(val_h)
        heldout_recall = compute_top_k_recall(final_pred, val_gt_topk, k=top_k)

    assert heldout_recall >= 0.80, f"Expected >= 80% top-8 recall, got {heldout_recall:.4f}"


def test_distillation_convergence_perplexity_retention_and_offline_cache(tmp_path: Path) -> None:
    """Milestone 3 Gate: Distilled ternary hybrid model converges & retains >= 90% baseline performance."""
    torch.manual_seed(99)
    cfg = _tiny_olmoe_config(num_layers=8, num_experts=8, top_k=2)
    teacher = OlmoeForCausalLM(cfg)
    # HF OlmoeTopKRouter initializes gate.weight to zeros on fresh configs; give teacher
    # realistic trained router weights and pre-fit briefly on the motif grammar so teacher has real structure.
    with torch.no_grad():
        for layer in teacher.model.layers:
            nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=0.04)

    # Structured repeatable motif corpus + held-out motif sequences sharing token transition grammar
    motif_a = torch.tensor([4, 9, 15, 23, 42, 7, 19, 31], dtype=torch.long)
    motif_b = torch.tensor([12, 5, 88, 64, 11, 3, 77, 50], dtype=torch.long)
    train_seqs = []
    for i in range(12):
        seq = torch.cat([motif_a if (i + j) % 2 == 0 else motif_b for j in range(4)])
        train_seqs.append(seq)
    train_batch = torch.stack(train_seqs, dim=0)  # [12, 32]
    val_batch = torch.stack(
        [
            torch.cat([motif_b, motif_a, motif_b, motif_a]),
            torch.cat([motif_a, motif_a, motif_b, motif_b]),
        ],
        dim=0,
    )

    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    teacher.train()
    for _ in range(15):
        teacher_opt.zero_grad()
        t_out = teacher(input_ids=train_batch, labels=train_batch, use_cache=False)
        t_out.loss.backward()
        teacher_opt.step()

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # First verify offline teacher caching path works cleanly on disk
    cache_dir = tmp_path / "teacher_cache"
    shards = cache_teacher_batches(teacher, [train_batch], cache_dir=cache_dir)
    assert len(shards) == 1 and shards[0].exists()
    cached_payload = torch.load(shards[0], weights_only=True)

    student, lookahead_bank, layer_kinds = build_student_from_teacher(
        teacher,
        use_rms_prescale=False,
        enable_hint_embedding=False,
    )
    assert layer_kinds == [1, 1, 1, 1, 1, 1, 1, 0]

    lookahead_ids = {id(p) for p in lookahead_bank.parameters()}
    backbone_params = [p for p in student.parameters() if id(p) not in lookahead_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 3e-3, "weight_decay": 0.0},
            {"params": list(lookahead_bank.parameters()), "lr": 2e-2, "weight_decay": 0.0},
        ]
    )
    weights = DistillLossWeights(kl_weight=1.0, hidden_mse_weight=0.5, lookahead_weight=0.5)

    first_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] | None = None
    for _ in range(50):
        optimizer.zero_grad()
        loss, metrics = distill_step(
            student=student,
            lookahead_bank=lookahead_bank,
            input_ids=cached_payload["input_ids"],
            teacher_logits=cached_payload["logits"].float(),
            teacher_hidden_states=tuple(h.float() for h in cached_payload["hidden_states"]),
            weights=weights,
        )
        loss.backward()
        optimizer.step()
        if first_metrics is None:
            first_metrics = metrics
        last_metrics = metrics

    assert first_metrics is not None and last_metrics is not None
    assert last_metrics["kl_loss"] < first_metrics["kl_loss"] * 0.5
    assert last_metrics["hidden_mse"] < first_metrics["hidden_mse"]

    # Evaluate Lookahead Recall@k strictly on HELD-OUT validation batch `val_batch` via real model hooks!
    student.eval()
    with torch.no_grad():
        _ = student(input_ids=val_batch, use_cache=False)
        _, heldout_look_stats = lookahead_bank.forward_from_captured()
    assert heldout_look_stats["mean_recall_at_k"] >= 0.80, (
        f"Expected >= 80% held-out lookahead recall on real hooked model, got {heldout_look_stats['mean_recall_at_k']:.4f}"
    )

    teacher_ppl = evaluate_perplexity(teacher, [val_batch])
    student_ppl = evaluate_perplexity(student, [val_batch])
    # Retain >= 90% of baseline teacher perplexity quality (teacher_ppl / student_ppl >= 0.90)
    retention_ratio = teacher_ppl / student_ppl
    assert retention_ratio >= 0.90, (
        f"Expected >= 0.90 perplexity retention, got {retention_ratio:.4f} "
        f"(teacher={teacher_ppl:.2f}, student={student_ppl:.2f})"
    )


def test_distilled_student_exports_to_terra_binary_and_matches_forward_parity(tmp_path: Path) -> None:
    """Verify Milestone 3 -> Milestone 1 exact forward parity between QAT student and exported `.bin` shards."""
    torch.manual_seed(2026)
    cfg = _tiny_olmoe_config(num_layers=8, num_experts=8, top_k=2)
    teacher = OlmoeForCausalLM(cfg)
    with torch.no_grad():
        for layer in teacher.model.layers:
            nn.init.normal_(layer.mlp.gate.weight, mean=0.0, std=0.04)

    student, lookahead_bank, _ = build_student_from_teacher(
        teacher,
        use_rms_prescale=True,
        enable_hint_embedding=True,
    )
    student.eval()

    prompt_ids = [3, 17, 42, 99]
    export_dir_f32 = tmp_path / "exported_student_f32"
    export_info = export_distilled_student(
        student=student,
        lookahead_bank=lookahead_bank,
        out_dir=export_dir_f32,
        use_rms_prescale=True,
        embed_dtype="f32",
        lm_head_dtype="f32",
        prompt_token_ids=prompt_ids,
    )

    assert (export_dir_f32 / "core.bin").is_file()
    assert (export_dir_f32 / "experts.bin").is_file()
    assert (export_dir_f32 / "golden_ref.json").is_file()
    for ent in export_info["expert_entries"]:
        assert ent.offset % 4096 == 0
        assert ent.padded_bytes % 4096 == 0

    with torch.no_grad():
        ids_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        qat_logits = student(input_ids=ids_tensor, use_cache=False).logits[0].float()
        ref_step_logits = torch.tensor(export_info["golden_ref"]["step_logits"], dtype=torch.float32)

    max_diff = float((qat_logits - ref_step_logits).abs().max().item())
    # Account for f16 KV cache storage in reference forward (machine epsilon ~4.88e-4)
    assert max_diff < 1e-3, f"Expected QAT student vs exported binary parity < 1e-3, got {max_diff}"

    # Default F16 embedding + lm_head export stays smaller and within 1e-2 parity
    export_dir_f16 = tmp_path / "exported_student_f16"
    export_info_f16 = export_distilled_student(
        student=student,
        lookahead_bank=lookahead_bank,
        out_dir=export_dir_f16,
        use_rms_prescale=True,
        prompt_token_ids=prompt_ids,
    )
    assert (export_dir_f16 / "core.bin").stat().st_size < (export_dir_f32 / "core.bin").stat().st_size
    ref_f16_logits = torch.tensor(export_info_f16["golden_ref"]["step_logits"], dtype=torch.float32)
    assert float((qat_logits - ref_f16_logits).abs().max().item()) < 5e-2


def test_phase3_bitlinear_l2_prescale_quant_lambda_hints_decay_and_capture_guard() -> None:
    torch.manual_seed(303)
    # 1. L2-optimal row prescale preserves ternary pattern bit-for-bit and lowers reconstruction L2
    w_orig = torch.randn(16, 64) * 0.03
    _, t_before, g_before = absmean_quantize_ste(w_orig)
    err_before = float((w_orig - t_before.float() * g_before.unsqueeze(-1)).pow(2).sum().item())

    w_scaled = w_orig.clone()
    apply_l2_optimal_row_prescale_(w_scaled)
    _, t_after, g_after = absmean_quantize_ste(w_scaled)
    err_after = float((w_orig - t_after.float() * g_after.unsqueeze(-1)).pow(2).sum().item())
    assert torch.equal(t_before, t_after)
    assert err_after <= err_before + 1e-7

    # 2. Progressive quant_lambda: 0 -> exact FP, 1 -> exact integer, defaults use_rms_prescale=False
    bl = BitLinear(64, 32)
    assert bl.use_rms_prescale is False
    x = torch.randn(2, 5, 64)
    set_model_quant_lambda(bl, 0.0)
    assert torch.allclose(bl(x), F.linear(x, bl.weight), atol=1e-5)
    set_model_quant_lambda(bl, 1.0)
    assert torch.allclose(bl(x), bl.forward_exact_integer(x), atol=1e-5)

    # 3. HintEmbedding outputs exact zeros wherever modality_flag == 0
    he = HintEmbedding(hidden_size=64, num_buckets=256, hint_dim=16)
    hints = torch.tensor(
        [
            [[0, 99, -2, 3], [1, 42, -1, 0], [0, 7, 0, 1], [2, 18, -3, 2]],
        ],
        dtype=torch.int32,
    )
    h_out = he(hints)
    assert torch.count_nonzero(h_out[0, 0]).item() == 0
    assert torch.count_nonzero(h_out[0, 2]).item() == 0
    assert float(h_out[0, 1].abs().sum().item()) > 0.0

    # 4. Multi-scale decay bias + eos_reset_mask prevents cross-document leakage
    rec = GatedLinearRecurrenceAttention(hidden_size=64, num_attention_heads=8, num_key_value_heads=4, rms_norm_eps=1e-5)
    assert rec.decay_proj.bias.max().item() > rec.decay_proj.bias.min().item() + 3.0
    seq_a = torch.randn(1, 10, 64)
    seq_b = seq_a.clone()
    seq_b[:, :5] = torch.randn(1, 5, 64) * 5.0
    reset_mask = torch.zeros((1, 10), dtype=torch.bool)
    reset_mask[:, 5] = True
    with torch.no_grad():
        out_a = rec.forward_parallel(seq_a, chunk_size=4, eos_reset_mask=reset_mask)
        out_b = rec.forward_parallel(seq_b, chunk_size=4, eos_reset_mask=reset_mask)
    assert torch.allclose(out_a[:, 5:], out_b[:, 5:], atol=1e-5, rtol=1e-5)

    # 5. LookaheadRouterBank capturing guard preserves captured forward activations
    cfg = _tiny_olmoe_config(num_layers=8, num_experts=8, top_k=2)
    base = OlmoeForCausalLM(cfg)
    bank = LookaheadRouterBank(num_layers=8, hidden_size=64, num_experts=8, top_k=2, lookahead_offset=3)
    bank.attach_hooks(base)
    ids1 = torch.randint(2, cfg.vocab_size, (1, 8))
    ids2 = torch.randint(2, cfg.vocab_size, (1, 8))
    bank.set_capturing(True)
    _ = base(input_ids=ids1, use_cache=False)
    snap0 = bank._captured_post_attn[0].detach().clone()
    bank.set_capturing(False)
    _ = base(input_ids=ids2, use_cache=False)
    assert torch.equal(bank._captured_post_attn[0], snap0)

    # 6. Colab 3-group optimizer + cosine floor scheduler
    from train.distill_colab import (
        build_colab_optimizer,
        build_cosine_min_lr_scheduler,
        collect_stage1_param_ids,
        configure_training_stage,
        evaluate_packed_teacher_ppl,
        verify_checkpoint_health,
    )

    opt = build_colab_optimizer(base, bank, lr=1e-3)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-3, 1e-4, 3e-3])
    sched = build_cosine_min_lr_scheduler(opt, num_warmup_steps=2, num_training_steps=10, min_lr_ratio=0.1)
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] >= 1e-4 - 1e-8


def test_checkpoint_health_verification_catches_underflow_and_nans(tmp_path: Path) -> None:
    from train.distill_colab import (
        build_colab_optimizer,
        collect_stage1_param_ids,
        configure_training_stage,
        evaluate_packed_teacher_ppl,
        verify_checkpoint_health,
    )

    torch.manual_seed(909)
    cfg = _tiny_olmoe_config(num_layers=8, num_experts=8, top_k=2)
    teacher = OlmoeForCausalLM(cfg).eval()
    student, bank, _ = build_student_from_teacher(
        teacher,
        use_rms_prescale=False,
        init_l2_scale=False,
        enable_hint_embedding=True,
    )
    device = torch.device("cpu")
    ids = torch.randint(2, cfg.vocab_size, (2, 16))
    resets = torch.zeros((2, 16), dtype=torch.bool)
    resets[:, 8] = True
    val_batches = [(ids, None, resets)]

    t_ppl = evaluate_packed_teacher_ppl(teacher, val_batches, device)
    s1_ids = collect_stage1_param_ids(student, bank)
    opt = build_colab_optimizer(student, bank, lr=1e-3)

    # Baseline step 0
    s_id, q_lam, l2_pre = configure_training_stage(
        student, s1_ids, step_idx=1, stage1_steps=2, quant_warmup_steps=2, l2_prescaled=False
    )
    rep0, probes0 = verify_checkpoint_health(
        step=0,
        prev_step=0,
        stage_id=s_id,
        active_quant_lambda=q_lam,
        student=student,
        lookahead_bank=bank,
        val_batches=val_batches,
        teacher_ppl=t_ppl,
        prev_probes=None,
        device=device,
        ckpt_root=tmp_path,
    )
    assert rep0["status"] == "OK"

    # Step 1 (Stage 1): ssm_recurrence and lookahead_bank move, bitlinear_experts stay frozen
    with torch.no_grad():
        t_out = teacher(input_ids=ids, output_hidden_states=True, use_cache=False)
    opt.zero_grad()
    loss, _ = distill_step(student, bank, ids, t_out.logits, t_out.hidden_states, eos_reset_mask=resets)
    loss.backward()
    opt.step()

    rep1, probes1 = verify_checkpoint_health(
        step=1,
        prev_step=0,
        stage_id=1,
        active_quant_lambda=0.0,
        student=student,
        lookahead_bank=bank,
        val_batches=val_batches,
        teacher_ppl=t_ppl,
        prev_probes=probes0,
        device=device,
        ckpt_root=tmp_path,
    )
    assert rep1["status"] == "OK"
    assert rep1["param_deltas"]["ssm_recurrence"] > 0.0
    assert rep1["param_deltas"]["lookahead_bank"] > 0.0
    assert rep1["param_deltas"]["bitlinear_experts"] == 0.0

    # Simulate Stage 2 STE underflow (zero expert delta when stage_id == 2) -> must FAIL
    rep_underflow, _ = verify_checkpoint_health(
        step=2,
        prev_step=1,
        stage_id=2,
        active_quant_lambda=0.5,
        student=student,
        lookahead_bank=bank,
        val_batches=val_batches,
        teacher_ppl=t_ppl,
        prev_probes=probes1,
        device=device,
    )
    assert rep_underflow["status"] == "FAIL"
    assert "zero_bitlinear_experts_update_in_stage2" in rep_underflow["reasons"]

    # Real Stage 2 step updates all subsystems -> must pass OK
    s_id, q_lam, l2_pre = configure_training_stage(
        student, s1_ids, step_idx=3, stage1_steps=2, quant_warmup_steps=2, l2_prescaled=l2_pre
    )
    opt.zero_grad()
    loss2, _ = distill_step(student, bank, ids, t_out.logits, t_out.hidden_states, eos_reset_mask=resets)
    loss2.backward()
    opt.step()

    rep2, probes2 = verify_checkpoint_health(
        step=3,
        prev_step=1,
        stage_id=s_id,
        active_quant_lambda=q_lam,
        student=student,
        lookahead_bank=bank,
        val_batches=val_batches,
        teacher_ppl=t_ppl,
        prev_probes=probes1,
        device=device,
        ckpt_root=tmp_path,
    )
    assert rep2["status"] == "OK"
    assert rep2["param_deltas"]["bitlinear_experts"] > 0.0
    assert (tmp_path / "verification_history.jsonl").is_file()

    # Inject NaN into a weight -> must FAIL with non_finite_params
    with torch.no_grad():
        next(student.parameters()).view(-1)[0] = float("nan")
    rep_nan, _ = verify_checkpoint_health(
        step=4,
        prev_step=3,
        stage_id=2,
        active_quant_lambda=1.0,
        student=student,
        lookahead_bank=bank,
        val_batches=val_batches,
        teacher_ppl=t_ppl,
        prev_probes=probes2,
        device=device,
    )
    assert rep_nan["status"] == "FAIL"
    assert any(r.startswith("non_finite_params") for r in rep_nan["reasons"])


