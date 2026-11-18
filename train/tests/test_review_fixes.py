"""Regression tests for architecture-review findings 14, 16 and 17 (see IMPLEMENTATION_REPORT.md)."""

from __future__ import annotations

import torch
from torch import nn

from train.bitlinear import (
    BitLinear,
    apply_l2_optimal_model_prescale_,
    apply_l2_optimal_row_prescale_,
    set_model_quant_lambda,
    set_model_ternary_l2_scale,
)
from train.distill_colab import StochasticBF16AdamW


def _bitlinear(seed: int = 0) -> BitLinear:
    torch.manual_seed(seed)
    return BitLinear.from_linear(nn.Linear(64, 32, bias=False), init_l2_scale=False)


def test_ternary_validation_is_continuous_through_the_prescale_step() -> None:
    """Finding 14: pure-ternary scoring before the prescale must equal the model after it."""
    layer = _bitlinear()
    x = torch.randn(4, 64)
    set_model_quant_lambda(layer, 1.0)

    naive = layer(x)  # plain absmean on un-prescaled weights: what the verifier used to score
    set_model_ternary_l2_scale(layer, True)
    scored = layer(x)  # what the verifier scores now
    set_model_ternary_l2_scale(layer, False)

    set_model_quant_lambda(layer, 1.0 - 1e-6)
    ramp_limit = layer(x)  # what training is converging to at the end of the ramp

    apply_l2_optimal_model_prescale_(layer)
    set_model_quant_lambda(layer, 1.0)
    after_prescale = layer(x)  # what the model computes from the prescale step on

    assert torch.allclose(scored, after_prescale, rtol=1e-4, atol=1e-5)
    assert torch.allclose(scored, ramp_limit, rtol=1e-3, atol=1e-4)
    # The old score was a different function: its rows are ~25% too small.
    ratio = (after_prescale.norm() / naive.norm()).item()
    assert 1.15 < ratio < 1.5, ratio


def test_model_prescale_is_idempotent_and_row_prescale_is_not() -> None:
    """Finding 17: repeating the model-level prescale must not compound the row scale."""
    layer = _bitlinear(1)
    apply_l2_optimal_model_prescale_(layer)
    once = layer.weight.detach().clone()
    apply_l2_optimal_model_prescale_(layer)
    assert torch.equal(layer.weight, once)

    # A layer built with init_l2_scale=True is already prescaled and must be left alone too.
    torch.manual_seed(2)
    pre = BitLinear.from_linear(nn.Linear(64, 32, bias=False), init_l2_scale=True)
    built = pre.weight.detach().clone()
    apply_l2_optimal_model_prescale_(pre)
    assert torch.equal(pre.weight, built)

    # The raw row operation does compound, which is why the guard exists.
    twice = apply_l2_optimal_row_prescale_(once.clone())
    assert (twice.norm() / once.norm()).item() > 1.1


def _zero_gradient_step_update(opt_cls: type, steady: int, spike_ratio: float) -> float:
    p = nn.Parameter(torch.zeros(1, 8))
    opt = opt_cls([p], lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    g = 1e-6
    for _ in range(steady):  # element 1 sees a steady small gradient
        grad = torch.zeros(1, 8)
        grad[0, 1] = g
        p.grad = grad
        opt.step()
    grad = torch.zeros(1, 8)  # element 0 of the same row spikes once
    grad[0, 0], grad[0, 1] = g * spike_ratio, g
    p.grad = grad
    opt.step()
    before = p.detach().clone()
    p.grad = torch.zeros(1, 8)  # the row gets exactly zero gradient: an expert that was not routed
    opt.step()
    return float((p.detach() - before)[0, 1])


def test_8bit_adam_never_divides_momentum_by_eps_alone() -> None:
    """Finding 16: before the fix this update was ~194x the FP32 AdamW one (spike_ratio=2500)."""
    for steady in (5, 20, 60):
        for spike_ratio in (500.0, 1500.0, 2300.0, 2500.0, 3000.0, 5000.0):
            ref = _zero_gradient_step_update(torch.optim.AdamW, steady, spike_ratio)
            got = _zero_gradient_step_update(StochasticBF16AdamW, steady, spike_ratio)
            assert abs(got) <= 3.0 * abs(ref), (steady, spike_ratio, ref, got)


def test_8bit_adam_tracks_fp32_adamw_on_a_small_regression() -> None:
    def final_loss(opt_cls: type) -> float:
        torch.manual_seed(0)
        target = torch.randn(16, 32)
        x = torch.randn(256, 32)
        y = x @ target.t()
        w = nn.Parameter(torch.zeros(16, 32))
        opt = opt_cls([w], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        for step in range(300):
            idx = torch.randperm(256, generator=torch.Generator().manual_seed(step))[:32]
            loss = ((x[idx] @ w.t() - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(((x @ w.detach().t() - y) ** 2).mean())

    ref, got = final_loss(torch.optim.AdamW), final_loss(StochasticBF16AdamW)
    assert got < 2.0 * ref + 1e-3, (ref, got)
