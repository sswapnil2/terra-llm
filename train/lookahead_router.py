"""Step 3C: 3-Layer Lookahead Router (`LookaheadRouterBank`) with Top-k CE & Load-Balancing Loss.

Predicts Layer `L + 3` top-`k` (`k = 8` in `OLMoE-1B-7B`) expert activations from Layer `L`'s
post-attention hidden state `h'_L` (matching Section 4.4 of `schemas/terra_format.md`) so the
Rust runtime can submit non-blocking `io_uring` / `F_NOCACHE` NVMe reads 3 layers ahead of time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_top_k_recall(
    pred_logits: torch.Tensor,
    target_topk_indices: torch.Tensor,
    k: int = 8,
) -> float:
    """Compute exact fraction of ground-truth top-`k` experts recalled by `topk(pred_logits, k)`.

    Args:
        pred_logits: `[..., num_experts]` predicted lookahead router logits at Layer `L`.
        target_topk_indices: `[..., k]` ground-truth expert indices selected at Layer `L + 3`.
        k: Number of experts per token (`8` for `OLMoE-1B-7B`).
    Returns:
        Scalar recall in `[0.0, 1.0]`.
    """
    num_experts = pred_logits.shape[-1]
    flat_logits = pred_logits.reshape(-1, num_experts)
    flat_targets = target_topk_indices.reshape(-1, k)

    pred_topk = torch.topk(flat_logits, k=k, dim=-1).indices  # [N, k]
    pred_multi_hot = torch.zeros((flat_logits.shape[0], num_experts), dtype=torch.bool, device=pred_logits.device)
    pred_multi_hot.scatter_(1, pred_topk, True)

    hits = pred_multi_hot.gather(1, flat_targets).float().sum(dim=-1)  # [N]
    return float((hits / float(k)).mean().item())


def compute_load_balancing_loss(
    pred_logits: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """Switch-style load-balancing auxiliary loss preventing lookahead router expert collapse.

    L_bal = E * sum_{e=1}^E (f_e * P_e), minimized (= 1.0) under uniform expert utilization.
    """
    num_experts = pred_logits.shape[-1]
    flat_logits = pred_logits.reshape(-1, num_experts).float()
    num_tokens = max(flat_logits.shape[0], 1)

    probs = F.softmax(flat_logits, dim=-1)  # [N, E]
    mean_prob_per_expert = probs.mean(dim=0)  # P_e: [E]

    with torch.no_grad():
        topk_idx = torch.topk(flat_logits, k=k, dim=-1).indices  # [N, k]
        dispatch_mask = F.one_hot(topk_idx, num_classes=num_experts).float()  # [N, k, E]
        frac_tokens_per_expert = dispatch_mask.sum(dim=(0, 1)) / float(num_tokens * k)  # f_e: [E]

    return float(num_experts) * torch.sum(frac_tokens_per_expert * mean_prob_per_expert)


def compute_lookahead_loss(
    pred_logits: torch.Tensor,
    target_router_logits: torch.Tensor,
    k: int = 8,
    topk_uniform_weight: float = 0.7,
    load_balance_coef: float = 0.01,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Cross-Entropy top-`k` lookahead loss + load-balancing auxiliary loss.

    Blends:
      1. Uniform multi-label Cross-Entropy over the exact top-`k` ground-truth experts
         (directly optimizes Recall@k across all `k` active experts equally), and
      2. Soft-target Cross-Entropy against Layer `L+3` router probabilities (dark knowledge).
    """
    num_experts = pred_logits.shape[-1]
    z_pred = pred_logits.reshape(-1, num_experts).float()
    z_targ = target_router_logits.detach().reshape(-1, num_experts).float()

    log_p_pred = F.log_softmax(z_pred, dim=-1)
    p_targ_soft = F.softmax(z_targ / temperature, dim=-1)
    target_topk = torch.topk(z_targ, k=k, dim=-1).indices  # [N, k]

    q_topk_uniform = torch.zeros_like(p_targ_soft)
    q_topk_uniform.scatter_(1, target_topk, 1.0 / float(k))

    blended_target = topk_uniform_weight * q_topk_uniform + (1.0 - topk_uniform_weight) * p_targ_soft
    ce_loss = -(blended_target * log_p_pred).sum(dim=-1).mean()
    bal_loss = compute_load_balancing_loss(z_pred, k=k)
    total = ce_loss + load_balance_coef * bal_loss

    recall = compute_top_k_recall(z_pred.detach(), target_topk, k=k)
    return total, {
        "ce_loss": float(ce_loss.item()),
        "bal_loss": float(bal_loss.item()),
        "recall_at_k": recall,
    }


class LookaheadRouterBank(nn.Module):
    """Auxiliary linear heads at Layer `L` predicting Layer `L + lookahead_offset` expert routing.

    Exports cleanly to `core.bin` as `layers.{l}.lookahead_router.weight` (`[num_experts, hidden_size]`).
    Can operate either explicitly on `(post_attn_states, router_logits)` lists OR automatically via
    forward hooks attached to an `OlmoeForCausalLM` model.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_experts: int,
        top_k: int = 8,
        lookahead_offset: int = 3,
    ) -> None:
        super().__init__()
        if num_layers <= lookahead_offset:
            raise ValueError(
                f"num_layers ({num_layers}) must exceed lookahead_offset ({lookahead_offset})."
            )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.lookahead_offset = lookahead_offset
        self.num_predictors = num_layers - lookahead_offset

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_experts, bias=False) for _ in range(self.num_predictors)]
        )
        for head in self.heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

        self._captured_post_attn: dict[int, torch.Tensor] = {}
        self._captured_router_logits: dict[int, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._capturing: bool = True

    def set_capturing(self, enabled: bool) -> None:
        """Enable or pause hook capture (pausing prevents gradient checkpointing backward overwrites)."""
        self._capturing = bool(enabled)

    def attach_hooks(self, olmoe_causal_lm: nn.Module) -> None:
        """Attach lightweight hooks capturing `h'_L` (input to `post_attention_layernorm`) and `mlp.gate` logits."""
        self.remove_hooks()
        layers = olmoe_causal_lm.model.layers

        for l_idx in range(self.num_predictors):
            def _make_pre_hook(layer_id: int):
                def _pre_hook(_mod: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
                    if self._capturing:
                        self._captured_post_attn[layer_id] = args[0]
                return _pre_hook

            h1 = layers[l_idx].post_attention_layernorm.register_forward_pre_hook(_make_pre_hook(l_idx))
            self._hook_handles.append(h1)

        for target_idx in range(self.lookahead_offset, self.num_layers):
            def _make_gate_hook(layer_id: int):
                def _gate_hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
                    if not self._capturing:
                        return
                    # In HF OlmoeTopKRouter, forward returns (router_logits, router_scores, router_indices)
                    if isinstance(output, tuple):
                        self._captured_router_logits[layer_id] = output[0]
                    else:
                        self._captured_router_logits[layer_id] = F.linear(inputs[0], _mod.weight)
                return _gate_hook

            h2 = layers[target_idx].mlp.gate.register_forward_hook(_make_gate_hook(target_idx))
            self._hook_handles.append(h2)

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._captured_post_attn.clear()
        self._captured_router_logits.clear()

    def predict_layer(self, src_layer_idx: int, post_attn_hidden: torch.Tensor) -> torch.Tensor:
        """Return predicted router logits `[..., num_experts]` for Layer `src_layer_idx + lookahead_offset`."""
        return self.heads[src_layer_idx](post_attn_hidden)

    def export_state_dict_entries(self) -> dict[str, torch.Tensor]:
        """Return canonical `core.bin` tensor dictionary `layers.{l}.lookahead_router.weight`."""
        return {
            f"layers.{l_idx}.lookahead_router.weight": self.heads[l_idx].weight.detach().float().cpu()
            for l_idx in range(self.num_predictors)
        }

    def forward_from_captured(
        self,
        topk_uniform_weight: float = 0.7,
        load_balance_coef: float = 0.01,
        detach_inputs: bool = True,
        override_target_router_logits: dict[int, torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute averaged lookahead loss & recall metrics across all hooked `L -> L + 3` pairs.

        By default supervises against the student's own detached Layer `L + 3` `mlp.gate` logits
        (matching runtime routing). Pass `override_target_router_logits` to supervise against
        frozen teacher router logits instead (`--lookahead-target teacher`).
        """
        if len(self._captured_post_attn) < self.num_predictors:
            raise RuntimeError(
                "LookaheadRouterBank hooks did not capture post-attention hidden states; "
                "call `attach_hooks(model)` before running `model(...)`."
            )

        losses: list[torch.Tensor] = []
        recalls: list[float] = []
        ce_vals: list[float] = []
        bal_vals: list[float] = []

        for src_l in range(self.num_predictors):
            dst_l = src_l + self.lookahead_offset
            h_src = self._captured_post_attn[src_l]
            if detach_inputs:
                h_src = h_src.detach()
            if override_target_router_logits is not None:
                g_dst = override_target_router_logits[dst_l]
            else:
                g_dst = self._captured_router_logits[dst_l]
            pred = self.heads[src_l](h_src.to(self.heads[src_l].weight.dtype))
            loss_l, stats_l = compute_lookahead_loss(
                pred,
                g_dst,
                k=self.top_k,
                topk_uniform_weight=topk_uniform_weight,
                load_balance_coef=load_balance_coef,
            )
            losses.append(loss_l)
            recalls.append(stats_l["recall_at_k"])
            ce_vals.append(stats_l["ce_loss"])
            bal_vals.append(stats_l["bal_loss"])

        self._captured_post_attn.clear()
        self._captured_router_logits.clear()

        mean_loss = torch.stack(losses).mean()
        return mean_loss, {
            "lookahead_loss": float(mean_loss.item()),
            "lookahead_ce": sum(ce_vals) / len(ce_vals),
            "lookahead_bal": sum(bal_vals) / len(bal_vals),
            "mean_recall_at_k": sum(recalls) / len(recalls),
            "min_recall_at_k": min(recalls),
        }
