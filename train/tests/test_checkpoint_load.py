"""`load_student_checkpoint` must rebuild whichever variant was saved and reproduce its logits."""

import copy

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from train.bitlinear import set_model_quant_lambda
from train.distill import build_student_from_teacher
from train.distill_colab import _unpage_loaded_optimizer_state, load_student_checkpoint, save_atomic_checkpoint
from train.hybrid_ssm import GatedLinearRecurrenceAttention


def _base() -> OlmoeForCausalLM:
    torch.manual_seed(0)
    cfg = OlmoeConfig(
        vocab_size=64, hidden_size=64, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=4, num_experts=8, num_local_experts=8, num_experts_per_tok=2,
        max_position_embeddings=128, eos_token_id=1, pad_token_id=1,
    )
    return OlmoeForCausalLM(cfg).eval()


@pytest.mark.parametrize("delta,gate,hints,lam", [(False, False, True, 1.0), (True, False, False, 0.0), (True, True, False, 0.0)])
def test_round_trip_reproduces_logits(tmp_path, delta, gate, hints, lam):
    base = _base()
    student, bank, kinds = build_student_from_teacher(
        copy.deepcopy(base), use_rms_prescale=False, init_l2_scale=False, enable_hint_embedding=hints, deepcopy_model=False
    )
    for mod in student.modules():
        if isinstance(mod, GatedLinearRecurrenceAttention):
            if delta:
                mod.enable_delta_rule()
                torch.nn.init.normal_(mod.erase_proj.weight, std=0.5)  # so a dropped erase gate changes logits
            if gate:
                mod.enable_write_gate()
                torch.nn.init.normal_(mod.write_proj.weight, std=0.5)
    set_model_quant_lambda(student, lam)
    student.eval()
    ckpt = save_atomic_checkpoint(
        tmp_path, 7, student, bank, torch.optim.SGD(student.parameters(), lr=0.0), None,
        {"quant_lambda": lam, "layer_kinds": kinds},
    )

    loaded, _, loaded_kinds, info = load_student_checkpoint(copy.deepcopy(base), ckpt)
    assert (info["delta_rule"], info["write_gate"], info["hints"], info["quant_lambda"]) == (delta, gate, hints, lam)
    assert loaded_kinds == kinds
    ids = torch.randint(2, 64, (2, 24))
    with torch.no_grad():
        want = student(input_ids=ids, use_cache=False).logits
        got = loaded.eval()(input_ids=ids, use_cache=False).logits
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_resumed_optimizer_state_is_not_treated_as_paged(tmp_path):
    marked = torch.zeros(3)
    marked.is_paged = True  # what bitsandbytes sets on CUDA-managed state
    torch.save({"state1": marked}, tmp_path / "opt.pt")
    loaded = torch.load(tmp_path / "opt.pt", weights_only=False)["state1"]
    assert loaded.is_paged  # the mark survives the round trip; the managed memory does not

    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.SGD([p], lr=0.1)
    opt.state[p]["state1"] = loaded
    _unpage_loaded_optimizer_state(opt)
    assert not loaded.is_paged
