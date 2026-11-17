"""Delta-rule recurrence: chunked parallel form must equal the token-by-token loop."""

import torch

from train.hybrid_ssm import GatedLinearRecurrenceAttention


def _block() -> GatedLinearRecurrenceAttention:
    torch.manual_seed(0)
    blk = GatedLinearRecurrenceAttention(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)
    blk.enable_delta_rule()
    blk.enable_write_gate()
    torch.nn.init.normal_(blk.write_proj.weight, std=0.5)
    blk.write_proj.bias.data.fill_(0.0)  # gate around 0.5 so it visibly matters
    torch.nn.init.normal_(blk.decay_proj.weight, std=0.5)
    torch.nn.init.normal_(blk.erase_proj.weight, std=0.5)
    blk.erase_proj.bias.data.fill_(0.5)  # strong erase so a wrong formula cannot hide
    return blk.eval()


def test_parallel_matches_recurrent():
    blk = _block()
    x = torch.randn(2, 37, 32)
    ref, _ = blk.forward_recurrent(x)
    for chunk in (1, 5, 16, 64):
        out = blk.forward_parallel(x, chunk_size=chunk)
        assert torch.allclose(out, ref, atol=1e-4), (chunk, (out - ref).abs().max())


def test_reset_mask_isolates_documents():
    blk = _block()
    x = torch.randn(1, 30, 32)
    reset = torch.zeros(1, 30, dtype=torch.bool)
    reset[0, 11] = True
    out = blk.forward_parallel(x, chunk_size=8, eos_reset_mask=reset)
    a, _ = blk.forward_recurrent(x[:, :11])
    b, _ = blk.forward_recurrent(x[:, 11:])
    assert torch.allclose(out, torch.cat([a, b], dim=1), atol=1e-4)


def test_erase_actually_changes_output():
    blk = _block()
    x = torch.randn(1, 20, 32)
    with_erase = blk.forward_parallel(x)
    blk.erase_proj = None
    assert not torch.allclose(with_erase, blk.forward_parallel(x), atol=1e-3)


def test_write_gate_scales_writes():
    blk = _block()
    x = torch.randn(1, 20, 32)
    gated = blk.forward_parallel(x)
    blk.write_proj.bias.data.fill_(-30.0)  # gate shut: nothing is written, so nothing can be read
    torch.nn.init.zeros_(blk.write_proj.weight)
    assert blk.forward_parallel(x).abs().max() < 1e-6
    blk.write_proj = None
    assert not torch.allclose(gated, blk.forward_parallel(x), atol=1e-3)
