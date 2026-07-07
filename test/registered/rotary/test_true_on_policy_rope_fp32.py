"""True-on-policy rotary must bit-match Megatron's fp32 rotary.

sglang's forward_native uses apply_rotary_emb (utils.py). It previously cast cos/sin to the
input bf16 dtype and did the o1=x1*cos-x2*sin multiply in bf16, which rounds ~1 bf16 ULP off
Megatron's fp32 rotary (rope_utils: t.float()*cos_ + _rotate_half(t).float()*sin_) on rare
query values. That ULP folds into the GDN recurrent state and broke decode/train bit-identity
(current_rollout_logprob_abs_diff onset at a single mid-sequence token). The fix computes the
apply in fp32. This test reproduces Megatron's fp32 neox rotary and asserts bit-identity.
"""
import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb


def _megatron_rotate_half(x):
    # rope_utils._rotate_half, rotary_interleaved=False (neox)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _megatron_bshd_fp32(t_bf16, cos_half, sin_half):
    # Megatron _apply_rotary_pos_emb_bshd: freqs rot_dim = full = 2*half; cos_=cos(freqs).float().
    # Duplicate the half cos/sin across the two chunk-halves (cat), then fp32 multiply, cast back.
    # unsqueeze the head dim to broadcast over num_heads (t is [tokens, heads, rot_dim])
    cos_full = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(-2).float()
    sin_full = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(-2).float()
    t = t_bf16.float()
    out = (t * cos_full) + (_megatron_rotate_half(t) * sin_full)
    return out.to(t_bf16.dtype)


def test_true_on_policy_rope_bit_matches_megatron_fp32():
    torch.manual_seed(0)
    num_tokens, num_heads, rot_dim = 512, 6, 64
    # bf16 query (post qk-norm), fp32 cos/sin cache (true-on-policy CUDA cache stays fp32)
    x = torch.randn(num_tokens, num_heads, rot_dim, dtype=torch.bfloat16)
    # cos/sin are head_size//2 = rot_dim//2 each (sglang forward_native chunks cos_sin in half)
    angle = torch.randn(num_tokens, rot_dim // 2, dtype=torch.float32)
    cos, sin = angle.cos(), angle.sin()

    sg = apply_rotary_emb(x, cos, sin, is_neox_style=True)
    meg = _megatron_bshd_fp32(x, cos, sin)
    max_abs = (sg.float() - meg.float()).abs().max().item()
    assert max_abs == 0.0, f"true-on-policy rope not bit-identical to Megatron fp32: {max_abs}"


if __name__ == "__main__":
    test_true_on_policy_rope_bit_matches_megatron_fp32()
    print("OK: true-on-policy rope bit-matches Megatron fp32")
