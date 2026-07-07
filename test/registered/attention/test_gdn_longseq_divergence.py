"""Long-sequence reproducer for the current_rollout_logprob_abs_diff root cause.

RL layer-bisection pinned the first Megatron-vs-sglang divergence to the GDN
core_attn_out at layer 40, decode position 6026 (all kernel inputs bit-identical;
the fused chunk-replay decode kernel OUTPUT differs by 1 bf16 ULP, then amplifies
through 23 downstream layers to 0.125 at the logits). The existing
test_gdn_chunk_replay_decode cases top out at 71 tokens and stay green; this test
drives the SAME incremental-decode-vs-full-sequence comparison out past 6026 folds
with real Qwen3.6 GDN dims (HV=48, HK=16) to reproduce the divergence locally.

Reports the FIRST token whose incremental decode output stops being bit-identical to
the full-sequence torch_chunk reference (bf16 and fp32), so the fix can be iterated
in seconds instead of a ~90-min RL dump.

Run on a B200 pod:
  python3 -m pytest test/registered/attention/test_gdn_longseq_divergence.py -s -x
"""

import unittest

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    torch_chunk_gated_delta_rule,
    torch_gdn_gating,
)

C = 64


class _FakeLayer:
    def __init__(self, HV, HK, K, V, A_log, dt_bias):
        self.layer_id = 0
        self.num_k_heads = HK
        self.num_v_heads = HV
        self.head_k_dim = K
        self.head_v_dim = V
        self.head_q_dim = K
        self.num_q_heads = HK
        self.q_dim = HK * K
        self.k_dim = HK * K
        self.v_dim = HV * V
        self.A_log = A_log
        self.dt_bias = dt_bias


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGDNLongSeqDivergence(unittest.TestCase):
    def _make_backend(self, max_bs=8):
        be = GDNAttnBackend.__new__(GDNAttnBackend)
        be.device = torch.device("cuda")
        be.gdn_arena_max_bs = max_bs
        be.gdn_arena_rows = max_bs + 1
        be.gdn_pad_row = max_bs
        be.gdn_arena = {}
        be.gdn_slot_to_row = {}
        be.gdn_free_rows = list(range(max_bs))
        be.gdn_max_batch = max_bs + 4
        be._gdn_row_map = torch.zeros(be.gdn_max_batch, dtype=torch.int32, device="cuda")
        be._gdn_row_map_host = torch.zeros(
            be.gdn_max_batch, dtype=torch.int32, device="cpu"
        ).pin_memory()
        be._gdn_out = None
        be._gdn_cur_row_map = None
        return be

    def _full_ref(self, mixed_all, a_all, b_all, layer):
        T = mixed_all.shape[0]
        qf, kf, vf = torch.split(
            mixed_all, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q = qf.view(1, T, layer.num_k_heads, layer.head_k_dim)
        k = kf.view(1, T, layer.num_k_heads, layer.head_k_dim)
        v = vf.view(1, T, layer.num_v_heads, layer.head_v_dim)
        g, beta = torch_gdn_gating(layer.A_log, a_all, b_all, layer.dt_bias)
        core, _, _ = torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, ssm_states=None,
            cache_indices=torch.zeros(1, dtype=torch.long, device="cuda"),
            query_start_loc=torch.tensor([0, T], dtype=torch.int32, device="cuda"),
        )
        return core[0]  # [T, HV, V]

    def test_longseq_decode_bit_identical(self):
        """Decode past pos 6026 and report the first token that diverges from full-seq torch_chunk."""
        with set_batch_invariant_mode(True):
            HV, HK, K, V = 48, 16, 128, 128  # real Qwen3.6 per-rank TP=4 GDN dims
            pl = 338  # RL prompt_len
            T = 6200  # past the observed onset at pos 6026
            torch.manual_seed(1234)
            A_log = torch.randn(HV, device="cuda")
            dt_bias = torch.randn(HV, device="cuda")
            layer = _FakeLayer(HV, HK, K, V, A_log, dt_bias)
            dim = layer.q_dim + layer.k_dim + layer.v_dim
            mixed = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)

            ref = self._full_ref(mixed, a, b, layer)

            be = self._make_backend()
            cidx = torch.tensor([7], dtype=torch.long, device="cuda")
            be._seed_gdn_replay_cache(
                layer, mixed[:pl], a[:pl], b[:pl],
                torch.tensor([0, pl], dtype=torch.int32, device="cuda"), cidx, None,
            )
            first_bf16_diff = None
            first_fp32_diff = None
            max_abs = 0.0
            n_bf16_diff = 0
            for p in range(pl, T):
                be._gdn_write_row_map(cidx.tolist())
                out = be._gdn_chunk_replay_decode(
                    layer, mixed[p : p + 1], a[p : p + 1], b[p : p + 1], cidx
                )
                o_bf16 = out[0, 0].to(torch.bfloat16)
                r_bf16 = ref[p].to(torch.bfloat16)
                d = (o_bf16.float() - r_bf16.float()).abs()
                dmax = d.max().item()
                if dmax > max_abs:
                    max_abs = dmax
                if dmax != 0.0:
                    n_bf16_diff += 1
                    if first_bf16_diff is None:
                        first_bf16_diff = (p, dmax)
                if first_fp32_diff is None and not torch.equal(out[0, 0], ref[p]):
                    first_fp32_diff = p
            print(f"\n[longseq] pl={pl} T={T} HV={HV}")
            print(f"[longseq] first bf16 diff: {first_bf16_diff}")
            print(f"[longseq] first fp32 diff pos: {first_fp32_diff}")
            print(f"[longseq] n bf16-diverging tokens: {n_bf16_diff}, max|d|={max_abs}")
            self.assertIsNone(
                first_bf16_diff,
                msg=f"incremental decode diverges from full-seq torch_chunk at {first_bf16_diff}",
            )


if __name__ == "__main__":
    unittest.main()
