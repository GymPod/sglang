from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch

from sglang.srt.layers.moe.expert_routing_mask import (
    apply_expert_routing_mask,
    decode_expert_routing_mask,
    expert_routing_mask_context,
    get_expert_routing_mask_backend_error,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


def _encode_int32(values):
    array = np.asarray(values, dtype=np.int32)
    return pybase64.b64encode(array.tobytes()).decode("utf-8")


def test_decode_expert_routing_mask_uses_returned_routed_experts_wire_format():
    encoded = _encode_int32([0, 1, 2, 3, 1, 2, 3, 4])

    mask = decode_expert_routing_mask(
        encoded,
        prompt_len=2,
        num_layers=2,
        top_k=2,
        num_experts=5,
    )

    assert mask.dtype == torch.int32
    assert mask.shape == (2, 2, 2)
    assert mask.tolist() == [[[0, 1], [2, 3]], [[1, 2], [3, 4]]]


def test_decode_expert_routing_mask_rejects_duplicates():
    encoded = _encode_int32([0, 0])

    with pytest.raises(ValueError, match="duplicate"):
        decode_expert_routing_mask(
            encoded,
            prompt_len=1,
            num_layers=1,
            top_k=2,
            num_experts=4,
        )


def test_apply_expert_routing_mask_masks_only_valid_rows():
    routing_mask = torch.tensor(
        [
            [[1, 3]],
            [[-1, -1]],
        ],
        dtype=torch.int32,
    )
    router_logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
    )

    with expert_routing_mask_context(routing_mask):
        masked = apply_expert_routing_mask(router_logits, layer_id=0)

    assert torch.isneginf(masked[0, 0])
    assert masked[0, 1] == router_logits[0, 1]
    assert torch.isneginf(masked[0, 2])
    assert masked[0, 3] == router_logits[0, 3]
    assert torch.equal(masked[1], router_logits[1])


def test_radix_cache_keeps_same_tokens_with_different_routing_metadata_separate():
    tree = RadixCache.create_simulated()
    key = RadixKey(token_ids=[1, 2, 3], extra_key=None)
    mask_a = torch.tensor([[[0, 1]], [[1, 2]], [[2, 3]]], dtype=torch.int32)
    mask_b = torch.tensor([[[0, 1]], [[3, 4]], [[2, 3]]], dtype=torch.int32)

    tree.insert(
        InsertParams(
            key=key,
            value=torch.tensor([10, 11, 12]),
            expert_routing_mask=mask_a,
        )
    )
    tree.insert(
        InsertParams(
            key=key,
            value=torch.tensor([20, 21, 22]),
            expert_routing_mask=mask_b,
        )
    )
    tree.insert(
        InsertParams(key=key, value=torch.tensor([30, 31, 32]))
    )

    match_a = tree.match_prefix(
        MatchPrefixParams(key=key, expert_routing_mask=mask_a)
    )
    match_b = tree.match_prefix(
        MatchPrefixParams(key=key, expert_routing_mask=mask_b)
    )
    match_unmasked = tree.match_prefix(MatchPrefixParams(key=key))

    assert match_a.device_indices.tolist() == [10, 11, 12]
    assert match_b.device_indices.tolist() == [10, 21, 22]
    assert match_unmasked.device_indices.tolist() == [30, 31, 32]


def test_radix_cache_supplied_mask_can_reuse_exact_actual_routes():
    tree = RadixCache.create_simulated()
    key = RadixKey(token_ids=[1, 2, 3], extra_key=None)
    mask = torch.tensor([[[0, 1]], [[1, 2]], [[2, 3]]], dtype=torch.int32)

    tree.insert(
        InsertParams(
            key=key,
            value=torch.tensor([10, 11, 12]),
            expert_routing_mask=mask,
            expert_routing_source="actual",
        )
    )

    match_masked = tree.match_prefix(
        MatchPrefixParams(key=key, expert_routing_mask=mask)
    )
    match_unmasked = tree.match_prefix(MatchPrefixParams(key=key))

    assert match_masked.device_indices.tolist() == [10, 11, 12]
    assert match_unmasked.device_indices.tolist() == [10, 11, 12]


def test_radix_cache_unmasked_request_does_not_reuse_supplied_only_routes():
    tree = RadixCache.create_simulated()
    key = RadixKey(token_ids=[1, 2, 3], extra_key=None)
    mask = torch.tensor([[[0, 1]], [[1, 2]], [[2, 3]]], dtype=torch.int32)

    tree.insert(
        InsertParams(
            key=key,
            value=torch.tensor([10, 11, 12]),
            expert_routing_mask=mask,
            expert_routing_source="supplied",
        )
    )
    assert tree.match_prefix(MatchPrefixParams(key=key)).device_indices.tolist() == []

    tree.insert(
        InsertParams(
            key=key,
            value=torch.tensor([20, 21, 22]),
            expert_routing_mask=mask,
            expert_routing_source="actual",
        )
    )
    assert tree.match_prefix(MatchPrefixParams(key=key)).device_indices.tolist() == [
        10,
        11,
        12,
    ]


def test_expert_routing_mask_requires_input_ids_request_path():
    with pytest.raises(ValueError, match="requires input_ids"):
        GenerateReqInput(
            text="hello",
            expert_routing_mask="AAAA",
        ).normalize_batch_and_arguments()

    with pytest.raises(ValueError, match="requires input_ids"):
        GenerateReqInput(
            input_embeds=[[0.0]],
            expert_routing_mask="AAAA",
        ).normalize_batch_and_arguments()

    req = GenerateReqInput(input_ids=[1, 2, 3], expert_routing_mask="AAAA")
    req.normalize_batch_and_arguments()
    assert req.expert_routing_mask == "AAAA"


def test_expert_routing_mask_backend_validation_rejects_fused_routing():
    assert (
        get_expert_routing_mask_backend_error(
            SimpleNamespace(moe_runner_backend="triton_kernel"),
            SimpleNamespace(hf_config=SimpleNamespace()),
        )
        is not None
    )
    assert (
        get_expert_routing_mask_backend_error(
            SimpleNamespace(moe_runner_backend="flashinfer_trtllm"),
            SimpleNamespace(hf_config=SimpleNamespace()),
        )
        is not None
    )
    assert (
        get_expert_routing_mask_backend_error(
            SimpleNamespace(moe_runner_backend="flashinfer_mxfp4"),
            SimpleNamespace(hf_config=SimpleNamespace()),
        )
        is not None
    )
    assert (
        get_expert_routing_mask_backend_error(
            SimpleNamespace(moe_runner_backend="flashinfer_mxfp4"),
            SimpleNamespace(
                hf_config=SimpleNamespace(
                    quantization_config={"quant_method": "mxfp4"}
                )
            ),
        )
        is None
    )
    assert (
        get_expert_routing_mask_backend_error(
            SimpleNamespace(moe_runner_backend="triton"),
            SimpleNamespace(hf_config=SimpleNamespace()),
        )
        is None
    )
