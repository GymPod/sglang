from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator, Optional

import numpy as np
import pybase64
import torch


_expert_routing_mask: contextvars.ContextVar[Optional[torch.Tensor]] = (
    contextvars.ContextVar("expert_routing_mask", default=None)
)


def _as_int(value) -> Optional[int]:
    return value if isinstance(value, int) else None


def get_num_experts_from_config(model_config) -> Optional[int]:
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", hf_config)
    for config in (hf_text_config, hf_config):
        if config is None:
            continue
        for name in (
            "n_routed_experts",
            "num_experts",
            "moe_num_experts",
            "num_local_experts",
        ):
            value = getattr(config, name, None)
            if isinstance(value, list):
                values = [_as_int(v) for v in value]
                if values and all(v is not None for v in values):
                    return max(values)
            elif isinstance(value, int):
                return value
    return None


def get_num_hidden_layers_from_config(model_config) -> Optional[int]:
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", hf_config)
    for config in (hf_text_config, hf_config):
        if config is None:
            continue
        value = getattr(config, "num_hidden_layers", None)
        if isinstance(value, int):
            return value
    return None


def get_num_experts_per_tok_from_config(model_config) -> Optional[int]:
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", hf_config)
    for config in (hf_text_config, hf_config):
        if config is None:
            continue
        for name in ("num_experts_per_tok", "top_k"):
            value = getattr(config, name, None)
            if isinstance(value, int):
                return value
    return None


def decode_expert_routing_mask(
    data: str,
    *,
    prompt_len: int,
    num_layers: int,
    top_k: int,
    num_experts: Optional[int],
) -> torch.Tensor:
    """Decode the user-supplied MoE routing mask.

    The wire format intentionally mirrors returned ``routed_experts``:
    base64-encoded raw little-endian int32 values, reshaped as
    ``[prompt_tokens, num_hidden_layers, num_experts_per_tok]``.
    """

    if not isinstance(data, str):
        raise ValueError("expert_routing_mask must be a base64 string.")

    try:
        raw = pybase64.b64decode(data.encode("utf-8"), validate=True)
    except Exception as exc:
        raise ValueError("expert_routing_mask is not valid base64.") from exc

    expected = prompt_len * num_layers * top_k
    if len(raw) != expected * np.dtype(np.int32).itemsize:
        actual = len(raw) // np.dtype(np.int32).itemsize
        raise ValueError(
            "expert_routing_mask must contain "
            f"{expected} int32 values, got {actual}. Expected shape is "
            f"[{prompt_len}, {num_layers}, {top_k}]."
        )

    array = np.frombuffer(raw, dtype=np.int32).copy()
    array = array.reshape(prompt_len, num_layers, top_k)

    if array.size > 0:
        if np.any(array < 0):
            raise ValueError("expert_routing_mask contains negative expert ids.")
        if num_experts is not None and np.any(array >= num_experts):
            raise ValueError(
                "expert_routing_mask contains expert ids outside the model's "
                f"[0, {num_experts}) range."
            )
        if top_k > 1:
            sorted_ids = np.sort(array, axis=-1)
            if np.any(np.diff(sorted_ids, axis=-1) == 0):
                raise ValueError(
                    "expert_routing_mask must not contain duplicate expert ids "
                    "within a token/layer row."
                )

    return torch.from_numpy(array).to(torch.int32)


@contextlib.contextmanager
def expert_routing_mask_context(
    routing_mask: Optional[torch.Tensor],
) -> Iterator[None]:
    token = _expert_routing_mask.set(routing_mask)
    try:
        yield
    finally:
        _expert_routing_mask.reset(token)


def get_current_expert_routing_mask() -> Optional[torch.Tensor]:
    return _expert_routing_mask.get()


def has_current_expert_routing_mask() -> bool:
    return _expert_routing_mask.get() is not None


def apply_expert_routing_mask(
    router_logits: torch.Tensor,
    layer_id: Optional[int],
) -> torch.Tensor:
    routing_mask = _expert_routing_mask.get()
    if routing_mask is None:
        return router_logits
    if layer_id is None:
        raise RuntimeError(
            "expert_routing_mask requires MoE TopK modules to provide layer_id."
        )
    if layer_id < 0 or layer_id >= routing_mask.shape[1]:
        raise RuntimeError(
            f"expert_routing_mask layer_id {layer_id} is outside mask shape "
            f"{tuple(routing_mask.shape)}."
        )
    if routing_mask.shape[0] != router_logits.shape[0]:
        raise RuntimeError(
            "expert_routing_mask token count does not match router logits: "
            f"{routing_mask.shape[0]} != {router_logits.shape[0]}."
        )

    allowed_ids = routing_mask[:, layer_id, :]
    valid_rows = allowed_ids[:, 0] >= 0
    if not torch.any(valid_rows):
        return router_logits

    num_experts = router_logits.shape[-1]
    active_allowed_ids = allowed_ids[valid_rows]
    if torch.any(active_allowed_ids < 0) or torch.any(active_allowed_ids >= num_experts):
        raise RuntimeError(
            "expert_routing_mask contains expert ids outside this layer's router "
            f"logit range [0, {num_experts})."
        )

    row_mask = torch.zeros_like(router_logits, dtype=torch.bool)
    row_mask[valid_rows] = True

    allowed_mask = torch.zeros_like(row_mask)
    active_allowed_mask = torch.zeros(
        (active_allowed_ids.shape[0], num_experts),
        dtype=torch.bool,
        device=router_logits.device,
    )
    active_allowed_mask.scatter_(1, active_allowed_ids.to(torch.long), True)
    allowed_mask[valid_rows] = active_allowed_mask

    masked_logits = router_logits.clone()
    masked_logits[row_mask & ~allowed_mask] = -torch.inf
    return masked_logits
