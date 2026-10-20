"""GQABwdOp-shaped Ascend example over the validated v6 builders.

Inputs/outputs follow the inspected TileOPs BSHD contract. LSE is FP32
base-2 [B,H,S], converted internally to the kernel's natural-log layout.
This module does not register a TileOPs backend or supply a forward kernel.
"""

import math

from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _compile_pair(seq_len, groups, is_causal, device_index, dtype):
    import tilelang

    from core_bwd import flash_attention_bwd_fused_dq_atomic_staged, flash_attention_bwd_preprocess

    with torch.npu.device(device_index):
        config = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
        preprocess = tilelang.compile(flash_attention_bwd_preprocess(seq_len * groups, 128, dtype=dtype), out_idx=[2], pass_configs=config)
        backward = tilelang.compile(
            flash_attention_bwd_fused_dq_atomic_staged(seq_len * groups, seq_len, 128, is_causal=is_causal, groups=groups, dtype=dtype),
            out_idx=[],
            pass_configs=config,
        )
    return preprocess, backward


def lse_log2_to_ln(lse):
    """Convert base-2 LSE to natural log without modifying caller input."""
    return lse * math.log(2.0)


@torch.no_grad()
def gqa_bwd(q, k, v, o, do, lse, *, is_causal=True):
    """GQABwdOp-shaped call: q/k/v/o/do, FP32 base-2 lse -> dq/dk/dv.

    Defaults: causal=True, scale=1/sqrt(D); no dropout/bias. D=128 and the
    original tile constraints remain implementation limits beyond the schema.
    """
    if not isinstance(is_causal, bool):
        raise ValueError("is_causal must be bool")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q/k must have BSHD rank four")
    batch, seq_len, heads, dim = q.shape
    if min(batch, seq_len, heads) <= 0 or dim != 128:
        raise ValueError("positive B/S/H and D=128 are required")
    heads_kv = k.shape[2]
    if heads_kv <= 0 or heads % heads_kv:
        raise ValueError("H_kv must be positive and divide H")
    if k.shape != (batch, seq_len, heads_kv, dim) or v.shape != k.shape:
        raise ValueError("k/v must have matching square-attention shapes")
    if o.shape != q.shape or do.shape != q.shape or lse.shape != (batch, heads, seq_len):
        raise ValueError("expected o/do like q and base-2 LSE in [B,H,S]")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype for x in (k, v, o, do)) or lse.dtype != torch.float32:
        raise ValueError("matching FP16/BF16 q/k/v/o/do and FP32 base-2 lse are required")
    if any(x.device != q.device for x in (k, v, o, do, lse)):
        raise ValueError("all inputs must be on the same device")
    if q.device.type not in ("npu", "privateuseone"):
        raise ValueError("Ascend NPU tensors are required")
    groups = heads // heads_kv
    if seq_len % 128 or (seq_len * groups) % 128 or seq_len * groups < 384:
        raise ValueError("128-row alignment and at least three query tiles are required")
    device_index = q.device.index if q.device.index is not None else torch.npu.current_device()
    with torch.npu.device(device_index):
        lse_ln = lse_log2_to_ln(lse)
        preprocess, backward = _compile_pair(seq_len, groups, bool(is_causal), device_index, str(q.dtype).split(".")[-1])
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        for b in range(batch):
            for kv_head in range(heads_kv):
                start, end = kv_head * groups, (kv_head + 1) * groups
                # Contiguous groups of Q heads share this KV head. Preserve
                # sequence-major rows s*groups+g, never merge batches/KV heads.
                q_group = q[b, :, start:end, :].contiguous().reshape(seq_len * groups, dim)
                do_group = do[b, :, start:end, :].contiguous().reshape(seq_len * groups, dim)
                o_group = o[b, :, start:end, :].contiguous().reshape(seq_len * groups, dim)
                k_group = k[b, :, kv_head, :].contiguous()
                v_group = v[b, :, kv_head, :].contiguous()
                lse_group = lse_ln[b, start:end, :].transpose(0, 1).contiguous().reshape(-1)
                delta = preprocess(o_group, do_group)
                dq_group = torch.zeros_like(q_group)
                dk_group, dv_group = torch.empty_like(k_group), torch.empty_like(v_group)
                backward(q_group, k_group, v_group, do_group, lse_group, delta, dq_group, dk_group, dv_group)
                dq[b, :, start:end, :] = dq_group.reshape(seq_len, groups, dim)
                dk[b, :, kv_head, :] = dk_group
                dv[b, :, kv_head, :] = dv_group
    return dq, dk, dv
