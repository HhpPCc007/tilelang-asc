"""TileOPs-shaped FP16/BF16 positive accuracy checks over the v5 grouped adapter.

Uses explicit FP32 attention primitives (CPU-checked against SDPA MATH), with the
inspected GQABwdWorkload BF16 atol/rtol. This is NOT a TileOPs API/control test.
The original fixed max_abs<0.02 tests are left unchanged.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from example_gqa_bwd_tileops import gqa_bwd, lse_log2_to_ln


def math_reference_groupwise(q, k, v, do):
    """Explicit FP32 primitives; never call NPU fused attention for reference."""
    batch, seq_len, heads, dim = q.shape
    heads_kv = k.shape[2]
    groups = heads // heads_kv
    output = torch.empty_like(q)
    lse_ln = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    positions = torch.arange(seq_len, device=q.device)
    future = positions[None, :] > positions[:, None]
    for b in range(batch):
        for h in range(heads_kv):
            start, end = h * groups, (h + 1) * groups
            with torch.enable_grad():
                qr = q[b, :, start:end, :].transpose(0, 1).unsqueeze(0).detach().float().requires_grad_(True)
                kr = k[b, :, h, :].unsqueeze(0).unsqueeze(0).detach().float().requires_grad_(True)
                vr = v[b, :, h, :].unsqueeze(0).unsqueeze(0).detach().float().requires_grad_(True)
                scores = (qr @ kr.transpose(-2, -1)) / math.sqrt(dim)
                scores = scores.masked_fill(future, -torch.inf)
                lse_group = scores.detach().logsumexp(-1)
                probability = scores.softmax(-1)
                result = probability @ vr
                upstream = do[b, :, start:end, :].transpose(0, 1).unsqueeze(0).float()
                grads = torch.autograd.grad(result, (qr, kr, vr), upstream)
            with torch.no_grad():
                output[b, :, start:end, :] = result[0].transpose(0, 1).to(q.dtype)
                dq[b, :, start:end, :] = grads[0][0].transpose(0, 1).to(q.dtype)
                dk[b, :, h, :] = grads[1][0, 0].to(q.dtype)
                dv[b, :, h, :] = grads[2][0, 0].to(q.dtype)
                lse_ln[b, start:end, :] = lse_group[0]
            del qr, kr, vr, result, grads, scores, probability, lse_group, upstream
    return output, lse_ln, (dq, dk, dv)


def test_math_groupwise_reference_matches_cpu_sdpa():
    """Independent CPU SDPA check; SDPA is never called on NPU by this file."""
    torch.manual_seed(7)
    q = torch.randn(2, 5, 6, 8, device="cpu")
    k, v = (torch.randn(2, 5, 2, 8, device="cpu") for _ in range(2))
    do = torch.randn_like(q)
    output, lse, refs = math_reference_groupwise(q, k, v, do)
    with torch.enable_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        qr, kr, vr = (x.detach().requires_grad_(True) for x in (q, k, v))
        full_output = F.scaled_dot_product_attention(
            qr.transpose(1, 2),
            kr.transpose(1, 2),
            vr.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        full_refs = torch.autograd.grad(full_output, (qr, kr, vr), do)
    scores = (q.transpose(1, 2) @ k.transpose(1, 2).repeat_interleave(3, dim=1).transpose(-2, -1)) / math.sqrt(8)
    positions = torch.arange(5)
    scores.masked_fill_(positions[None, :] > positions[:, None], -torch.inf)
    torch.testing.assert_close(output, full_output)
    torch.testing.assert_close(lse, scores.logsumexp(-1))
    for actual, expected in zip(refs, full_refs):
        torch.testing.assert_close(actual, expected)


def test_lse_base2_conversion_cpu():
    expected_ln = torch.tensor([-4.0, 0.0, 1.0, 8.0, 20.0], dtype=torch.float32)
    base2 = expected_ln * math.log2(math.e)
    before = base2.clone()
    actual_ln = lse_log2_to_ln(base2)
    torch.testing.assert_close(actual_ln, expected_ln, rtol=1e-6, atol=2e-6)
    assert torch.equal(base2, before)
    assert actual_ln.dtype == torch.float32


@pytest.fixture(scope="module")
def npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    return "npu"


def check_case(npu, batch, seq_len, heads, heads_kv, dtype):
    torch.manual_seed(42)
    q = torch.randn(batch, seq_len, heads, 128, device=npu, dtype=dtype)
    k, v = (torch.randn(batch, seq_len, heads_kv, 128, device=npu, dtype=dtype) for _ in range(2))
    do = torch.randn_like(q)
    print("reference_backend=explicit_fp32_matmul_softmax_autograd; CPU SDPA cross-check only; input_lse=base2")
    print(f"case: B={batch}, S={seq_len}, H={heads}, Hkv={heads_kv}, D=128, dtype={dtype}, causal, seed=42")
    o, lse, refs = math_reference_groupwise(q, k, v, do)
    # Match TileOPs workload: its externally supplied LSE uses base 2.
    lse_base2 = lse * math.log2(math.e)
    actuals = gqa_bwd(q, k, v, o, do, lse_base2, is_causal=True)
    torch.npu.synchronize()
    failures = []
    # From the inspected TileOPs GQABwdWorkload:
    # atol=max(0.005, reference_tolerance), rtol=reference_tolerance.
    atol, rtol = (0.005, 0.001) if dtype == torch.float16 else (0.016, 0.016)
    for name, actual, expected in zip(("dQ", "dK", "dV"), actuals, refs):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == dtype
        assert actual.device == expected.device == q.device
        assert torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item()
        a, ref = actual.detach().float().cpu(), expected.detach().float().cpu()
        error = (a - ref).abs()
        allowed = atol + rtol * ref.abs()
        mismatch = error > allowed
        print(
            f"{name}: max_abs_vs_rounded_ref={error.max().item():.9f}, "
            f"mismatches={mismatch.sum().item()}/{mismatch.numel()}, "
            f"max_normalized_error={(error / allowed).max().item():.9f}, atol={atol}, rtol={rtol}"
        )
        if mismatch.any():
            failures.append(f"{name}: {mismatch.sum().item()} elementwise mismatches")
            flat = (error / allowed).flatten().argmax().item()
            print(f"{name}: worst flat_index={flat}, actual={a.flatten()[flat].item()}, ref={ref.flatten()[flat].item()}")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,seq_len,heads,heads_kv",
    [
        pytest.param(1, 512, 2, 2, id="B1-S512-H2-Hkv2"),
        pytest.param(2, 256, 4, 2, id="B2-S256-H4-Hkv2"),
        pytest.param(2, 512, 8, 2, id="B2-S512-H8-Hkv2"),
    ],
)
def test_contract_smoke(npu, batch, seq_len, heads, heads_kv, dtype):
    check_case(npu, batch, seq_len, heads, heads_kv, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,seq_len,heads,heads_kv",
    [
        pytest.param(4, 512, 32, 8, id="llama-8b-short"),
        pytest.param(2, 2048, 32, 8, id="llama-8b-long"),
        pytest.param(2, 512, 64, 8, id="llama-70b-short"),
        pytest.param(1, 2048, 64, 8, id="llama-70b-long"),
        pytest.param(4, 512, 32, 32, id="llama-8b-short-mha"),
        pytest.param(2, 2048, 32, 32, id="llama-8b-long-mha"),
        pytest.param(2, 512, 64, 64, id="llama-70b-short-mha"),
        pytest.param(1, 2048, 64, 64, id="llama-70b-long-mha"),
    ],
)
def test_default_workload(npu, batch, seq_len, heads, heads_kv, dtype):
    check_case(npu, batch, seq_len, heads, heads_kv, dtype)
