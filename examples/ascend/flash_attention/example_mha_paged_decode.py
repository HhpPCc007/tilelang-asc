"""Paged MHA decode on Ascend NPU: correctness + bandwidth benchmark.

Decode is memory-bound, so the benchmark reports effective HBM bandwidth
(cache bytes the kernel must move divided by kernel time) alongside latency.
"""

import argparse
import math

import torch
import tilelang

from tilelang.profiler import do_bench

from core_paged_decode import paged_decode_fwd, run_decode


def make_inputs(batch, heads, seqlen_kv, seqlen_q, dim, page_size, dtype,
                device, real_lengths=None, fragment_pages=True):
    """Inputs per call: a fragmented page table and per-request cache lengths."""
    num_pages = seqlen_kv // page_size
    table_width = max(num_pages, 1)
    q = torch.randn(batch, seqlen_q, heads, dim, device=device, dtype=dtype) * 0.5
    k = torch.randn(seqlen_kv, heads, dim, device=device, dtype=dtype) * 0.5
    v = torch.randn(seqlen_kv, heads, dim, device=device, dtype=dtype) * 0.5
    if real_lengths is None:
        real_lengths = [seqlen_kv] * batch
    real = torch.tensor(real_lengths, dtype=torch.int32, device=device)
    if fragment_pages:
        # A shuffled page id per row of the table; every page is used once.
        table = torch.stack(
            [torch.randperm(table_width, device=device)[:table_width] for _ in range(batch)]
        ).to(torch.int32)
    else:
        table = torch.arange(table_width, device=device, dtype=torch.int32).repeat(batch, 1)
    return q, k, v, real, table


def ref_program(Q, KP, VP, real_seqlen_kv, block_table, page_size, is_causal=False):
    """Reassemble paged K/V per batch, then dense attention in float32."""
    batch, seqlen_q, heads, dim = Q.shape
    pool_rows = KP.shape[0]
    outs = []
    for b in range(batch):
        kv_len = int(real_seqlen_kv[b])
        num_pages = math.ceil(kv_len / page_size)
        k_logical = torch.zeros(pool_rows, heads, dim, dtype=Q.dtype, device=Q.device)
        v_logical = torch.zeros(pool_rows, heads, dim, dtype=Q.dtype, device=Q.device)
        for p in range(num_pages):
            start = int(block_table[b, p]) * page_size
            end = min(start + page_size, pool_rows)
            k_logical[p * page_size:p * page_size + (end - start)] = KP[start:end]
            v_logical[p * page_size:p * page_size + (end - start)] = VP[start:end]
        q_b = Q[b].float().transpose(0, 1)
        k_b = k_logical[:kv_len].float().transpose(0, 1)
        v_b = v_logical[:kv_len].float().transpose(0, 1)
        scores = q_b @ k_b.transpose(1, 2) * dim**-0.5
        if is_causal:
            last_key = torch.arange(seqlen_q, device=Q.device)[:, None] + kv_len - seqlen_q
            visible = torch.arange(kv_len, device=Q.device)[None, :] <= last_key
            scores = scores.masked_fill(~visible, float("-inf"))
        probs = scores.softmax(-1).nan_to_num(0.0)
        outs.append((probs @ v_b).transpose(0, 1).unsqueeze(0).to(Q.dtype))
    return torch.cat(outs, dim=0)


def check_close(got, expected, atol=0.01, rtol=0.05):
    """The mixed criterion attention values need: pure relative tolerance blows up
    on near-zero outputs, where normal fp16 accumulation noise dominates."""
    diff = (got.float() - expected.float()).abs()
    bad = (diff > atol + rtol * expected.float().abs()).sum().item()
    assert bad == 0, f"{bad} elements exceed |d| <= {atol} + {rtol}*|exp| (max |d|={diff.max().item():.2e})"


def run_case(batch, heads, seqlen_kv, seqlen_q, dim, page_size, dtype,
             is_causal=False, real_lengths=None, device=torch.device("npu")):
    """Compile, check correctness against the reference, return the kernel and inputs."""
    kernel = paged_decode_fwd(
        batch, seqlen_q, heads, dim, page_size,
        seqlen_kv // page_size, seqlen_kv, dtype="float16" if dtype == torch.float16 else "bfloat16",
    )
    q, k, v, real, table = make_inputs(batch, heads, seqlen_kv, seqlen_q, dim, page_size,
                                       dtype, device, real_lengths)
    got = run_decode(kernel, q, k, v, real, table, page_size, is_causal=is_causal)
    expected = ref_program(q, k, v, real, table, page_size, is_causal)
    check_close(got, expected)
    return kernel, (q, k, v, real, table)


def run_regression_perf(batch=2, heads=32, seqlen_kv=16384, seqlen_q=1, dim=128,
                        page_size=256, dtype=torch.float16, device=torch.device("npu")):
    """Compile + benchmark the decode kernel, reporting latency and effective bandwidth."""
    kernel, (q, k, v, real, table) = run_case(
        batch, heads, seqlen_kv, seqlen_q, dim, page_size, dtype, device=device)

    def call():
        run_decode(kernel, q, k, v, real, table, page_size)

    prof = do_bench(call, backend="msprof_detail", _n_warmup=10, _n_repeat=100)
    # Effective bytes: K + V pool rows actually read, plus Q and O.
    total_rl = int(real.sum())
    eff_bytes = 2 * total_rl * heads * dim * q.element_size() + q.numel() * q.element_size() * 2
    gbps = eff_bytes / (prof.dur_ns / 1e3) / 1e3
    print(f"    [B={batch} H={heads} kv={seqlen_kv} S_q={seqlen_q} D={dim} p={page_size} "
          f"{str(dtype).removeprefix('torch.')}] {prof.dur_us:.2f} us/iter  |  {gbps:.0f} GB/s effective")
    return prof.dur_ns / 1e6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paged MHA decode on Ascend NPU")
    parser.add_argument("--no-perf", action="store_true", help="correctness checks only")
    args = parser.parse_args()

    device = torch.device("npu")
    torch.manual_seed(42)
    print("correctness:")
    run_case(1, 8, 2048, 1, 128, 128, torch.float16, device=device)
    print("  basic fp16 B1 H8 kv2048 p128 ............ PASS")
    run_case(2, 8, 1024, 1, 64, 256, torch.float16,
             real_lengths=[700, 3], device=device)
    print("  NaN pool, rl=[700, 3], p256 D64 ......... PASS")
    run_case(2, 8, 1024, 4, 64, 256, torch.float16,
             is_causal=True, real_lengths=[1024, 3], device=device)
    print("  causal S_q=4, no-key rows ............... PASS")
    run_case(2, 8, 2048, 1, 128, 128, torch.bfloat16, device=device)
    print("  bf16 B2 H8 kv2048 p128 .................. PASS")

    if not args.no_perf:
        print("benchmark:")
        run_regression_perf(batch=8, heads=8, seqlen_kv=2048, device=device)
        run_regression_perf(batch=2, heads=32, seqlen_kv=16384, page_size=256, device=device)
