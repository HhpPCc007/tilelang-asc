"""pytest test for example_mha_paged_decode.py — paged MHA decode kernel.

Covers the paged contract: a fragmented page table, per-request cache lengths
shorter than the static bound, NaN-poisoned unread pool rows, causal masks with
queries that see no key, and both 16-bit element types.
"""

import pytest
import torch

from example_mha_paged_decode import make_inputs, ref_program, run_case, check_close
from core_paged_decode import paged_decode_fwd, run_decode


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_decode_basic(dtype):
    kernel, (q, k, v, real, table) = run_case(
        1, 8, 2048, 1, 128, 128, dtype, device=torch.device("npu"))
    got = run_decode(kernel, q, k, v, real, table, 128)
    check_close(got, ref_program(q, k, v, real, table, 128))


def test_paged_decode_cache_shorter_than_bound():
    """A cache far shorter than the table; unread pool rows hold NaN and must
    not reach the output."""
    batch, heads, seqlen_kv, seqlen_q, dim, page_size = 2, 8, 1024, 1, 64, 256
    device = torch.device("npu")
    kernel = paged_decode_fwd(
        batch, seqlen_q, heads, dim, page_size, seqlen_kv // page_size, seqlen_kv)
    q, k, v, _full, table = make_inputs(
        batch, heads, seqlen_kv, seqlen_q, dim, page_size, torch.float16, device)
    real = torch.tensor([700, 3], dtype=torch.int32, device=device)
    pos = torch.arange(table.shape[1] * page_size, device=device)
    rows = table[:, pos // page_size] * page_size + pos % page_size
    read = torch.zeros(seqlen_kv, dtype=torch.bool, device=device)
    read[rows[pos[None, :] < real[:, None]]] = True
    k[~read] = float("nan")
    v[~read] = float("nan")

    got = run_decode(kernel, q, k, v, real, table, page_size)
    assert torch.isfinite(got).all(), "output is not finite for a partly filled cache"
    check_close(got, ref_program(q, k, v, real, table, page_size))


def test_paged_decode_causal_no_key_rows():
    """Causal queries over a shorter cache: rows that see no key output zeros."""
    batch, heads, seqlen_kv, seqlen_q, dim, page_size = 2, 8, 1024, 4, 64, 256
    device = torch.device("npu")
    real_lengths = [1024, 3]
    kernel, (q, k, v, real, table) = run_case(
        batch, heads, seqlen_kv, seqlen_q, dim, page_size, torch.float16,
        is_causal=True, real_lengths=real_lengths, device=device)
    got = run_decode(kernel, q, k, v, real, table, page_size, is_causal=True)
    # Batch 1 keeps 3 keys: its last query row sees none of them and must be zero.
    assert (got[1, -1] == 0).all()
    check_close(got, ref_program(q, k, v, real, table, page_size, is_causal=True))


if __name__ == "__main__":
    test_paged_decode_basic(torch.float16)
    test_paged_decode_basic(torch.bfloat16)
    test_paged_decode_cache_shorter_than_bound()
    test_paged_decode_causal_no_key_rows()
    print("PASS: test_mha_paged_decode")
