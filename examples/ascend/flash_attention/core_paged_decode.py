"""Paged MHA decode kernel for Ascend NPU (Cube + Vector cores).

One block serves one ``(batch, head)`` pair of
``O = softmax(Q @ K^T / sqrt(d)) @ V`` over a paged KV cache::

    Q:   [B, S_q, H, D]
    K/V: pools of [pool_rows, H, D] rows addressed through a page table
    O:   [B, S_q, H, D]

Every 128-row key tile is scored with an online softmax; probabilities are
written directly in NZ layout so the second GEMM consumes them without an
ND-to-NZ conversion. A key tile never straddles a page: pages of 256 rows walk
as two BC=128 sub-blocks whose page id ``block_table[b, kv // SP]`` and
in-page offset the block computes at run time.

Two inputs are the caller's business because the kernel walks whole tiles:

* ``BIAS`` — an additive float32 mask of shape ``[B * BR, NUM_KV * BC]`` built
  by :func:`build_attention_bias`: 0 where the key is visible, -1e30 where the
  causal end or the cache length cuts it. Online softmax turns a fully cut tile
  into an exact zero contribution.
* **Sanitized pools** — rows no request reads may hold NaN; see
  :func:`sanitize_pools`.

The padding rows of the staged query tile (``BR`` rows, only the first
``S_q`` real) are never copied from Q, so the caller slices the output back to
``S_q`` rows; a causal query that sees no key outputs zeros because its softmax
denominator is zero (see the example wrapper).
"""

from __future__ import annotations

import torch
import tilelang

import tilelang.ascend.language as T
from tilelang.layout import make_ascend_compact_nz_layout
from tilelang.ascend.language import simd as S

# Query rows one block stages, and key rows per tile. Baked into the compiled
# program.
BR = 16
BC = 128
VL = 64  # lanes one PAT_ALL vector operation reads or writes


def paged_decode_fwd(
    batch: int,
    seqlen_q: int,
    heads: int,
    dim: int,
    page_size: int,
    max_pages_per_req: int,
    pool_rows: int,
    dtype: str = "float16",
):
    """Compile the paged decode kernel for one shape specialization.

    The returned callable takes ``(Q, BT, RL, BIAS, KP, VP, O)`` with ``O`` a
    preallocated ``(B, H, BR, D)`` buffer, and fills every ``BR`` row of it.
    """
    B = batch
    S_Q = seqlen_q
    H = heads
    D = dim
    PAGE = page_size
    W = max_pages_per_req
    POOL_ROWS = pool_rows
    SP = PAGE // BC  # sub-blocks per page
    NUM_KV = W * SP  # key tiles per request
    ROWS = BR // 2  # rows per AIV of the dual-lane vector ops
    out_dtype = dtype
    accum_dtype = "float32"
    # NZ packing converts through a 16-bit type whose bit layout matches the
    # element type for both float16 and bfloat16; only the names differ.
    pack_dtype = "bfloat16" if dtype == "bfloat16" else "float16"
    pack_vec = pack_dtype + "x128"
    scale = dim**-0.5

    assert page_size % BC == 0, f"page_size must be a multiple of {BC}, got {page_size}"
    assert seqlen_q <= BR, f"seqlen_q must be at most {BR}, got {seqlen_q}"
    assert dim % VL == 0, f"dim must be a multiple of {VL}, got {dim}"

    @T.macro
    def qk(base, h, KP, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(KP[base:base + BC, h, 0:D], K_shared[0:BC, 0:D])
        T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
        T.copy(K_shared[0:BC, 0:D], qk_b_l0)
        T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])

    @T.macro
    def softmax(S_ub, P_nz_ub, bias_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                pack_dtype,
            )

            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")

            for r in range(0, ROWS, 2):
                b0lo = S.vld(bias_ub[r, 0], "NORM")
                b0hi = S.vld(bias_ub[r, VL], "NORM")
                b1lo = S.vld(bias_ub[r + 1, 0], "NORM")
                b1hi = S.vld(bias_ub[r + 1, VL], "NORM")
                s00 = S.vadd(S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full),
                             b0lo, softmax_full)
                s01 = S.vadd(S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full),
                             b0hi, softmax_full)
                s10 = S.vadd(S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full),
                             b1lo, softmax_full)
                s11 = S.vadd(S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full),
                             b1hi, softmax_full)
                S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")
                S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")

                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            merged_max = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(p00, max0, softmax_full)
                e10 = S.vexpdif(p10, max1, softmax_full)
                e01 = S.vexpdif(p01, max0, softmax_full)
                e11 = S.vexpdif(p11, max1, softmax_full)
                row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                S.vsts(row_sum_ub[r], row_sum0, softmax_one, "ONEPT_B32")
                S.vsts(row_sum_ub[r + 1], row_sum1, softmax_one, "ONEPT_B32")

                even_pk_0 = S.vcvt(e00, pack_dtype, softmax_full, sat=False, part=0)
                even_pk_1 = S.vcvt(e10, pack_dtype, softmax_full, sat=False, part=0)
                odd_pk_0 = S.vcvt(e01, pack_dtype, softmax_full, sat=False, part=1)
                odd_pk_1 = S.vcvt(e11, pack_dtype, softmax_full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_pk_0, "uint16x128"),
                    T.reinterpret(odd_pk_0, "uint16x128"),
                    softmax_b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_pk_1, "uint16x128"),
                    T.reinterpret(odd_pk_1, "uint16x128"),
                    softmax_b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, pack_vec),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, pack_vec),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )

            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv(base, h, VP, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(VP[base:base + BC, h, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")

    @T.macro
    def kv_body(b, kv, base, h, KP, VP, Q_shared, K_shared, V_shared, P_shared,
                qk_a_l0, qk_b_l0, pv_a_l0, pv_b_l0, qk_acc_l0c, pv_acc_l0c,
                S_ub, P_nz_ub, bias_ub, O_tmp_ub, O_ub, m_ub, l_ub, alpha_ub,
                old_m_ub, row_sum_ub, BIAS):
        T.copy(BIAS[b * BR:(b + 1) * BR, kv * BC:(kv + 1) * BC], bias_ub[0:BR, 0:BC])
        qk(base, h, KP, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        softmax(S_ub, P_nz_ub, bias_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(base, h, VP, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def finalize_cast(O, b, h, O_ub, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        with T.SimdVF(latency=193):
            cast_full = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D, VL):
                    ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)
                    S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")
        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[b, h, 0:BR, 0:D])

    @T.prim_func
    def mha_decode_paged(
        Q: T.Buffer((B, S_Q, H, D), dtype),
        BT: T.Buffer((B, W), "int32"),
        RL: T.Buffer((B,), "int32"),
        BIAS: T.Buffer((B * BR, NUM_KV * BC), accum_dtype),
        KP: T.Buffer((POOL_ROWS, H, D), dtype),
        VP: T.Buffer((POOL_ROWS, H, D), dtype),
        O: T.Buffer((B, H, BR, D), out_dtype),
    ):
        with T.Kernel(B * H) as bx:
            b: T.int32 = bx // H
            h: T.int32 = bx % H
            # W and POOL_ROWS appear only in the shape annotations above; the
            # annotation evaluator resolves names through the function's closure,
            # so pin them with these (otherwise dead) TIR scalars.
            page_width: T.int32 = W
            pool_rows: T.int32 = POOL_ROWS
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, D), dtype)
            V_shared = T.alloc_l1((D, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BC), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            bias_ub = T.alloc_shared((BR, BC), accum_dtype)
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            m_ub = T.alloc_shared((VL,), accum_dtype)
            l_ub = T.alloc_shared((VL,), accum_dtype)
            alpha_ub = T.alloc_shared((VL,), accum_dtype)
            old_m_ub = T.alloc_shared((VL,), accum_dtype)
            row_sum_ub = T.alloc_shared((VL,), accum_dtype)
            T.annotate_buffer_versions(
                {
                    K_shared: 3,
                    V_shared: 3,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 3,
                    P_nz_ub: 2,
                    alpha_ub: 3,
                    O_tmp_ub: 2,
                }
            )

            T.copy(Q[b, 0:S_Q, h, 0:D], Q_shared[0:S_Q, 0:D])
            init_m_l(m_ub, l_ub)
            clear_O_ub(O_ub)

            rl = RL[b]
            for kv in T.serial(NUM_KV):
                with T.If(kv * BC < rl):
                    with T.Then():
                        pg = kv // SP
                        off = (kv % SP) * BC
                        base = BT[b, pg] * PAGE + off
                        kv_body(b, kv, base, h, KP, VP, Q_shared, K_shared, V_shared, P_shared,
                                qk_a_l0, qk_b_l0, pv_a_l0, pv_b_l0, qk_acc_l0c, pv_acc_l0c,
                                S_ub, P_nz_ub, bias_ub, O_tmp_ub, O_ub, m_ub, l_ub, alpha_ub,
                                old_m_ub, row_sum_ub, BIAS)

            O_cast_ub = T.alloc_shared((ROWS, D), out_dtype)
            finalize_cast(O, b, h, O_ub, O_cast_ub, l_ub)

    return tilelang.compile(mha_decode_paged)


def build_attention_bias(
    batch: int,
    seqlen_q: int,
    num_kv: int,
    real_seqlen_kv: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """The additive float32 mask the kernel consumes, of shape ``[B * BR, NUM_KV * BC]``.

    0 where the key is visible, -1e30 where the causal end or the cache length
    cuts it. A causal query ``i`` sees keys up to ``i + real_seqlen_kv - seqlen_q``.
    """
    dev = real_seqlen_kv.device
    kv_pos = torch.arange(num_kv * BC, device=dev, dtype=torch.float32).view(1, 1, -1)
    row_idx = torch.arange(BR, device=dev, dtype=torch.float32).view(1, BR, 1)
    rl_f = real_seqlen_kv.to(torch.float32).view(batch, 1, 1)
    if is_causal:
        last_key = row_idx + rl_f - seqlen_q
        visible = kv_pos <= last_key
    else:
        visible = (kv_pos < rl_f).expand(batch, BR, num_kv * BC)
    return torch.where(visible, 0.0, -1.0e30).view(batch * BR, num_kv * BC)


def sanitize_pools(
    k: torch.Tensor,
    v: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero every pool row no request reads, so a cut tile contributes an exact zero.

    Pool rows past a request's cache end may hold NaN; a tile the cache length
    cuts would otherwise poison its scores through masked-but-nonzero lanes.
    Returns possibly-cloned pools; hand them to the kernel untouched.
    """
    pool_rows = k.shape[0]
    max_pages = block_table.shape[1]
    pos = torch.arange(max_pages * page_size, device=k.device)
    rows = block_table[:, pos // page_size] * page_size + pos % page_size
    read = torch.zeros(pool_rows, dtype=torch.bool, device=k.device)
    read[rows[pos[None, :] < real_seqlen_kv[:, None]]] = True
    if not bool((~read).any()):
        return k, v
    unread = ~read
    k_safe = k.clone()
    v_safe = v.clone()
    k_safe[unread] = 0
    v_safe[unread] = 0
    return k_safe, v_safe


def run_decode(
    kernel,
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """One end-to-end decode call: bias, sanitized pools, kernel, output slice.

    Returns a fresh ``(B, S_q, H, D)`` tensor. A causal query that sees no key
    keeps exact zeros (its softmax denominator is 0).
    """
    batch, seqlen_q, heads, dim = q.shape
    num_kv = block_table.shape[1] * (page_size // BC)
    bias = build_attention_bias(batch, seqlen_q, num_kv, real_seqlen_kv, is_causal=is_causal)
    k_safe, v_safe = sanitize_pools(k_pool, v_pool, real_seqlen_kv, block_table, page_size)
    o_buf = torch.empty(batch, heads, BR, dim, dtype=q.dtype, device=q.device)
    kernel(q, block_table, real_seqlen_kv, bias, k_safe, v_safe, o_buf)
    out = o_buf[:, :, :seqlen_q, :].permute(0, 2, 1, 3).contiguous()
    if is_causal:
        q_rows = torch.arange(seqlen_q, device=q.device)
        no_key = (q_rows[None, :] + real_seqlen_kv[:, None] - seqlen_q) < 0
        out[no_key] = 0
    return out
