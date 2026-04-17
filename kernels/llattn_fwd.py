"""
LocalLatentAttention — fused forward kernel (Triton).
"""

import triton
import triton.language as tl


@triton.jit
def llattn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    GQ_ptr, GK_ptr, GV_ptr,
    gate_local_w_ptr, gate_remote_w_ptr,
    r_out_w_ptr,
    Out_ptr, LSE_local_ptr, LSE_remote_ptr, W_local_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_gqb, stride_gqh, stride_gqs, stride_gqd,
    stride_gkb, stride_gkh, stride_gkn, stride_gkd,
    stride_gvb, stride_gvh, stride_gvn, stride_gvd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    stride_lrb, stride_lrh, stride_lrs,
    stride_wb,  stride_wh,  stride_ws,
    SEQ_LEN:         tl.constexpr,
    N_HEADS:         tl.constexpr,
    LATENT_HEADS:    tl.constexpr,
    N_LATENTS:       tl.constexpr,
    HEAD_DIM:        tl.constexpr,
    LATENT_HEAD_DIM: tl.constexpr,
    WINDOW:          tl.constexpr,
    CHUNK_SIZE:      tl.constexpr,
    scale,
    latent_scale,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    batch_idx = pid_bh // N_HEADS
    local_h   = pid_bh %  N_HEADS
    latent_h  = local_h * LATENT_HEADS // N_HEADS

    q_start = pid_m * BLOCK_M

    Q_base  = Q_ptr  + batch_idx * stride_qb + local_h  * stride_qh
    K_base  = K_ptr  + batch_idx * stride_kb + local_h  * stride_kh
    V_base  = V_ptr  + batch_idx * stride_vb + local_h  * stride_vh
    GQ_base = GQ_ptr + batch_idx * stride_gqb + latent_h * stride_gqh
    GK_base = GK_ptr + batch_idx * stride_gkb + latent_h * stride_gkh
    GV_base = GV_ptr + batch_idx * stride_gvb + latent_h * stride_gvh

    d_offs  = tl.arange(0, HEAD_DIM)
    ld_offs = tl.arange(0, LATENT_HEAD_DIM)
    nl_offs = tl.arange(0, N_LATENTS)

    # ── Gate weights [Dh] ────────────────────────────────────────────────────
    gate_l_w = tl.load(gate_local_w_ptr  + local_h * HEAD_DIM + d_offs).to(tl.float32)
    gate_r_w = tl.load(gate_remote_w_ptr + local_h * HEAD_DIM + d_offs).to(tl.float32)

    # ── r_out_w slice [Dh, Ld] ──────────────────────────────────────────────
    # r_out_w layout: [Nh*Dh, Lh*Ld], stride = Lh*Ld per row
    r_row = local_h  * HEAD_DIM        + d_offs [:, None]
    r_col = latent_h * LATENT_HEAD_DIM + ld_offs[None, :]
    r_out_w = tl.load(
        r_out_w_ptr + r_row * (LATENT_HEADS * LATENT_HEAD_DIM) + r_col
    ).to(tl.float32)   # [Dh, Ld]

    # ── Latent K, V [Nl, Ld] ────────────────────────────────────────────────
    gk = tl.load(
        GK_base + nl_offs[:, None] * stride_gkn + ld_offs[None, :] * stride_gkd
    ).to(tl.float32)
    gv = tl.load(
        GV_base + nl_offs[:, None] * stride_gvn + ld_offs[None, :] * stride_gvd
    ).to(tl.float32)

    # ── Query block [BLOCK_M, Dh] and [BLOCK_M, Ld] ─────────────────────────
    m_offs = tl.arange(0, BLOCK_M)
    s_offs = q_start + m_offs
    q_mask = s_offs < SEQ_LEN

    q  = tl.load(Q_base  + s_offs[:, None] * stride_qs  + d_offs [None, :] * stride_qd,
                 mask=q_mask[:, None], other=0.0).to(tl.float32)
    gq = tl.load(GQ_base + s_offs[:, None] * stride_gqs + ld_offs[None, :] * stride_gqd,
                 mask=q_mask[:, None], other=0.0).to(tl.float32)

    # ── LOCAL ATTENTION (online softmax) ─────────────────────────────────────
    acc_local = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i       = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i       = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_lo = tl.maximum(0, q_start - WINDOW)
    k_hi = q_start  # strict causal: keys [k_lo, q_start)

    for k_start in range(k_lo, k_hi, BLOCK_N):
        k_offs = k_start + tl.arange(0, BLOCK_N)
        kv_mask = k_offs < k_hi

        k_blk = tl.load(K_base + k_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                        mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_blk = tl.load(V_base + k_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                        mask=kv_mask[:, None], other=0.0).to(tl.float32)

        s = tl.dot(q.to(tl.bfloat16), tl.trans(k_blk).to(tl.bfloat16)).to(tl.float32) * scale

        dist  = s_offs[:, None] - k_offs[None, :]
        valid = (dist > 0) & (dist <= WINDOW) & kv_mask[None, :]
        s     = tl.where(valid, s, -1e9)   # large negative, NOT -inf → avoids NaN

        blk_m = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, blk_m)

        # safe exp: subtract running max
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid, p, 0.0)        # zero out masked positions explicitly

        alpha     = tl.exp(m_i - m_new)
        # when m_i == -inf (no prior keys): alpha would be exp(-inf) = 0,
        # but we want alpha=1 so acc_local is not zeroed out spuriously.
        # However since acc_local starts at 0 and l_i=0, multiplying by 0 is fine.
        l_i       = l_i * alpha + tl.sum(p, axis=1)
        acc_local = acc_local * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_blk.to(tl.bfloat16)).to(tl.float32)
        m_i       = m_new

    # Normalize — if no keys attended (l_i=0), output is 0 (acc_local was never written)
    safe_l    = tl.where(l_i > 0.0, l_i, 1.0)
    acc_local = acc_local / safe_l[:, None]
    # lse for backward: -inf is ok for "no keys" — bwd will use it safely
    lse_local = tl.where(l_i > 0.0, m_i + tl.log(l_i), -1e9)

    # ── LATENT ATTENTION ─────────────────────────────────────────────────────
    # gq: [BM, Ld],  gk: [Nl, Ld]
    lat_s = tl.dot(gq.to(tl.bfloat16), tl.trans(gk).to(tl.bfloat16)).to(tl.float32) * latent_scale

    q_chunk = s_offs // CHUNK_SIZE
    fut_mask = nl_offs[None, :] >= q_chunk[:, None]   # mask future latents
    lat_s = tl.where(fut_mask, -1e9, lat_s)

    # softmax over latents
    lat_max   = tl.max(lat_s, axis=1)                   # [BM]
    has_valid = lat_max > -1e8                           # True if at least one valid latent
    lat_max   = tl.where(has_valid, lat_max, 0.0)

    p_lat  = tl.exp(lat_s - lat_max[:, None])
    p_lat  = tl.where(fut_mask, 0.0, p_lat)
    l_lat  = tl.sum(p_lat, axis=1)
    safe_ll = tl.where(l_lat > 0.0, l_lat, 1.0)
    p_lat_n = p_lat / safe_ll[:, None]
    lse_remote = tl.where(l_lat > 0.0, lat_max + tl.log(l_lat), -1e9)

    # Weighted sum over GV: [BM, Ld]
    lat_out = tl.dot(p_lat_n.to(tl.bfloat16), gv.to(tl.bfloat16)).to(tl.float32)
    lat_out = tl.where(has_valid[:, None], lat_out, 0.0)

    # Project to head space: [BM, Dh] = lat_out @ r_out_w.T
    # r_out_w: [Dh, Ld]  → tl.trans → [Ld, Dh]
    acc_remote = tl.dot(lat_out.to(tl.bfloat16), tl.trans(r_out_w).to(tl.bfloat16)).to(tl.float32)
    acc_remote = tl.where(has_valid[:, None], acc_remote, 0.0)

    # ── GATE (online softmax over 2 logits) ───────────────────────────────────
    s_l = tl.sum(acc_local  * gate_l_w[None, :], axis=1)   # [BM]
    s_r = tl.sum(acc_remote * gate_r_w[None, :], axis=1)   # [BM]

    g_max  = tl.maximum(s_l, s_r)
    exp_l  = tl.exp(s_l - g_max)
    exp_r  = tl.exp(s_r - g_max)
    Z      = exp_l + exp_r
    w_l    = exp_l / Z
    w_r    = exp_r / Z

    out = w_l[:, None] * acc_local + w_r[:, None] * acc_remote

    # ── Store outputs ─────────────────────────────────────────────────────────
    Out_base = Out_ptr + batch_idx * stride_ob + local_h * stride_oh
    tl.store(Out_base + s_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
             out.to(tl.bfloat16), mask=q_mask[:, None])

    tl.store(LSE_local_ptr  + batch_idx * stride_lsb + local_h  * stride_lsh + s_offs * stride_lss,
             lse_local,  mask=q_mask)
    tl.store(LSE_remote_ptr + batch_idx * stride_lrb + latent_h * stride_lrh + s_offs * stride_lrs,
             lse_remote, mask=q_mask)
    tl.store(W_local_ptr    + batch_idx * stride_wb  + local_h  * stride_wh  + s_offs * stride_ws,
             w_l, mask=q_mask)
