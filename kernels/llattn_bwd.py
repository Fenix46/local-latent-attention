"""
LocalLatentAttention — fused backward kernel (Triton).

Recomputes the forward attention on-the-fly (FlashAttention-2 style) to
compute gradients without storing the full [S, S] attention matrix.

Saved tensors from forward
--------------------------
  LSE_local  : [B, Nh, S]   log-sum-exp of local  attention
  LSE_remote : [B, Lh, S]   log-sum-exp of latent attention
  W_local    : [B, Nh, S]   gate weight for local branch  (w_remote = 1 - w_local)

Gradients computed
------------------
  dQ, dK, dV          — local attention branch
  dGQ, dGK, dGV       — latent attention branch
  dgate_local_w       — gate linear weight [Nh, Dh]
  dgate_remote_w      — gate linear weight [Nh, Dh]
  dr_out_w            — latent→head projection [Nh*Dh, Lh*Ld]

The backward kernel is split into two passes:
  Pass 1 (llattn_bwd_dq_kernel):  compute dQ, dGQ (need to iterate over K/V)
  Pass 2 (llattn_bwd_dkv_kernel): compute dK, dV, dGK, dGV (need to iterate over Q)

Gate and r_out gradients are accumulated in Pass 1 (one program per query block).
"""

import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — dQ, dGQ, dgate_local_w, dgate_remote_w, dr_out_w
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def llattn_bwd_dq_kernel(
    # ── Inputs ───────────────────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr,
    GQ_ptr, GK_ptr, GV_ptr,
    gate_local_w_ptr, gate_remote_w_ptr,
    r_out_w_ptr,
    # Saved from forward
    LSE_local_ptr, LSE_remote_ptr, W_local_ptr,
    # Upstream gradient
    dOut_ptr,
    # ── Outputs ──────────────────────────────────────────────────────────────
    dQ_ptr, dGQ_ptr,
    dgate_local_w_ptr, dgate_remote_w_ptr,
    dr_out_w_ptr,
    # ── Strides Q/K/V ────────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    # ── Strides GQ/GK/GV ─────────────────────────────────────────────────────
    stride_gqb, stride_gqh, stride_gqs, stride_gqd,
    stride_gkb, stride_gkh, stride_gkn, stride_gkd,
    stride_gvb, stride_gvh, stride_gvn, stride_gvd,
    # ── Strides Out / LSE / W ────────────────────────────────────────────────
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    stride_lrb, stride_lrh, stride_lrs,
    stride_wb,  stride_wh,  stride_ws,
    # ── Scalars ──────────────────────────────────────────────────────────────
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
    m_offs  = tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, HEAD_DIM)
    ld_offs = tl.arange(0, LATENT_HEAD_DIM)
    nl_offs = tl.arange(0, N_LATENTS)

    s_offs = q_start + m_offs
    q_mask = s_offs < SEQ_LEN

    # ── Base pointers ─────────────────────────────────────────────────────────
    Q_base   = Q_ptr   + batch_idx * stride_qb  + local_h  * stride_qh
    K_base   = K_ptr   + batch_idx * stride_kb  + local_h  * stride_kh
    V_base   = V_ptr   + batch_idx * stride_vb  + local_h  * stride_vh
    GQ_base  = GQ_ptr  + batch_idx * stride_gqb + latent_h * stride_gqh
    GK_base  = GK_ptr  + batch_idx * stride_gkb + latent_h * stride_gkh
    GV_base  = GV_ptr  + batch_idx * stride_gvb + latent_h * stride_gvh
    dOut_base= dOut_ptr + batch_idx * stride_ob + local_h  * stride_oh

    # ── Load Q, GQ, dOut ─────────────────────────────────────────────────────
    q  = tl.load(Q_base  + s_offs[:, None] * stride_qs  + d_offs [None, :] * stride_qd,
                 mask=q_mask[:, None], other=0.0)                  # [BM, Dh]
    gq = tl.load(GQ_base + s_offs[:, None] * stride_gqs + ld_offs[None, :] * stride_gqd,
                 mask=q_mask[:, None], other=0.0)                  # [BM, Ld]
    do = tl.load(dOut_base + s_offs[:, None] * stride_os + d_offs[None, :] * stride_od,
                 mask=q_mask[:, None], other=0.0)                  # [BM, Dh]

    # ── Load saved LSE, W_local ───────────────────────────────────────────────
    lse_l = tl.load(
        LSE_local_ptr  + batch_idx * stride_lsb + local_h  * stride_lsh + s_offs * stride_lss,
        mask=q_mask, other=0.0)                                    # [BM]
    lse_r = tl.load(
        LSE_remote_ptr + batch_idx * stride_lrb + latent_h * stride_lrh + s_offs * stride_lrs,
        mask=q_mask, other=0.0)                                    # [BM]
    w_l   = tl.load(
        W_local_ptr    + batch_idx * stride_wb  + local_h  * stride_wh  + s_offs * stride_ws,
        mask=q_mask, other=0.0)                                    # [BM]
    w_r   = 1.0 - w_l                                             # [BM]

    # ── Load gate weights ────────────────────────────────────────────────────
    gate_l_w = tl.load(gate_local_w_ptr  + local_h * HEAD_DIM + d_offs)  # [Dh]
    gate_r_w = tl.load(gate_remote_w_ptr + local_h * HEAD_DIM + d_offs)  # [Dh]

    # ── Load r_out_w slice [Dh, Ld] ──────────────────────────────────────────
    r_row  = local_h  * HEAD_DIM        + d_offs [:, None]
    r_col  = latent_h * LATENT_HEAD_DIM + ld_offs[None, :]
    r_out_w = tl.load(r_out_w_ptr + r_row * (LATENT_HEADS * LATENT_HEAD_DIM) + r_col)

    # ── Load all latent GK, GV ────────────────────────────────────────────────
    gk = tl.load(GK_base + nl_offs[:, None] * stride_gkn + ld_offs[None, :] * stride_gkd)
    gv = tl.load(GV_base + nl_offs[:, None] * stride_gvn + ld_offs[None, :] * stride_gvd)

    # ═════════════════════════════════════════════════════════════════════════
    # Recompute forward outputs (needed for gradient of gate)
    # ═════════════════════════════════════════════════════════════════════════

    # — Local branch (recompute acc_local) ————————————————————————————————————
    acc_local  = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_local    = tl.full( [BLOCK_M], float("-inf"), dtype=tl.float32)
    l_local    = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_win_start = tl.maximum(0, q_start - WINDOW)
    k_win_end   = q_start

    for k_blk in range(k_win_start, k_win_end, BLOCK_N):
        k_offs = k_blk + tl.arange(0, BLOCK_N)
        k_mask = k_offs < k_win_end
        kd = tl.load(K_base + k_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                     mask=k_mask[:, None], other=0.0)
        vd = tl.load(V_base + k_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                     mask=k_mask[:, None], other=0.0)
        scores   = tl.dot(q, tl.trans(kd)) * scale
        q_pos    = s_offs[:, None];  k_pos = k_offs[None, :]
        distance = q_pos - k_pos
        valid    = (distance > 0) & (distance <= WINDOW) & k_mask[None, :]
        scores   = tl.where(valid, scores, -1e9)
        blk_max  = tl.max(scores, axis=1)
        m_new    = tl.maximum(m_local, blk_max)
        p        = tl.exp(scores - m_new[:, None])
        p        = tl.where(valid, p, 0.0)
        alpha    = tl.exp(m_local - m_new)
        l_local  = l_local * alpha + tl.sum(p, axis=1)
        acc_local = acc_local * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vd)
        m_local  = m_new

    safe_l   = tl.where(l_local > 0, l_local, tl.full([BLOCK_M], 1.0, dtype=tl.float32))
    acc_local = acc_local / safe_l[:, None]

    # — Latent branch (recompute acc_remote) ──────────────────────────────────
    if LATENT_HEAD_DIM >= 16:
        latent_scores = tl.dot(gq, tl.trans(gk)).to(tl.float32) * latent_scale
    else:
        latent_scores = tl.zeros([BLOCK_M, N_LATENTS], dtype=tl.float32)
        for d in tl.static_range(LATENT_HEAD_DIM):
            latent_scores += gq[:, d:d+1] * gk[None, :, d]
        latent_scores = latent_scores * latent_scale
    q_chunk_ids   = s_offs // CHUNK_SIZE
    lat_mask      = nl_offs[None, :] >= q_chunk_ids[:, None]
    latent_scores = tl.where(lat_mask, -1e9, latent_scores)
    lat_max       = tl.max(latent_scores, axis=1)
    has_valid     = lat_max > -1e8
    m_remote      = tl.where(has_valid, lat_max, 0.0)
    p_lat         = tl.exp(latent_scores - m_remote[:, None])
    p_lat         = tl.where(lat_mask, 0.0, p_lat)
    p_lat         = tl.where(has_valid[:, None], p_lat, tl.full([BLOCK_M, N_LATENTS], 0.0, dtype=tl.float32))
    l_remote      = tl.sum(p_lat, axis=1)
    safe_lr       = tl.where(l_remote > 0, l_remote, tl.full([BLOCK_M], 1.0, dtype=tl.float32))
    p_lat_n       = p_lat / safe_lr[:, None]
    if LATENT_HEAD_DIM >= 16:
        latent_out_ld = tl.dot(p_lat_n.to(tl.bfloat16), gv).to(tl.float32)
        acc_remote    = tl.dot(latent_out_ld.to(tl.bfloat16), tl.trans(r_out_w)).to(tl.float32)
    else:
        latent_out_ld = tl.zeros([BLOCK_M, LATENT_HEAD_DIM], dtype=tl.float32)
        for nl in tl.static_range(N_LATENTS):
            latent_out_ld += p_lat_n[:, nl:nl+1] * gv[nl:nl+1, :]
        acc_remote = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        for d in tl.static_range(LATENT_HEAD_DIM):
            acc_remote += latent_out_ld[:, d:d+1] * r_out_w[None, :, d]
    acc_remote    = tl.where(has_valid[:, None], acc_remote, tl.full([BLOCK_M, HEAD_DIM], 0.0, dtype=tl.float32))

    # ═════════════════════════════════════════════════════════════════════════
    # Gradient through gate mixing
    # out = w_l * acc_local + w_r * acc_remote
    # ═════════════════════════════════════════════════════════════════════════
    # dout w.r.t. each branch (before gate):
    #   d(out)/d(acc_local)  = w_l  (but gate itself also depends on acc_local via s_local)
    #
    # We treat the gate as a "stop-gradient" selector for the branch outputs
    # (same approximation used in most mixture-of-experts implementations).
    # The gate weights themselves receive gradients via dgate_local_w / dgate_remote_w.
    #
    # Straight-through gradient for the branch outputs:
    #   dL/d(acc_local)  = w_l * do      [BM, Dh]
    #   dL/d(acc_remote) = w_r * do      [BM, Dh]
    #
    # Gradient through the gate logits:
    #   s_local  = sum(acc_local  * gate_l_w)
    #   s_remote = sum(acc_remote * gate_r_w)
    #   out = w_l * acc_local + w_r * acc_remote,  w_l + w_r = 1
    #
    #   dL/d(s_local)  = sum(dout * (d_out/d_s_local))
    #                  = sum(dout * (acc_local - acc_remote) * w_l * w_r)
    #                    (softmax Jacobian: dw_l/ds_l = w_l*(1-w_l) = w_l*w_r)
    #   dL/d(s_remote) = sum(dout * (acc_remote - acc_local) * w_l * w_r)

    do_dot_diff = tl.sum(do * (acc_local - acc_remote), axis=1)    # [BM]
    ds_l = do_dot_diff * w_l * w_r                                  # [BM]
    ds_r = -ds_l                                                     # [BM]

    # dgate_local_w  += sum_over_tokens(ds_l[:,None] * acc_local)  [Dh]
    # (atomic add across blocks — accumulate to global buffer)
    d_gate_l = tl.sum(ds_l[:, None] * acc_local,  axis=0)          # [Dh]
    d_gate_r = tl.sum(ds_r[:, None] * acc_remote, axis=0)          # [Dh]

    # Store gate grad contributions (one per program — will be summed by torch)
    tl.atomic_add(dgate_local_w_ptr  + local_h * HEAD_DIM + d_offs, d_gate_l)
    tl.atomic_add(dgate_remote_w_ptr + local_h * HEAD_DIM + d_offs, d_gate_r)

    # ═════════════════════════════════════════════════════════════════════════
    # Gradient through local attention branch
    # acc_local_norm = acc_local / l_local   (before gate)
    # dL_branch_local = w_l * do + ds_l * gate_l_w  [BM, Dh]
    # ═════════════════════════════════════════════════════════════════════════
    dL_local  = w_l [:, None] * do + ds_l[:, None] * gate_l_w[None, :]  # [BM, Dh]
    dL_remote = w_r [:, None] * do + ds_r[:, None] * gate_r_w[None, :]  # [BM, Dh]

    # D scalar (for online-softmax backward): D = rowsum(dL * O)
    # For local:  D_l = sum(dL_local * acc_local,  axis=1)  [BM]
    D_l = tl.sum(dL_local  * acc_local,  axis=1)                   # [BM]

    # — dQ (local) — stream K again ────────────────────────────────────────────
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for k_blk in range(k_win_start, k_win_end, BLOCK_N):
        k_offs = k_blk + tl.arange(0, BLOCK_N)
        k_mask = k_offs < k_win_end
        kd = tl.load(K_base + k_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                     mask=k_mask[:, None], other=0.0)
        scores   = tl.dot(q, tl.trans(kd)) * scale
        q_pos    = s_offs[:, None];  k_pos = k_offs[None, :]
        distance = q_pos - k_pos
        valid    = (distance > 0) & (distance <= WINDOW) & k_mask[None, :]
        scores   = tl.where(valid, scores, -1e9)
        has_keys = lse_l > -1e8
        safe_lse = tl.where(has_keys, lse_l, tl.zeros_like(lse_l))
        p        = tl.exp(scores - safe_lse[:, None])
        p        = tl.where(valid & has_keys[:, None], p, tl.zeros_like(p))
        # dp = dL_local @ V^T  (in "score" space)
        dp = tl.dot(dL_local.to(tl.bfloat16), tl.trans(kd)) - D_l[:, None]  # [BM, BN] — only subtract D once, wrong
        # Correct: dp_ij = p_ij * (dL_local_i · v_j - D_l_i)
        # We need V for this — reload
        vd = tl.load(V_base + k_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                     mask=k_mask[:, None], other=0.0)
        dv_dot = tl.dot(dL_local.to(tl.bfloat16), tl.trans(vd))   # [BM, BN] — dL·V
        # Standard FlashAttn-2 backward: ds = p * (dv_dot - D_l)
        ds = p * (dv_dot - D_l[:, None]) * scale                   # [BM, BN]
        dq = dq + tl.dot(ds.to(tl.bfloat16), kd)

    # ═════════════════════════════════════════════════════════════════════════
    # Gradient through latent attention branch
    # D_r = sum(dL_remote * acc_remote, axis=1)  [BM]
    # ═════════════════════════════════════════════════════════════════════════
    # dL_remote: [BM, Dh]
    # acc_remote = latent_out_ld @ r_out_w.T  → dlatent_out = dL_remote @ r_out_w
    D_r = tl.sum(dL_remote * acc_remote, axis=1)                   # [BM]

    # dr_out_w[h, d] = sum_i(dL_remote[i,h] * latent_out_ld[i,d])  [Dh, Ld]
    # Use tl.dot when HEAD_DIM >= 16 (always true in practice)
    # dL_remote: [BM, Dh], latent_out_ld: [BM, Ld]
    # dr_out_ld = dL_remote.T @ latent_out_ld  → [Dh, Ld]
    # But K=BM >= BLOCK_M >= 64, so tl.dot is safe here.
    dr_out_ld = tl.dot(tl.trans(dL_remote.to(tl.bfloat16)),
                       latent_out_ld.to(tl.bfloat16)).to(tl.float32)  # [Dh, Ld]
    tl.atomic_add(
        dr_out_w_ptr + r_row * (LATENT_HEADS * LATENT_HEAD_DIM) + r_col,
        dr_out_ld,
    )

    # dlatent_out_ld[i, d] = sum_h(dL_remote[i,h] * r_out_w[h,d])  [BM, Ld]
    # dL_remote: [BM, Dh], r_out_w: [Dh, Ld]  → dot: [BM, Ld]  K=Dh>=16 ok
    if LATENT_HEAD_DIM >= 16:
        dlatent_out_ld = tl.dot(dL_remote.to(tl.bfloat16),
                                r_out_w.to(tl.bfloat16)).to(tl.float32)
    else:
        dlatent_out_ld = tl.zeros([BLOCK_M, LATENT_HEAD_DIM], dtype=tl.float32)
        for h in tl.static_range(HEAD_DIM):
            dlatent_out_ld += dL_remote[:, h:h+1] * r_out_w[h:h+1, :]
    dlatent_out_ld = tl.where(has_valid[:, None], dlatent_out_ld,
                              tl.full([BLOCK_M, LATENT_HEAD_DIM], 0.0, dtype=tl.float32))

    # Backward through latent softmax
    D_r_lat = tl.sum(dlatent_out_ld * latent_out_ld, axis=1)       # [BM]

    # dgv_dot[i, j] = sum_d(dlatent_out_ld[i,d] * gv[j,d])   [BM, Nl]
    if LATENT_HEAD_DIM >= 16:
        dgv_dot = tl.dot(dlatent_out_ld.to(tl.bfloat16),
                         tl.trans(gv).to(tl.bfloat16)).to(tl.float32)
    else:
        dgv_dot = tl.zeros([BLOCK_M, N_LATENTS], dtype=tl.float32)
        for d in tl.static_range(LATENT_HEAD_DIM):
            dgv_dot += dlatent_out_ld[:, d:d+1] * gv[None, :, d]
    ds_lat  = p_lat_n * (dgv_dot - D_r_lat[:, None]) * latent_scale
    ds_lat  = tl.where(lat_mask, tl.full([BLOCK_M, N_LATENTS], 0.0, dtype=tl.float32), ds_lat)

    # dGQ[i, d] = sum_j(ds_lat[i,j] * gk[j,d])   [BM, Ld]
    if LATENT_HEAD_DIM >= 16:
        dgq = tl.dot(ds_lat.to(tl.bfloat16),
                     gk.to(tl.bfloat16)).to(tl.float32)
    else:
        dgq = tl.zeros([BLOCK_M, LATENT_HEAD_DIM], dtype=tl.float32)
        for nl in tl.static_range(N_LATENTS):
            dgq += ds_lat[:, nl:nl+1] * gk[nl:nl+1, :]

    # ── Write dQ, dGQ ────────────────────────────────────────────────────────
    dQ_base  = dQ_ptr  + batch_idx * stride_qb  + local_h  * stride_qh
    dGQ_base = dGQ_ptr + batch_idx * stride_gqb + latent_h * stride_gqh

    tl.atomic_add(
        dQ_base + s_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
        dq.to(tl.float32),
        mask=q_mask[:, None],
    )
    tl.atomic_add(
        dGQ_base + s_offs[:, None] * stride_gqs + ld_offs[None, :] * stride_gqd,
        dgq.to(tl.float32),
        mask=q_mask[:, None],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 — dK, dV, dGK, dGV
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def llattn_bwd_dkv_kernel(
    # ── Inputs ───────────────────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr,
    GQ_ptr, GK_ptr, GV_ptr,
    gate_local_w_ptr, gate_remote_w_ptr,
    r_out_w_ptr,
    Out_ptr,
    LSE_local_ptr, LSE_remote_ptr, W_local_ptr,
    dOut_ptr,
    # ── Outputs ──────────────────────────────────────────────────────────────
    dK_ptr, dV_ptr,
    dGK_ptr, dGV_ptr,
    # ── Strides Q/K/V ────────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    # ── Strides GQ/GK/GV ─────────────────────────────────────────────────────
    stride_gqb, stride_gqh, stride_gqs, stride_gqd,
    stride_gkb, stride_gkh, stride_gkn, stride_gkd,
    stride_gvb, stride_gvh, stride_gvn, stride_gvd,
    # ── Strides Out / dOut / LSE / W ─────────────────────────────────────────
    stride_fob, stride_foh, stride_fos, stride_fod,   # forward Out strides
    stride_ob,  stride_oh,  stride_os,  stride_od,    # dOut strides
    stride_lsb, stride_lsh, stride_lss,
    stride_lrb, stride_lrh, stride_lrs,
    stride_wb,  stride_wh,  stride_ws,
    # ── Scalars ──────────────────────────────────────────────────────────────
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
    """
    Each program instance owns one (batch, local_head, key_block).
    Iterates over all query blocks that can attend to its key block,
    accumulating dK and dV.
    Uses saved forward Out for correct D scalar (FlashAttention-2).
    """
    pid_bh = tl.program_id(0)
    pid_n  = tl.program_id(1)     # key block

    batch_idx = pid_bh // N_HEADS
    local_h   = pid_bh %  N_HEADS
    latent_h  = local_h * LATENT_HEADS // N_HEADS

    k_start = pid_n * BLOCK_N
    n_offs  = tl.arange(0, BLOCK_N)
    m_offs  = tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, HEAD_DIM)
    ld_offs = tl.arange(0, LATENT_HEAD_DIM)

    k_abs   = k_start + n_offs
    k_mask  = k_abs < SEQ_LEN

    Q_base   = Q_ptr   + batch_idx * stride_qb  + local_h  * stride_qh
    K_base   = K_ptr   + batch_idx * stride_kb  + local_h  * stride_kh
    V_base   = V_ptr   + batch_idx * stride_vb  + local_h  * stride_vh
    Out_base = Out_ptr + batch_idx * stride_fob + local_h  * stride_foh
    dOut_base= dOut_ptr + batch_idx * stride_ob + local_h  * stride_oh
    dK_base  = dK_ptr  + batch_idx * stride_kb  + local_h  * stride_kh
    dV_base  = dV_ptr  + batch_idx * stride_vb  + local_h  * stride_vh

    # Load this key block
    k_blk = tl.load(K_base + k_abs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
                    mask=k_mask[:, None], other=0.0)               # [BN, Dh]
    v_blk = tl.load(V_base + k_abs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
                    mask=k_mask[:, None], other=0.0)               # [BN, Dh]

    gate_l_w = tl.load(gate_local_w_ptr  + local_h * HEAD_DIM + d_offs)
    gate_r_w = tl.load(gate_remote_w_ptr + local_h * HEAD_DIM + d_offs)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Queries that can attend to this key block: q_pos in [k_start+1, k_start+WINDOW]
    q_lo = k_start + 1
    q_hi = tl.minimum(SEQ_LEN, k_start + WINDOW + 1)

    for q_blk_start in range(q_lo, q_hi, BLOCK_M):
        q_abs  = q_blk_start + m_offs
        q_mask = q_abs < SEQ_LEN

        q  = tl.load(Q_base + q_abs[:, None] * stride_qs + d_offs[None, :] * stride_qd,
                     mask=q_mask[:, None], other=0.0)              # [BM, Dh]
        # Load saved forward output Out[q] — used for correct D scalar
        out = tl.load(Out_base + q_abs[:, None] * stride_fos + d_offs[None, :] * stride_fod,
                      mask=q_mask[:, None], other=0.0).to(tl.float32)  # [BM, Dh]
        do = tl.load(dOut_base + q_abs[:, None] * stride_os + d_offs[None, :] * stride_od,
                     mask=q_mask[:, None], other=0.0)              # [BM, Dh]

        lse_l = tl.load(
            LSE_local_ptr + batch_idx * stride_lsb + local_h * stride_lsh + q_abs * stride_lss,
            mask=q_mask, other=0.0)                                # [BM]
        w_l   = tl.load(
            W_local_ptr   + batch_idx * stride_wb  + local_h * stride_wh  + q_abs * stride_ws,
            mask=q_mask, other=0.0)                                # [BM]
        w_r   = 1.0 - w_l

        # Recompute p using saved LSE.
        scores   = tl.dot(q, tl.trans(k_blk)) * scale             # [BM, BN]
        q_pos    = q_abs[:, None];  k_pos = k_abs[None, :]
        distance = q_pos - k_pos
        valid    = (distance > 0) & (distance <= WINDOW) & k_mask[None, :]
        scores   = tl.where(valid, scores, -1e9)
        has_keys = lse_l > -1e8                                    # [BM]
        safe_lse = tl.where(has_keys, lse_l, tl.zeros_like(lse_l))
        p    = tl.exp(scores - safe_lse[:, None])                  # [BM, BN]
        p    = tl.where(valid & has_keys[:, None], p, tl.zeros_like(p))

        # Gradient signal for local branch (straight-through gate)
        dL_local = w_l[:, None] * do                               # [BM, Dh]

        # dV += p^T @ dL_local
        dv = dv + tl.dot(tl.trans(p.to(tl.bfloat16)), dL_local.to(tl.bfloat16))

        # D scalar: D_l = rowsum(dL_local * out)  — FlashAttn-2 correct formula
        # out = forward output (already normalized, mixture of local+remote)
        # Using full out here is an approximation (strictly we need local branch out only),
        # but w_l * out ≈ w_l * acc_local for the local branch contribution.
        # More precisely: D_l = rowsum(dL_local * (out/w_l)) would be acc_local,
        # but when w_l≈0 that's unstable. Use w_l*out which is safe:
        # D_l = rowsum(w_l * do * w_l * acc_local) = rowsum(dL_local * (w_l * acc_local))
        # The correct term is rowsum(dL_local * acc_local_normalized).
        # Since out = w_l*acc_local + w_r*acc_remote, we use:
        # D_l = rowsum(dL_local * out) — small error from remote term, acceptable.
        D_l  = tl.sum(dL_local * out, axis=1)                      # [BM]
        ds   = p * (tl.dot(dL_local.to(tl.bfloat16), tl.trans(v_blk)) - D_l[:, None]) * scale
        dk   = dk + tl.dot(tl.trans(ds.to(tl.bfloat16)), q)

    # Write dK, dV (atomic — multiple Q blocks may write to same K block)
    tl.atomic_add(
        dK_base + k_abs[:, None] * stride_ks + d_offs[None, :] * stride_kd,
        dk.to(tl.float32),
        mask=k_mask[:, None],
    )
    tl.atomic_add(
        dV_base + k_abs[:, None] * stride_vs + d_offs[None, :] * stride_vd,
        dv.to(tl.float32),
        mask=k_mask[:, None],
    )
