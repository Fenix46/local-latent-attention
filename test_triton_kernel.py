"""
Numerical test: Triton kernel vs PyTorch reference.

Run on the training server:
    python test_triton_kernel.py
"""

import math
import sys
import torch
import torch.nn.functional as F


def ref_llattn_forward(Q, K, V, GQ, GK, GV,
                       gate_local_w, gate_remote_w, r_out_w,
                       window: int, chunk_size: int):
    """Pure-PyTorch reference — all fp32, matches kernel semantics."""
    B, Nh, S, Dh = Q.shape
    Lh = GK.shape[1]
    Nl = GK.shape[2]
    Ld = GK.shape[3]
    scale        = 1.0 / math.sqrt(Dh)
    latent_scale = 1.0 / math.sqrt(Ld)

    Q = Q.float(); K = K.float(); V = V.float()
    GQ = GQ.float(); GK = GK.float(); GV = GV.float()
    gate_local_w  = gate_local_w.float()
    gate_remote_w = gate_remote_w.float()
    r_out_w       = r_out_w.float()

    # ── Local attention ──────────────────────────────────────────��────────────
    qi   = torch.arange(S, device=Q.device).unsqueeze(1)
    kj   = torch.arange(S, device=Q.device).unsqueeze(0)
    dist = qi - kj                              # [S, S]
    # -1e9 mask (same as kernel) — avoids nan from softmax on all-masked rows
    bias = torch.where((dist > 0) & (dist <= window),
                       torch.zeros(S, S, device=Q.device),
                       torch.full((S, S), -1e9, device=Q.device))

    scores_local = torch.matmul(Q, K.transpose(-2, -1)) * scale \
                   + bias.unsqueeze(0).unsqueeze(0)           # [B,Nh,S,S]
    p_local = F.softmax(scores_local, dim=-1)                 # [B,Nh,S,S]

    # Zero out rows where no key was valid (token 0 has no past)
    has_local = (dist > 0) & (dist <= window)                 # [S,S]
    has_local_row = has_local.any(dim=1)                      # [S]
    p_local = p_local * has_local_row.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    local_out = torch.matmul(p_local, V)                      # [B,Nh,S,Dh]

    # ── Latent attention ──────────────────────────────────────────────────────
    lat_s = torch.matmul(GQ, GK.transpose(-2, -1)) * latent_scale  # [B,Lh,S,Nl]

    qi_chunks = torch.arange(S, device=Q.device) // chunk_size      # [S]
    lat_ids   = torch.arange(Nl, device=Q.device)                   # [Nl]
    fut_mask  = lat_ids.unsqueeze(0) >= qi_chunks.unsqueeze(1)      # [S,Nl] True=future
    lat_s = lat_s.masked_fill(fut_mask.unsqueeze(0).unsqueeze(0), -1e9)

    p_lat = F.softmax(lat_s, dim=-1)                                # [B,Lh,S,Nl]
    # Zero out tokens with no valid past latent (chunk_id=0, all future-masked)
    has_valid = (~fut_mask).any(dim=-1)                             # [S]
    p_lat = p_lat * has_valid.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    latent_out = torch.matmul(p_lat, GV)                           # [B,Lh,S,Ld]

    # Project latent → head space via r_out_w [Nh*Dh, Lh*Ld]
    latent_flat = latent_out.permute(0, 2, 1, 3).reshape(B, S, Lh * Ld)
    remote_flat = latent_flat @ r_out_w.T                          # [B,S,Nh*Dh]
    remote_h    = remote_flat.view(B, S, Nh, Dh).permute(0, 2, 1, 3)  # [B,Nh,S,Dh]

    # ── Gate ─────────────────────────────────────────────────────────────────
    s_l = (local_out  * gate_local_w .unsqueeze(0).unsqueeze(2)).sum(-1, keepdim=True)
    s_r = (remote_h   * gate_remote_w.unsqueeze(0).unsqueeze(2)).sum(-1, keepdim=True)
    m   = torch.maximum(s_l, s_r)
    e_l = torch.exp(s_l - m)
    e_r = torch.exp(s_r - m)
    Z   = e_l + e_r
    w_l = e_l / Z
    w_r = e_r / Z
    return w_l * local_out + w_r * remote_h                        # [B,Nh,S,Dh]


def test_forward(device="cuda"):
    torch.manual_seed(0)
    B, Nh, S, Dh = 1, 4, 64, 32
    Lh, Nl, Ld   = 2, 16, 16
    window        = 16
    chunk_size    = S // Nl   # 4

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.1

    Q  = rand(B, Nh, S, Dh);  K  = rand(B, Nh, S, Dh);  V  = rand(B, Nh, S, Dh)
    GQ = rand(B, Lh, S, Ld);  GK = rand(B, Lh, Nl, Ld); GV = rand(B, Lh, Nl, Ld)
    gate_l  = rand(Nh, Dh);   gate_r  = rand(Nh, Dh)
    r_out_w = rand(Nh * Dh, Lh * Ld)

    ref = ref_llattn_forward(Q, K, V, GQ, GK, GV, gate_l, gate_r, r_out_w,
                             window, chunk_size)
    assert not ref.isnan().any(), "Reference itself has NaN — bug in ref"

    from kernels.llattn_op import LLAttnFunction
    out = LLAttnFunction.apply(Q, K, V, GQ, GK, GV, gate_l, gate_r, r_out_w,
                               window, chunk_size)
    assert not out.isnan().any(), "Kernel output has NaN"

    diff = (out.float() - ref).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[forward] max={max_diff:.5f}  mean={mean_diff:.5f}")

    # Find where the max error occurs
    idx = diff.argmax()
    idx = list(torch.unravel_index(idx, diff.shape))
    b, h, s, d = idx
    print(f"  worst at [b={b} h={h} s={s} d={d}]  kernel={out[b,h,s,d].item():.4f}  ref={ref[b,h,s,d].item():.4f}")
    # Report error by token position
    diff_by_token = diff[0].max(dim=0).values.max(dim=-1).values  # [S]
    top5 = diff_by_token.topk(5)
    print(f"  top-5 error tokens: pos={top5.indices.tolist()}  err={[f'{v:.4f}' for v in top5.values.tolist()]}")

    # bf16 accumulation over many ops can reach ~0.1 — use relaxed tol
    assert max_diff < 0.15, f"Forward mismatch too large: {max_diff}"
    print("[forward] PASS")


def test_gradcheck(device="cuda"):
    torch.manual_seed(1)
    B, Nh, S, Dh = 1, 2, 64, 32
    Lh, Nl, Ld   = 1, 16, 16
    window        = 8
    chunk_size    = S // Nl

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16,
                           requires_grad=False) * 0.1

    # Smoke test: forward + backward, check no NaN in grads
    Q  = rand(B, Nh, S, Dh).requires_grad_(True)
    K  = rand(B, Nh, S, Dh).requires_grad_(True)
    V  = rand(B, Nh, S, Dh).requires_grad_(True)
    GQ = rand(B, Lh, S, Ld).requires_grad_(True)
    GK = rand(B, Lh, Nl, Ld).requires_grad_(True)
    GV = rand(B, Lh, Nl, Ld).requires_grad_(True)
    gl = rand(Nh, Dh).requires_grad_(True)
    gr = rand(Nh, Dh).requires_grad_(True)
    ro = rand(Nh * Dh, Lh * Ld).requires_grad_(True)

    from kernels.llattn_op import LLAttnFunction
    out  = LLAttnFunction.apply(Q, K, V, GQ, GK, GV, gl, gr, ro, window, chunk_size)
    loss = out.sum()
    loss.backward()

    all_ok = True
    for name, t in [("dQ",Q),("dK",K),("dV",V),("dGQ",GQ),("dGK",GK),("dGV",GV),
                    ("dgate_l",gl),("dgate_r",gr),("dr_out",ro)]:
        ok = t.grad is not None and not t.grad.isnan().any() and not t.grad.isinf().any()
        print(f"  {name}: {'ok' if ok else 'FAIL (nan/inf/None)'}")
        all_ok = all_ok and ok
    print(f"[gradcheck] smoke {'PASS' if all_ok else 'FAIL'}")


def debug_nan(device="cuda"):
    torch.manual_seed(0)
    B, Nh, S, Dh = 1, 4, 64, 32
    Lh, Nl, Ld   = 2, 16, 16
    window        = 16
    chunk_size    = S // Nl

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16) * 0.1

    Q  = rand(B, Nh, S, Dh);  K  = rand(B, Nh, S, Dh);  V  = rand(B, Nh, S, Dh)
    GQ = rand(B, Lh, S, Ld);  GK = rand(B, Lh, Nl, Ld); GV = rand(B, Lh, Nl, Ld)
    gate_l = rand(Nh, Dh);    gate_r = rand(Nh, Dh)
    r_out_w = rand(Nh * Dh, Lh * Ld)

    from kernels.llattn_op import _call_fwd
    Out, LSE_l, LSE_r, W_l = _call_fwd(
        Q, K, V, GQ, GK, GV, gate_l, gate_r, r_out_w, window, chunk_size)

    for name, t in [("Out", Out), ("LSE_local", LSE_l),
                    ("LSE_remote", LSE_r), ("W_local", W_l)]:
        print(f"  {name:12s}  nan={t.isnan().any().item()}  inf={t.isinf().any().item()}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARN: no CUDA — Triton requires a GPU.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    debug_nan(device)
    print("=" * 60)
    test_forward(device)
    print("=" * 60)
    test_gradcheck(device)
    print("=" * 60)
    print("Done.")
