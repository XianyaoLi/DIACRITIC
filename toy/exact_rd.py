"""
Reveal toy (`toy_reveal`, deterministic reveal) -- exact computations (Fig 0 panels A & B).
Pure numpy; no training. Run:  python exact_rd.py | tee results_exact_rd.log
Checks: R_E(0)=log2 R (Thm 1/2), BA == closed form on the reduced source, BA on the full history source == reduced
(rate--distortion reduction theorem numerical check, M-independent), deterministic frontier R^det_{G|O}(D) (Appendix A.1).

Source at decision time T:
  U_beh ~ Unif[M], g(u) = ceil(u R / M) in [R]
  theta_nui ~ Unif[N] independent
  History H_T = (z0 = theta_nui, b(g) bits, approach action a0 in A0)   -- determines g exactly (A2 holds)
  O_T = X_T only, independent of everything               -> G independent of O_T
  Expert at T: deterministic action g(u). Distortion: Hamming on class.
"""
import numpy as np
from itertools import product
from math import log2, ceil

def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

def closed_form_RD(R, D):
    """R-ary uniform source, Hamming distortion. Valid for 0 <= D <= 1-1/R."""
    if R == 1:
        return np.zeros_like(D)
    D = np.asarray(D, float)
    out = log2(R) - h2(D) - D * log2(R - 1) if R > 2 else log2(R) - h2(D)
    out[D >= 1 - 1 / R] = 0.0
    return np.maximum(out, 0.0)

def blahut_arimoto(p_x, d, beta, iters=2000, tol=1e-10):
    """
    Standard BA for R(D) of source p_x with distortion matrix d[x, xhat].
    Returns (rate_bits, distortion) for the slope parameter beta (>= 0).
    """
    nx, nxh = d.shape
    q = np.full(nxh, 1.0 / nxh)          # output marginal
    ed = np.exp(-beta * d)
    for _ in range(iters):
        # conditional p(xhat | x) ∝ q(xhat) exp(-beta d)
        pc = q[None, :] * ed
        pc /= pc.sum(1, keepdims=True)
        q_new = p_x @ pc
        if np.abs(q_new - q).max() < tol:
            q = q_new
            break
        q = q_new
    pc = q[None, :] * ed
    pc /= pc.sum(1, keepdims=True)
    joint = p_x[:, None] * pc
    D = (joint * d).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, pc / q[None, :], 1.0)
        Rb = (joint * np.log2(ratio)).sum()
    return max(Rb, 0.0), D

def rd_curve(p_x, d, betas):
    pts = np.array([blahut_arimoto(p_x, d, b) for b in betas])
    return pts[:, 0], pts[:, 1]

# ---------- sources ----------
def reduced_source(R):
    """Source G ~ Unif[R], reproduction [R], Hamming."""
    p = np.full(R, 1.0 / R)
    d = 1.0 - np.eye(R)
    return p, d

def history_source(M, R, N, A0_size=2):
    """
    Full history source H = (z0, u, a0). Reproduction = class label in [R].
    Distortion d(h, ghat) = 1[g(u) != ghat]. BA on this gives I(C;H|O_T)
    where C is the reproduction. rate--distortion reduction theorem predicts it equals the reduced curve.
    """
    g_of_u = np.array([ceil((u + 1) * R / M) for u in range(M)]) - 1   # 0-indexed classes
    rows, dist = [], []
    for z, u, a0 in product(range(N), range(M), range(A0_size)):
        rows.append(1.0 / (N * M * A0_size))
        dist.append([0.0 if g_of_u[u] == gh else 1.0 for gh in range(R)])
    return np.array(rows), np.array(dist), g_of_u

# ---------- deterministic frontier (encoder = function of G) ----------
def set_partitions(elems):
    if not elems:
        yield []
        return
    first, rest = elems[0], elems[1:]
    for smaller in set_partitions(rest):
        for i, block in enumerate(smaller):
            yield smaller[:i] + [[first] + block] + smaller[i + 1:]
        yield [[first]] + smaller

def deterministic_frontier_G(R):
    """
    Enumerate all deterministic maps [R] -> codes (equivalently set partitions of [R]).
    For each block, decoder outputs one class of the block -> block error (|block|-1)/R.
    Rate = H(C) = entropy of block-size distribution. Return Pareto-optimal (rate, D).
    """
    pts = []
    for part in set_partitions(list(range(R))):
        sizes = np.array([len(b) for b in part], float) / R
        rate = -(sizes * np.log2(sizes)).sum()
        D = sum((len(b) - 1) for b in part) / R
        pts.append((rate, D, len(part)))
    pts = np.array(pts)
    # Pareto: minimal rate for each distortion level
    pareto = []
    for D in np.unique(pts[:, 1]):
        sub = pts[pts[:, 1] == D]
        pareto.append(sub[sub[:, 0].argmin()])
    return np.array(pareto)

# ---------- entropy identities ----------
def entropy_identities(M, R, N):
    HU = log2(M)
    HG = log2(R)
    H_U_given_G = log2(M / R)
    H_nui = log2(N)
    return dict(H_Ubeh=HU, H_G_given_O=HG, harmless_beh=H_U_given_G, harmless_nui=H_nui,
                check=abs(HU - (HG + H_U_given_G)) < 1e-12)

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    betas = np.concatenate([np.linspace(0, 2, 21), np.linspace(2.2, 12, 50)])

    print("=== Panel A: R_E(0) = log2 R, H(U_beh) fixed at log2 M ===")
    for M in [16]:
        for R in [1, 2, 4, 8]:
            ident = entropy_identities(M, R, 4)
            p, d = reduced_source(R)
            r0, D0 = blahut_arimoto(p, d, beta=60.0)
            print(f"M={M:2d} R={R}: H(U_beh)={ident['H_Ubeh']:.2f}  H(G|O)={ident['H_G_given_O']:.2f}  "
                  f"harmless_beh={ident['harmless_beh']:.2f}  BA R(D≈0)={r0:.4f} (D={D0:.1e})  identity_ok={ident['check']}")

    print("\n=== Panel B: closed form vs BA (reduced source), R=4 ===")
    R = 4
    p, d = reduced_source(R)
    rates, Ds = rd_curve(p, d, betas)
    cf = closed_form_RD(R, Ds)
    err = np.abs(rates - cf)
    print(f"max |BA - closed form| over curve = {err.max():.2e}")
    for i in range(0, len(Ds), 10):
        print(f"  D={Ds[i]:.3f}  BA={rates[i]:.4f}  closed={cf[i]:.4f}")

    print("\n=== rate--distortion reduction theorem numerical check: BA on full history source == reduced ===")
    for M in [4, 8, 16]:
        pH, dH, _ = history_source(M, R=4, N=4)
        rH, DH = rd_curve(pH, dH, betas)
        cfH = closed_form_RD(4, DH)
        print(f"M={M:2d} (R=4, N=4, |H|={len(pH)}): max |BA_history - closed form| = {np.abs(rH - cfH).max():.2e}; "
              f"R(D≈0)={rH[-1]:.4f}")

    print("\n=== Deterministic frontier R^det_{G|O}(D), R=4 (Pareto points: rate, D, #codes) ===")
    det = deterministic_frontier_G(4)
    for rate, D, k in det:
        print(f"  rate={rate:.4f}  D={D:.3f}  K={int(k)}   (stochastic lower bound at this D: {closed_form_RD(4, np.array([D]))[0]:.4f})")

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        # Panel A
        Rs = [1, 2, 4, 8]
        ax[0].bar(np.arange(4) - 0.2, [log2(16)] * 4, width=0.4, label=r"$H(U_{beh})$ (M=16)")
        ax[0].bar(np.arange(4) + 0.2, [log2(r) for r in Rs], width=0.4, label=r"$H(G_E\mid O)=R_E(0)$")
        ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"R={r}" for r in Rs])
        ax[0].set_ylabel("bits"); ax[0].set_title("A. Same world uncertainty, less behavioral uncertainty")
        ax[0].legend()
        # Panel B
        ax[1].plot(Ds, cf, "k-", lw=2, label="closed form")
        ax[1].plot(Ds, rates, "o", ms=3, label="BA (reduced source G)")
        for M in [4, 8, 16]:
            pH, dH, _ = history_source(M, 4, 4)
            rH, DH = rd_curve(pH, dH, betas)
            ax[1].plot(DH, rH, "--", lw=1, label=f"BA (history source, M={M})")
        ax[1].plot(det[:, 1], det[:, 0], "rs-", label=r"$R^{det}_{G\mid O}$ (Pareto)")
        ax[1].set_xlabel("Hamming distortion D"); ax[1].set_ylabel("rate (bits)")
        ax[1].set_title("B. Rate–distortion, R=4 (M-independent)"); ax[1].legend(fontsize=7)
        plt.tight_layout()
        import os; out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fig0_AB.png")
        plt.savefig(out, dpi=150)
        print("\nsaved", out)
    except Exception as e:
        print("plotting skipped:", e)
