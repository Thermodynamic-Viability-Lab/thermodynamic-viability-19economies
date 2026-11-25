"""
Exact variance-based Global Sensitivity Analysis (Saltelli-Jansen 2010 + Bootstrap 95% CI)
and Annealed Sequential Monte Carlo Approximate Bayesian Computation (Beaumont et al. 2009 ABC-SMC).
"""
import numpy as np
from scipy.stats import qmc


def build_saltelli_matrix(bounds, n_base_pow2, seed=2026):
    """
    Constructs scrambled Sobol QMC matrices (A, B, AB_0..AB_{k-1}) mapped to `bounds` (k x 2).
    Returns:
      X_all: ((k + 2) * N, k) float64 parameter matrix
      N: int (= 2^m)
      k: int (= len(bounds))
    Order in X_all: [A (N rows), B (N rows), AB_0 (N rows), ..., AB_{k-1} (N rows)].
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    k = bounds.shape[0]
    m = int(round(np.log2(n_base_pow2)))
    assert (1 << m) == n_base_pow2, "n_base_pow2 must be an exact power of 2 for QMC balance"
    N = 1 << m

    eng = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    U = eng.random_base2(m=m)
    lo = bounds[:, 0]
    span = bounds[:, 1] - bounds[:, 0]

    A = lo + U[:, :k] * span
    B = lo + U[:, k:] * span

    blocks = [A, B]
    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        blocks.append(AB_i)
    return np.vstack(blocks), N, k


def compute_jansen_sobol_indices(Y_all, N, k, n_boot=800, seed=99):
    """
    Computes first-order S_i and total-effect S_Ti via Jansen (2010) / Saltelli (2010)
    with paired-index Bootstrap 95% confidence intervals.
    Y_all has length (k + 2) * N matching `build_saltelli_matrix` order.
    """
    Y_A = Y_all[:N]
    Y_B = Y_all[N:2 * N]
    Y_AB = Y_all[2 * N:].reshape(k, N)  # (k, N)

    def _calc(ya, yb, yab):
        vy = np.var(np.concatenate([ya, yb]))
        if vy < 1e-18:
            return np.zeros(k), np.zeros(k), 0.0
        # Jansen (2010) / Saltelli (2010) formulas:
        # S_i  = 1 - mean((yb - yab_i)^2) / (2 * Var(Y))
        # S_Ti =     mean((ya - yab_i)^2) / (2 * Var(Y))
        si = 1.0 - np.mean((yb[None, :] - yab) ** 2, axis=1) / (2.0 * vy)
        sti = np.mean((ya[None, :] - yab) ** 2, axis=1) / (2.0 * vy)
        return si, sti, float(vy)

    si, sti, var_y = _calc(Y_A, Y_B, Y_AB)

    rng = np.random.default_rng(seed)
    boot_si = np.zeros((n_boot, k), dtype=np.float64)
    boot_sti = np.zeros((n_boot, k), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        bsi, bsti, _ = _calc(Y_A[idx], Y_B[idx], Y_AB[:, idx])
        boot_si[b] = bsi
        boot_sti[b] = bsti

    si_lo, si_hi = np.percentile(boot_si, [2.5, 97.5], axis=0)
    sti_lo, sti_hi = np.percentile(boot_sti, [2.5, 97.5], axis=0)

    return {
        "var_y": var_y,
        "S_i": si,
        "S_i_ci95": np.stack([si_lo, si_hi], axis=1),
        "S_Ti": sti,
        "S_Ti_ci95": np.stack([sti_lo, sti_hi], axis=1),
    }


def weighted_cov(samples, weights):
    w = weights / np.sum(weights)
    mu = np.sum(samples * w[:, None], axis=0)
    diff = samples - mu[None, :]
    denom = max(1.0 - np.sum(w ** 2), 1e-8)
    cov = (diff.T * w[None, :]) @ diff / denom
    # Small ridge for Cholesky stability
    diag_ridge = 1e-6 * np.maximum(np.diag(cov), 1e-8)
    cov += np.diag(diag_ridge)
    return cov


def compute_ess(weights):
    w = weights / np.sum(weights)
    return float(1.0 / np.sum(w ** 2))


def systematic_resample_indices(weights, n_out, rng):
    w = weights / np.sum(weights)
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    u0 = rng.random() / n_out
    u = u0 + np.arange(n_out, dtype=np.float64) / n_out
    return np.searchsorted(cdf, u)


def systematic_resample(particles, weights, n_out, rng):
    idx = systematic_resample_indices(weights, n_out, rng)
    return particles[idx].copy()


def propose_abc_smc_batch(parents, parent_weights, bounds, n_draw, rng):
    """
    Beaumont et al. (2009) component-wise multivariate Gaussian kernel perturbation
    from weighted parents, rejecting out-of-prior draws to enforce exact truncated uniform prior.
    Returns (proposed_particles [n_draw, d], proposal_log_weights).
    """
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    cov = 2.0 * weighted_cov(parents, parent_weights)
    L = np.linalg.cholesky(cov)
    inv_L = np.linalg.inv(L)

    n_parents = parents.shape[0]
    w_norm = parent_weights / np.sum(parent_weights)
    cdf = np.cumsum(w_norm)
    cdf[-1] = 1.0

    accepted = []
    tot = 0
    while tot < n_draw:
        need = n_draw - tot
        m_try = int(need * 1.6) + 256
        p_idx = np.searchsorted(cdf, rng.random(m_try))
        z = rng.standard_normal((m_try, d))
        cand = parents[p_idx] + z @ L.T
        in_b = np.all((cand >= lo[None, :]) & (cand <= hi[None, :]), axis=1)
        valid = cand[in_b]
        if len(valid) > 0:
            accepted.append(valid)
            tot += len(valid)

    props = np.vstack(accepted)[:n_draw]

    # Exact Beaumont importance density evaluation in batches to cap RAM
    sub_sz = min(len(parents), 1024)
    if len(parents) > sub_sz:
        sub_idx = rng.choice(len(parents), size=sub_sz, replace=False, p=w_norm)
        ref_p = parents[sub_idx]
        ref_w = np.full(sub_sz, 1.0 / sub_sz)
    else:
        ref_p = parents
        ref_w = w_norm

    ref_white = ref_p @ inv_L.T
    prop_white = props @ inv_L.T
    log_weights = np.zeros(n_draw, dtype=np.float64)
    log_ref_w = np.log(np.maximum(ref_w, 1e-300))

    chunk = 4096
    for i0 in range(0, n_draw, chunk):
        i1 = min(n_draw, i0 + chunk)
        pw = prop_white[i0:i1]  # (c, d)
        # Squared Mahalanobis distances (c, M)
        d2 = (
            np.sum(pw ** 2, axis=1, keepdims=True)
            + np.sum(ref_white ** 2, axis=1)[None, :]
            - 2.0 * (pw @ ref_white.T)
        )
        log_k = -0.5 * np.maximum(d2, 0.0) + log_ref_w[None, :]
        m_max = np.max(log_k, axis=1)
        lse = m_max + np.log(np.sum(np.exp(log_k - m_max[:, None]), axis=1))
        log_weights[i0:i1] = -lse

    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return props, weights
