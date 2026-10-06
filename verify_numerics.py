#!/usr/bin/env python3
"""Numerical checks for the paper

    Invariant Cubic Tensors and Bi-invariant Statistical Structures on
    Compact Simple Lie Groups (S. K.C and V. Mishra), revised version.

The script recomputes the numerical values quoted in the paper and checks
the algebraic identities used in Sections 4 and 5.  It needs NumPy and SciPy.

    python verify_numerics.py          all checks (about five minutes)
    python verify_numerics.py --fast   without the Monte Carlo block [D]

Output blocks (numbering of the revised manuscript):
  [A] cubic form d and product D on su(n): Definition 5.1, Lemma 5.2
  [B] curvature identity (Corollary 5.4), invariant cubics on so(n)
      (Theorem 5.3, Remark 5.7), Pfaffian on so(6), tetrahedral group
      (Remark 5.6)
  [C] c(beta) and lambda(beta) by Weyl integration: Example 5.13
  [D] Monte Carlo on SU(3): Examples 5.12 and 5.13, Remark 5.15
  [E] Kantorovich bound: Proposition 4.4
  [F] powers and torus averages on SU(n): Theorem 5.14
"""
import itertools
import sys

import numpy as np
from scipy.linalg import expm, null_space

np.set_printoptions(precision=5, suppress=True, linewidth=150)
FAST = "--fast" in sys.argv
rng = np.random.default_rng(2026)


def gell_mann():
    """The Gell-Mann matrices, normalised by tr(L_a L_b) = 2 delta_ab."""
    gm = np.zeros((8, 3, 3), dtype=complex)
    gm[0][0, 1] = 1
    gm[0][1, 0] = 1
    gm[1][0, 1] = -1j
    gm[1][1, 0] = 1j
    gm[2][0, 0] = 1
    gm[2][1, 1] = -1
    gm[3][0, 2] = 1
    gm[3][2, 0] = 1
    gm[4][0, 2] = -1j
    gm[4][2, 0] = 1j
    gm[5][1, 2] = 1
    gm[5][2, 1] = 1
    gm[6][1, 2] = -1j
    gm[6][2, 1] = 1j
    gm[7] = np.diag([1, 1, -2]) / np.sqrt(3)
    return gm


def rand_su(n):
    """A random element of su(n)."""
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    X = (Z - Z.conj().T) / 2
    return X - np.trace(X) / n * np.eye(n)


def ip(X, Y):
    """Frobenius inner product Re tr(X^* Y)."""
    return np.real(np.trace(X.conj().T @ Y))


def dform(X, Y, Z):
    """d(X,Y,Z) = (i/2) tr(X{Y,Z})."""
    return 0.5j * np.trace(X @ (Y @ Z + Z @ Y))


def Dprod(X, Y):
    """D(X,Y) = -(i/2) ({X,Y} - (2/n) tr(XY) I)."""
    n = len(X)
    return -0.5j * (X @ Y + Y @ X - (2.0 / n) * np.trace(X @ Y) * np.eye(n))


def br(X, Y):
    return X @ Y - Y @ X


gm = gell_mann()
E = [1j * m / np.sqrt(2) for m in gm]  # orthonormal basis of su(3)
dsym = np.real(np.array([[[0.25 * np.trace(gm[a] @ (gm[b] @ gm[c] + gm[c] @ gm[b]))
                           for c in range(8)] for b in range(8)] for a in range(8)]))
dE = np.real(np.array([[[dform(E[a], E[b], E[c])
                         for c in range(8)] for b in range(8)] for a in range(8)]))


def check_a():
    print("[A] cubic form d and product D on su(n)  (Definition 5.1, Lemma 5.2)")
    for n in (2, 3, 4, 5):
        X = rand_su(n)
        Y = rand_su(n)
        Z = rand_su(n)
        W = rand_su(n)
        k = expm(rand_su(n))

        def Ad(A):
            return k @ A @ k.conj().T

        d0 = dform(X, Y, Z)
        DXY = Dprod(X, Y)
        checks = [
            abs(d0.imag) < 1e-12,
            np.isclose(d0, -np.trace(X @ Y @ Z).imag),
            np.allclose([dform(*p) for p in itertools.permutations((X, Y, Z))], d0),
            np.isclose(dform(Ad(X), Ad(Y), Ad(Z)), d0),
            np.isclose(dform(X, X, X), 1j * np.trace(X @ X @ X)),
            np.allclose(DXY, -DXY.conj().T) and abs(np.trace(DXY)) < 1e-12,
            np.isclose(ip(DXY, Z), d0.real),
            np.allclose(br(W, DXY), Dprod(br(W, X), Y) + Dprod(X, br(W, Y))),
            np.isclose(dform(X.conj(), Y.conj(), Z.conj()), -d0),
        ]
        note = "   (d vanishes for n=2)" if n == 2 else ""
        print("   su(%d): all identities hold: %s;  |d(X,X,X)| = %.4f" % (n, all(checks), abs(dform(X, X, X))) + note)
    print("   d(E_a,E_b,E_c) = d_abc/sqrt(2):", np.allclose(dE, dsym / np.sqrt(2)),
          "; sum_abc d(E_a,E_b,E_c)^2 = %.6f (20/3 = %.6f)" % ((dE ** 2).sum(), 20 / 3))
    print("   d_118 = %.5f, d_344 = %.5f, d_888 = %.5f" % (dsym[0, 0, 7], dsym[2, 3, 3], dsym[7, 7, 7]))
    print("   diagonal of (d_ab8):", np.round(np.diag(dsym[7]), 5), "; off-diagonal max:",
          np.abs(dsym[7] - np.diag(np.diag(dsym[7]))).max())
    print("   trace sum_a d_aac = 0:", np.allclose(np.einsum('aac->c', dsym), 0))


def pfaffian(A):
    n = len(A)
    if n == 0:
        return 1.0
    total = 0.0
    for j in range(1, n):
        minor = np.delete(np.delete(A, [0, j], 0), [0, j], 1)
        total += (-1) ** (j + 1) * A[0, j] * pfaffian(minor)
    return total


def so_basis(n):
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            M = np.zeros((n, n))
            M[i, j] = 1
            M[j, i] = -1
            basis.append(M / np.sqrt(2))
    return basis


def rotation(axis, angle):
    a = np.array(axis, dtype=float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return expm(angle * K)


def check_b():
    print("\n[B] curvature identity (Corollary 5.4), invariant cubics (Theorem 5.3), tetrahedral group (Remark 5.6)")
    for n in (3, 4):
        X = rand_su(n)
        Y = rand_su(n)
        Z = rand_su(n)
        a = 0.37

        def conn(U, V):
            # left-invariant connection nabla_U V = (1/2)[U,V] + a D(U,V)
            return 0.5 * br(U, V) + a * Dprod(U, V)

        R_direct = conn(X, conn(Y, Z)) - conn(Y, conn(X, Z)) - conn(br(X, Y), Z)
        R_formula = -0.25 * br(br(X, Y), Z) + a * a * (Dprod(X, Dprod(Y, Z)) - Dprod(Y, Dprod(X, Z)))
        print("   su(%d): R = -1/4[[X,Y],Z] + a^2 [D_X,D_Y]Z :" % n, np.allclose(R_direct, R_formula))

    A6 = rng.normal(size=(6, 6))
    A6 = A6 - A6.T
    M = rng.normal(size=(6, 6))
    Q6 = expm(M - M.T)
    print("   so(6): Pf(QAQ^T) = Pf(A):", np.isclose(pfaffian(Q6 @ A6 @ Q6.T), pfaffian(A6)),
          "; Pf is cubic:", np.isclose(pfaffian(2 * A6), 8 * pfaffian(A6)))

    for n in (3, 4, 5, 6):
        basis = so_basis(n)
        m = len(basis)
        # matrices of ad(B_k) in the orthonormal basis
        ads = [np.array([[np.trace(Bi.T @ (Bk @ Bj - Bj @ Bk)) for Bj in basis] for Bi in basis])
               for Bk in basis]
        combos = list(itertools.combinations_with_replacement(range(m), 3))
        pos = {c: i for i, c in enumerate(combos)}
        rows = []
        for ad in ads:
            for (i, j, k) in combos:
                row = np.zeros(len(combos))
                for l in range(m):
                    if ad[l, i]:
                        row[pos[tuple(sorted((l, j, k)))]] += ad[l, i]
                    if ad[l, j]:
                        row[pos[tuple(sorted((i, l, k)))]] += ad[l, j]
                    if ad[l, k]:
                        row[pos[tuple(sorted((i, j, l)))]] += ad[l, k]
                rows.append(row)
        dim_cubics = null_space(np.array(rows)).shape[1]
        comm = null_space(np.vstack([np.kron(ad, np.eye(m)) - np.kron(np.eye(m), ad.T) for ad in ads])).shape[1]
        note = "  (so(4) is not simple)" if n == 4 else ""
        print("   so(%d): dim of ad-invariant symmetric cubic forms = %d;  commutant of ad has dim %d"
              % (n, dim_cubics, comm) + note)

    # rotation group of the tetrahedron, generated by two half-turns and a third-turn
    gens = [rotation([1, 0, 0], np.pi), rotation([0, 0, 1], np.pi), rotation([1, 1, 1], 2 * np.pi / 3)]
    group = [np.eye(3)]
    grew = True
    while grew:
        grew = False
        for g in list(group):
            for h in gens:
                gh = g @ h
                if not any(np.allclose(gh, x) for x in group):
                    group.append(gh)
                    grew = True
    cubic = np.zeros((3, 3, 3))
    for p in itertools.permutations((0, 1, 2)):
        cubic[p] = 1.0  # the polarisation of x1 x2 x3
    comm = null_space(np.vstack([np.kron(g, np.eye(3)) - np.kron(np.eye(3), g.T) for g in group])).shape[1]
    invariant = all(np.allclose(np.einsum('ia,jb,kc,abc->ijk', g, g, g, cubic), cubic) for g in group)
    print("   tetrahedral group: order %d, commutant dim %d (irreducible), x1x2x3 invariant:" % (len(group), comm),
          invariant)


def weyl(beta, N=600, kind="im"):
    """E[tr H0^2] and E[tr H0^3] under rho ~ exp(beta Im tr x) (kind="im") or
    exp(beta Re tr x) (kind="re") on SU(3), by the Weyl integration formula.
    H0 is the trace-free part of the matrix representing grad f, see Example 5.13."""
    phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
    P1, P2 = np.meshgrid(phi, phi, indexing='ij')
    P3 = -P1 - P2
    z = [np.exp(1j * P1), np.exp(1j * P2), np.exp(1j * P3)]
    vdm = np.abs(z[0] - z[1]) ** 2 * np.abs(z[0] - z[2]) ** 2 * np.abs(z[1] - z[2]) ** 2
    cs = np.stack([np.cos(P1), np.cos(P2), np.cos(P3)])
    sn = np.stack([np.sin(P1), np.sin(P2), np.sin(P3)])
    if kind == "im":
        H = cs
        logw = beta * sn.sum(0)
    else:
        H = -sn
        logw = beta * cs.sum(0)
    H0 = H - H.mean(0)
    t2 = (H0 ** 2).sum(0)
    t3 = (H0 ** 3).sum(0)
    w = vdm * np.exp(logw - logw.max())
    w /= w.sum()
    return (t2 * w).sum(), (t3 * w).sum()


def check_c():
    print("\n[C] Weyl integration on SU(3): rho_beta ~ exp(beta Im tr x)  (Example 5.13)")
    t2, t3 = weyl(0.0)
    print("   Haar: E[tr H0^2] = %.10f (4/3),  E[tr H0^3] = %.10f (5/9)" % (t2, t3))
    for b in (0.25, 0.5, 1.0, -1.0):
        t2, t3 = weyl(b)
        print("   beta=%5.2f: c = %.7f (beta^2/6 = %.7f),  lambda = %+.7f (-beta^3/12 = %+.7f)"
              % (b, b * b * t2 / 8, b * b / 6, -(3 / 20) * b ** 3 * t3, -b ** 3 / 12))
    t2, t3 = weyl(1.0, kind="re")
    print("   rho ~ exp(beta Re tr x), beta=1: c = %.7f, lambda = %.1e (vanishes by inversion symmetry)"
          % (t2 / 8, -(3 / 20) * t3))
    b = 0.01
    t2, t3 = weyl(b)
    tau = (-(3 / 20) * b ** 3 * t3) / (b * b * t2 / 8)
    print("   lambda/(c beta) at beta=%.2f: %.6f (-1/2 by the expansions of Example 5.13)" % (b, tau / b))


def check_d():
    print("\n[D] Monte Carlo on SU(3): 4e6 Haar samples (20 batches of 2e5), seed 12345  (Remark 5.15)")
    print("   standard errors from the per-sample variances, delta method for weighted estimates")
    mrng = np.random.default_rng(12345)

    def haar_su3(N):
        Z = (mrng.normal(size=(N, 3, 3)) + 1j * mrng.normal(size=(N, 3, 3))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        dg = np.einsum('nii->ni', R)
        Q = Q * (dg / np.abs(dg))[:, None, :]
        return Q * (np.linalg.det(Q) ** (-1 / 3))[:, None, None]

    def mean_and_error(y):
        """Mean of i.i.d. values and its standard error."""
        return y.mean(), y.std(ddof=1) / np.sqrt(len(y))

    def weighted_mean_and_error(w, y):
        """Self-normalised importance sampling estimate of E[y] and its standard error."""
        m = (w * y).sum() / w.sum()
        return m, np.sqrt((w ** 2 * (y - m) ** 2).sum()) / w.sum()

    batches, batch_size = 20, 200000
    T_parts, cubic_parts, imtr_parts, score_parts = [], [], [], []
    for _ in range(batches):
        g = haar_su3(batch_size)
        Tb = np.einsum('aij,nji->na', gm, g).real  # T_a = Re tr(Lambda_a x)
        sb = -Tb / np.sqrt(2)  # scores -Im tr(E_a x) of the transformation model with beta = 1
        T_parts.append(Tb)
        cubic_parts.append(np.einsum('abc,na,nb,nc->n', dsym, Tb, Tb, Tb) / (dsym ** 2).sum())
        imtr_parts.append(np.einsum('nii->n', g).imag)
        score_parts.append(np.einsum('abc,na,nb,nc->n', dE, sb, sb, sb) / (dE ** 2).sum())
    T = np.concatenate(T_parts)
    N = len(T)

    # Example 5.12 at theta = 0
    F0 = T.T @ T / N
    print("   Example 5.12, theta=0: mean diagonal of F = %.5f +- %.5f (1/3);"
          % mean_and_error((T ** 2).mean(axis=1)),
          "max |off-diagonal| of F = %.5f" % np.abs(F0 - np.diag(np.diag(F0))).max())
    print("   Example 5.12, theta=0: fit C_abc = gamma*d_abc: gamma = %.5f +- %.5f (1/6 = %.5f)"
          % (*mean_and_error(np.concatenate(cubic_parts)), 1 / 6))

    # Example 5.12 at theta = e_8, by importance weights exp(T_8)
    w = np.exp(T[:, 7])
    Tc = T - (w[:, None] * T).sum(axis=0) / w.sum()
    F8 = (Tc * w[:, None]).T @ Tc / w.sum()
    # F(e_8) is diagonal in the Gell-Mann basis and constant on a = 1-3, a = 4-7 and a = 8.
    blocks = [list(range(0, 3)), list(range(3, 7)), [7]]
    (m1, s1), (m2, s2), (m3, s3) = [weighted_mean_and_error(w, (Tc[:, b] ** 2).mean(axis=1)) for b in blocks]
    print("   Example 5.12, theta=e_8: eigenvalues of F:", np.linalg.eigvalsh(F8))
    print("      block means a=1-3: %.5f +- %.5f (x3), a=4-7: %.5f +- %.5f (x4), a=8: %.5f +- %.5f (x1);"
          % (m1, s1, m2, s2, m3, s3), "max |off-diagonal| = %.5f;  kappa = %.3f"
          % (np.abs(F8 - np.diag(np.diag(F8))).max(), m1 / m3))
    print("      first-order values 1/3 + d_aa8/6:", np.sort(1 / 3 + np.diag(dsym[7]) / 6))

    # Example 5.13 with beta = 1, by importance weights exp(Im tr x)
    wb = np.exp(np.concatenate(imtr_parts))
    c = weighted_mean_and_error(wb, (T ** 2).mean(axis=1) / 2)
    lam_hat = weighted_mean_and_error(wb, np.concatenate(score_parts))
    print("   Example 5.13, beta=1: c = %.5f +- %.5f,  lambda = %.5f +- %.5f" % (*c, *lam_hat))


def check_e():
    print("\n[E] Kantorovich bound  (Proposition 4.4)")
    dim = 7
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    ev = np.sort(rng.uniform(0.5, 9, size=dim))
    F = Q @ np.diag(ev) @ Q.T
    kappa = ev[-1] / ev[0]

    def cosine(u):
        Fu = np.linalg.solve(F, u)
        return (u @ Fu) / np.linalg.norm(u) / np.linalg.norm(Fu)

    v = np.sqrt(ev[-1]) * Q[:, -1] + np.sqrt(ev[0]) * Q[:, 0]  # the equality case
    print("   equality case: cos = %.8f, bound 2 sqrt(k)/(k+1) = %.8f, min over 20000 random v = %.6f"
          % (cosine(v), 2 * np.sqrt(kappa) / (kappa + 1), min(cosine(rng.normal(size=dim)) for _ in range(20000))))
    for eps in (0.1, 0.5, 0.9):
        k = (1 + eps) / (1 - eps)
        print("   eps=%.1f: 2 sqrt(k)/(k+1) at k=(1+eps)/(1-eps) equals sqrt(1-eps^2): %s"
              % (eps, np.isclose(2 * np.sqrt(k) / (k + 1), np.sqrt(1 - eps ** 2))))


def su_basis(n):
    """An orthonormal basis of su(n) for <X,Y> = -tr(XY) (generalised Gell-Mann matrices times i/sqrt2)."""
    B = []
    for j in range(n):
        for k in range(j + 1, n):
            S = np.zeros((n, n), dtype=complex)
            S[j, k] = S[k, j] = 1
            A = np.zeros((n, n), dtype=complex)
            A[j, k] = -1j
            A[k, j] = 1j
            B += [S, A]
    for m in range(1, n):
        Dm = np.zeros((n, n), dtype=complex)
        Dm[:m, :m] = np.eye(m)
        Dm[m, m] = -m
        B.append(Dm * np.sqrt(2.0 / (m * (m + 1))))
    return [1j * L / np.sqrt(2) for L in B]


def torus_moments(n, k, beta, N, haar):
    """E[tr S0^2] and E[tr S0^3] for the density exp(beta Im tr(y + y^2)), y = x^k, on SU(n).
    S0 is the trace-free part of diag(cos(k phi_j) + 2 cos(2 k phi_j)), exp(i phi_j) the eigenvalues of x.
    haar=True: x is Haar distributed (Weyl integration formula with the Vandermonde weight).
    haar=False: the eigenvalues are uniform on the maximal torus.
    Uniform N-point grid on each of the n-1 free angles."""
    phi = 2 * np.pi * np.arange(N) / N
    grids = np.meshgrid(*([phi] * (n - 2)), indexing="ij")
    s0 = s2 = s3 = 0.0
    for p1 in phi:
        P = np.stack([np.full(grids[0].shape, p1).ravel()] + [g.ravel() for g in grids])
        P = np.vstack([P, -P.sum(0)])
        w = np.exp(beta * (np.sin(k * P) + np.sin(2 * k * P)).sum(0))
        if haar:
            z = np.exp(1j * P)
            for j in range(n):
                for m in range(j + 1, n):
                    w = w * np.abs(z[j] - z[m]) ** 2
        a = np.cos(k * P) + 2 * np.cos(2 * k * P)
        a0 = a - a.mean(0)
        s0 += w.sum()
        s2 += (w * (a0 ** 2).sum(0)).sum()
        s3 += (w * (a0 ** 3).sum(0)).sum()
    return s2 / s0, s3 / s0


def check_f():
    print("\n[F] Realisation on SU(n): powers and torus averages  (Theorem 5.14)")
    for n in (4, 5, 6):
        basis = su_basis(n)
        norm_d = sum(abs(dform(X, Y, Z)) ** 2 for X in basis for Y in basis for Z in basis)
        print("   su(%d): sum_abc d(E_a,E_b,E_c)^2 = %.10f  ((n^2-4)(n^2-1)/(2n) = %.10f)"
              % (n, norm_d, (n * n - 4) * (n * n - 1) / (2 * n)))
    print("   Step 4, uniform measure on the maximal torus, beta = 0:")
    for n in (3, 4, 5, 6, 7):
        t2, t3 = torus_moments(n, 1, 0.0, 14, False)
        e2 = 2.5 * (n - 1)
        e3 = 4.0 if n == 3 else 3 * (n - 1) * (n - 2) / (2 * n)
        norm_d = (n * n - 4) * (n * n - 1) / (2 * n)
        print("   n=%d: E[tr S0^2] = %.10f (%.10f),  E[tr S0^3] = %.10f (%.10f),  A(0) = %.6f,  B(0) = %.6f"
              % (n, t2, e2, t3, e3, t2 / (n * n - 1), t3 / norm_d))
    print("   identity (5) at beta = 0.25, Haar average of class functions of x^k against the torus average:")
    for n, k, N in ((3, 5, 600), (4, 7, 180)):
        h2, h3 = torus_moments(n, k, 0.25, N, True)
        t2, t3 = torus_moments(n, 1, 0.25, N, False)
        print("   SU(%d), k=%d: E[tr S0^2] = %.10f (torus %.10f),  E[tr S0^3] = %.10f (torus %.10f)"
              % (n, k, h2, t2, h3, t3))
    # one realised ratio on SU(3): solve g(beta) = -tau/k on the torus, then integrate over SU(3) directly
    k, tau = 10, 5.0

    def g(b):
        t2, t3 = torus_moments(3, 1, b, 400, False)
        return b * (t3 / (20 / 3)) / (t2 / 8)

    lo, hi = -2.0, 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if g(mid) < -tau / k:
            lo = mid
        else:
            hi = mid
    b = (lo + hi) / 2
    h2, h3 = torus_moments(3, k, b, 900, True)
    c = (k * b) ** 2 * h2 / 8
    lam = -(k * b) ** 3 * h3 / (20 / 3)
    print("   SU(3), k=%d, beta = %.5f: c = %.4f, lambda = %.4f, lambda/c = %.6f  (Remark 5.15)"
          % (k, b, c, lam, lam / c))


if __name__ == "__main__":
    check_a()
    check_b()
    check_c()
    if not FAST:
        check_d()
    check_e()
    check_f()
    print("\nDone.")
