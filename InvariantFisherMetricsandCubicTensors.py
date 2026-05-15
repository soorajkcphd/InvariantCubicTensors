
# Numerical Illustrations — three-panel figure.
# (a) Alignment bound tightness: random Fisher matrices on R^9, theory vs empirical
#     — with error bars (±1 std dev)
# (b) Alignment dynamics during optimization on so(3) with controlled κ
#     — cosine alignment vs iteration, with theoretical dotted lines
# (c) Smoothness dichotomy: so(3) vs sl(2) Lipschitz constants

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'text.usetex': False,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
})

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8))

# ============================================================
# Panel (a): Alignment bound tightness — with error bars
# ============================================================
ax = axes[0]
d = 9
kappas = np.logspace(0, 2.5, 30)
theoretical = 2 * np.sqrt(kappas) / (kappas + 1)
empirical_min = []
empirical_mean = []
empirical_std = []
n_trials = 500

for kappa in kappas:
    alignments = []
    for _ in range(n_trials):
        eigvals = np.linspace(1.0, kappa, d)
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        F_inv = Q @ np.diag(1.0 / eigvals) @ Q.T
        g = np.random.randn(d)
        g = g / np.linalg.norm(g)
        ng = F_inv @ g
        cos_align = np.dot(g, ng) / (np.linalg.norm(g) * np.linalg.norm(ng))
        alignments.append(cos_align)
    empirical_min.append(np.min(alignments))
    empirical_mean.append(np.mean(alignments))
    empirical_std.append(np.std(alignments))

empirical_mean = np.array(empirical_mean)
empirical_std = np.array(empirical_std)
empirical_min = np.array(empirical_min)

ax.semilogx(kappas, theoretical, 'k-', linewidth=2.0,
            label=r'Bound: $2\sqrt{\kappa}/(\kappa{+}1)$', zorder=4)
ax.fill_between(kappas, empirical_mean - empirical_std,
                empirical_mean + empirical_std,
                color='#1f77b4', alpha=0.18, zorder=1,
                label=r'Mean $\pm\,1\sigma$ (500 trials)')
ax.semilogx(kappas, empirical_mean, 's', color='#1f77b4', markersize=2.5,
            alpha=0.7, label='Empirical mean', zorder=2)
ax.semilogx(kappas, empirical_min, 'o', color='#d62728', markersize=3.5,
            alpha=0.7, label='Empirical minimum', zorder=3)

ax.set_xlabel(r'Fisher condition number $\kappa$')
ax.set_ylabel('Cosine alignment')
ax.set_ylim([0, 1.05])
ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_title('(a) Alignment bound tightness')
print("Panel (a) done.")

# ============================================================
# Panel (b): Alignment dynamics during optimization on so(3)
#   — plots cosine(P_g(∇J), F^{-1}∇J) vs iteration
# ============================================================
ax = axes[1]

# so(3) basis (orthonormal under Frobenius)
E1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float) / np.sqrt(2)
E2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float) / np.sqrt(2)
E3 = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float) / np.sqrt(2)
basis = [E1, E2, E3]
d_g = 3

T = 600
n_seeds = 10
sigma = 0.02

kappa_values = [1.0, 2.0, 5.0, 15.0, 50.0]
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

for kappa, color in zip(kappa_values, colors):
    # Fixed Fisher matrix with prescribed condition number
    np.random.seed(7)
    Q_rot, _ = np.linalg.qr(np.random.randn(d_g, d_g))
    eigvals = np.array([1.0, np.sqrt(kappa), kappa])
    F_inv = Q_rot @ np.diag(1.0 / eigvals) @ Q_rot.T

    all_alignments = np.zeros((n_seeds, T))

    for seed in range(n_seeds):
        np.random.seed(seed * 100 + int(kappa * 10))
        coords = np.random.randn(d_g) * 0.3

        for t in range(T):
            theta_mat = sum(coords[k] * basis[k] for k in range(d_g))
            exp_theta = expm(theta_mat)

            # Objective: J(θ) = -tr(W exp(θ))
            # Use a fixed W generated once per seed
            if t == 0:
                np.random.seed(seed * 31 + 77)
                W_raw = np.random.randn(3, 3)
                W_opt = expm((W_raw - W_raw.T) / 2)  # target in SO(3)
                np.random.seed(seed * 100 + int(kappa * 10) + t + 1)

            J_current = -np.trace(W_opt @ exp_theta)

            # Finite-difference gradient in Lie algebra coordinates
            grad_coords = np.zeros(d_g)
            eps_fd = 1e-5
            for k in range(d_g):
                c_plus = coords.copy()
                c_plus[k] += eps_fd
                J_plus = -np.trace(W_opt @ expm(sum(c_plus[j] * basis[j] for j in range(d_g))))
                grad_coords[k] = (J_plus - J_current) / eps_fd

            # Projected gradient = grad_coords (already in g coordinates)
            proj_grad = grad_coords
            # Natural gradient = F^{-1} @ projected gradient
            nat_grad = F_inv @ proj_grad

            norm_proj = np.linalg.norm(proj_grad)
            norm_nat = np.linalg.norm(nat_grad)
            if norm_proj > 1e-12 and norm_nat > 1e-12:
                all_alignments[seed, t] = np.dot(proj_grad, nat_grad) / (norm_proj * norm_nat)
            else:
                all_alignments[seed, t] = 1.0

            # SGD update
            lr = 0.01 / np.sqrt(t + 1)
            noise = np.random.randn(d_g) * sigma
            coords = coords - lr * (proj_grad + noise)

    mean_align = np.mean(all_alignments, axis=0)
    std_align = np.std(all_alignments, axis=0)
    window = 10
    sm = np.convolve(mean_align, np.ones(window) / window, mode='valid')
    sm_std = np.convolve(std_align, np.ones(window) / window, mode='valid')
    x_range = np.arange(len(sm))

    label = rf'$\kappa={kappa:.0f}$' if kappa >= 2 else r'$\kappa=1$ (isotropic)'
    ax.plot(x_range, sm, color=color, linewidth=1.2, label=label, alpha=0.85)
    ax.fill_between(x_range, sm - sm_std, sm + sm_std,
                    color=color, alpha=0.10)

    # Theoretical alignment bound as dotted horizontal line
    alpha_k = 2 * np.sqrt(kappa) / (kappa + 1)
    if kappa > 1:
        ax.axhline(y=alpha_k, color=color, linestyle=':', alpha=0.5, linewidth=1.0)

ax.set_xlabel('Iteration $t$')
ax.set_ylabel(r'$\cos(P_{\mathfrak{g}}(\nabla J),\, F^{-1}\nabla J)$')
ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.15, 1.05])
ax.set_title(r'(b) Alignment during optimization on $\mathfrak{so}(3)$')
print("Panel (b) done.")

# ============================================================
# Panel (c): Compact vs non-compact Lipschitz scaling
# ============================================================
ax = axes[2]
R_values = np.linspace(0.1, 4.0, 25)
n_samples = 2000
lip_compact = []
lip_noncompact = []

for R in R_values:
    # so(3): compact
    ratios = []
    for _ in range(n_samples):
        A = np.random.randn(3, 3)
        theta1 = (A - A.T) / 2
        theta1 = theta1 / np.linalg.norm(theta1, 'fro') * R * np.random.rand()
        A = np.random.randn(3, 3)
        theta2 = (A - A.T) / 2
        theta2 = theta2 / np.linalg.norm(theta2, 'fro') * R * np.random.rand()
        diff_exp = np.linalg.norm(expm(theta1) - expm(theta2), 'fro')
        diff_theta = np.linalg.norm(theta1 - theta2, 'fro')
        if diff_theta > 1e-10:
            ratios.append(diff_exp / diff_theta)
    lip_compact.append(np.max(ratios))

    # sl(2): non-compact
    ratios = []
    for _ in range(n_samples):
        a, b, c = np.random.randn(3)
        theta1 = np.array([[a, b], [c, -a]])
        theta1 = theta1 / np.linalg.norm(theta1, 'fro') * R * np.random.rand()
        a, b, c = np.random.randn(3)
        theta2 = np.array([[a, b], [c, -a]])
        theta2 = theta2 / np.linalg.norm(theta2, 'fro') * R * np.random.rand()
        diff_exp = np.linalg.norm(expm(theta1) - expm(theta2), 'fro')
        diff_theta = np.linalg.norm(theta1 - theta2, 'fro')
        if diff_theta > 1e-10:
            ratios.append(diff_exp / diff_theta)
    lip_noncompact.append(np.max(ratios))

ax.plot(R_values, lip_compact, 'o-', color='#2ca02c', markersize=4,
        linewidth=1.5, label=r'$\mathfrak{so}(3)$ (compact)', zorder=3)
ax.plot(R_values, lip_noncompact, 's-', color='#d62728', markersize=4,
        linewidth=1.5, label=r'$\mathfrak{sl}(2)$ (non-compact)', zorder=3)
ax.plot(R_values, np.ones_like(R_values), '--', color='#2ca02c',
        alpha=0.5, linewidth=1, label=r'Theory: $1$ (compact)')
ax.plot(R_values, np.exp(R_values), '--', color='#d62728',
        alpha=0.5, linewidth=1, label=r'Theory: $e^R$ (non-compact)')

ax.set_xlabel(r'Parameter radius $R = \|\theta\|_F$')
ax.set_ylabel('Empirical Lipschitz constant')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title('(c) Smoothness dichotomy: compact vs. non-compact')
print("Panel (c) done.")

# ============================================================
# Save
# ============================================================
fig.tight_layout(w_pad=2.5)
fig.savefig('fig_numerical_illustrations.pdf', bbox_inches='tight')
fig.savefig('fig_numerical_illustrations.png', bbox_inches='tight')
plt.close()
print("Combined figure saved.")
