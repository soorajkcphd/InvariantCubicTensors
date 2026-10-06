# InvariantCubicTensors

Numerical checks for the paper

    Invariant Cubic Tensors and Bi-invariant Statistical Structures on
    Compact Simple Lie Groups
    Sooraj K.C and Vivek Mishra
    Differential Geometry and its Applications, revised version of
    manuscript DGA-D-26-00227

## verify_numerics.py

The script recomputes the numerical values quoted in the revised paper and
checks the algebraic identities used in Sections 4 and 5. Result numbers
refer to the revised manuscript.

    [A] The cubic form d(X,Y,Z) = (i/2) tr(X{Y,Z}) and the product D on
        su(n): reality, symmetry, Ad-invariance, duality, derivation
        identity, sign under complex conjugation, and the normalisation
        d(E_a,E_b,E_c) = d_abc / sqrt(2).
        (Definition 5.1, Lemma 5.2)

    [B] Curvature of the alpha-connections, the dimension of the
        ad-invariant symmetric cubic forms on so(3), so(4), so(5) and
        so(6), the Pfaffian on so(6), and the invariance of x1 x2 x3
        under the tetrahedral group.
        (Corollary 5.4, Theorem 5.3, Remarks 5.6 and 5.7)

    [C] c(beta) and lambda(beta) for the transformation model
        rho_beta ~ exp(beta Im tr x) on SU(3), by Weyl integration,
        and the value of lambda/(c beta) at small beta, which tends
        to -1/2. (Example 5.13)

    [D] Monte Carlo with 4e6 Haar samples of SU(3): the exponential family
        of Example 5.12 at theta = 0 and theta = e_8, and Example 5.13 at
        beta = 1.
        (Examples 5.12 and 5.13, Remark 5.15)

    [E] The Kantorovich bound 2 sqrt(kappa) / (kappa + 1), its equality
        case, and the bound sqrt(1 - eps^2) under approximate isotropy.
        (Proposition 4.4)

    [F] The realisation theorem on SU(n): the averages of tr S0^2 and
        tr S0^3 over the maximal torus for n = 3, ..., 7 (Step 4 of the
        proof), the identity (5) between Haar averages of class functions
        of x^k and torus averages, on SU(3) with k = 5 and on SU(4) with
        k = 7, and the realised ratio lambda/c = 5 on SU(3) quoted in
        Remark 5.15. The normalisation
        sum d(E_a,E_b,E_c)^2 = (n^2-4)(n^2-1)/(2n) is checked for
        n = 4, 5, 6. (Theorem 5.14)

verify_output.txt is the output of a complete run. All random numbers are
seeded, so a run reproduces this file up to rounding differences between
platforms.

## Installation

    git clone https://github.com/soorajkcphd/InvariantCubicTensors.git
    cd InvariantCubicTensors
    pip install -r requirements.txt

## Usage

    python verify_numerics.py          # all checks (about five minutes)
    python verify_numerics.py --fast   # without the Monte Carlo block [D]

## Requirements

Python 3.9 or later, NumPy 1.24 or later, SciPy 1.10 or later.
Matplotlib 3.7 or later is needed only for the first-version script below.

## First version

InvariantFisherMetricsandCubicTensors.py and the figure files
fig_numerical_illustrations.pdf and fig_numerical_illustrations.png belong
to the first version of the manuscript (Figure 1 there). The figure and the
corresponding section were removed in the revision.

## Citation

    @article{kc2026invariant,
      title   = {Invariant Cubic Tensors and Bi-invariant Statistical Structures
                 on Compact Simple Lie Groups},
      author  = {K.C, Sooraj and Mishra, Vivek},
      journal = {Differential Geometry and its Applications},
      year    = {2026},
      note    = {Submitted}
    }

## License

MIT License
