# InvariantFisherMetricsandCubicTensors
# Invariant Fisher Metrics and Cubic Tensors on Lie-Algebraic Statistical Submanifolds

This repository contains the numerical experiments for the paper:

> **Invariant Fisher Metrics and Cubic Tensors on Lie-Algebraic Statistical Submanifolds**
> Sooraj K.C and Vivek Mishra
> *Submitted to Annals of Global Analysis and Geometry*

## Overview

The paper studies invariant differential-geometric structures on statistical
submanifolds parametrised by matrix Lie algebras. It proves that Fisher
isotropy is both necessary and sufficient for the Fisher-Rao metric to induce
the orthogonal Lie-algebraic projection structure, classifies the
Amari-Chentsov cubic tensor under adjoint-induced isotropy via Casimir
invariant theory, and establishes a sharp compact/non-compact distortion
dichotomy for the matrix exponential map.

## Numerical Experiments

The script InvariantFisherMetricsandCubicTensors.py reproduces the
three-panel Figure 1 (Section 7):

| Panel | Description | Result validated |
|---|---|---|
| (a) Alignment bound tightness | 500 random Fisher matrices in R^{9x9} per kappa | Proposition 4.5 -- Kantorovich bound 2*sqrt(kappa)/(kappa+1) |
| (b) Alignment along trajectories | Parametric path on so(3) with J(theta) = -tr(W exp(theta)) | Proposition 4.5 -- bound holds uniformly |
| (c) Smoothness dichotomy | Empirical Lipschitz constants on so(3) vs sl(2) | Lemma 7.1 -- compact: O(1), non-compact: O(e^R) |

## Repository Structure

```
InvariantFisherMetricsandCubicTensors/
    README.md
    requirements.txt
    InvariantFisherMetricsandCubicTensors.py
    fig_numerical_illustrations.pdf
    fig_numerical_illustrations.png
```

## Installation

```bash
git clone https://github.com/soorajkcphd/InvariantFisherMetricsandCubicTensors.git
cd InvariantFisherMetricsandCubicTensors
pip install -r requirements.txt
```

## Usage

```bash
python InvariantFisherMetricsandCubicTensors.py
```

This generates fig_numerical_illustrations.pdf and fig_numerical_illustrations.png.

## Requirements

- Python >= 3.9
- NumPy >= 1.24
- SciPy >= 1.10 (matrix exponential via scipy.linalg.expm)
- Matplotlib >= 3.7

No specialized optimization software is required.

## Citation

If you use this code, please cite:

```bibtex
@article{kc2026invariant,
  title={Invariant Fisher Metrics and Cubic Tensors on Lie-Algebraic Statistical Submanifolds},
  author={K.C, Sooraj and Mishra, Vivek},
  journal={Annals of Global Analysis and Geometry},
  year={2026},
  note={Submitted}
}
```

## License

MIT License
