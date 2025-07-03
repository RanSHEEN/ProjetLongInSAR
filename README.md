# ProjetLongInSAR

**Interferometric SAR Parameter Estimation via Deep Learning**
Université Paris-Saclay – M2 TRIED
Project Report Presented on March 28, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background & Literature Review](#background--literature-review)
3. [Methodology](#methodology)

   * [Data Generation & Input Preprocessing](#data-generation--input-preprocessing)
   * [Model Architecture](#model-architecture)
   * [Activation Functions](#activation-functions)
   * [Training Procedure](#training-procedure)
   * [Evaluation Protocol](#evaluation-protocol)
4. [Results & Discussion](#results--discussion)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Dependencies & Environment](#dependencies--environment)
8. [Contributing](#contributing)
9. [License](#license)
10. [References](#references)

---

## Introduction

Interferometric Synthetic Aperture Radar (InSAR) provides millimetric precision for surface displacement monitoring. This project, conducted by Félix Courtin and Ran Xin under the supervision of Arnaud Breloy, investigates replacing the computationally expensive Covariance Fitting for Interferometric Phase Linking (COFI-PL) pipeline with a Multi-Layer Perceptron (MLP) that directly estimates phase parameters from simulated SAR patches.

While COFI-PL achieves high accuracy, its cost—per-pixel covariance inversion and iterative optimization—limits large-scale deployment. We propose a deep learning approach to approximate phase linking, aiming to retain accuracy while drastically reducing runtime.

---

## Background & Literature Review

1. **InSAR Fundamentals**: Two complex SAR images $I_1$, $I_2$ yield an interferogram by $I_1 \cdot I_2^*$, mapping phase differences to surface deformation estimates.
2. **Phase Linking (IPL)**: Exploits temporal redundancy across $n$ acquisitions \\({x\_i}\_{i=1}^n\\) to build and regularize a covariance matrix $\Sigma$, then solves $\min_{w}\,d(	ilde\Sigma,\Psi\circ\,ww^H)$ on the complex torus $T^p$.
3. **Optimization Techniques**: MM algorithms, Riemannian gradients, and conjugate-gradient schemes have been applied to solve the COFI-PL cost reliably (see Vu et al., 2024) \[1].

---

## Methodology

### Data Generation & Input Preprocessing

- **Synthetic Patches**: Each sample vector $x_i \in \mathbb{C}^p$ simulates a multi‐temporal SAR pixel stack.  
- **Normalization**: Real and imaginary parts are normalized independently to zero mean and unit variance, then de‐normalized for evaluation.  
- **Correlation Parameter** $\rho$: Covariance matrices vary $\rho \in [0.95,\,0.99]$ during training; evaluation extends to $\rho \in [0.7,\,0.99]$.  

### Model Architecture

* **MLP Structure**: Three hidden linear layers:

  First. Projection to higher-dimensional space (captures covariance structure).
  Second–Third. Dimension reduction back to $p$.

* **Complex Weights**: Fully complex-valued parameters and operations to preserve phase information.

### Activation Functions

We benchmarked seven complex-domain activations:

* **CartReLU**: ReLU on real and imaginary parts independently.
* **ZReLU**: Passes values in first complex quadrant only.
* **ModReLU**: Threshold on magnitude with learnable bias, preserves phase.
* **Cardioid**: Scales magnitude by $	frac{1+\cos(\arg z)}{2}$.
* **CartTanh**: Tanh on each part separately.
* **Adjusted Sinusoidal**: $f(x)=x+\alpha\sin(\pi x)$, adds periodic perturbation.
* **ModMVN**: Phase quantization like MVN but multiplies back original magnitude.

### Training Procedure

* **On-the-fly Data Generation**: New simulations each epoch to maximize variance and prevent overfitting.
* **Optimizer**: Adam with learning rate $10^{-3}$.
* **Loss**: Mean Squared Error (MSE) between network output $\hat w$ and COFI-PL ground truth $w$.

### Evaluation Protocol

- **Test Set**: 1,000 patches with $\rho \in [0.7,\,0.99]$.  
- **Metrics**:  
  - $\mathrm{MSE}\bigl(w_{\text{true}},\,w_{\mathrm{COFI\text{-}PL}}\bigr)$ vs.\  
    $\mathrm{MSE}\bigl(w_{\text{true}},\,w_{\mathrm{MLP}}\bigr)$.  
  - Convergence speed across activation functions.  

---

## Results & Discussion

- **Accuracy**: For high coherence ($\rho > 0.95$), the MLP’s MSE matches COFI‐PL within the same order of magnitude.  
- **Robustness**: Under lower coherence ($\rho < 0.9$), certain activations (e.g., CartReLU) show faster convergence.  
- **Recommendations**:  
  1. Use **ModReLU** for balanced performance across all $\rho$ ranges.  
  2. Apply **ZReLU** if fastest initial convergence is required.

**Limitations & Future Work**: Current tests use $5	imes5$ patches over 10 acquisitions. Scaling to larger patches and lower coherence regimes requires architectural adaptation or dual-network designs for real/imag parts.

---

## Usage

```
# Clone & enter directory
git clone https://github.com/RanSHEEN/ProjetLongInSAR.git
cd ProjetLongInSAR

# Environment
conda env create -f InSAR-env-0225.yml
conda activate insar-env

# Generate data
python src/simulations/run_simulation.py --model AffSin --num-samples 1000

# Train MLP
python main.py --config configs/affsin_config.yaml

# Estimate α on test set
python estim_alpha.py --checkpoint outputs/affsin_best.pth

# Compare all methods
python compar_methods.py --results-dir outputs/
```

---

## Project Structure

```
ProjetLongInSAR/
├── InSAR-env-0225.yml       # Conda environment spec
├── src/                     # Source code: simulations, model, utils
├── configs/                 # YAML configs for each activation
├── simulations/             # Data generation scripts
├── outputs/                 # Logs, checkpoints, metrics
├── model-archiv/            # Archived model weights
├── main.py                  # Training/evaluation entry point
├── compar_methods.py        # Performance comparison scripts
├── estim_alpha.py           # Parameter α estimation module
├── quick_test.py            # Fast sanity-check script
├── requirements.txt         # Pip requirements (if any)
└── README.md                # This file
```

---

## Dependencies & Environment

* Python ≥ 3.8
* PyTorch ≥ 1.12
* NumPy, SciPy, matplotlib
* Conda environment provided in `InSAR-env-0225.yml`

---

## Contributing

We welcome improvements! Please fork, branch, and submit pull requests. Ensure tests pass and adhere to PEP8 style.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
