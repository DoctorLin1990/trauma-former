# Trauma-Former: Real-time Prediction of Trauma-Induced Coagulopathy Using an Inverted Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://zenodo.org)

> **Huang X\*, Chen W\*, Wei G, Lin W#.**
> *Real-time prediction of trauma-induced coagulopathy using an inverted transformer (Trauma-Former): a methodological feasibility and simulation study based on the ADEMP framework.*
> \*Equal first authorship. #Corresponding author: DoctorLin1990@163.com

---

## ⚠️ Critical Interpretability Warning

All performance metrics in this repository (AUROC 0.939, early warning time 18.1 min) are **upper-bound estimates** from a deliberately simplified, linearly-structured synthetic data generator. They **do not** represent real-world clinical performance and **must not** be cited as evidence of diagnostic accuracy.

The paramount finding is the **PPV collapse from 0.89 → 0.48** when evaluated under a realistic 25% TIC prevalence — identifying alarm fatigue as the primary translational barrier. External validation on real-world prehospital data is the absolute prerequisite for any clinical application.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Bug Fixes in v3](#3-bug-fixes-in-v3)
4. [Requirements & Installation](#4-requirements--installation)
5. [Data Generation](#5-data-generation)
6. [One-Command Reproduction](#6-one-command-reproduction)
7. [Step-by-Step Reproduction](#7-step-by-step-reproduction)
8. [Expected Results](#8-expected-results)
9. [Data Availability Statement](#9-data-availability-statement)
10. [Code Availability Statement](#10-code-availability-statement)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Overview

Trauma-Former is an **inverted Transformer (iTransformer)** architecture for real-time prediction of trauma-induced coagulopathy (TIC) from continuous 1 Hz vital-sign streams (HR, SBP, DBP, SpO₂). Each variable's 60-second history is embedded as an independent token; self-attention operates **across variables** to model inter-signal coupling (Section 2.5).

This repository provides:

- A physiologically parameterised **Ornstein–Uhlenbeck synthetic data generator** (Supplementary S1)
- Full **Trauma-Former implementation** (1.52 M parameters) with all nine baseline models
- **Patient-level 5-fold cross-validation** with bootstrap 95% CI and Monte Carlo standard errors
- **Independent test set evaluation** at 25% TIC prevalence (PPV collapse analysis, Table 3)
- **Binary missingness indicator sensitivity analysis** (Section 2.6 / Supplementary Figure S2)
- Bayesian hyperparameter optimisation (Optuna, 50 trials, Supplementary S2)
- Robustness tests: Gaussian noise, MCAR missingness, HR sensor dropout (Figure 4)
- Network latency simulation: 5G URLLC vs 4G LTE (Section 2.4)
- Interpretability: cross-variable attention extraction + t-SNE (Figure 5)
- Non-linear stress test (Supplementary S3 / Table S3.2)

---

## 2. Repository Structure

```
trauma_former/
├── configs/                        # YAML hyperparameter files for all models
│   ├── trauma_former.yaml          # Best Bayesian-search config (Table S2.2)
│   ├── final_config.yaml           # Master experiment config (Supplementary S2.4)
│   ├── lstm.yaml                   # BiLSTM  (S2.3.4)
│   ├── gru.yaml                    # BiGRU   (S2.3.3)
│   ├── cnn.yaml                    # 1D-CNN  (S2.3.2)
│   ├── xgboost.yaml                # XGBoost (S2.3.5)
│   ├── patchtst.yaml               # PatchTST (S2.3.6)
│   ├── informer.yaml               # Informer (S2.3.6)
│   └── lr_trend.yaml               # LR-trend (S2.3.1)
│
├── data/
│   ├── synthetic_generator.py      # OU simulator  ← BUG-FIXED (Supplementary S1)
│   ├── generate_datasets.py        # Generates development_set.npz & test_set.npz
│   ├── dataset.py                  # TICDataset (sliding windows, masking)
│   └── preprocessing.py           # Z-score normalizer, interpolation
│
├── simulator/
│   └── ou_generator.py            # CLI shim (matches S1.7 command syntax)
│
├── models/
│   ├── trauma_former.py            # iTransformer (Algorithm 1)
│   └── baselines/
│       ├── lstm.py                 # BiLSTM
│       ├── gru.py                  # BiGRU
│       ├── cnn.py                  # 1D-CNN
│       ├── lr_trend.py             # LR-trend (12 linear-regression features)
│       ├── xgboost_model.py        # XGBoost (20 summary features)
│       ├── patchtst.py             # PatchTST (self-contained)
│       ├── informer.py             # Informer ← BUG-FIXED (self-contained, no external pkg)
│       └── shock_index.py         # HR/SBP threshold  ← BUG-FIXED (class renamed)
│
├── training/
│   ├── train_cv.py                 # Patient-level 5-fold CV (Section 2.8)
│   ├── trainer.py                  # AdamW training loop with early stopping
│   ├── hyperparameter_search.py    # Optuna Bayesian search (50 trials, S2.2.1)
│   └── utils.py                    # Seed, device, logger helpers
│
├── evaluation/
│   ├── metrics.py                  # AUROC, AUPRC, Brier, calibration, MCSE, CI
│   ├── alert_rule.py              # EWT computation  ← BUG-FIXED (persistence units)
│   ├── decision_curve.py           # Net-benefit DCA (Figure 3D)
│   ├── interpretability.py         # Attention hook extraction + t-SNE (Figure 5)
│   ├── network_simulation.py       # 5G/4G latency & packet-loss simulation
│   └── robustness_tests.py         # Noise / MCAR / sensor-dropout utilities
│
├── experiments/
│   ├── run_cv.py                   # CLI wrapper for train_cv.run_cv()
│   ├── run_test_set.py             # Table 3: 25% prevalence evaluation
│   ├── run_ablation.py             # Section 3.7 ablation studies
│   ├── run_robustness.py           # Figure 4 robustness curves
│   ├── run_alert_analysis.py      # Section 3.4 EWT  ← BUG-FIXED (persistence units)
│   └── missingness_indicator_analysis.py  # NEW: Section 2.6 / Figure S2
│
├── supplementary/
│   └── S3_nonlinear/
│       ├── nonlinear_generator.py  # Power-law decompensation generator (S3.2)
│       ├── spline_imputer.py       # Cubic spline imputation (S3.3)
│       └── run_stress_test.py      # Table S3.2 reproduction (S3.4)
│
├── results/                        # Auto-created; stores CSVs, JSON, checkpoints
│   ├── figures/
│   ├── models/                     # trauma_former_best.pt saved here
│   └── logs/
│
├── train_all_models.py             # One-command pipeline (Table 2 + checkpoint)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 3. Bug Fixes in v3

The following bugs were identified by cross-audit against the paper and all supplementary materials:

| # | Severity | Location | Description | Fix |
|---|----------|----------|-------------|-----|
| 1 | **Critical** | `data/synthetic_generator.py` | TIC drift added post-hoc to X[t] instead of being incorporated into OU mean function μᵢ(t) inside the Euler-Maruyama loop (violates Supplementary Eq. 3) | Drift now enters `mu_t` inside the time loop |
| 2 | **Critical** | `models/baselines/shock_index.py` | Class named `ShockIndex` but `train_cv.py` imports `ShockIndexModel` → `ImportError` at runtime | Renamed to `ShockIndexModel` |
| 3 | **Critical** | `models/baselines/informer.py` | `from informer import Informer` requires a non-pip-installable external GitHub package → `ImportError` for all users | Self-contained ProbSparse implementation using PyTorch MHA |
| 4 | **Critical** | `evaluation/alert_rule.py` + `experiments/run_alert_analysis.py` | `persistence_sec = persistence * 60` converted minutes→seconds, then searched for that many consecutive samples in a minute-level series (stride=60 s) — effectively never fired | `compute_early_warning_time` now accepts `samples_per_minute` parameter; `run_alert_analysis.py` passes `samples_per_minute=1` |
| 5 | **Moderate** | `configs/lstm.yaml`, `cnn.yaml`, `patchtst.yaml`, `informer.yaml` | `weight_decay: 0.01` (should be `1.0e-4` per paper Section 2.7: "identical optimisation settings" as Trauma-Former) | Corrected to `1.0e-4` |
| 6 | **Moderate** | `configs/lstm.yaml`, `cnn.yaml`, `patchtst.yaml`, `informer.yaml` | `max_epochs: 100` (should be 200 per Supplementary Table S2.2 and Section 2.7) | Corrected to 200 |
| 7 | **Moderate** | Repository root | `simulator/ou_generator.py` path referenced in S1.7 (`python simulator/ou_generator.py --n_episodes 1240 …`) did not exist | Added CLI shim at `simulator/ou_generator.py` |
| 8 | **Moderate** | Repository root | Binary missingness indicator analysis (Section 2.6 / Supplementary Figure S2) described in paper but not implemented | Added `experiments/missingness_indicator_analysis.py` |
| 9 | **Minor** | Repository root | `configs/final_config.yaml` referenced in Supplementary S2.4 command but missing | Added master config file |
| 10 | **Minor** | Paper S3.7 | S3.7 says code is at `supplementary/S6/run_stress_test.py` but actual path is `supplementary/S3_nonlinear/run_stress_test.py` | Documented here; code path is correct |

---

## 4. Requirements & Installation

### Hardware

| Component | Minimum | Used in paper |
|-----------|---------|---------------|
| GPU | NVIDIA GPU with ≥ 8 GB VRAM | NVIDIA A100 SXM4 80 GB |
| CPU | 8-core modern CPU | Intel Xeon Platinum 8370C, 32 cores |
| RAM | 16 GB | 256 GB DDR4 ECC |

CPU-only execution is fully supported but ~4× slower.

### Software Installation

```bash
# 1. Clone repository
git clone https://github.com/DoctorLin1990/trauma-former.git
cd trauma-former

# 2. Create conda environment (recommended)
conda create -n traumaformer python=3.10 -y
conda activate traumaformer

# 3a. Install PyTorch — GPU (CUDA 11.8)
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3b. CPU-only alternative
# pip install torch==2.1.0

# 4. Install all remaining dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch, sklearn, xgboost, optuna; print('OK')"
```

### Dependency Versions (Supplementary Table S2.6)

| Package | Version |
|---------|---------|
| Python | 3.10.12 |
| PyTorch | 2.1.0+cu118 |
| scikit-learn | 1.3.2 |
| XGBoost | 2.0.2 |
| Optuna | 3.3.0 |
| NumPy | 1.26.2 |
| SciPy | 1.11.4 |
| pandas | 2.1.3 |
| matplotlib | 3.8.2 |
| PyYAML | 6.0.1 |

> **Note on Informer:** The original paper states "Informer was implemented using its official public repository." In this repository, `models/baselines/informer.py` is a **self-contained implementation** requiring no external packages. It uses PyTorch's `nn.MultiheadAttention` with a ProbSparse-approximated attention mechanism, consistent with Zhou et al. (2021) and the classification adaptation in Supplementary S2.3.6.

---

## 5. Data Generation

No real patient data are used. All data are generated via an Ornstein–Uhlenbeck stochastic simulator parameterised from published physiological ranges (Supplementary S1).

```bash
# Generate both cohorts (≈2 min on a modern CPU)
python data/generate_datasets.py
```

This creates:

| File | N episodes | TIC prevalence | Seed | Purpose |
|------|-----------|---------------|------|---------|
| `data/development_set.npz` | 1,240 | 50% | 42 | 5-fold CV (Section 2.8) |
| `data/test_set.npz` | 1,000 | 25% | 43 | PPV-collapse evaluation (Section 3.3) |

Each `.npz` contains:
- `data`: `float32` array `(N, 1800, 4)` — 30 min at 1 Hz, columns `[HR, SBP, DBP, SpO₂]`
- `labels`: `int32` array `(N,)` — `1 = TIC`, `0 = control`

Alternatively, using the S1.7 command syntax:
```bash
python simulator/ou_generator.py --n_episodes 1240 --prevalence 0.5 \
    --seed 42 --output data/dev_cohort.pkl
```

> `data/generate_datasets.py` is called automatically by `train_all_models.py` if files are absent.

---

## 6. One-Command Reproduction

The following command reproduces **Table 2** (5-fold CV on all 9 models), saves the Trauma-Former checkpoint, and prints a formatted summary:

```bash
# Full pipeline (~3 h on A100; ~12 h on CPU)
python train_all_models.py --seed 42

# Smoke test: 1-fold only (~10 min on GPU)
python train_all_models.py --seed 42 --quick
```

After completion, run the remaining experiments in sequence:

```bash
# Table 3: PPV collapse at 25% prevalence
python experiments/run_test_set.py \
    --model_path results/models/trauma_former_best.pt \
    --test_data  data/test_set.npz \
    --dev_data   data/development_set.npz

# Supplementary Figure S2: Missingness indicator analysis (Section 2.6)
python experiments/missingness_indicator_analysis.py \
    --model_path results/models/trauma_former_best.pt \
    --test_data  data/test_set.npz \
    --dev_data   data/development_set.npz \
    --missing_rate 0.30

# Figure 4: Robustness to noise / missingness / sensor dropout
python experiments/run_robustness.py \
    --model_path results/models/trauma_former_best.pt \
    --dev_data   data/development_set.npz

# Section 3.7: Ablation study
python experiments/run_ablation.py \
    --data data/development_set.npz --seed 42

# Section 3.4: Early warning time and alert statistics (Supplementary Figure S4)
python experiments/run_alert_analysis.py \
    --model_path results/models/trauma_former_best.pt \
    --dev_data   data/development_set.npz

# Supplementary S3: Non-linear stress test (Table S3.2)
python supplementary/S3_nonlinear/run_stress_test.py \
    --seed 42 --missing_rate 0.30 --n_episodes 1000

# Supplementary S2: Bayesian hyperparameter search (50 trials, ~6 h on A100)
python training/hyperparameter_search.py \
    --data data/development_set.npz \
    --n_trials 50 --seed 42 \
    --output results/optuna_study.pkl
```

---

## 7. Step-by-Step Reproduction

### 7.1 Train a single model (5-fold CV)

```bash
# Trauma-Former (primary model)
python experiments/run_cv.py \
    --config configs/trauma_former.yaml \
    --model  trauma-former \
    --data   data/development_set.npz \
    --seed   42

# LR-trend (key diagnostic baseline)
python experiments/run_cv.py \
    --config configs/lr_trend.yaml \
    --model  lr-trend \
    --data   data/development_set.npz

# Any other baseline
python experiments/run_cv.py \
    --config configs/gru.yaml --model gru --data data/development_set.npz
python experiments/run_cv.py \
    --config configs/cnn.yaml --model cnn --data data/development_set.npz
python experiments/run_cv.py \
    --config configs/xgboost.yaml --model xgboost --data data/development_set.npz
```

### 7.2 Model naming convention

| CLI name | Python class | Config file |
|----------|-------------|-------------|
| `trauma-former` | `TraumaFormer` | `configs/trauma_former.yaml` |
| `lr-trend` | `LRTrendModel` | `configs/lr_trend.yaml` |
| `lstm` | `LSTMModel` | `configs/lstm.yaml` |
| `gru` | `GRUModel` | `configs/gru.yaml` |
| `cnn` | `CNNModel` | `configs/cnn.yaml` |
| `xgboost` | `XGBoostModel` | `configs/xgboost.yaml` |
| `patchtst` | `PatchTSTModel` | `configs/patchtst.yaml` |
| `informer` | `InformerModel` | `configs/informer.yaml` |
| `shock-index` | `ShockIndexModel` | *(config unused)* |

### 7.3 Random seed convention

| Scope | Seed |
|-------|------|
| Global | 42 |
| CV fold 1–5 | 42, 43, 44, 45, 46 |
| Development set generation | 42 |
| Test set generation | 43 |
| Bootstrap CI | 42 |

Results may differ slightly on different hardware due to floating-point non-determinism in cuDNN.

---

## 8. Expected Results

### Table 2 — Development cohort (50% prevalence, 5-fold patient-level CV)

| Model | AUROC (95% CI) | MCSE | AUPRC | Sens. | Spec. | F1 | Brier |
|-------|---------------|------|-------|-------|-------|----|-------|
| **Trauma-Former** | **0.939 (0.920–0.950)** | 0.003 | 0.880 | 0.910 | 0.880 | 0.890 | 0.110 |
| LR-trend† | 0.917 (0.890–0.940) | 0.004 | 0.830 | 0.860 | 0.850 | 0.860 | 0.140 |
| LSTM | 0.871 (0.850–0.890) | 0.006 | 0.780 | 0.840 | 0.810 | 0.820 | 0.160 |
| 1D-CNN | 0.868 (0.840–0.890) | 0.007 | 0.770 | 0.830 | 0.800 | 0.810 | 0.160 |
| PatchTST | 0.863 (0.840–0.885) | 0.007 | 0.760 | 0.820 | 0.810 | 0.800 | 0.165 |
| Informer | 0.860 (0.830–0.880) | 0.008 | 0.760 | 0.810 | 0.790 | 0.800 | 0.170 |
| GRU | 0.854 (0.830–0.880) | 0.007 | 0.760 | 0.810 | 0.790 | 0.800 | 0.170 |
| XGBoost‡ | 0.821 (0.790–0.850) | 0.009 | 0.690 | 0.760 | 0.770 | 0.760 | 0.200 |
| Shock index | 0.785 (0.730–0.830) | — | 0.420 | 0.550 | 0.850 | 0.670 | — |

†LR-trend AUROC gap vs Trauma-Former: only 0.022 — confirms the synthetic task is predominantly detectable by monotonic trend features.
‡XGBoost uses 20 hand-crafted features vs full 60-s waveform for DL models.

### Table 3 — Independent test set (25% TIC prevalence) — Key Result

| Metric | Value (95% CI) |
|--------|----------------|
| AUROC | 0.931 (0.91–0.95) |
| AUPRC | 0.66 (0.62–0.70) |
| Sensitivity | 0.89 (0.85–0.93) |
| Specificity | 0.86 (0.83–0.89) |
| **PPV** | **0.48 (0.43–0.53)** ← collapsed from 0.89 at 50% prevalence |
| NPV | 0.98 (0.97–0.99) |
| F1 | 0.62 |
| Brier | 0.13 |

### Supplementary Figure S2 — Missingness Indicator Sensitivity Analysis

| Condition | AUROC | PPV |
|-----------|-------|-----|
| Standard masking (30% MCAR) | 0.931 | 0.48 |
| + Binary missingness indicators | 0.926 | 0.50 |
| Marginal PPV improvement | — | +0.02 |

### Supplementary Table S3.2 — Non-linear Stress Test

| Model | AUROC (linear) | AUROC (non-linear) | Δ AUROC |
|-------|---------------|-------------------|---------|
| Trauma-Former | 0.939 | 0.815 | −0.124 |
| 1D-CNN | 0.868 | 0.790 | −0.078 |
| GRU | 0.854 | 0.703 | −0.151 |

> ⚠️ Non-linear results use a single 80/20 split (not 5-fold CV). Not directly comparable to Table 2.

---

## 9. Data Availability Statement

This study uses **exclusively synthetic data** generated by mathematical simulation. No patient records, electronic health records, or identifiable health information were used or are required.

The synthetic data generator (`data/synthetic_generator.py`) is fully open-source. All parameters are documented in Supplementary S1 (Tables S1.1–S1.3). Datasets can be reproduced exactly by running:

```bash
python data/generate_datasets.py
```

Upon manuscript acceptance, the complete codebase will be deposited in **Zenodo** with a permanent DOI for long-term archival access.

---

## 10. Code Availability Statement

All source code is released under the MIT License at:
**https://github.com/DoctorLin1990/trauma-former**

This repository adheres to open-science standards recommended by TRIPOD+AI and PROBAST+AI. All random seeds are fixed for exact reproducibility. The version described in this README corresponds to `v3` (post-peer-review bug-fix release).

---

## 11. Citation

If you use this code or synthetic data generator, please cite:

```bibtex
@article{huang2025traumaformer,
  title   = {Real-time prediction of trauma-induced coagulopathy using an inverted
             transformer ({Trauma-Former}): a methodological feasibility and simulation
             study based on the {ADEMP} framework},
  author  = {Huang, Xiaolei and Chen, Wenliang and Wei, Guan and Lin, Wenjia},
  journal = {[Journal — to be updated upon acceptance]},
  year    = {2025},
  note    = {Code: \url{https://github.com/DoctorLin1990/trauma-former}}
}
```

---

## 12. License

Source code: **MIT License** — see [LICENSE](LICENSE).
Synthetic data and generated results: **CC0 1.0 Public Domain**.

---

## Funding

Fujian Medical University QiHang Fund (Grant No. 2023QH1130). The funding body had no role in study design, simulation, analysis, or manuscript writing.

## Ethical Statement

This study used exclusively synthetically generated data. No human participants, animal subjects, or identifiable records were involved. No IRB approval was required.

## Contact

Corresponding author: **Wenjia Lin, MD** — DoctorLin1990@163.com
Department of Emergency Medicine, The Second Affiliated Hospital of Fujian Medical University, Quanzhou, Fujian, China
