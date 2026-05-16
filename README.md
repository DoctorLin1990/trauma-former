# Trauma-Former: Real-time Prediction of Trauma-Induced Coagulopathy Using an Inverted Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)

> **Huang X\*, Chen W\*, Wei G, Lin W#.**  
> *Real-time prediction of trauma-induced coagulopathy using an inverted transformer (Trauma-Former): a methodological feasibility and simulation study based on the ADEMP framework.*  
> \*Equal contribution. #Corresponding author: DoctorLin1990@163.com

---

## ⚠️ Critical Interpretability Notice

All performance metrics reported in this repository (AUROC 0.939, early warning time 18.1 min) are **upper-bound estimates** derived from a deliberately simplified, linearly-structured synthetic data generator. They **do not** represent real-world clinical performance and **must not** be cited as evidence of diagnostic accuracy. The paramount finding of this study is the **collapse of PPV from 0.89 to 0.48** when evaluated under a realistic 25% TIC prevalence — identifying alarm fatigue as the primary translational barrier. External validation on real-world prehospital data is the absolute prerequisite for any clinical application.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements & Installation](#3-requirements--installation)
4. [Data Generation](#4-data-generation)
5. [One-Command Reproduction](#5-one-command-reproduction)
6. [Step-by-Step Reproduction](#6-step-by-step-reproduction)
7. [Expected Results](#7-expected-results)
8. [Data Availability Statement](#8-data-availability-statement)
9. [Code Availability Statement](#9-code-availability-statement)
10. [Citation](#10-citation)
11. [License](#11-license)

---

## 1. Overview

Trauma-Former is an **inverted Transformer (iTransformer)** architecture for real-time prediction of trauma-induced coagulopathy (TIC) from continuous 1 Hz vital-sign streams (HR, SBP, DBP, SpO₂). The model embeds each variable's 60-second history as an independent token and applies self-attention across variables to model inter-signal coupling.

This repository provides:

- A physiologically parameterised **Ornstein–Uhlenbeck synthetic data generator** (1,240 development + 1,000 test episodes)
- Full **Trauma-Former implementation** (1.52 M parameters) with all nine baseline models (LR-trend, LSTM, GRU, 1D-CNN, XGBoost, PatchTST, Informer, Shock Index)
- **Patient-level 5-fold cross-validation** pipeline with bootstrap 95% CI and Monte Carlo standard errors (MCSE)
- **Independent test set evaluation** at 25% TIC prevalence (PPV collapse analysis)
- Bayesian hyperparameter optimisation (Optuna, 50 trials)
- Robustness tests: Gaussian noise, MCAR missingness, HR sensor dropout
- Network latency simulation (5G URLLC vs 4G LTE)
- Interpretability: cross-variable attention extraction + t-SNE
- Non-linear stress test (Supplementary S3)

---

## 2. Repository Structure

```
trauma_former/
├── configs/                  # YAML hyperparameter files for all models
│   ├── trauma_former.yaml    # Best Bayesian-search configuration (Table S2.2)
│   ├── lstm.yaml
│   ├── gru.yaml
│   ├── cnn.yaml
│   ├── xgboost.yaml
│   ├── patchtst.yaml
│   ├── informer.yaml
│   └── lr_trend.yaml
├── data/
│   ├── synthetic_generator.py # OU simulator (Supplementary S1)
│   ├── generate_datasets.py   # Generates development_set.npz & test_set.npz
│   ├── dataset.py             # TICDataset (sliding windows, masking)
│   └── preprocessing.py      # Z-score normalizer, interpolation
├── models/
│   ├── trauma_former.py       # iTransformer (Algorithm 1 in paper)
│   └── baselines/
│       ├── lstm.py            # BiLSTM (S2.3.4)
│       ├── gru.py             # BiGRU  (S2.3.3)
│       ├── cnn.py             # 1D-CNN (S2.3.2)
│       ├── lr_trend.py        # LR-trend: 12 linear-regression features (S2.3.1)
│       ├── xgboost_model.py   # XGBoost: 20 summary features (S2.3.5)
│       ├── patchtst.py        # PatchTST (S2.3.6)
│       ├── informer.py        # Informer (S2.3.6)
│       └── shock_index.py     # HR/SBP threshold rule
├── training/
│   ├── train_cv.py            # Patient-level 5-fold CV (Section 2.8)
│   ├── trainer.py             # AdamW training loop with early stopping
│   ├── hyperparameter_search.py # Optuna Bayesian search (50 trials, S2.2.1)
│   └── utils.py               # Seed, device, logger helpers
├── evaluation/
│   ├── metrics.py             # AUROC, AUPRC, Brier, calibration, MCSE, bootstrap CI
│   ├── alert_rule.py          # EWT computation (3-min persistence rule)
│   ├── decision_curve.py      # Net-benefit DCA (Figure 3D)
│   ├── interpretability.py    # Attention hook extraction + t-SNE (Figure 5)
│   ├── network_simulation.py  # 5G/4G latency & packet-loss simulation (Section 2.4)
│   └── robustness_tests.py    # Noise / MCAR / sensor-dropout utilities
├── experiments/
│   ├── run_cv.py              # CLI wrapper for train_cv.run_cv()
│   ├── run_test_set.py        # Table 3: 25% prevalence evaluation
│   ├── run_ablation.py        # Section 3.7 ablation studies
│   ├── run_robustness.py      # Figure 4 robustness curves
│   └── run_alert_analysis.py  # Section 3.4 EWT + alert statistics
├── supplementary/
│   └── S3_nonlinear/
│       ├── nonlinear_generator.py # Power-law decompensation generator (S3.2)
│       ├── spline_imputer.py      # Cubic spline imputation (S3.3)
│       └── run_stress_test.py     # Table S3.2 reproduction (S3.4)
├── results/                   # Auto-created; stores CSVs, JSON, checkpoints
│   ├── figures/
│   ├── models/                # trauma_former_best.pt saved here
│   └── logs/
├── train_all_models.py        # One-command pipeline (generates Table 2 + checkpoint)
├── requirements.txt
└── README.md
```

---

## 3. Requirements & Installation

### Hardware

| Component | Minimum | Used in paper |
|-----------|---------|---------------|
| GPU       | NVIDIA GPU with ≥ 8 GB VRAM | NVIDIA A100 SXM4 80 GB |
| CPU       | 8-core modern CPU | Intel Xeon Platinum 8370C, 32 cores |
| RAM       | 16 GB | 256 GB DDR4 ECC |

CPU-only execution is supported (significantly slower; ~4× wall-clock time).

### Software

```bash
# 1. Clone repository
git clone https://github.com/DoctorLin1990/trauma-former.git
cd trauma-former

# 2. Create conda environment (recommended)
conda create -n traumaformer python=3.10 -y
conda activate traumaformer

# 3. Install PyTorch (GPU — CUDA 11.8)
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3b. CPU-only alternative
# pip install torch==2.1.0

# 4. Install all remaining dependencies
pip install -r requirements.txt
```

### Dependency versions (Supplementary Table S2.6)

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

---

## 4. Data Generation

No real patient data are used in this study. All data are generated synthetically via an Ornstein–Uhlenbeck (OU) stochastic simulator parameterised from published physiological ranges (Supplementary S1).

```bash
# Generate both cohorts (~2 min on a standard CPU)
python data/generate_datasets.py
```

This creates:

| File | N episodes | TIC prevalence | Seed | Purpose |
|------|-----------|---------------|------|---------|
| `data/development_set.npz` | 1,240 | 50% | 42 | 5-fold CV (Section 2.8) |
| `data/test_set.npz` | 1,000 | 25% | 43 | PPV-collapse evaluation (Section 3.3) |

Each `.npz` contains:
- `data`: `float32` array of shape `(N, 1800, 4)` — 30 min at 1 Hz, columns `[HR, SBP, DBP, SpO₂]`
- `labels`: `int32` array of shape `(N,)` — `1 = TIC`, `0 = control`

> **Note:** `generate_datasets.py` is called automatically by `train_all_models.py` if the files are absent.

---

## 5. One-Command Reproduction

The following command reproduces **Table 2** (5-fold CV on all 9 models), saves the Trauma-Former checkpoint (`results/models/trauma_former_best.pt`), and prints a formatted summary to stdout.

```bash
# Full pipeline (~3 h on A100, ~12 h on CPU)
python train_all_models.py --seed 42

# Smoke test: 1-fold only (~10 min on GPU)
python train_all_models.py --seed 42 --quick
```

After `train_all_models.py` completes, run the remaining experiments:

```bash
# Table 3: PPV collapse at 25% prevalence
python experiments/run_test_set.py \
    --model_path results/models/trauma_former_best.pt \
    --test_data  data/test_set.npz \
    --dev_data   data/development_set.npz

# Figure 4: Robustness to noise / missingness / sensor dropout
python experiments/run_robustness.py \
    --model_path results/models/trauma_former_best.pt \
    --dev_data   data/development_set.npz

# Section 3.7: Ablation study
python experiments/run_ablation.py \
    --data data/development_set.npz --seed 42

# Section 3.4: Early warning time and alert statistics
python experiments/run_alert_analysis.py \
    --model_path results/models/trauma_former_best.pt \
    --dev_data   data/development_set.npz

# Supplementary S3: Non-linear stress test (Table S3.2)
python supplementary/S3_nonlinear/run_stress_test.py \
    --seed 42 --missing_rate 0.30 --n_episodes 1000

# Bayesian hyperparameter search (50 trials, ~6 h on A100)
python training/hyperparameter_search.py \
    --data data/development_set.npz \
    --n_trials 50 --seed 42 \
    --output results/optuna_study.pkl
```

---

## 6. Step-by-Step Reproduction

### 6.1 Train a single model (5-fold CV)

```bash
# Trauma-Former
python experiments/run_cv.py \
    --config configs/trauma_former.yaml \
    --model  trauma-former \
    --data   data/development_set.npz \
    --seed   42

# LR-trend
python experiments/run_cv.py \
    --config configs/lr_trend.yaml \
    --model  lr-trend \
    --data   data/development_set.npz \
    --seed   42

# XGBoost
python experiments/run_cv.py \
    --config configs/xgboost.yaml \
    --model  xgboost \
    --data   data/development_set.npz \
    --seed   42
```

### 6.2 Model naming convention

| CLI name | Class | Config |
|----------|-------|--------|
| `trauma-former` | `TraumaFormer` | `configs/trauma_former.yaml` |
| `lr-trend` | `LRTrendModel` | `configs/lr_trend.yaml` |
| `lstm` | `LSTMModel` | `configs/lstm.yaml` |
| `gru` | `GRUModel` | `configs/gru.yaml` |
| `cnn` | `CNNModel` | `configs/cnn.yaml` |
| `xgboost` | `XGBoostModel` | `configs/xgboost.yaml` |
| `patchtst` | `PatchTSTModel` | `configs/patchtst.yaml` |
| `informer` | `InformerModel` | `configs/informer.yaml` |
| `shock-index` | `ShockIndexModel` | *(config unused)* |

### 6.3 Random seed convention

All experiments use `--seed 42` (global) with fold-level seeds 42, 43, 44, 45, 46 as specified in Supplementary Table S2.6. Results may differ slightly on different hardware due to floating-point non-determinism in cuDNN operations.

---

## 7. Expected Results

### Table 2 — Development cohort (50% prevalence, 5-fold CV)

| Model | AUROC (95% CI) | MCSE | AUPRC | PPV | Brier |
|-------|---------------|------|-------|-----|-------|
| **Trauma-Former** | **0.939 (0.920–0.950)** | 0.003 | 0.880 | 0.890 | 0.110 |
| LR-trend | 0.917 (0.890–0.940) | 0.004 | 0.830 | 0.860 | 0.140 |
| LSTM | 0.871 (0.850–0.890) | 0.006 | 0.780 | 0.840 | 0.160 |
| 1D-CNN | 0.868 (0.840–0.890) | 0.007 | 0.770 | 0.830 | 0.160 |
| PatchTST | 0.868 (0.840–0.890) | 0.007 | 0.770 | 0.830 | 0.160 |
| Informer | 0.860 (0.830–0.880) | 0.008 | 0.760 | 0.810 | 0.170 |
| GRU | 0.854 (0.830–0.880) | 0.007 | 0.760 | 0.810 | 0.170 |
| XGBoost | 0.821 (0.790–0.850) | 0.009 | 0.690 | 0.760 | 0.200 |
| Shock index | 0.785 (0.730–0.830) | — | 0.420 | — | — |

### Table 3 — Independent test set (25% TIC prevalence)

| Metric | Value (95% CI) |
|--------|---------------|
| AUROC | 0.931 (0.91–0.95) |
| AUPRC | 0.66 (0.62–0.70) |
| Sensitivity | 0.89 (0.85–0.93) |
| Specificity | 0.86 (0.83–0.89) |
| **PPV** | **0.48 (0.43–0.53)** ← collapsed from 0.89 |
| NPV | 0.98 (0.97–0.99) |
| F1 | 0.62 |
| Brier | 0.13 |

### Supplementary Table S3.2 — Non-linear stress test

| Model | AUROC (linear) | AUROC (non-linear) | Δ AUROC |
|-------|---------------|-------------------|---------|
| Trauma-Former | 0.939 | 0.815 | −0.124 |
| 1D-CNN | 0.868 | 0.790 | −0.078 |
| GRU | 0.854 | 0.703 | −0.151 |

> ⚠️ Non-linear results use a single 80/20 split (not 5-fold CV); values are not directly comparable to Table 2.

---

## 8. Data Availability Statement

This study uses **exclusively synthetic data** generated by mathematical simulation. No patient records, electronic health records, or other identifiable health information were used or are required.

The synthetic data generator source code (`data/synthetic_generator.py`) is fully open-source. All parameters are specified in `configs/` and documented in Supplementary S1 (OU process parameters, Tables S1.1–S1.3). Any researcher can reproduce the exact datasets by running:

```bash
python data/generate_datasets.py
```

with `--seed 42` (development set) and `--seed 43` (test set) as defaults.

Upon manuscript acceptance, the complete codebase will be deposited in **Zenodo** with a permanent DOI for long-term archival access.

---

## 9. Code Availability Statement

All source code is released under the MIT License at:  
**https://github.com/DoctorLin1990/trauma-former**

The repository includes:
- All model implementations (PyTorch ≥ 2.1)
- Training and evaluation pipelines
- Hyperparameter search (Optuna)
- Synthetic data generator and fidelity validation
- Figure generation scripts

This repository adheres to the open-science standards recommended by TRIPOD+AI and PROBAST+AI. All random seeds are fixed for exact reproducibility.

---

## 10. Citation

If you use this code or synthetic data generator in your research, please cite:

```bibtex
@article{huang2025traumaformer,
  title   = {Real-time prediction of trauma-induced coagulopathy using an inverted
             transformer (Trauma-Former): a methodological feasibility and simulation
             study based on the ADEMP framework},
  author  = {Huang, Xiaolei and Chen, Wenliang and Wei, Guan and Lin, Wenjia},
  journal = {[Journal name — to be updated upon acceptance]},
  year    = {2025},
  note    = {Preprint. Code: https://github.com/DoctorLin1990/trauma-former}
}
```

---

## 11. License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The synthetic data and all generated results are dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

---

## Funding

This study was funded by the **Fujian Medical University QiHang Fund** (Grant No. 2023QH1130). The funding body had no role in study design, data simulation, analysis, interpretation, or manuscript writing.

## Ethical Statement

This study used exclusively synthetically generated data. No human participants, animal subjects, or identifiable electronic medical records were involved. No institutional review board approval was required.

## Contact

Corresponding author: **Wenjia Lin, MD** — DoctorLin1990@163.com  
Department of Emergency Medicine, The Second Affiliated Hospital of Fujian Medical University, Quanzhou, Fujian, China
