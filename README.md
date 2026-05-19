# Trauma-Former: Real-time Prediction of Trauma-Induced Coagulopathy Using an Inverted Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![Optuna 3.3](https://img.shields.io/badge/Optuna-3.3.0-purple.svg)](https://optuna.org/)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://zenodo.org)

> **Huang X\*, Chen W\*, Wei G, Lin W#.**  
> *Real-time prediction of trauma-induced coagulopathy using an inverted transformer (Trauma-Former): a methodological feasibility and simulation study based on the ADEMP framework.*  
> \*Equal first authorship. #Corresponding author: DoctorLin1990@163.com

---

## ⚠️ Critical Interpretability Warning

All performance metrics in this repository (AUROC 0.939, early warning time 18.1 min) are **upper-bound estimates** derived from a deliberately simplified, linearly structured synthetic data generator. They **do not** represent real-world clinical performance and **must not** be cited as evidence of diagnostic accuracy.

The paramount finding is the **PPV collapse from 0.89 → 0.48** when the model is evaluated under a realistic 25% TIC prevalence — identifying alarm fatigue as the primary translational barrier. External validation on real-world prehospital data is the absolute prerequisite for any clinical application.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Bug Fixes (v3)](#3-bug-fixes-v3)
4. [Requirements and Installation](#4-requirements-and-installation)
5. [Data Generation](#5-data-generation)
6. [One-Command Reproduction](#6-one-command-reproduction)
7. [Step-by-Step Reproduction](#7-step-by-step-reproduction)
8. [Expected Results](#8-expected-results)
9. [Running Tests](#9-running-tests)
10. [Generating Figures](#10-generating-figures)
11. [Data Availability Statement](#11-data-availability-statement)
12. [Code Availability Statement](#12-code-availability-statement)
13. [Citation](#13-citation)
14. [License](#14-license)

---

## 1. Overview

Trauma-Former is an **inverted Transformer (iTransformer)** architecture for real-time prediction of trauma-induced coagulopathy (TIC) from continuous 1 Hz vital-sign streams (HR, SBP, DBP, SpO₂). Each variable's 60-second history is embedded as an independent token; self-attention operates **across variables** to model inter-signal coupling (paper Section 2.5).

This repository provides:

- A physiologically parameterised **Ornstein–Uhlenbeck (OU) synthetic data generator** (Supplementary S1) with the TIC drift superimposed correctly inside the Euler–Maruyama loop
- Full **Trauma-Former implementation** (iTransformer, 1.52 M parameters) with all nine baseline models (LR-trend, LSTM, GRU, 1D-CNN, XGBoost, PatchTST, Informer, Shock Index)
- **Patient-level 5-fold cross-validation** with percentile bootstrap 95% CI and Monte Carlo standard errors (MCSE)
- **Independent test set evaluation** at 25% TIC prevalence (PPV collapse analysis, Table 3)
- **Binary missingness indicator sensitivity analysis** (Section 2.6 / Figure S2)
- Bayesian hyperparameter optimisation (Optuna, 50 trials, Supplementary S2)
- Robustness tests: Gaussian noise, MCAR missingness, HR sensor dropout (Figure 4)
- Network latency simulation: 5G URLLC vs 4G LTE (Section 2.4)
- Interpretability: cross-variable attention extraction and t-SNE (Figure 5)
- Non-linear stress test (Supplementary S3 / Table S3.2)
- Supplementary figures S1–S6 generation scripts

---

## 2. Repository Structure

```
trauma_former/
├── configs/                         # YAML hyperparameter files for all models
│   ├── trauma_former.yaml           # Best Bayesian-search config (Table S2.2)
│   ├── final_config.yaml            # Master experiment config (Supplementary S2.4)
│   ├── lstm.yaml                    # BiLSTM   (S2.3.4)
│   ├── gru.yaml                     # BiGRU    (S2.3.3)
│   ├── cnn.yaml                     # 1D-CNN   (S2.3.2)
│   ├── xgboost.yaml                 # XGBoost  (S2.3.5)
│   ├── patchtst.yaml                # PatchTST (S2.3.6)
│   ├── informer.yaml                # Informer (S2.3.6)
│   └── lr_trend.yaml                # LR-trend (S2.3.1)
│
├── data/
│   ├── synthetic_generator.py       # OU simulator (Supplementary S1) — BUG-FIXED v3
│   ├── generate_datasets.py         # Generates development_set.npz & test_set.npz
│   ├── dataset.py                   # TICDataset (sliding windows, masking)
│   └── preprocessing.py            # Z-score normalizer, interpolation, masking
│
├── simulator/
│   └── ou_generator.py             # CLI shim (matches Supplementary S1.7 command)
│
├── models/
│   ├── trauma_former.py             # iTransformer (Algorithm 1) — main model
│   └── baselines/
│       ├── lstm.py                  # Bidirectional LSTM (S2.3.4)
│       ├── gru.py                   # Bidirectional GRU  (S2.3.3)
│       ├── cnn.py                   # 1D-CNN             (S2.3.2)
│       ├── lr_trend.py              # LR-trend, 12 features (S2.3.1)
│       ├── xgboost_model.py         # XGBoost, 20 features  (S2.3.5)
│       ├── patchtst.py              # PatchTST (self-contained) (S2.3.6)
│       ├── informer.py              # Informer (self-contained) — BUG-FIXED v3
│       └── shock_index.py           # HR/SBP threshold — BUG-FIXED v3 (renamed)
│
├── training/
│   ├── train_cv.py                  # Patient-level 5-fold CV (Section 2.8)
│   ├── trainer.py                   # AdamW loop with AUROC early stopping
│   ├── hyperparameter_search.py     # Optuna Bayesian search (50 trials, S2.2.1)
│   └── utils.py                     # Seed, device, logger helpers
│
├── evaluation/
│   ├── metrics.py                   # AUROC, AUPRC, Brier, PPV, MCSE, Hellinger
│   ├── alert_rule.py                # Alert threshold + persistence rule — BUG-FIXED v3
│   ├── interpretability.py          # Attention extraction, t-SNE (Figure 5)
│   ├── decision_curve.py            # Decision curve analysis (Figure 3D)
│   ├── robustness_tests.py          # Noise, MCAR, sensor dropout (Figure 4)
│   └── network_simulation.py        # 5G/4G latency simulation (Section 2.4)
│
├── experiments/
│   ├── run_cv.py                    # Cross-validation for a single model
│   ├── run_test_set.py              # Table 3: 25% prevalence evaluation
│   ├── run_ablation.py              # Section 3.7 ablation studies
│   ├── run_robustness.py            # Figure 4: robustness experiments
│   ├── run_alert_analysis.py        # Section 3.4: early warning time
│   └── missingness_indicator_analysis.py  # Section 2.6 / Figure S2
│
├── supplementary/
│   └── S3_nonlinear/
│       ├── nonlinear_generator.py   # Power-law OU generator (Table S3.1)
│       ├── spline_imputer.py        # Cubic spline imputation (S3.3)
│       └── run_stress_test.py       # Reproduces Table S3.2 / Figure S6
│
├── figures/
│   └── generate_all_figures.py      # Generates Figures S1–S6 (TIFF, 300 DPI)
│
├── tests/
│   └── test_pipeline.py             # Full smoke-test suite (pytest)
│
├── train_all_models.py              # One-command pipeline (all models + tables)
├── requirements.txt                 # Pinned Python dependencies
└── LICENSE                          # MIT License
```

> **Note:** `data/development_set.npz` and `data/test_set.npz` are not committed to version control. They are generated deterministically by `data/generate_datasets.py` (see [Section 5](#5-data-generation)).

---

## 3. Bug Fixes (v3)

This repository is **version 3** of the codebase. The following bugs present in earlier versions have been corrected:

| ID | File | Description |
|----|------|-------------|
| BUG-1 | `data/synthetic_generator.py` | TIC drift was applied **post-hoc** after the OU loop. Corrected to update the time-varying mean μᵢ(t) **inside** the Euler–Maruyama step, consistent with Supplementary Eq. 3. |
| BUG-2 | `evaluation/alert_rule.py` | `compute_early_warning_time` converted persistence (minutes) to seconds then searched for that many **consecutive samples** in a minute-stride series — essentially never firing. Corrected via `samples_per_minute` parameter. |
| BUG-3 | `models/baselines/shock_index.py` | Class was named `ShockIndex`; renamed to `ShockIndexModel` to match import in `train_cv.py`. |
| BUG-4 | `configs/informer.yaml`, `configs/patchtst.yaml` | `weight_decay` was set to `0.01`; corrected to `1.0e-4` per Supplementary Table S2.2 (identical settings for all baselines). |
| BUG-5 | `configs/informer.yaml`, `configs/patchtst.yaml` | `max_epochs` was set to `100`; corrected to `200` per Supplementary Table S2.2. |
| BUG-6 | `models/baselines/informer.py` | Original required external `informer` package. Replaced with a self-contained ProbSparse attention implementation. |
| BUG-7 | Repository root | Spurious directory `{data,models/...}` (shell brace expansion artefact) removed. |
| BUG-8 | `experiments/` | `missingness_indicator_analysis.py` was entirely absent from earlier versions; added to reproduce Section 2.6 / Figure S2. |
| BUG-9 | `evaluation/metrics.py` | `compute_all_metrics`, `bootstrap_ci`, `monte_carlo_standard_error`, and `multivariate_hellinger_distance` were missing; added. |
| BUG-10 | `tests/` | Test directory was empty (`__init__.py` only); full smoke-test suite added. |
| BUG-11 | `figures/` | Figure generation scripts were absent; `figures/generate_all_figures.py` added for S1–S6. |

---

## 4. Requirements and Installation

### System requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10 (recommended) or 3.9–3.11 |
| CUDA (optional) | 11.8+ for GPU acceleration |
| RAM | ≥ 8 GB (≥ 16 GB recommended for full training) |
| GPU | NVIDIA A100 used in paper; any CUDA GPU or CPU works |
| Disk | ≈ 500 MB for datasets + model checkpoints |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/DoctorLin1990/trauma-former.git
cd trauma-former

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. GPU build (optional, for CUDA 11.8)
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 5. Smoke-test the installation
python -m pytest tests/test_pipeline.py -v --tb=short
```

### Key dependencies (pinned)

```
torch==2.1.0
scikit-learn==1.3.2
xgboost==2.0.2
optuna==3.3.0
numpy==1.26.2
scipy==1.11.4
matplotlib==3.8.2
```

See `requirements.txt` for the complete pinned environment.

---

## 5. Data Generation

All data used in this study are **fully synthetic** — no patient records are required. The two datasets are generated deterministically from fixed random seeds.

```bash
# Generate development set (1,240 episodes, 50% TIC, seed=42)
# and test set (1,000 episodes, 25% TIC, seed=43)
python data/generate_datasets.py
```

Expected output:

```
Generating development set …
  development_set.npz: (1240, 1800, 4)  |  TIC=620  Control=620  Prevalence=50.0%
  Saved → data/development_set.npz  [~60.0 s]
Generating test set …
  test_set.npz: (1000, 1800, 4)  |  TIC=250  Control=750  Prevalence=25.0%
  Saved → data/test_set.npz  [~50.0 s]
Dataset generation complete.
```

The data array shape is `(n_episodes, 1800, 4)` where:
- axis 0: patient episode index
- axis 1: time step (1800 s = 30 min @ 1 Hz)
- axis 2: vital sign channel [HR, SBP, DBP, SpO₂]

The CLI wrapper (Supplementary S1.7) is also available:

```bash
python simulator/ou_generator.py --n_episodes 1240 --prevalence 0.5 \
    --seed 42 --output data/dev_cohort.pkl
```

---

## 6. One-Command Reproduction

To reproduce all results from Tables 2 and 3 in a single command:

```bash
# Full pipeline: train all 9 models + generate Table 2 + Table 3 + save checkpoints
# Expected runtime: ~3 hours on NVIDIA A100; longer on CPU
python train_all_models.py --seed 42

# Quick smoke-test version (1 fold only, ~20 minutes on GPU):
python train_all_models.py --seed 42 --quick
```

Results are saved to `results/`:
- `results/table2_cv_results.csv` — Table 2 (5-fold CV, balanced cohort)
- `results/table3_test_set.json` — Table 3 (25% prevalence test set)
- `results/models/trauma_former_best.pt` — Best Trauma-Former checkpoint

---

## 7. Step-by-Step Reproduction

### Step 1: Generate data

```bash
python data/generate_datasets.py
```

### Step 2: Bayesian hyperparameter search (optional — best config already in `configs/trauma_former.yaml`)

```bash
python training/hyperparameter_search.py \
    --data data/development_set.npz \
    --n_trials 50 --seed 42 \
    --output results/optuna_study.pkl
```

### Step 3: Cross-validation for each model (Table 2)

```bash
for MODEL in trauma-former lr-trend lstm gru cnn xgboost patchtst informer shock-index; do
    python experiments/run_cv.py \
        --config configs/trauma_former.yaml \
        --model  $MODEL \
        --data   data/development_set.npz \
        --seed   42
done
```

### Step 4: Test-set evaluation at 25% prevalence (Table 3)

```bash
python experiments/run_test_set.py \
    --model_path results/models/trauma_former_best.pt \
    --test_data  data/test_set.npz \
    --dev_data   data/development_set.npz \
    --seed       42 \
    --output     results/table3_test_set.json
```

### Step 5: Ablation studies (Section 3.7)

```bash
python experiments/run_ablation.py \
    --data data/development_set.npz \
    --seed 42
```

### Step 6: Robustness experiments (Figure 4)

```bash
python experiments/run_robustness.py \
    --model_path results/models/trauma_former_best.pt \
    --test_data  data/test_set.npz \
    --seed       42
```

### Step 7: Alert rule analysis (Section 3.4 / Figure S4)

```bash
python experiments/run_alert_analysis.py \
    --model_path results/models/trauma_former_best.pt \
    --test_data  data/test_set.npz \
    --dev_data   data/development_set.npz \
    --seed       42
```

### Step 8: Missingness indicator sensitivity analysis (Section 2.6 / Figure S2)

```bash
python experiments/missingness_indicator_analysis.py \
    --model_path  results/models/trauma_former_best.pt \
    --test_data   data/test_set.npz \
    --dev_data    data/development_set.npz \
    --missing_rate 0.30 \
    --seed        42
```

### Step 9: Non-linear stress test (Supplementary S3 / Figure S6)

```bash
python supplementary/S3_nonlinear/run_stress_test.py \
    --seed 42 --missing_rate 0.30 --n_episodes 1000
```

### Step 10: Generate all supplementary figures (S1–S6)

```bash
python figures/generate_all_figures.py
```

---

## 8. Expected Results

### Table 2 — Development cohort (5-fold patient-level CV, 50% prevalence)

| Model | AUROC (95% CI) | MCSE | AUPRC | Sens. | Spec. | F1 | Brier |
|-------|---------------|------|-------|-------|-------|----|-------|
| **Trauma-Former** | **0.939 (0.920–0.950)** | **0.003** | **0.880** | **0.910** | **0.880** | **0.890** | **0.110** |
| LR-trend | 0.917 (0.890–0.940) | 0.004 | 0.830 | 0.860 | 0.850 | 0.860 | 0.140 |
| LSTM | 0.871 (0.850–0.890) | 0.006 | 0.780 | 0.840 | 0.810 | 0.820 | 0.160 |
| 1D-CNN | 0.868 (0.840–0.890) | 0.007 | 0.770 | 0.830 | 0.800 | 0.810 | 0.160 |
| PatchTST | 0.863 (0.840–0.885) | 0.007 | 0.760 | 0.820 | 0.810 | 0.800 | 0.165 |
| Informer | 0.860 (0.830–0.880) | 0.008 | 0.760 | 0.810 | 0.790 | 0.800 | 0.170 |
| GRU | 0.854 (0.830–0.880) | 0.007 | 0.760 | 0.810 | 0.790 | 0.800 | 0.170 |
| XGBoost | 0.821 (0.790–0.850) | 0.009 | 0.690 | 0.760 | 0.770 | 0.760 | 0.200 |
| Shock index | 0.785 (0.730–0.830) | — | 0.420 | 0.550 | 0.850 | 0.670 | — |

> The LR-trend AUROC gap of only 0.022 confirms that the task is predominantly solved by detecting monotonic vital-sign trends.

### Table 3 — Independent test set (25% TIC prevalence)

| Metric | Value (95% CI) |
|--------|---------------|
| AUROC | 0.931 (0.91–0.95) |
| AUPRC | 0.66 (0.62–0.70) |
| Sensitivity | 0.89 (0.85–0.93) |
| Specificity | 0.86 (0.83–0.89) |
| **PPV** | **0.48 (0.43–0.53)** ← primary translational finding |
| NPV | 0.98 (0.97–0.99) |
| F1-score | 0.62 |
| Brier score | 0.13 |
| MCSE (AUROC / PPV) | 0.004 / 0.012 |

> **PPV collapses from 0.89 → 0.48** when moving from balanced (50%) to realistic (25%) prevalence: one in two alerts is false-positive.

### Supplementary Table S3.2 — Non-linear stress test

| Model | AUROC (linear) | AUROC (non-linear) | ΔAUROC |
|-------|---------------|-------------------|--------|
| Trauma-Former | 0.939 | 0.815 | −0.124 |
| 1D-CNN | 0.868 | 0.790 | −0.078 |
| GRU | 0.854 | 0.703 | −0.151 |

> ⚠️ Non-linear results are post-hoc and not directly comparable (different validation protocol, simultaneous confounds — see Supplementary S3.6).

---

## 9. Running Tests

```bash
# Full test suite
python -m pytest tests/test_pipeline.py -v

# Individual test class
python -m pytest tests/test_pipeline.py::TestOUSimulator -v
python -m pytest tests/test_pipeline.py::TestMetrics -v
python -m pytest tests/test_pipeline.py::TestAlertRule -v

# Coverage report (optional, requires pytest-cov)
pip install pytest-cov
python -m pytest tests/test_pipeline.py --cov=. --cov-report=term-missing
```

The test suite covers:

- OU simulator shape, ranges, reproducibility, TIC drift direction, motion artifacts
- Preprocessing: z-score normalizer, interpolation for gaps ≤5 s and >5 s
- All model architectures (forward pass, output range, parameter count)
- Metric functions (AUROC, Brier, MCSE, Hellinger, bootstrap CI)
- Alert rule logic (persistence requirement, EWT computation)
- Dataset class (sliding windows, patient ID, labels)
- Missingness indicator augmentation (8-channel forward pass)
- End-to-end mini CV fold smoke test

---

## 10. Generating Figures

```bash
# Generate all supplementary figures S1–S6 (TIFF, 300 DPI)
python figures/generate_all_figures.py
```

Output files in `figures/`:

| File | Content | Paper Reference |
|------|---------|----------------|
| `Figure_S1_fidelity_validation.tiff` | Correlation error heatmap + HR trajectories | Supplementary Figure S1 |
| `Figure_S2_missingness_indicators.tiff` | AUROC and PPV: masking vs. indicators | Supplementary Figure S2 |
| `Figure_S3_hyperparameter_search.tiff` | Optuna convergence + top-10 configs | Supplementary Figure S3 |
| `Figure_S4_alert_persistence.tiff` | EWT vs. FPR trade-off | Supplementary Figure S4 |
| `Figure_S5_error_analysis.tiff` | FP / FN / TP / TN case examples | Supplementary Figure S5 |
| `Figure_S6_nonlinear_stress.tiff` | Linear vs. non-linear AUROC and Brier | Supplementary Figure S6 |

If experiment results (`results/*.json`, `results/*.csv`) are available, figures are generated from actual results. Otherwise, expected values from the paper are used to produce illustrative figures.

---

## 11. Data Availability Statement

This study used **exclusively synthetically generated data**. No patient records, electronic health records, or identifiable health information were used or are contained in this repository.

The synthetic datasets (development set: 1,240 episodes; test set: 1,000 episodes) are fully reproducible from the OU simulator with fixed random seeds (development: `seed=42`; test: `seed=43`). They can be regenerated in approximately 2 minutes using:

```bash
python data/generate_datasets.py
```

All simulator parameters are documented in Supplementary Material S1 (Additional file 1) and in `data/synthetic_generator.py`. No proprietary data, restricted-access databases, or third-party datasets are required.

Upon manuscript acceptance, the complete codebase will be deposited in Zenodo with a permanent DOI for long-term archival.

---

## 12. Code Availability Statement

The complete source code — including the OU simulator, all model implementations, training and evaluation pipelines, hyperparameter search, alert rule, robustness tests, network simulation, figure generation, and smoke-test suite — is publicly available at:

**https://github.com/DoctorLin1990/trauma-former**

The repository is version-controlled (Git), fully self-contained (no external proprietary packages), and specifies the exact Python environment in `requirements.txt`. All random seeds are fixed to ensure exact reproducibility.

---

## 13. Citation

If you use this code or the synthetic data generator in your research, please cite:

```bibtex
@article{huang2024traumaformer,
  title   = {Real-time prediction of trauma-induced coagulopathy using an
             inverted transformer (Trauma-Former): a methodological feasibility
             and simulation study based on the ADEMP framework},
  author  = {Huang, Xiaolei and Chen, Wenliang and Wei, Guan and Lin, Wenjia},
  journal = {[Journal name]},
  year    = {2025},
  note    = {Manuscript under review},
  url     = {https://github.com/DoctorLin1990/trauma-former}
}
```

---

## 14. License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for full terms.

---

## Funding

This study was funded by the Fujian Medical University QiHang Fund (Grant No. 2023QH1130). The funding body had no role in the design of the study, data simulation, execution of the analysis, interpretation of data, or in writing the manuscript.

---

## Contact

Corresponding author: **Wenjia Lin, MD**  
Email: DoctorLin1990@163.com  
Department of Emergency Medicine, The Second Affiliated Hospital of Fujian Medical University, Quanzhou, Fujian, China
