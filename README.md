# FlowGuard — Reproducibility Guide

## CCS 2026 Artifact Evaluation

This guide enables complete reproduction of all results reported in the paper.

---

## 1. Environment Setup

```bash
# Tested on: Python 3.10+, Ubuntu 22.04 / macOS 14+
# Hardware: ≥8 GB RAM, no GPU required (Autoencoder falls back to PCA)

# Clone and install
git clone <repository-url> flowguard
cd flowguard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: install Slither for enhanced parsing
# pip install slither-analyzer
# Requires solc: https://docs.soliditylang.org/en/latest/installing-solidity.html
```

## 2. Dataset Preparation

### Option A: Full Datasets (Recommended)

```bash
# SmartBugs Curated
git clone https://github.com/smartbugs/smartbugs-curated data/raw/smartbugs

# ScrawlD
git clone https://github.com/sujeetc/ScrawlD data/raw/scrawld

# Smart Contract Vulnerabilities 
git clone https://github.com/Messi-Q/Smart-Contract-Dataset

```

### Option B: Synthetic Only (Quick Verification)

No dataset download needed — the pipeline auto-generates synthetic contracts
with known vulnerability patterns when real datasets are unavailable.

## 3. Running the Full Pipeline

```bash
# Full run (synthetic data if datasets not found)
python main.py

# With specific dataset
python main.py --dataset smartbugs

# Single experiment
python main.py --experiment E4

# Regenerate figures from saved results
python main.py --figures-only

# Custom seed for reproducibility verification
python main.py --seed 123
```

## 4. Expected Outputs

```
outputs/
├── results/
│   ├── all_results.json          # Consolidated experiment results
│   ├── e1_static_detection.json  # RQ1: Static detection effectiveness
│   ├── e2_smartbugs.json         # RQ1: SmartBugs complementarity
│   ├── e3_behavioral.json        # RQ2: Behavioral detection (5-fold CV)
│   ├── e4_fusion.json            # RQ3: Fusion benefit + bootstrap test
│   ├── e5_correlation.json       # RQ4: Correlation rule validity
│   ├── e6_scalability.json       # RQ5: Timing measurements
│   ├── e6_timings.csv            # Raw timing data
│   ├── e7_ablation.json          # RQ6: Ablation study
│   ├── risk_rankings.csv         # Final risk-ranked output
│   └── dataset_summary.json      # Dataset statistics
├── figures/
│   ├── fig1_detection_by_category.pdf
│   ├── fig2_behavioral_comparison.pdf
│   ├── fig3_fusion_benefit.pdf     # KEY FIGURE
│   ├── fig4_scalability.pdf
│   ├── fig5_ablation.pdf
│   └── fig6_risk_distribution.pdf
├── reports/
│   └── final_report.json
└── logs/
    └── flowguard_*.log
```

## 5. Verifying Key Claims

### Claim 1: Fusion Outperforms Single Layers (E4)
```bash
python -c "
import json
r = json.load(open('outputs/results/e4_fusion.json'))
print('Static F1:', r['configs']['static_only']['f1'])
print('Behavioral F1:', r['configs']['behavioral_only']['f1'])
print('Fusion F1:', r['configs']['full_fusion']['f1'])
print('Lift:', r['fusion_lift_f1'])
print('Significant:', r['bootstrap']['significant'])
"
```

### Claim 2: Behavioral Ensemble > Individual Models (E3)
```bash
python -c "
import json
r = json.load(open('outputs/results/e3_behavioral.json'))
for m, v in r['average'].items():
    print(f'{m}: AUROC={v[\"auroc\"][\"mean\"]:.3f} F1={v[\"f1\"][\"mean\"]:.3f}')
"
```

### Claim 3: Each Component Contributes (E7)
```bash
python -c "
import json
r = json.load(open('outputs/results/e7_ablation.json'))
print(f'Full system F1: {r[\"full_system_f1\"]:.3f}')
for comp, v in r['ablations'].items():
    print(f'  Without {comp}: F1={v[\"f1\"]:.3f} (Δ={v[\"delta_f1\"]:+.3f})')
"
```

## 6. Configuration

All constants are in `config/default.yaml`. Key parameters:

| Parameter | Value | Location |
|-----------|-------|----------|
| Fusion weights (α, β, γ) | 0.40, 0.35, 0.25 | `fusion.alpha/beta/gamma` |
| Trust weights (w₁–w₄) | 0.28, 0.32, 0.25, 0.30 | `behavioral.trust_weights` |
| Detection threshold | 0.65 | `behavioral.detection_threshold` |
| IF weight / XGB weight / AE weight | 0.35 / 0.45 / 0.20 | `behavioral.anomaly_models.*.ensemble_weight` |
| Cross-validation folds | 5 | `experiments.n_folds` |
| Bootstrap iterations | 1000 | `experiments.bootstrap_iterations` |

## 7. Project Structure

```
flowguard/
├── main.py                              # Entry point
├── config/default.yaml                  # All hyperparameters
├── requirements.txt                     # Dependencies
├── README.md                            # This file
├── src/
│   ├── parsers/
│   │   ├── dataset_loader.py            # Dataset loaders + synthetic generator
│   │   └── solidity_parser.py           # Regex + Slither parser
│   ├── static_analysis/
│   │   ├── stg_builder.py               # STG construction (NetworkX)
│   │   └── detectors.py                 # FG-1 through FG-8 detectors
│   ├── behavioral/
│   │   ├── feature_extractor.py         # 32-feature extraction + trace simulator
│   │   └── anomaly_detection.py         # IF, LOF, XGBoost, Autoencoder
│   ├── fusion/
│   │   └── correlation_engine.py        # Trust scorer + correlation + ranking
│   ├── experiments/
│   │   └── runner.py                    # E1–E7 experiment implementations
│   ├── figures/
│   │   └── plot_results.py              # Publication figures (300 DPI)
│   └── utils/
│       └── helpers.py                   # Config, logging, timing utilities
├── data/
│   ├── raw/                             # Downloaded datasets
│   ├── processed/                       # Parsed/extracted features
│   └── labels/                          # Ground truth annotations
├── outputs/                             # All generated output
└── tests/                               # Unit tests
```

## 8. Estimated Runtime

| Phase | Synthetic (50 contracts) | Full (1000+ contracts) |
|-------|--------------------------|------------------------|
| Data loading | <1s | 10–30s |
| Static analysis | 2–5s | 1–5 min |
| Behavioral simulation | 5–10s | 2–5 min |
| Anomaly detection (5-fold CV) | 10–30s | 2–10 min |
| Fusion + correlation | <5s | 30–60s |
| Figures | 5–10s | 10–20s |
| **Total** | **~1 min** | **~15–30 min** |
