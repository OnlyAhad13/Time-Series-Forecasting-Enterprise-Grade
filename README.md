<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Weights_&_Biases-Integrated-FFCC33?style=flat-square&logo=weightsandbiases&logoColor=black" alt="W&B"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-1.24+-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat-square&logo=plotly&logoColor=white" alt="Matplotlib"/>
</p>

<h1 align="center">🔮 Production-Grade Time Series Forecasting</h1>

<p align="center">
  <strong>A comprehensive, research-grade deep learning framework for multi-horizon time series forecasting</strong>
</p>

<p align="center">
  <em>Implementing MAANG/NVIDIA-level engineering standards with state-of-the-art neural architectures</em>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Model Zoo](#-model-zoo)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Evaluation & Metrics](#-evaluation--metrics)
- [Advanced Usage](#-advanced-usage)
- [Performance Optimization](#-performance-optimization)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This repository provides a **production-ready, modular framework** for training and deploying deep learning models for time series forecasting. Designed with research reproducibility and enterprise deployment in mind, it supports:

| Capability | Description |
|------------|-------------|
| **Multi-horizon Forecasting** | Predict arbitrary future horizons (7, 14, 28+ steps) |
| **Probabilistic Outputs** | Quantile regression and Gaussian likelihood for uncertainty estimation |
| **Panel Data Support** | Simultaneously forecast thousands of related time series |
| **External Covariates** | Incorporate holidays, weather, economic indicators |
| **Mixed Precision Training** | Accelerated training with `torch.cuda.amp` |

### Why This Framework?

> Traditional forecasting libraries often sacrifice either **research flexibility** or **production robustness**. This framework bridges that gap with clean abstractions, comprehensive logging, and battle-tested training loops.

---

## ✨ Key Features

### 📊 Data Pipeline
- **Timezone Normalization** — Consistent temporal alignment across data sources
- **Intelligent Gap Filling** — Forward fill, interpolation, or zero-fill for missing timestamps
- **Outlier Detection** — IQR and Z-score based anomaly handling with configurable thresholds
- **Chronological Splitting** — Train/validation/test splits with **zero data leakage**
- **Series-specific Scaling** — Robust/Standard scalers fitted per series for panel data

### 🔧 Feature Engineering
| Feature Type | Implementation |
|--------------|----------------|
| **Lag Features** | Configurable lookback periods (e.g., `[1, 7, 14, 28, 365]`) |
| **Rolling Statistics** | Mean, std, min, max over sliding windows |
| **Cyclical Encodings** | Fourier features for hour, day, week, month, year |
| **Holiday Indicators** | Country-specific holiday detection via `holidays` library |
| **Category Embeddings** | Learned embeddings for categorical covariates |

### 🎓 Training Infrastructure
- **AdamW Optimizer** with decoupled weight decay
- **Learning Rate Schedulers**: OneCycleLR, ReduceLROnPlateau, CosineAnnealing
- **Gradient Clipping** for training stability
- **Early Stopping** with configurable patience and delta
- **Mixed Precision (AMP)** for 2x+ speedup on modern GPUs
- **Full Reproducibility** — Deterministic operations with seed control
- **Weights & Biases Integration** — Comprehensive experiment tracking

### 📈 Evaluation Suite
- **Point Metrics**: MAE, RMSE, MAPE, sMAPE
- **Probabilistic Metrics**: Quantile Loss (Pinball), Prediction Interval Coverage
- **Walk-forward Backtesting** — Realistic out-of-sample evaluation
- **Visualization Toolkit** — Forecast plots, residual analysis, model comparisons

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CONFIGURATION LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                      │
│  │ DataConfig  │  │ ModelConfig │  │TrainingConfig│                     │
│  └─────────────┘  └─────────────┘  └─────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                 │
│  Raw CSV → Preprocessing → Feature Engineering → Scaling → DataLoader  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐           │
│  │  Naive   │  │LSTM/GRU  │  │   TCN    │  │  Transformer  │           │
│  │ Baseline │  │ Seq2Seq  │  │          │  │               │           │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          TRAINING ENGINE                                │
│  Trainer → Loss Functions → Schedulers → Callbacks → Checkpointing     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION & OUTPUT                             │
│  Metrics Computation → Backtesting → Visualization → Model Export      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Model Zoo

### Implemented Architectures

| Model | Description | Best For |
|-------|-------------|----------|
| **Seasonal Naive** | Baseline using seasonal lag | Benchmarking |
| **LSTM Seq2Seq** | Encoder-decoder with teacher forcing | Medium sequences, strong baselines |
| **GRU Seq2Seq** | Lighter alternative to LSTM | Resource-constrained settings |
| **TCN** | Dilated causal convolutions | Long sequences, parallelizable |
| **Transformer** | Self-attention mechanism | Complex dependencies, large datasets |

### Output Heads

All models support three output types:

```python
# Point Forecast
output_type = "point"  # → (batch, horizon)

# Quantile Regression
output_type = "quantile"
quantiles = [0.1, 0.5, 0.9]  # → (batch, horizon, num_quantiles)

# Gaussian Likelihood
output_type = "gaussian"  # → (μ, σ) each (batch, horizon)
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/OnlyAhad13/Time-Series-Forecasting-Enterprise-Grade.git
cd Time-Series-Forecasting-Enterprise-Grade

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as editable package
pip install -e .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `pandas` | ≥2.0.0 | Data manipulation |
| `scikit-learn` | ≥1.3.0 | Preprocessing utilities |
| `wandb` | ≥0.15.0 | Experiment tracking |
| `holidays` | ≥0.35 | Holiday feature engineering |
| `matplotlib` | ≥3.7.0 | Visualization |

---

## 🚀 Quick Start

### 1. Prepare Your Data

Your CSV should contain:

| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | ✅ | Datetime column |
| `target` | ✅ | Value to forecast |
| `series_id` | ❌ | Identifier for panel data |
| Covariates | ❌ | Additional features |

```csv
timestamp,value,series_id,temperature
2024-01-01 00:00:00,42.5,store_1,72.4
2024-01-01 01:00:00,43.2,store_1,71.8
...
```

### 2. Train a Model

```bash
# LSTM with default configuration
python experiments/train.py \
    --config config/example_config.yaml \
    --model lstm \
    --experiment my_lstm_experiment

# Transformer with custom settings
python experiments/train.py \
    --config config/example_config.yaml \
    --model transformer \
    --experiment transformer_v1 \
    --no-wandb  # Disable W&B logging
```

### 3. Evaluate

```bash
python experiments/evaluate.py \
    --checkpoint checkpoints/my_lstm_experiment_best.pt \
    --data data/test.csv \
    --output outputs/evaluation_results
```

### 4. Compare Models

```bash
python experiments/compare_models.py \
    --data data/timeseries.csv \
    --models lstm gru tcn transformer \
    --epochs 50 \
    --output outputs/model_comparison
```

---

## ⚙️ Configuration

All experiments are controlled via **YAML configuration files**:

```yaml
# config/example_config.yaml

data:
  data_path: "data/processed/dataset.csv"
  timestamp_col: "timestamp"
  target_col: "value"
  series_id_col: "series_id"        # null for single series
  freq: "H"                          # Hourly frequency
  
  # Preprocessing
  fill_method: "interpolate"         # forward, interpolate, zero
  outlier_method: "iqr"              # iqr, zscore, none
  outlier_threshold: 3.0
  
  # Splits
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  
  # Feature engineering
  lags: [1, 2, 3, 7, 14, 28]
  rolling_windows: [7, 14, 28]
  rolling_stats: ["mean", "std", "min", "max"]

model:
  model_type: "lstm"                 # lstm, gru, tcn, transformer
  input_dim: 1                       # Auto-updated based on features
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
  lookback: 168                      # 7 days of hourly data
  horizon: 24                        # Predict next 24 hours
  output_type: "quantile"            # point, quantile, gaussian
  quantiles: [0.1, 0.5, 0.9]

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-5
  grad_clip: 1.0
  teacher_forcing_ratio: 0.5         # For LSTM/GRU
  
  scheduler: "onecycle"              # onecycle, plateau, cosine
  mixed_precision: true
  
  early_stopping: true
  patience: 10
  
  use_wandb: true
  wandb_project: "ts-forecasting"

experiment_name: "lstm_baseline"
```

---

## 📁 Project Structure

```
Time-Series-Forecasting-Enterprise-Grade/
│
├── 📂 config/                      # Configuration management
│   ├── base_config.py              # Dataclass-based configs
│   └── *.yaml                      # Experiment configurations
│
├── 📂 data/                        # Data handling
│   ├── preprocessing.py            # Cleaning, normalization, splitting
│   ├── dataset.py                  # PyTorch Dataset implementation
│   └── dataloader.py               # DataLoader factory
│
├── 📂 features/                    # Feature engineering
│   ├── temporal.py                 # Time-based features
│   ├── lag_features.py             # Lag and rolling statistics
│   └── categorical.py              # Category embeddings
│
├── 📂 models/                      # Neural network architectures
│   ├── base_model.py               # Abstract base class
│   ├── naive.py                    # Seasonal naive baseline
│   ├── lstm_seq2seq.py             # LSTM/GRU encoder-decoder
│   ├── tcn.py                      # Temporal Convolutional Network
│   ├── transformer.py              # Transformer forecaster
│   └── create_model.py             # Model factory
│
├── 📂 training/                    # Training infrastructure
│   ├── trainer.py                  # Main training loop
│   ├── losses.py                   # MSE, Quantile, Gaussian NLL
│   └── callbacks.py                # EarlyStopping, ModelCheckpoint
│
├── 📂 evaluation/                  # Evaluation tools
│   ├── metrics.py                  # MAE, RMSE, MAPE, sMAPE
│   ├── backtesting.py              # Walk-forward validation
│   └── visualization.py            # Plotting utilities
│
├── 📂 utils/                       # Utilities
│   ├── reproducibility.py          # Seed management
│   ├── logger.py                   # Logging and W&B init
│   └── scalers.py                  # TimeSeriesScaler
│
├── 📂 experiments/                 # Entry point scripts
│   ├── train.py                    # Training script
│   └── evaluate.py                 # Evaluation script
│
├── 📂 tests/                       # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── 📂 checkpoints/                 # Saved model weights
├── 📂 logs/                        # Training logs
├── 📂 outputs/                     # Evaluation outputs
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔬 Methodology

### Data Leakage Prevention

> **Critical for time series**: Standard random splits cause information leakage.

This framework enforces:

1. **Chronological Splits** — Train → Validate → Test in temporal order
2. **Fit on Train Only** — Scalers fitted exclusively on training data
3. **Temporal Feature Boundaries** — Lag features respect split boundaries

### Teacher Forcing

For sequence-to-sequence models (LSTM/GRU), we implement **scheduled sampling**:

```python
# During training, randomly use ground truth or model predictions
if random() < teacher_forcing_ratio:
    decoder_input = ground_truth[t]
else:
    decoder_input = model_prediction[t-1]
```

This bridges the train-test distribution gap and improves convergence.

### Loss Functions

| Output Type | Loss Function | Formula |
|-------------|---------------|---------|
| Point | MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ |
| Quantile | Pinball | $\sum \max[\tau(y-\hat{y}), (\tau-1)(y-\hat{y})]$ |
| Gaussian | NLL | $\frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$ |

---

## 📈 Evaluation & Metrics

### Point Forecast Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | $\frac{1}{n}\sum|y - \hat{y}|$ | Average absolute error |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | Penalizes large errors |
| **MAPE** | $\frac{100}{n}\sum|\frac{y - \hat{y}}{y}|$ | Percentage error |
| **sMAPE** | $\frac{200}{n}\sum\frac{|y - \hat{y}|}{|y| + |\hat{y}|}$ | Symmetric percentage |

### Probabilistic Metrics

- **Quantile Loss**: Asymmetric penalty for over/under prediction
- **Coverage**: Fraction of actuals within prediction intervals
- **Sharpness**: Width of prediction intervals

### Walk-Forward Backtesting

```
Time ─────────────────────────────────────────────────────►
      │ Train │ Val │ Test │
            │ Train │ Val │ Test │
                  │ Train │ Val │ Test │
                        │ Train │ Val │ Test │
```

Expanding window evaluation for realistic performance estimation.

---

## 🔧 Advanced Usage

### Panel Data (Multiple Series)

```python
# In config:
data:
  series_id_col: "store_id"  # Each store forecasted separately
```

### Probabilistic Forecasting

```python
model:
  output_type: "quantile"
  quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]  # 90% prediction interval
```

### Custom Feature Engineering

```python
data:
  lags: [1, 2, 3, 7, 14, 28, 365]       # Include year-over-year
  rolling_windows: [7, 30, 90, 365]     # Multi-scale statistics
```

### Gradient Accumulation

```python
training:
  batch_size: 16
  accumulation_steps: 4  # Effective batch size = 64
```

---

## ⚡ Performance Optimization

### GPU Acceleration

```python
training:
  mixed_precision: true  # 2x speedup with minimal precision loss
```

### Long Sequences

```python
model:
  model_type: "tcn"  # O(log n) receptive field growth
  model_params:
    kernel_size: 7
    dilation_base: 2
```

### Large-Scale Panel Data

```python
training:
  batch_size: 256
  num_workers: 8       # Parallel data loading
```

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{ts_forecasting_2025,
  title     = {Production-Grade Time Series Forecasting with PyTorch},
  author    = {Syed Abdul Ahad},
  year      = {2025},
  url       = {https://github.com/OnlyAhad13/Time-Series-Forecasting-Enterprise-Grade},
  note      = {A comprehensive deep learning framework for multi-horizon forecasting}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Ensure** all tests pass (`pytest tests/ -v`)
5. **Submit** a pull request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run linting
black . --check
flake8 .
mypy .

# Run tests
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ for the forecasting community</strong>
</p>

<p align="center">
  <a href="https://github.com/OnlyAhad13/Time-Series-Forecasting-Enterprise-Grade/issues">Report Bug</a>
  ·
  <a href="https://github.com/OnlyAhad13/Time-Series-Forecasting-Enterprise-Grade/issues">Request Feature</a>
</p>
