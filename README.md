# Production-Grade Time Series Forecasting with PyTorch

A comprehensive, production-ready time series forecasting framework implementing MAANG/NVIDIA-level engineering standards.

## Features

### 🎯 Problem Scope
- Multi-step forecasting (configurable horizon: 7/14/28+ steps)
- Single series and panel (multiple series) forecasting
- Point forecasts and probabilistic forecasts (quantiles, Gaussian)

### 📊 Data Handling
- Timezone normalization
- Missing timestamp detection & intelligent filling
- Outlier detection and handling (IQR, Z-score)
- Chronological train/val/test splits (no data leakage)
- Robust scaling with series-specific scalers
- External covariates support (holidays, weather, macro signals)

### 🔧 Feature Engineering
- Lag features with configurable lookback periods
- Rolling window statistics (mean, std, min, max)
- Cyclical time encodings (sin/cos of hour/day/week/month)
- Holiday indicators
- Category embeddings for static features

### 🤖 Models Implemented
1. **Seasonal Naive** - Baseline for comparison
2. **LSTM/GRU Seq2Seq** - Encoder-decoder with multi-horizon decoding
3. **TCN** - Temporal Convolutional Network with dilated causal convolutions
4. **Transformer** - Self-attention based forecaster

All models support:
- Dropout and Layer/Weight Normalization
- Point, quantile, and Gaussian likelihood outputs
- Gradient clipping
- Mixed precision training

### 🎓 Training Best Practices
- AdamW optimizer with weight decay
- Multiple LR schedulers (OneCycle, ReduceLROnPlateau, Cosine)
- Gradient clipping for stability
- Early stopping with patience
- Mixed precision training (torch.cuda.amp)
- Comprehensive logging with Weights & Biases
- Full reproducibility (seed setting, deterministic operations)

### 📈 Evaluation
- **Metrics**: MAE, RMSE, MAPE, sMAPE
- **Probabilistic**: Quantile Loss (Pinball), Coverage
- Walk-forward backtesting
- Comprehensive visualizations (forecasts, residuals, comparisons)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ts-forecasting.git
cd ts-forecasting

# Install dependencies
pip install -r requirements.txt

# Install as package (optional)
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

Your data should be a CSV with at least:
- Timestamp column
- Target value column
- Optional: Series ID column for panel data
- Optional: Covariate columns

Example:
```csv
timestamp,value,series_id
2024-01-01 00:00:00,42.5,A
2024-01-01 01:00:00,43.2,A
...
```

### 2. Train a Model

```bash
# Train LSTM model
python experiments/train.py \\
    --data data/timeseries.csv \\
    --model lstm \\
    --experiment lstm_baseline \\
    --config config/example_config.yaml

# Train Transformer model
python experiments/train.py \\
    --data data/timeseries.csv \\
    --model transformer \\
    --experiment transformer_baseline
```

### 3. Evaluate Model

```bash
python experiments/evaluate.py \\
    --checkpoint checkpoints/lstm_baseline_best.pt \\
    --data data/test.csv \\
    --output outputs/evaluation
```

### 4. Compare Multiple Models

```bash
python experiments/compare_models.py \\
    --data data/timeseries.csv \\
    --models lstm gru tcn transformer \\
    --epochs 50 \\
    --output outputs/comparison
```

## Project Structure

```
ts_forecasting/
├── config/                 # Configuration files
│   ├── base_config.py
│   └── example_config.yaml
├── data/                   # Data handling
│   ├── dataset.py         # PyTorch datasets
│   ├── dataloader.py      # DataLoader utilities
│   └── preprocessing.py   # Preprocessing pipeline
├── features/               # Feature engineering
│   ├── temporal.py        # Time-based features
│   ├── lag_features.py    # Lag & rolling features
│   └── categorical.py     # Embeddings
├── models/                 # Model implementations
│   ├── base_model.py      # Abstract base
│   ├── naive.py           # Baseline
│   ├── lstm_seq2seq.py    # LSTM/GRU
│   ├── tcn.py             # TCN
│   └── transformer.py     # Transformer
├── training/               # Training utilities
│   ├── trainer.py         # Training loop
│   ├── losses.py          # Loss functions
│   └── callbacks.py       # Callbacks
├── evaluation/             # Evaluation tools
│   ├── metrics.py         # Metrics
│   ├── backtesting.py     # Walk-forward validation
│   └── visualization.py   # Plotting
├── utils/                  # Utilities
│   ├── reproducibility.py # Seed setting
│   ├── logger.py          # Logging
│   └── scalers.py         # Scaling
├── experiments/            # Main scripts
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   └── compare_models.py  # Model comparison
└── tests/                  # Unit tests
    ├── test_data.py
    ├── test_models.py
    └── test_metrics.py
```

## Configuration

All experiments are controlled via YAML configuration files. Example:

```yaml
data:
  freq: "H"              # Hourly data
  lookback: 168          # Use 7 days (168 hours) of history
  horizon: 24            # Predict next 24 hours
  
model:
  model_type: "lstm"
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
  
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  early_stopping: true
  patience: 10
```

## Advanced Usage

### Panel Data (Multiple Series)

```python
config.data.series_id_col = "series_id"  # Enable panel forecasting
```

### Probabilistic Forecasting

```python
config.model.output_type = "quantile"
config.model.quantiles = [0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
```

### Custom Features

```python
config.data.lags = [1, 2, 3, 7, 14, 28, 365]  # Year-over-year lag
config.data.rolling_windows = [7, 30, 90]     # Weekly, monthly, quarterly
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Design Decisions

### 1. No Data Leakage
- Chronological splits only
- Scalers fit on training data only
- Features computed respecting time boundaries

### 2. Reproducibility
- All random seeds set
- Deterministic operations enabled
- Configuration tracking

### 3. Modularity
- Clean separation of concerns
- Easy to extend with new models
- Pluggable components

### 4. Production Readiness
- Comprehensive error handling
- Extensive logging
- Type hints throughout
- Unit tests for critical paths

## Performance Tips

### For Large Datasets
```python
config.training.accumulation_steps = 4  # Gradient accumulation
config.training.mixed_precision = True  # Use AMP
```

### For Long Sequences
```python
config.model.model_type = "tcn"  # TCN handles long sequences well
config.model.model_params = {"kernel_size": 7}
```

### For Many Series
```python
config.training.batch_size = 128  # Larger batches for efficiency
config.data.series_id_col = "series_id"  # Enable panel mode
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ts_forecasting,
  title = {Production-Grade Time Series Forecasting with PyTorch},
  author = {Syed Abdul Ahad},
  year = {2025},
  url = {https://github.com/OnlyAhad13/Time-Series-Forecasting-Enterprise-Grade}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue.

---
