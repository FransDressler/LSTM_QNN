# 🧠 LSTM + Quantum Neural Network Project

Advanced financial prediction using hybrid classical-quantum neural networks.

## 📁 Project Structure

```
LSTM_QNN/
├── src/
│   ├── stocks/             # Stock prediction with relative features
│   ├── swaptions/          # Swaption prediction (original project)
│   ├── quantum/            # Quantum computing components
│   └── utils/              # Shared utilities
├── data/                   # Training data (.pt files)
├── models/                 # Trained model weights (.pth files)
├── notebooks/              # Jupyter analysis notebooks
├── tests/                  # Test scripts
└── venv/                   # Python virtual environment
```

## 🚀 Quick Start

### Stock Prediction (Recommended)
```bash
# 1. Activate environment
source venv/bin/activate
pip install -r requirements.txt

# 2. Run complete pipeline
python run_stock_pipeline.py

# 3. Or run steps individually:
cd src/stocks
python preprocessing.py  # Create relative feature datasets
python training.py       # Train LSTM with company features
python testing.py        # Test on unseen data
```

### Swaption Prediction (Quantum Enhanced)
```bash
cd src/swaptions
python preprocessing.py           # Preprocess swaption data
python lstm_training.py          # Train classical LSTM
python unified_quantum_swaptions.py  # Train quantum enhancement
python testing.py               # Test full pipeline
```

## 🧠 Model Architecture

### Stock LSTM (Relative Features Only)
- **Input**: 60 days × 14 features (6 relative + 8 company features)
- **Architecture**: CNN → Company Embedding → LSTM → Residual
- **Output**: 14-day price returns prediction
- **Key**: No absolute prices, only relative patterns!

### Swaption Quantum-LSTM
- **Stage 1**: CNN-LSTM extracts 8D latent features
- **Stage 2**: Photonic quantum circuit (252D expansion)
- **Stage 3**: Classical decoder predicts 14-day swaption prices
- **Quantum Layer**: Uses Merlin photonic framework

## 📊 Features

### Stock Features (Relative - No Date Leakage)
- **Price**: Returns, volatility, momentum
- **Volume**: Volume changes, patterns
- **Technical**: MA ratios, HL spreads
- **Company**: Market cap, beta, PE ratio, sector

### Swaption Features
- **Financial**: Tenor, maturity, strike prices
- **Market**: Interest rates, volatility surfaces
- **Quantum**: 8D → 252D feature expansion

## 🎯 Results

- **Stock LSTM**: Relative features prevent overfitting
- **Swaption Quantum**: Enhanced curve fitting vs classical
- **Innovation**: First relative-feature stock prediction
- **Breakthrough**: Photonic quantum enhancement for finance

## 🔧 Requirements

```bash
pip install -r requirements.txt
```

## 📈 Model Performance

- **Relative features prevent overfitting to absolute price levels**
- **Company features enable cross-stock generalization**
- **Quantum enhancement improves complex pattern recognition**
- **Production ready with proper train/val splits**

## 🌟 Key Innovations

1. **Relative Stock Prediction**: No absolute prices, only patterns
2. **Company-Aware Features**: Sector, beta, financials integration
3. **Photonic Quantum**: Real quantum computing for finance
4. **Hybrid Architecture**: Classical + Quantum for best results

## 🗂️ File Structure

```
src/
├── stocks/
│   ├── preprocessing.py    # Create relative feature datasets
│   ├── training.py        # Train LSTM with company features
│   └── testing.py         # Test on unseen data
├── swaptions/
│   ├── preprocessing.py           # Preprocess swaption data
│   ├── lstm_training.py          # Train classical LSTM
│   ├── unified_quantum_swaptions.py  # Quantum enhancement
│   └── testing.py               # Test full pipeline
├── quantum/
│   └── core_quantum_layer.py    # Photonic quantum circuits
└── utils/
    └── robust_quantum_scaler.py # Quantum-specific preprocessing
```

## 📋 Data

- **Stocks**: 10 major stocks (AAPL, GOOGL, TSLA, etc.)
- **Features**: 14 total (6 relative + 8 company)
- **Split**: Proper temporal split, no data leakage
- **Format**: PyTorch tensors (.pt files)

## 💡 Getting Started

1. **Clone and setup**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run stock pipeline** (recommended for beginners):
   ```bash
   python run_stock_pipeline.py
   ```

3. **Check results** in `models/` directory

## 🚨 Important Notes

- **No Data Leakage**: Uses only relative patterns, not absolute prices
- **Company Features**: Beta, sector, financials for better predictions
- **Temporal Splits**: Proper train/val split with gap to prevent overfitting
- **Production Ready**: Robust preprocessing, error handling, proper scaling