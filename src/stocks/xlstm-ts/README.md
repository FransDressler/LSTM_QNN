# xLSTM-TS: Extended LSTM for Time Series Stock Prediction

Implementation of the xLSTM-TS model as described in "An Evaluation of Deep Learning Models for Stock Market Trend Prediction" ([ArXiv paper](https://arxiv.org/html/2408.12408v1)).

## Features

- **Wavelet Denoising**: Daubechies db4 wavelet for signal preprocessing
- **xLSTM Architecture**: Exponential gating with sLSTM and mLSTM cells
- **Trend Prediction**: Binary classification for up/down market movements
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, MAE, RMSE, MASE
- **Multi-step Forecasting**: Predict multiple time steps ahead
- **Backtesting**: Historical performance evaluation

## Model Architecture

### xLSTM Components
- **sLSTM (scalar LSTM)**: Enhanced with exponential gating
- **mLSTM (matrix LSTM)**: Matrix memory with attention mechanisms
- **Residual Connections**: Skip connections for better gradient flow
- **Layer Normalization**: Improved training stability

### Key Parameters
- **Sequence Length**: 150 days (as per paper)
- **Embedding Dimension**: 64
- **Hidden Size**: 64
- **Batch Size**: 16
- **Input**: Normalized price sequences

## Usage

### Training a Model
```python
from xlstm_ts.main import train_model

# Train on S&P 500
model_path, metrics = train_model("^GSPC", epochs=100)

# Train on specific stock
model_path, metrics = train_model("AAPL", epochs=100)
```

### Making Predictions
```python
from xlstm_ts.predictor import xLSTMPredictor

predictor = xLSTMPredictor(model_path)

# Single prediction
prediction = predictor.predict_trend("AAPL")
print(f"Trend: {prediction['predicted_trend']}")
print(f"Confidence: {prediction['confidence']:.2%}")

# Multi-step prediction
multi_step = predictor.predict_multiple_steps("AAPL", steps=5)

# Get investment recommendation
recommendation = predictor.get_recommendation("AAPL")
```

### Command Line Usage
```bash
# Train model
python -m xlstm_ts.main train AAPL 100

# Make predictions
python -m xlstm_ts.main predict model.pth AAPL

# Compare multiple stocks
python -m xlstm_ts.main compare
```

## Performance

Based on the original paper results:
- **S&P 500 Daily**: 71.28% accuracy
- **EWZ Daily**: 72.87% accuracy
- **Superior Performance**: Outperforms TCN, N-BEATS, TFT, N-HiTS, TiDE

## File Structure

```
xlstm-ts/
├── __init__.py          # Package initialization
├── model.py             # xLSTM-TS model architecture
├── preprocessing.py     # Wavelet denoising and data loading
├── training.py          # Training pipeline and optimization
├── predictor.py         # Prediction and inference
├── metrics.py           # Evaluation metrics
├── main.py              # Main execution script
└── README.md           # This file
```

## Dependencies

- PyTorch >= 2.0.0
- PyWavelets >= 1.4.0 (for wavelet denoising)
- yfinance >= 0.2.0 (for stock data)
- scikit-learn >= 1.2.0
- numpy, pandas, matplotlib, seaborn

## Technical Details

### Preprocessing Pipeline
1. **Data Collection**: Yahoo Finance API via yfinance
2. **Wavelet Denoising**: db4 wavelet with soft thresholding
3. **Trend Calculation**: Binary labels from price differences
4. **Normalization**: MinMax scaling to [0,1]
5. **Sequence Creation**: 150-day sliding windows

### Model Training
- **Loss Function**: Cross-entropy for trend classification
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Early Stopping**: Patience of 20 epochs
- **Gradient Clipping**: Max norm of 1.0

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Trend-Specific**: Up/Down trend sensitivity and specificity
- **Regression**: MAE, RMSE, MASE (for price analysis)
- **Additional**: Confusion matrix, classification report

## Limitations

- Model performance varies across different market conditions
- Predictions should not be used for actual trading without additional validation
- Highly volatile markets may pose challenges
- Past performance doesn't guarantee future results

## Future Enhancements

- Integration with economic indicators
- Multi-asset portfolio optimization
- Real-time prediction pipeline
- Advanced risk management features