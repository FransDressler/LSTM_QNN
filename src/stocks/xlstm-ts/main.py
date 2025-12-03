#!/usr/bin/env python3
"""
xLSTM-TS Stock Prediction Main Script
Based on the paper: "An Evaluation of Deep Learning Models for Stock Market Trend Prediction"
"""

import torch
import os
import sys
from datetime import datetime

from .model import xLSTMTS
from .training import xLSTMTrainer
from .predictor import xLSTMPredictor
from .metrics import ModelMetrics


def train_model(symbol: str = "^GSPC", epochs: int = 100):
    """
    Train xLSTM-TS model on stock data
    """
    print(f"Training xLSTM-TS model on {symbol}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    model = xLSTMTS(
        input_size=1,
        embedding_dim=64,
        num_layers=2,
        hidden_size=64,
        num_classes=2,
        dropout=0.1
    )

    trainer = xLSTMTrainer(model)

    train_loader, test_loader = trainer.prepare_data(
        symbol=symbol,
        batch_size=16,
        period="2y",
        test_size=0.2
    )

    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        patience=20
    )

    print("\nEvaluating model...")
    test_metrics = trainer.evaluate(test_loader)

    metrics_calculator = ModelMetrics()
    metrics_calculator.print_detailed_report(
        test_metrics['y_true'],
        test_metrics['y_pred'],
        "xLSTM-TS"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"xlstm_ts_model_{symbol.replace('^', '')}_{timestamp}.pth"

    trainer.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

    trainer.plot_training_history(f"training_history_{symbol.replace('^', '')}_{timestamp}.png")

    return model_path, test_metrics


def predict_stock(model_path: str, symbol: str):
    """
    Make predictions using trained model
    """
    print(f"\nMaking predictions for {symbol}...")

    predictor = xLSTMPredictor(model_path)

    prediction = predictor.predict_trend(symbol, period="1y")
    print(f"Single prediction: {prediction}")

    multi_step = predictor.predict_multiple_steps(symbol, steps=5)
    print(f"\nMulti-step predictions:")
    for step in multi_step:
        print(f"  Step {step['step']}: {step['predicted_trend']} (confidence: {step['confidence']:.2%})")

    recommendation = predictor.get_recommendation(symbol)
    print(f"\nRecommendation: {recommendation['recommendation']}")

    predictor.visualize_prediction(symbol, period="6m")

    return prediction, multi_step, recommendation


def compare_with_sp500():
    """
    Compare different stocks with S&P 500
    """
    symbols = ["^GSPC", "AAPL", "MSFT", "GOOGL", "TSLA"]

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Training and evaluating {symbol}")
        print(f"{'='*50}")

        try:
            model_path, metrics = train_model(symbol, epochs=50)
            predict_stock(model_path, symbol)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue


def main():
    """
    Main execution function
    """
    print("xLSTM-TS Stock Prediction System")
    print("=" * 40)

    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "^GSPC"
            epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
            model_path, _ = train_model(symbol, epochs)

        elif sys.argv[1] == "predict":
            model_path = sys.argv[2]
            symbol = sys.argv[3] if len(sys.argv) > 3 else "^GSPC"
            predict_stock(model_path, symbol)

        elif sys.argv[1] == "compare":
            compare_with_sp500()

        else:
            print("Usage:")
            print("  python main.py train [SYMBOL] [EPOCHS]")
            print("  python main.py predict MODEL_PATH [SYMBOL]")
            print("  python main.py compare")
    else:
        # Default: train on S&P 500
        print("Default mode: Training on S&P 500")
        model_path, _ = train_model("^GSPC", epochs=100)
        predict_stock(model_path, "^GSPC")


if __name__ == "__main__":
    main()