import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from .model import xLSTMTS
from .preprocessing import StockDataLoader
import yfinance as yf


class xLSTMPredictor:
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.data_loader = StockDataLoader(sequence_length=150)
        self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = xLSTMTS()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict_trend(self, symbol: str, period: str = "1y") -> Dict:
        raw_data = self.data_loader.load_data(symbol, period)
        trends, denoised_prices = self.data_loader.preprocess_price_data(raw_data)

        normalized_prices = self.data_loader.scaler.fit_transform(
            denoised_prices.reshape(-1, 1)
        ).flatten()

        if len(normalized_prices) < 150:
            raise ValueError("Not enough data points. Need at least 150 days of data.")

        last_sequence = normalized_prices[-150:].reshape(1, 150, 1)
        last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)

        with torch.no_grad():
            output = self.model(last_sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()

        confidence = probabilities[0, predicted_class].item()
        trend_direction = "UP" if predicted_class == 1 else "DOWN"

        return {
            'symbol': symbol,
            'predicted_trend': trend_direction,
            'confidence': confidence,
            'probabilities': {
                'down': probabilities[0, 0].item(),
                'up': probabilities[0, 1].item()
            },
            'last_price': raw_data['Close'].iloc[-1],
            'denoised_last_price': denoised_prices[-1]
        }

    def predict_multiple_steps(self, symbol: str, steps: int = 5, period: str = "1y") -> List[Dict]:
        predictions = []

        raw_data = self.data_loader.load_data(symbol, period)
        trends, denoised_prices = self.data_loader.preprocess_price_data(raw_data)

        normalized_prices = self.data_loader.scaler.fit_transform(
            denoised_prices.reshape(-1, 1)
        ).flatten()

        current_sequence = normalized_prices[-150:]

        for step in range(steps):
            sequence_tensor = torch.FloatTensor(current_sequence.reshape(1, 150, 1)).to(self.device)

            with torch.no_grad():
                output = self.model(sequence_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()

            confidence = probabilities[0, predicted_class].item()
            trend_direction = "UP" if predicted_class == 1 else "DOWN"

            predictions.append({
                'step': step + 1,
                'predicted_trend': trend_direction,
                'confidence': confidence,
                'probabilities': {
                    'down': probabilities[0, 0].item(),
                    'up': probabilities[0, 1].item()
                }
            })

            synthetic_next_value = current_sequence[-1] + (0.001 if predicted_class == 1 else -0.001)
            current_sequence = np.append(current_sequence[1:], synthetic_next_value)

        return predictions

    def backtest_predictions(self, symbol: str, period: str = "2y", test_period: str = "3m") -> Dict:
        ticker = yf.Ticker(symbol)

        train_end_date = pd.Timestamp.now() - pd.DateOffset(months=3)
        train_data = ticker.history(period=period, end=train_end_date)
        test_data = ticker.history(period=test_period)

        trends, denoised_prices = self.data_loader.preprocess_price_data(train_data)
        normalized_prices = self.data_loader.scaler.fit_transform(
            denoised_prices.reshape(-1, 1)
        ).flatten()

        predictions = []
        actual_trends = []

        test_trends, _ = self.data_loader.preprocess_price_data(test_data)

        sequence = normalized_prices[-150:]

        for i in range(min(len(test_trends), 30)):
            sequence_tensor = torch.FloatTensor(sequence.reshape(1, 150, 1)).to(self.device)

            with torch.no_grad():
                output = self.model(sequence_tensor)
                predicted_class = torch.argmax(output, dim=1).item()

            predictions.append(predicted_class)
            actual_trends.append(test_trends[i])

            if i < len(test_data) - 1:
                actual_next_price = test_data['Close'].iloc[i + 1]
                normalized_next = self.data_loader.scaler.transform([[actual_next_price]])[0, 0]
                sequence = np.append(sequence[1:], normalized_next)

        correct_predictions = sum(p == a for p, a in zip(predictions, actual_trends))
        accuracy = correct_predictions / len(predictions) if predictions else 0

        return {
            'accuracy': accuracy,
            'total_predictions': len(predictions),
            'correct_predictions': correct_predictions,
            'predictions': predictions,
            'actual_trends': actual_trends,
            'dates': test_data.index[:len(predictions)].tolist()
        }

    def visualize_prediction(self, symbol: str, period: str = "6m", save_path: Optional[str] = None):
        raw_data = self.data_loader.load_data(symbol, period)
        trends, denoised_prices = self.data_loader.preprocess_price_data(raw_data)

        prediction_result = self.predict_trend(symbol, period)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(raw_data.index, raw_data['Close'], label='Original Price', alpha=0.7)
        ax1.plot(raw_data.index, denoised_prices, label='Denoised Price', linewidth=2)
        ax1.set_title(f'{symbol} - Price Data and Denoising')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)

        trend_colors = ['red' if t == 0 else 'green' for t in trends]
        ax2.scatter(raw_data.index[1:], trends, c=trend_colors, alpha=0.6, s=10)
        ax2.set_title('Historical Trends (Red=Down, Green=Up)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Trend Direction')
        ax2.grid(True)

        fig.suptitle(f'Prediction: {prediction_result["predicted_trend"]} '
                    f'(Confidence: {prediction_result["confidence"]:.2%})', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

        return prediction_result

    def get_recommendation(self, symbol: str, period: str = "1y") -> Dict:
        prediction = self.predict_trend(symbol, period)
        multi_step = self.predict_multiple_steps(symbol, steps=5, period=period)

        avg_confidence = np.mean([p['confidence'] for p in multi_step])
        up_predictions = sum(1 for p in multi_step if p['predicted_trend'] == 'UP')

        if prediction['confidence'] > 0.7 and avg_confidence > 0.6:
            if prediction['predicted_trend'] == 'UP' and up_predictions >= 3:
                recommendation = "STRONG BUY"
            elif prediction['predicted_trend'] == 'UP':
                recommendation = "BUY"
            elif prediction['predicted_trend'] == 'DOWN' and up_predictions <= 2:
                recommendation = "STRONG SELL"
            else:
                recommendation = "SELL"
        else:
            recommendation = "HOLD"

        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'current_prediction': prediction,
            'multi_step_analysis': {
                'avg_confidence': avg_confidence,
                'up_predictions': up_predictions,
                'total_steps': len(multi_step),
                'predictions': multi_step
            }
        }