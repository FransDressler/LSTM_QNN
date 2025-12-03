import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from typing import Tuple, List, Optional


class WaveletDenoiser:
    def __init__(self, wavelet='db4', threshold_mode='soft'):
        self.wavelet = wavelet
        self.threshold_mode = threshold_mode

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        coeffs = pywt.wavedec(signal, self.wavelet, level=4)

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(detail, threshold, mode=self.threshold_mode)
                           for detail in coeffs_thresh[1:]]

        return pywt.waverec(coeffs_thresh, self.wavelet)


class StockDataLoader:
    def __init__(self, sequence_length: int = 150):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.denoiser = WaveletDenoiser()

    def load_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data

    def preprocess_price_data(self, data: pd.DataFrame) -> np.ndarray:
        close_prices = data['Close'].values

        denoised_prices = self.denoiser.denoise(close_prices)

        price_changes = np.diff(denoised_prices)
        trends = np.where(price_changes > 0, 1, 0)

        return trends, denoised_prices

    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(targets[i-1])

        return np.array(X), np.array(y)

    def prepare_dataset(self, symbol: str, period: str = "2y",
                       interval: str = "1d") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_data = self.load_data(symbol, period, interval)

        trends, denoised_prices = self.preprocess_price_data(raw_data)

        normalized_prices = self.scaler.fit_transform(denoised_prices.reshape(-1, 1)).flatten()

        X, y = self.create_sequences(normalized_prices[:-1], trends)

        return X, y, denoised_prices


def train_test_split_temporal(X: np.ndarray, y: np.ndarray,
                             test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test