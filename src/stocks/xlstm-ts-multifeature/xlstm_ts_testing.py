#!/usr/bin/env python3
"""
📈 Asset Model Testing (ETFs & Stocks)
=====================================

Tests trained asset model on unseen data.

Author: Claude & Frans
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import json

# Import model class
from xlstm_ts_training import StockXLSTMTS

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

DATA_STORAGE = CONFIG['data_storage']
ETF_SYMBOLS = CONFIG['assets']['etfs']['symbols']
STOCK_SYMBOLS = CONFIG['assets']['stocks']['symbols']

# Import asset feature functions from preprocessing
from xlstm_ts_preprocessing import get_asset_features, calculate_relative_features


def main():
    print("📈 ASSET X-LSTM-TS MODEL TESTING (7-DAY PREDICTION + PRICE HISTORY)")
    print("=" * 72)

    # Load X-LSTM model - prioritize best model (state_dict)
    try:
        # First try to load best state dict and create model architecture
        from xlstm_ts_training import StockXLSTMTS
        model = StockXLSTMTS(
            input_dim=18,       # 10 relative + 8 asset features
            hidden_dim=128,     # Updated architecture
            latent_dim=32,      # Updated architecture
            output_dim=7,       # 7-day prediction
            dropout=0.3,
            num_layers=3
        )
        model.load_state_dict(torch.load('../../../models/best_stock_xlstm.pth', map_location='cpu'))
        print("✅ X-LSTM-TS Best Model (state_dict) loaded")
        model.eval()
    except:
        # Fallback: Try to load complete model
        try:
            model = torch.load('../../../models/stock_xlstm_complete.pth', map_location='cpu')
            print("✅ X-LSTM-TS Complete Model loaded")
            model.eval()
        except:
            print("⚠️ No compatible trained model found for new 7-day + price history architecture")
            print("   You need to run training.py first to train a new model with the updated features!")
            print("   The old model is incompatible due to:")
            print("   - New feature count: 17 → 18 (added price history)")
            print("   - New output: 1-day → 7-day predictions")
            print("   - Updated architecture: hidden_dim 96→128, latent_dim 24→32")
            return

    # Load metadata including scalers
    try:
        with open('../../../data/scalers.json', 'r') as f:
            metadata = json.load(f)
        print("✅ Metadata loaded (scalers, config)")
    except:
        metadata = None
        print("⚠️ No metadata found - using raw features")

    # Test both ETFs and Stocks
    test_assets = [('etf', symbol) for symbol in ETF_SYMBOLS] + [('stock', symbol) for symbol in STOCK_SYMBOLS[:6]]  # Limit stocks for display
    results = {}

    cols = 3
    rows = (len(test_assets) + cols - 1) // cols  # Ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))

    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [axes]  # Single subplot
    elif rows == 1:
        axes = [axes]  # Single row, multiple columns
    elif cols == 1:
        axes = [[ax] for ax in axes]  # Single column, multiple rows
    else:
        pass  # Multiple rows and columns, axes is already 2D

    fig.suptitle('📈 Mixed Asset X-LSTM-TS Predictions (ETFs & Stocks)', fontsize=16)

    for idx, (asset_type, symbol) in enumerate(test_assets):
        row, col = idx // 3, idx % 3

        # Handle axis indexing based on subplot configuration
        if rows == 1 and cols == 1:
            ax = axes[0]
        elif rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row][0]
        else:
            ax = axes[row, col]

        print(f"\n📊 Testing {symbol}...")

        # Get recent data (unseen by model) + extra days for actual future comparison
        end_date = datetime.now() - timedelta(days=15)  # Go back 15 days to have future data
        start_date = end_date - timedelta(days=150)  # More historical data

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if len(data) < 80:  # Need more data now
                continue

            # Calculate features exactly like preprocessing
            relative_features = calculate_relative_features(data)
            asset_features = get_asset_features(symbol, asset_type)

            # Asset vector (properly normalized like preprocessing)
            if asset_type == 'etf':
                asset_vec = np.array([
                    asset_features['nav_normalized'], asset_features['beta_normalized'],
                    asset_features['volume_normalized'], asset_features['expense_ratio_normalized'],
                    asset_features['large_cap_etf'], asset_features['growth_etf'],
                    asset_features['tech_heavy_etf'], asset_features['high_volatility_etf']
                ])
            else:  # stock
                asset_vec = np.array([
                    asset_features['market_cap_normalized'], asset_features['beta_normalized'],
                    asset_features['pe_ratio_normalized'], asset_features['profit_margin_normalized'],
                    asset_features['sector_tech'], asset_features['sector_consumer'],
                    asset_features['sector_communication'], asset_features['sector_other']
                ])

            # Use data up to 20 days ago as input (so we have actual future data for comparison)
            # Since we already went back 15 days in the data fetch, we can use recent data
            prediction_start_idx = -20  # Use data ending 20 days ago to have actual future data
            actual_future_available = len(data) + prediction_start_idx >= 14

            if not actual_future_available or len(data) + prediction_start_idx - 60 < 0:
                prediction_start_idx = -60  # Fall back to older data
                actual_future_available = False
                print(f"   ⚠️ {symbol}: No actual future data available, using older prediction point")

            # Ensure we have enough data for 60-day sequence
            if len(data) < 60:
                print(f"   ❌ {symbol}: Not enough data ({len(data)} days)")
                continue

            # Get exactly 60 days ending at prediction_start_idx
            start_idx = max(0, len(data) + prediction_start_idx - 60)
            end_idx = len(data) + prediction_start_idx

            # Get VIX data for sentiment features
            vix_data_test = None
            try:
                vix_ticker = yf.Ticker("^VIX")
                vix_data_test = vix_ticker.history(start=start_date, end=end_date)['Close']
                # Align with asset data
                vix_data_test = vix_data_test.reindex(data.index, method='ffill').fillna(method='bfill')
            except:
                print(f"   ⚠️ {symbol}: VIX data unavailable for testing")

            # Calculate all relative features including sentiment (with denoising)
            all_relative_features = calculate_relative_features(data, vix_data_test, use_denoising=True)

            # Prepare input sequence (exactly 60 days) with sentiment + price history features
            input_relative = np.stack([
                all_relative_features['returns'][start_idx:end_idx],
                all_relative_features['volume_change'][start_idx:end_idx],
                all_relative_features['volatility'][start_idx:end_idx],
                all_relative_features['hl_spread_pct'][start_idx:end_idx],
                all_relative_features['ma_ratio'][start_idx:end_idx],
                all_relative_features['momentum'][start_idx:end_idx],
                all_relative_features['vix_normalized'][start_idx:end_idx],
                all_relative_features['vix_change'][start_idx:end_idx],
                all_relative_features['fear_greed_synthetic'][start_idx:end_idx],
                all_relative_features['price_history_norm'][start_idx:end_idx]
            ], axis=1)  # Shape: (60, 10)

            # Ensure we have exactly 60 timesteps
            if input_relative.shape[0] != 60:
                print(f"   ❌ {symbol}: Input shape mismatch: {input_relative.shape[0]} != 60")
                continue

            # Apply same scaling as in preprocessing
            if metadata:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.min_ = np.array(metadata['relative_scaler_min'])
                scaler.scale_ = np.array(metadata['relative_scaler_scale'])
                input_relative_scaled = scaler.transform(input_relative)
            else:
                input_relative_scaled = input_relative

            # Asset features (8) - already normalized
            asset_seq = np.tile(asset_vec, (60, 1))  # (60, 8)

            # Combine (18 total features: 10 relative + 8 asset)
            input_features = np.concatenate([input_relative_scaled, asset_seq], axis=1)  # (60, 18)

            # Predict percentage changes for next 7 days
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)  # (1, 60, 18)
            with torch.no_grad():
                prediction = model(input_tensor)
                pred_percentages = prediction[0].numpy()  # 7-day percentage predictions

            # Get the price at prediction start point and actual future prices
            prediction_price = data['Close'].iloc[end_idx - 1]  # Last price in our input sequence

            # Convert percentage predictions to future prices
            pred_prices = [prediction_price]
            for pct_change in pred_percentages:
                new_price = pred_prices[-1] * (1 + pct_change / 100)
                pred_prices.append(new_price)

            # Get actual future 7-day prices if available
            print(f"   🔍 Debug {symbol}: end_idx={end_idx}, data_len={len(data)}, prediction_start_idx={prediction_start_idx}")

            if actual_future_available and end_idx < len(data):
                future_end = min(end_idx + 7, len(data))
                actual_future_prices = data['Close'].iloc[end_idx:future_end].values

                print(f"   🔍 Debug {symbol}: Got {len(actual_future_prices)} actual future prices")

                # Pad if we don't have full 7 days
                if len(actual_future_prices) < 7:
                    print(f"   ⚠️ {symbol}: Only {len(actual_future_prices)}/7 future days available")
                    # Extend with last known price for missing days
                    actual_future_prices = np.concatenate([actual_future_prices,
                                                         [actual_future_prices[-1]] * (7 - len(actual_future_prices))])
            else:
                actual_future_prices = None
                actual_future_available = False
                print(f"   ⚠️ {symbol}: No actual future data available")

            # Plot
            hist_days = range(-59, 1)
            future_days = range(1, 8)  # 7-day horizon

            # Historical prices (60 days leading up to prediction point)
            hist_prices = data['Close'].iloc[start_idx:end_idx].values
            ax.plot(hist_days, hist_prices, 'o-', color='gray', alpha=0.7, label='Historical', markersize=3)

            # 7-day predictions
            ax.plot(future_days, pred_prices[1:], 's-', color='blue', label='X-LSTM 7-Day', markersize=4)

            # Actual future prices (if available)
            if actual_future_available:
                ax.plot(future_days, actual_future_prices, 'o-', color='green', label='Actual Future', markersize=4)

                # Calculate prediction accuracy (MAPE over 7 days)
                prediction_error = np.mean(np.abs((np.array(pred_prices[1:]) - actual_future_prices) / actual_future_prices * 100))
                accuracy_text = f'MAPE: {prediction_error:.1f}%'
            else:
                accuracy_text = 'No future data'

            # Styling
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax.set_title(f'{symbol}\nCurrent: ${prediction_price:.2f}\n{accuracy_text}')
            ax.set_xlabel('Days (0 = Today, 1-7 = Next Week)')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            results[symbol] = {
                'prediction_price': prediction_price,
                'predicted_7day_prices': pred_prices[1:],
                'predicted_percentages': pred_percentages,
                'actual_future_prices': actual_future_prices if actual_future_available else None,
                'asset_beta': asset_features['beta_normalized'],
                'prediction_accuracy': prediction_error if actual_future_available else None
            }

            # Calculate predicted 7-day change
            pred_7day_change = (pred_prices[-1] - prediction_price) / prediction_price * 100

            if actual_future_available:
                actual_7day_change = (actual_future_prices[-1] - prediction_price) / prediction_price * 100
                print(f"   ✅ {symbol}: ${prediction_price:.2f} → Pred: ${pred_prices[-1]:.2f} ({pred_7day_change:+.1f}%) | Actual: ${actual_future_prices[-1]:.2f} ({actual_7day_change:+.1f}%) | MAPE: {prediction_error:.1f}%")
            else:
                print(f"   ✅ {symbol}: ${prediction_price:.2f} → ${pred_prices[-1]:.2f} ({pred_7day_change:+.1f}%) | Beta: {asset_features['beta_normalized']:+.2f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    plt.tight_layout()
    plt.savefig('../../../models/stock_xlstm_predictions.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Summary
    print(f"\n📈 PREDICTION vs ACTUAL COMPARISON (7-Day Horizon):")
    print("=" * 85)
    print(f"{'Stock':<6} {'Current $':<10} {'Pred $':<8} {'Actual $':<9} {'Pred %':<8} {'Actual %':<9} {'MAPE':<6} {'Beta':<8}")
    print("-" * 85)

    accurate_predictions = []
    for symbol, result in results.items():
        start = result['prediction_price']
        pred = result['predicted_7day_prices'][-1]  # 7-day prediction
        pred_change = (pred - start) / start * 100
        beta = result['asset_beta']

        if result['actual_future_prices'] is not None:
            actual = result['actual_future_prices'][-1]  # 7-day actual
            actual_change = (actual - start) / start * 100
            mape = result['prediction_accuracy']
            accurate_predictions.append(mape)

            print(f"{symbol:<6} ${start:<9.2f} ${pred:<7.2f} ${actual:<8.2f} {pred_change:+6.1f}% {actual_change:+7.1f}% {mape:5.1f}% {beta:+6.2f}")
        else:
            print(f"{symbol:<6} ${start:<9.2f} ${pred:<7.2f} {'N/A':<8} {pred_change:+6.1f}% {'N/A':<7} {'N/A':<4} {beta:+6.2f}")

    print("-" * 85)

    if accurate_predictions:
        avg_mape = np.mean(accurate_predictions)
        print(f"\n📊 MODEL PERFORMANCE:")
        print(f"   📈 Average MAPE (Mean Absolute Percentage Error): {avg_mape:.1f}%")
        print(f"   🎯 Best prediction: {min(accurate_predictions):.1f}% MAPE")
        print(f"   📉 Worst prediction: {max(accurate_predictions):.1f}% MAPE")
        print(f"   📊 Stocks with actual data: {len(accurate_predictions)}/{len(results)}")

        if avg_mape < 10:
            print(f"   ✅ Excellent performance (< 10% MAPE)")
        elif avg_mape < 20:
            print(f"   👍 Good performance (< 20% MAPE)")
        else:
            print(f"   📈 Room for improvement (> 20% MAPE)")
    else:
        print(f"   ⚠️ No actual future data available for comparison")

    all_pred_pcts = [p for r in results.values() for p in r['predicted_percentages']]
    print(f"\n📊 Model Output Statistics:")
    print(f"   💡 Daily percentage changes range: {np.min(all_pred_pcts):.2f}% to {np.max(all_pred_pcts):.2f}%")
    print(f"   📊 Mean daily prediction: {np.mean(all_pred_pcts):.2f}% ± {np.std(all_pred_pcts):.2f}%")

    print(f"\n💾 Comparison chart saved to: ../../../models/stock_xlstm_predictions.png")

if __name__ == "__main__":
    main()