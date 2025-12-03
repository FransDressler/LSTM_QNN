#!/usr/bin/env python3
"""
📊 Stock Data Preprocessing (Relative Features Only)
==================================================

Creates relative-feature training data:
- No absolute prices or dates
- Only relative patterns and company features
- Prevents date leakage and overfitting

Author: Claude & Frans
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import MinMaxScaler
import requests
import pywt
import warnings
warnings.filterwarnings('ignore')

# Load configuration from JSON
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Extract configuration values
MODEL_CONFIG = CONFIG['model_config']
SEQUENCE_LENGTH = MODEL_CONFIG['sequence_length']
PREDICTION_LENGTH = MODEL_CONFIG['prediction_length']
TRAIN_SPLIT = MODEL_CONFIG['train_split']
MAX_YEARS = MODEL_CONFIG['max_years']

# Asset configuration
ETF_SYMBOLS = CONFIG['assets']['etfs']['symbols']
STOCK_SYMBOLS = CONFIG['assets']['stocks']['symbols']
ETF_HARDCODED = CONFIG['assets']['etfs']['hardcoded_features']
ETF_FEATURES = CONFIG['assets']['etfs']['features']
STOCK_FEATURES = CONFIG['assets']['stocks']['features']

# Data storage paths
DATA_STORAGE = CONFIG['data_storage']

def get_asset_features(symbol, asset_type):
    """Get asset-specific features based on type (ETF or Stock)"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if asset_type == 'etf':
            return _get_etf_features(symbol, info)
        elif asset_type == 'stock':
            return _get_stock_features(symbol, info)
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

    except Exception as e:
        # Return default values based on asset type
        if asset_type == 'etf':
            return {key: 0.0 for key in ETF_FEATURES.keys()}
        else:
            return {key: 0.0 for key in STOCK_FEATURES.keys()}

def _get_etf_features(symbol, info):
    """ETF-specific feature extraction"""
    # ETF-spezifische Metadaten
    nav = info.get('navPrice', 100)
    beta = info.get('beta', 1.0)
    volume = info.get('volume', 1e6)
    expense_ratio = info.get('totalExpenseRatio', 0.001)

    # None checks
    beta = beta if beta is not None else 1.0
    expense_ratio = expense_ratio if expense_ratio is not None else 0.001

    # Hardcoded ETF characteristics from config
    etf_type = ETF_HARDCODED.get(symbol, {'large_cap': 0.0, 'growth': 0.0, 'tech_heavy': 0.0, 'volatility': 0.0})

    return {
        'nav_normalized': float(np.clip(nav / 200 - 1, -1, 1)),
        'beta_normalized': float(np.clip((beta - 1.0) / 1.0, -1, 1)),
        'volume_normalized': float(np.clip(np.log10(volume / 1e6) / 2, -1, 1)),
        'expense_ratio_normalized': float(np.clip((expense_ratio - 0.001) / 0.002, -1, 1)),
        'large_cap_etf': etf_type['large_cap'],
        'growth_etf': etf_type['growth'],
        'tech_heavy_etf': etf_type['tech_heavy'],
        'high_volatility_etf': etf_type['volatility'],
    }

def _get_stock_features(symbol, info):
    """Stock-specific feature extraction"""
    # Stock-spezifische Metadaten
    market_cap = info.get('marketCap', 1e12)
    beta = info.get('beta', 1.0)
    pe = info.get('forwardPE', 25)
    margin = info.get('profitMargins', 0.15)

    # None checks
    pe = pe if pe is not None else 25
    margin = margin if margin is not None else 0.15

    return {
        'market_cap_normalized': float(np.clip(market_cap / 2.5e12 - 1, -1, 1)),
        'beta_normalized': float(np.clip((beta - 1.0) / 1.5, -1, 1)),
        'pe_ratio_normalized': float(np.clip((pe - 25) / 40, -1, 1)),
        'profit_margin_normalized': float(np.clip((margin - 0.15) / 0.3, -1, 1)),
        'sector_tech': 1.0 if info.get('sector') == 'Technology' else -0.3,
        'sector_consumer': 1.0 if info.get('sector') == 'Consumer Cyclical' else -0.3,
        'sector_communication': 1.0 if info.get('sector') == 'Communication Services' else -0.3,
        'sector_other': 1.0 if info.get('sector') not in ['Technology', 'Consumer Cyclical', 'Communication Services'] else -0.3,
    }

def wavelet_denoise(data, wavelet='db4', levels=3, threshold_mode='soft'):
    """
    Wavelet denoising for financial time series

    Args:
        data: 1D array or pandas Series
        wavelet: Wavelet type (db4 good for financial data)
        levels: Number of decomposition levels
        threshold_mode: 'soft' or 'hard' thresholding
    """
    if len(data) < 2**levels:
        # Not enough data for wavelet decomposition
        return data

    # Convert to numpy if pandas Series
    if hasattr(data, 'values'):
        is_series = True
        index = data.index
        data_vals = data.values
    else:
        is_series = False
        data_vals = np.array(data)

    # Handle NaN values
    nan_mask = np.isnan(data_vals)
    if nan_mask.all():
        return data

    # Interpolate NaN values for wavelet processing
    data_clean = data_vals.copy()
    if nan_mask.any():
        data_clean = pd.Series(data_vals).interpolate().fillna(method='bfill').fillna(method='ffill').values

    try:
        # Wavelet decomposition
        coeffs = pywt.wavedec(data_clean, wavelet, level=levels, mode='symmetric')

        # Estimate noise level using median absolute deviation of finest detail
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(data_clean)))

        # Apply soft thresholding to detail coefficients
        coeffs_thresh = coeffs.copy()
        for i in range(1, len(coeffs)):  # Skip approximation coefficients
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, threshold_mode)

        # Reconstruct signal
        denoised = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')

        # Ensure same length as input
        denoised = denoised[:len(data_vals)]

        # Restore original NaN positions
        denoised[nan_mask] = np.nan

        # Return in original format
        if is_series:
            return pd.Series(denoised, index=index)
        else:
            return denoised

    except Exception as e:
        print(f"   ⚠️ Wavelet denoising failed: {e}, returning original data")
        return data

def get_vix_data(start_date, end_date):
    """Fetch VIX (CBOE Volatility Index) data"""
    try:
        vix_ticker = yf.Ticker("^VIX")
        vix_data = vix_ticker.history(start=start_date, end=end_date)
        return vix_data['Close'].fillna(method='ffill')
    except Exception as e:
        print(f"   ⚠️ VIX data fetch failed: {e}")
        return None

def get_fear_greed_index(days=100):
    """Fetch Fear & Greed Index from CNN API (simplified version)"""
    try:
        # Alternative: Create synthetic fear/greed based on VIX and market conditions
        # This is more reliable than depending on external APIs
        print(f"   💡 Using VIX-based fear/greed calculation (more reliable)")
        return None  # Will be calculated from VIX below
    except Exception as e:
        print(f"   ⚠️ Fear & Greed Index fetch failed: {e}")
        return None

def calculate_sentiment_features(vix_data, market_returns, use_denoising=True):
    """Calculate market sentiment features from VIX and market data with denoising"""
    if vix_data is None or len(vix_data) == 0:
        return {
            'vix_normalized': np.zeros(len(market_returns)),
            'vix_change': np.zeros(len(market_returns)),
            'fear_greed_synthetic': np.zeros(len(market_returns))
        }

    if use_denoising:
        print(f"   🧠 Applying wavelet denoising to sentiment data...")
        # Strategy: VIX normalized - weich denoisen für Regime-Erkennung
        vix_denoised = wavelet_denoise(vix_data, levels=2, threshold_mode='soft')
        vix_normalized = np.clip((vix_denoised - 20) / 25, -1, 1)

        # VIX change - both versions as you suggested
        vix_change_raw = vix_data.pct_change().fillna(0)  # "oh shit, Marktpanik heute"
        # Note: We'll just use raw for now, could add both versions later

        vix_change = vix_change_raw  # Keep the shock signal

        # Fear & Greed - denoise for smooth regime detection
        market_momentum = pd.Series(market_returns).rolling(5).mean().fillna(0)
        vix_inverted = (-vix_normalized + 0.5)  # Invert denoised VIX

        fear_greed_raw = (vix_inverted * 0.6 + market_momentum * 0.4)
        fear_greed_synthetic = wavelet_denoise(fear_greed_raw, levels=2, threshold_mode='soft')
        fear_greed_synthetic = np.clip(fear_greed_synthetic, -1, 1)
    else:
        # Original raw sentiment features
        vix_normalized = np.clip((vix_data - 20) / 25, -1, 1)
        vix_change = vix_data.pct_change().fillna(0)

        market_momentum = pd.Series(market_returns).rolling(5).mean().fillna(0)
        vix_inverted = (-vix_normalized + 0.5)

        fear_greed_synthetic = (vix_inverted * 0.6 + market_momentum * 0.4)
        fear_greed_synthetic = np.clip(fear_greed_synthetic, -1, 1)

    return {
        'vix_normalized': vix_normalized.values if hasattr(vix_normalized, 'values') else vix_normalized,
        'vix_change': vix_change.values if hasattr(vix_change, 'values') else vix_change,
        'fear_greed_synthetic': fear_greed_synthetic.values if hasattr(fear_greed_synthetic, 'values') else fear_greed_synthetic
    }

def calculate_relative_features(data, vix_data=None, use_denoising=True):
    """Calculate relative features including sentiment indicators with optional denoising"""

    # Strategy 1: Denoise prices first, then calculate features
    if use_denoising:
        print(f"   🧠 Applying wavelet denoising to price data...")
        close_denoised = wavelet_denoise(data['Close'], levels=3)
        high_denoised = wavelet_denoise(data['High'], levels=3)
        low_denoised = wavelet_denoise(data['Low'], levels=3)
        volume_denoised = wavelet_denoise(data['Volume'], levels=3)

        # Calculate features from denoised prices
        close_returns = close_denoised.pct_change().fillna(0)
        volume_change = volume_denoised.pct_change().fillna(0)
        volatility = close_returns.rolling(window=5, min_periods=1).std().fillna(0)
        hl_spread_pct = ((high_denoised - low_denoised) / close_denoised).fillna(0)

        ma_5 = close_denoised.rolling(5).mean()
        ma_20 = close_denoised.rolling(20).mean()
        ma_ratio = (ma_5 / ma_20 - 1).fillna(0)

        momentum = close_returns.rolling(window=10).mean().fillna(0)
    else:
        # Original raw features
        close_returns = data['Close'].pct_change().fillna(0)
        volume_change = data['Volume'].pct_change().fillna(0)
        volatility = close_returns.rolling(window=5, min_periods=1).std().fillna(0)
        hl_spread_pct = ((data['High'] - data['Low']) / data['Close']).fillna(0)

        ma_5 = data['Close'].rolling(5).mean()
        ma_20 = data['Close'].rolling(20).mean()
        ma_ratio = (ma_5 / ma_20 - 1).fillna(0)

        momentum = close_returns.rolling(window=10).mean().fillna(0)

    # Calculate sentiment features with denoising
    sentiment_features = calculate_sentiment_features(vix_data, close_returns, use_denoising=use_denoising)

    return {
        'returns': close_returns.values,
        'volume_change': volume_change.values,
        'volatility': volatility.values,
        'hl_spread_pct': hl_spread_pct.values,
        'ma_ratio': ma_ratio.values,
        'momentum': momentum.values,
        'vix_normalized': sentiment_features['vix_normalized'],
        'vix_change': sentiment_features['vix_change'],
        'fear_greed_synthetic': sentiment_features['fear_greed_synthetic']
    }

def main():
    print("📊 ASSET PREPROCESSING - TEMPORAL SPLIT WITH SLIDING WINDOWS + SENTIMENT")
    print("=" * 70)
    print(f"Processing ETFs: {ETF_SYMBOLS}")
    print(f"Processing Stocks: {STOCK_SYMBOLS}")

    # Create data directories
    import os
    os.makedirs(DATA_STORAGE['preprocessed_data_dir'], exist_ok=True)
    os.makedirs(DATA_STORAGE['plots_dir'], exist_ok=True)

    # Fetch VIX data once for all assets (market-wide sentiment)
    print(f"\n🧠 Fetching VIX sentiment data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=MAX_YEARS * 365)
    vix_data = get_vix_data(start_date, end_date)

    if vix_data is not None:
        print(f"   ✅ VIX data fetched: {len(vix_data)} days")
    else:
        print(f"   ⚠️ VIX data unavailable - using zero sentiment features")

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []

    # Process both ETFs and Stocks
    all_symbols = [('etf', symbol) for symbol in ETF_SYMBOLS] + [('stock', symbol) for symbol in STOCK_SYMBOLS]

    for asset_type, symbol in all_symbols:
        print(f"\n📈 Processing {symbol}...")

        # Get maximum available data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=MAX_YEARS * 365)

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if len(data) < 300:  # Mindestens ~1 Jahr Daten
                print(f"   ⚠️ {symbol}: Nicht genug Daten ({len(data)} Tage)")
                continue

            # Align VIX data with asset data dates
            asset_vix_data = None
            if vix_data is not None:
                try:
                    # Align VIX dates with asset data dates
                    vix_aligned = vix_data.reindex(data.index, method='ffill').fillna(method='bfill')
                    asset_vix_data = vix_aligned
                except:
                    print(f"   ⚠️ {symbol}: VIX alignment failed")

            relative_features = calculate_relative_features(data, asset_vix_data)
            asset_features = get_asset_features(symbol, asset_type)

            # TEMPORAL SPLIT: Konfigurierbar via TRAIN_SPLIT
            total_days = len(data)
            split_idx = int(total_days * TRAIN_SPLIT)

            print(f"   📊 {symbol}: {total_days} Tage gesamt, Split bei Tag {split_idx}")

            # Asset vector (normalized) - works for both ETF and Stock
            asset_vec = np.array(list(asset_features.values()))

            # SLIDING WINDOW mit STRIDE=1 für maximale Datenausnutzung
            def create_sequences_sliding(start_idx, end_idx):
                sequences_x, sequences_y = [], []

                # Sliding window: jeden Tag um 1 verschieben
                for i in range(start_idx, end_idx - SEQUENCE_LENGTH - PREDICTION_LENGTH + 1):
                    # Relative features (9) - now includes sentiment features
                    input_features = np.stack([
                        relative_features['returns'][i:i+SEQUENCE_LENGTH],
                        relative_features['volume_change'][i:i+SEQUENCE_LENGTH],
                        relative_features['volatility'][i:i+SEQUENCE_LENGTH],
                        relative_features['hl_spread_pct'][i:i+SEQUENCE_LENGTH],
                        relative_features['ma_ratio'][i:i+SEQUENCE_LENGTH],
                        relative_features['momentum'][i:i+SEQUENCE_LENGTH],
                        relative_features['vix_normalized'][i:i+SEQUENCE_LENGTH],
                        relative_features['vix_change'][i:i+SEQUENCE_LENGTH],
                        relative_features['fear_greed_synthetic'][i:i+SEQUENCE_LENGTH]
                    ], axis=1)

                    # Asset features (8) für jede Sequenz
                    asset_seq = np.tile(asset_vec, (SEQUENCE_LENGTH, 1))

                    # Combine (17 total: 9 relative + 8 asset)
                    full_features = np.concatenate([input_features, asset_seq], axis=1)

                    # Target: future percentage changes
                    current_price = data['Close'].iloc[i+SEQUENCE_LENGTH-1]
                    future_prices = data['Close'].iloc[i+SEQUENCE_LENGTH:i+SEQUENCE_LENGTH+PREDICTION_LENGTH]
                    percentage_changes = ((future_prices / current_price) - 1) * 100

                    sequences_x.append(full_features)
                    sequences_y.append(percentage_changes.values)

                return np.array(sequences_x), np.array(sequences_y)

            # Training: Erste 80% der Daten (zeitlich älter)
            X_train, y_train = create_sequences_sliding(20, split_idx)  # Start bei Tag 20 für MA-Berechnungen

            # Validation: Letzte 20% der Daten (zeitlich neuer)
            X_val, y_val = create_sequences_sliding(split_idx, total_days)

            if len(X_train) > 0 and len(X_val) > 0:
                all_X_train.append(X_train)
                all_y_train.append(y_train)
                all_X_val.append(X_val)
                all_y_val.append(y_val)
                print(f"   ✅ {symbol} ({asset_type}): {len(X_train)} train, {len(X_val)} val sequences")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Guard against empty sequences
    if not all_X_train or not all_X_val:
        print("\n❌ Not enough sequences collected for training/validation. Aborting.")
        return

    # Combine and normalize
    X_train_combined = np.concatenate(all_X_train, axis=0)
    y_train_combined = np.concatenate(all_y_train, axis=0)
    X_val_combined = np.concatenate(all_X_val, axis=0)
    y_val_combined = np.concatenate(all_y_val, axis=0)

    print(f"\n⚖️ Normalizing features...")

    # Scale relative features (0:9) to [-1, +1] for stable training
    X_train_relative = X_train_combined[:, :, :9].reshape(-1, 9)

    # Clean infinity and NaN values
    X_train_relative = np.nan_to_num(X_train_relative, nan=0.0, posinf=1.0, neginf=-1.0)

    # Remove extreme outliers (> 10 standard deviations)
    for i in range(X_train_relative.shape[1]):
        col = X_train_relative[:, i]
        std_val = np.std(col)
        mean_val = np.mean(col)
        mask = np.abs(col - mean_val) < 10 * std_val
        if not mask.all():
            X_train_relative[~mask, i] = np.clip(col[~mask], mean_val - 10*std_val, mean_val + 10*std_val)

    relative_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_relative_norm = relative_scaler.fit_transform(X_train_relative)

    X_val_relative = X_val_combined[:, :, :9].reshape(-1, 9)

    # Clean validation data the same way as training
    X_val_relative = np.nan_to_num(X_val_relative, nan=0.0, posinf=1.0, neginf=-1.0)

    # Apply same outlier clipping to validation (for consistency)
    for i in range(X_val_relative.shape[1]):
        col = X_val_relative[:, i]
        std_val = np.std(col)
        mean_val = np.mean(col)
        mask = np.abs(col - mean_val) < 10 * std_val
        if not mask.all():
            X_val_relative[~mask, i] = np.clip(col[~mask], mean_val - 10*std_val, mean_val + 10*std_val)

    X_val_relative_norm = relative_scaler.transform(X_val_relative)

    # Reshape back and combine
    X_train_relative_norm = X_train_relative_norm.reshape(X_train_combined.shape[0], SEQUENCE_LENGTH, 9)
    X_val_relative_norm = X_val_relative_norm.reshape(X_val_combined.shape[0], SEQUENCE_LENGTH, 9)

    X_train_final = np.concatenate([X_train_relative_norm, X_train_combined[:, :, 9:]], axis=2)
    X_val_final = np.concatenate([X_val_relative_norm, X_val_combined[:, :, 9:]], axis=2)

    # Keep percentage targets as-is (no normalization needed)
    y_train_final = y_train_combined
    y_val_final = y_val_combined

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_final)
    y_train_tensor = torch.FloatTensor(y_train_final)
    X_val_tensor = torch.FloatTensor(X_val_final)
    y_val_tensor = torch.FloatTensor(y_val_final)

    # Shuffle
    train_indices = torch.randperm(len(X_train_tensor))
    X_train_tensor = X_train_tensor[train_indices]
    y_train_tensor = y_train_tensor[train_indices]

    # Save to structured directories
    print(f"\n💾 Saving data to {DATA_STORAGE['preprocessed_data_dir']}...")

    torch.save(X_train_tensor, DATA_STORAGE['preprocessed_data_dir'] + 'X_train_mixed.pt')
    torch.save(y_train_tensor, DATA_STORAGE['preprocessed_data_dir'] + 'y_train_mixed.pt')
    torch.save(X_val_tensor, DATA_STORAGE['preprocessed_data_dir'] + 'X_val_mixed.pt')
    torch.save(y_val_tensor, DATA_STORAGE['preprocessed_data_dir'] + 'y_val_mixed.pt')

    # Save comprehensive metadata
    metadata = {
        'config': CONFIG,
        'relative_scaler_min': relative_scaler.min_.tolist(),
        'relative_scaler_scale': relative_scaler.scale_.tolist(),
        'scaled_feature_names': list(CONFIG['relative_features'].keys()),
        'etf_feature_names': list(ETF_FEATURES.keys()),
        'stock_feature_names': list(STOCK_FEATURES.keys()),
        'processing_info': {
            'train_samples': len(X_train_tensor),
            'val_samples': len(X_val_tensor),
            'etf_symbols': ETF_SYMBOLS,
            'stock_symbols': STOCK_SYMBOLS,
            'train_split': TRAIN_SPLIT
        }
    }
    with open(DATA_STORAGE['scalers_file'], 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Preprocessing complete!")
    print(f"   📊 Training: {X_train_tensor.shape}")
    print(f"   📊 Validation: {X_val_tensor.shape}")
    print(f"   💡 Features: 9 relative (6 base + 3 sentiment) + 8 asset meta = 17 total")
    print(f"   🧠 Sentiment features: VIX normalized, VIX change, Fear/Greed synthetic")
    print(f"   📊 Targets: percentage changes (mean={y_train_tensor.mean():.3f}%, std={y_train_tensor.std():.3f}%)")

    print(f"\n🔍 Data Quality:")
    print(f"   📈 Training samples: {len(X_train_tensor):,}")
    print(f"   📊 Validation samples: {len(X_val_tensor):,}")
    print(f"   📊 Input ranges: [{X_train_tensor.min():.2f}, {X_train_tensor.max():.2f}]")
    print(f"   🎯 Target ranges: [{y_train_tensor.min():.1f}%, {y_train_tensor.max():.1f}%]")
    print(f"   ❌ Any NaN: {torch.isnan(X_train_tensor).any().item()}")

    print(f"\n🚀 Ready for TCN training with sentiment features!")
    print(f"   💡 Features scaled to [-1, +1] for stable training")
    print(f"   🧠 VIX & Fear/Greed features help combat regime noise")
    print(f"   💡 Targets are percentage changes (directly interpretable)")

    # Plot normalized ETF features for analysis
    import matplotlib.pyplot as plt

    print(f"\n📊 Creating asset features plots...")

    # ETF Features Plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('📈 Normalized ETF Features Analysis', fontsize=16)

    etf_data = {}
    for symbol in ETF_SYMBOLS:
        try:
            features = get_asset_features(symbol, 'etf')
            etf_data[symbol] = list(features.values())
        except:
            continue

    if etf_data:
        feature_names = ['NAV', 'Beta', 'Volume', 'Expense Ratio', 'Large Cap', 'Growth', 'Tech Heavy', 'High Vol']

        for i, feature_name in enumerate(feature_names):
            row, col = i // 4, i % 4
            ax = axes[row, col]

            values = [etf_data[symbol][i] for symbol in etf_data.keys()]
            symbols_list = list(etf_data.keys())

            bars = ax.bar(symbols_list, values, alpha=0.7)
            ax.set_title(f'{feature_name} (Normalized)')
            ax.set_ylabel('Normalized Value [-1, +1]')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_ylim(-1.1, 1.1)

            # Color bars based on value
            for j, bar in enumerate(bars):
                if values[j] > 0.5:
                    bar.set_color('green')
                elif values[j] < -0.5:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')

        plt.tight_layout()
        etf_plot_path = DATA_STORAGE['plots_dir'] + 'normalized_etf_features.png'
        plt.savefig(etf_plot_path, dpi=200, bbox_inches='tight')
        print(f"   📊 ETF features plot saved: {etf_plot_path}")
        plt.show()

        # Print feature values for inspection
        print(f"\n📋 ETF Feature Values:")
        for symbol in ETF_SYMBOLS:
            if symbol in etf_data:
                print(f"   {symbol}: {[f'{val:+.2f}' for val in etf_data[symbol]]}")

    # Stock Features Plot (first 6 features only for display)
    if STOCK_SYMBOLS:
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        fig2.suptitle('📈 Normalized Stock Features Analysis', fontsize=16)

        stock_data = {}
        for symbol in STOCK_SYMBOLS[:6]:  # Limit to 6 for display
            try:
                features = get_asset_features(symbol, 'stock')
                # Show first 6 features only
                stock_data[symbol] = list(features.values())[:6]
            except:
                continue

        if stock_data:
            stock_feature_names = ['Market Cap', 'Beta', 'PE Ratio', 'Profit Margin', 'Tech Sector', 'Consumer Sector']

            for i, feature_name in enumerate(stock_feature_names):
                row, col = i // 3, i % 3
                ax = axes2[row, col]

                values = [stock_data[symbol][i] for symbol in stock_data.keys()]
                symbols_list = list(stock_data.keys())

                bars = ax.bar(symbols_list, values, alpha=0.7)
                ax.set_title(f'{feature_name} (Normalized)')
                ax.set_ylabel('Normalized Value [-1, +1]')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_ylim(-1.1, 1.1)

                # Color bars based on value
                for j, bar in enumerate(bars):
                    if values[j] > 0.5:
                        bar.set_color('green')
                    elif values[j] < -0.5:
                        bar.set_color('red')
                    else:
                        bar.set_color('blue')

            plt.tight_layout()
            stock_plot_path = DATA_STORAGE['plots_dir'] + 'normalized_stock_features.png'
            plt.savefig(stock_plot_path, dpi=200, bbox_inches='tight')
            print(f"   📊 Stock features plot saved: {stock_plot_path}")
            plt.show()

            print(f"\n📋 Stock Feature Values:")
            for symbol in list(stock_data.keys())[:3]:  # Show first 3
                print(f"   {symbol}: {[f'{val:+.2f}' for val in stock_data[symbol]]}")

    if not etf_data and not stock_data:
        print(f"   ⚠️ Could not create asset features plots")

if __name__ == "__main__":
    main()