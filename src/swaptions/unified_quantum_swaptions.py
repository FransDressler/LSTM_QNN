#!/usr/bin/env python3
"""
🌌 Unified Quantum-Classical Swaption Predictor
==============================================

Complete end-to-end quantum-enhanced swaption prediction with:
- Unified quantum-classical optimization
- Financial-domain-specific decoder
- Live training visualization every 5 epochs
- Comprehensive performance analysis

Author: Claude & Frans
Date: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import warnings
from pathlib import Path
import os
 
# Quantum imports
try:
    import merlin as ML
    from merlin.core.computation_space import ComputationSpace
    MERLIN_AVAILABLE = True
    print("🌌 Merlin Quantum Framework loaded successfully!")
except ImportError as e:
    MERLIN_AVAILABLE = False
    raise RuntimeError(f"🚫 Merlin required for quantum processing: {e}")

# Setup für schöne Plots
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

print("="*80)
print("🌌 UNIFIED QUANTUM-CLASSICAL SWAPTION PREDICTOR")
print("="*80)


class FinancialSwaptionDecoder(nn.Module):
    """
    🏦 Financial-Domain-Specific Decoder for Swaption Predictions

    Designed specifically for quantum features → swaption price predictions
    Incorporates financial domain knowledge through specialized branches:
    - Market Risk Extraction
    - Price Trend Analysis
    - Volatility Modeling
    - Multi-Horizon Prediction
    """

    def __init__(self, quantum_dim=252):
        super().__init__()

        print(f"🏦 Initializing Financial Swaption Decoder...")
        print(f"   📊 Quantum input dimension: {quantum_dim}")

        # Financial feature extraction branches
        self.market_risk_extractor = nn.Sequential(
            nn.Linear(quantum_dim, 96),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Dropout(0.15)
        )

        self.price_trend_extractor = nn.Sequential(
            nn.Linear(quantum_dim, 96),
            nn.Tanh(),  # Bounded for trend direction
            nn.Dropout(0.1)
        )

        self.volatility_extractor = nn.Sequential(
            nn.Linear(quantum_dim, 64),
            nn.Softplus(),  # Always positive for volatility
            nn.Dropout(0.1)
        )

        # Multi-horizon prediction heads
        total_features = 96 + 96 + 64  # 256 combined features

        self.short_term_processor = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.long_term_processor = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Final prediction fusion
        self.prediction_fusion = nn.Sequential(
            nn.Linear(64 + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 14)  # 14-day predictions
        )

        # Financial output constraints (learnable)
        self.output_scaler = nn.Parameter(torch.tensor(0.05))  # Realistic price changes
        # Note: Removed output_bias - using multi-task level loss instead

        total_params = sum(p.numel() for p in self.parameters())
        print(f"   ⚙️  Financial decoder parameters: {total_params:,}")
        print(f"   🎯 Specialized branches: Risk + Trend + Volatility")
        print(f"   📈 Multi-horizon: Short-term (1-7 days) + Long-term (8-14 days)")

    def forward(self, quantum_features):
        """
        Forward pass through financial decoder

        Args:
            quantum_features: (batch, quantum_dim) - Quantum measurement outcomes

        Returns:
            predictions: (batch, 14) - 14-day swaption price predictions
        """
        # Extract specialized financial aspects from quantum features
        risk_features = self.market_risk_extractor(quantum_features)      # Market risk signals
        trend_features = self.price_trend_extractor(quantum_features)     # Price trend direction
        volatility_features = self.volatility_extractor(quantum_features) # Volatility patterns

        # Combine all financial aspects
        financial_features = torch.cat([risk_features, trend_features, volatility_features], dim=1)

        # Multi-horizon processing (different logic for short vs long term)
        short_term_features = self.short_term_processor(financial_features)  # Days 1-7
        long_term_features = self.long_term_processor(financial_features)    # Days 8-14

        # Fusion of temporal perspectives
        temporal_fusion = torch.cat([short_term_features, long_term_features], dim=1)

        # Final predictions with financial constraints
        raw_predictions = self.prediction_fusion(temporal_fusion)

        # Apply learnable financial bounds (prevents unrealistic price jumps)
        bounded_predictions = torch.tanh(raw_predictions) * torch.abs(self.output_scaler)

        # Return predictions (bias correction now handled by multi-task level loss)
        return bounded_predictions

    def get_financial_insights(self, quantum_features):
        """
        Extract interpretable financial insights from quantum features

        Returns:
            dict with risk, trend, volatility components
        """
        with torch.no_grad():
            risk = self.market_risk_extractor(quantum_features)
            trend = self.price_trend_extractor(quantum_features)
            volatility = self.volatility_extractor(quantum_features)

            return {
                'market_risk': torch.norm(risk, dim=1),
                'price_trend': torch.mean(trend, dim=1),
                'volatility': torch.mean(volatility, dim=1),
                'output_scaling': self.output_scaler.item()
            }


class LevelPredictionLSTM(nn.Module):
    """
    📈 Simple LSTM for Level Prediction

    Takes original X values (not quantum-scaled) and predicts the overall level
    for bias correction of quantum predictions.
    """

    def __init__(self, input_dim=8, hidden_dim=32, output_dim=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=1, batch_first=True, dropout=0.1)
        self.level_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, output_dim)  # Single level prediction
        )

        print(f"📈 Level Prediction LSTM initialized:")
        print(f"   Input: {input_dim}D → LSTM: {hidden_dim}D → Output: {output_dim}D")

    def forward(self, original_x):
        """
        Args:
            original_x: (batch, 8) - Original X values (not quantum scaled)
        Returns:
            level_pred: (batch, 1) - Predicted level for bias correction
        """
        # Add sequence dimension for LSTM (batch, seq=1, features)
        x_seq = original_x.unsqueeze(1)  # (batch, 1, 8)

        lstm_out, _ = self.lstm(x_seq)  # (batch, 1, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        level_prediction = self.level_head(last_hidden)  # (batch, 1)

        return level_prediction


class UnifiedQuantumSwaptionPredictor(nn.Module):
    """
    🌌 Unified Quantum-Classical Swaption Predictor

    End-to-end model combining:
    1. Quantum feature extraction (photonic quantum computing)
    2. Financial-specific decoding (domain-aware neural network)

    Trained as unified block against swaption prediction loss
    """

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

        if not MERLIN_AVAILABLE:
            raise RuntimeError("🚫 Merlin Quantum Framework required!")

        # 1. Quantum Feature Extractor
        self._build_quantum_layer()

        # 2. Financial Decoder
        self.financial_decoder = FinancialSwaptionDecoder(
            quantum_dim=self.quantum_output_dim
        )

        # Removed: Level Prediction LSTM (using simple shift instead)

        # 3. Training statistics
        self.training_epoch = 0
        self._rescale_warning_shown = False

        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            quantum_params = sum(p.numel() for p in self.quantum_enhancer.parameters())
            classical_params = sum(p.numel() for p in self.financial_decoder.parameters())

            print(f"\n🏗️ Unified Quantum-Classical Architecture:")
            print(f"   🌌 Quantum parameters: {quantum_params:,}")
            print(f"   🏦 Financial parameters: {classical_params:,}")
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   🎯 End-to-end optimization: Quantum ↔ Financial")

    def _build_quantum_layer(self):
        """Build quantum feature extractor"""
        if self.verbose:
            print("\n🔧 Building Quantum Feature Extractor...")

        self.quantum_enhancer = ML.QuantumLayer.simple(
            input_size=8,              # Original LSTM features (no augmentation)
            n_params=90,               # Trainable quantum parameters
            no_bunching=True           # Unbunched space (stable)
        )

        self.quantum_output_dim = self.quantum_enhancer.output_size

        if self.verbose:
            print(f"   ✅ Quantum circuit created!")
            print(f"   📊 Input: 8D LSTM latents → Output: {self.quantum_output_dim}D quantum features")
            print(f"   ⚙️  Trainable quantum parameters: 90")
            print(f"   🎯 Photonic quantum computing with angle encoding")

    def forward(self, lstm_latents):
        """
        Standard forward pass: LSTM latents → Quantum → Financial → Predictions

        Args:
            lstm_latents: (batch, 8) - LSTM latent vectors

        Returns:
            predictions: (batch, 14) - 14-day swaption predictions
        """
        # Step 1: Quantum feature extraction with automatic rescaling
        quantum_features = self._quantum_forward_with_rescaling(lstm_latents)

        # Step 2: Financial decoding
        raw_predictions = self.financial_decoder(quantum_features)

        # Step 3: Inverse scaling to restore original scale
        final_predictions = self._inverse_scale_predictions(raw_predictions)

        return final_predictions

    def _quantum_forward_with_rescaling(self, lstm_latents):
        """Forward pass with invertible MinMax scaling to [0, 2π]"""

        # Simple MinMax scaling: min/max → [0, 2π]
        if lstm_latents.min() < -0.05 or lstm_latents.max() > 2*np.pi + 0.05:
            # Store scaling parameters for exact reversal
            data_min = lstm_latents.min()
            data_max = lstm_latents.max()
            data_range = data_max - data_min

            if data_range > 1e-10:
                # Forward: original → [0, 2π]
                lstm_latents_scaled = (lstm_latents - data_min) / data_range * 2 * np.pi
            else:
                # Fallback for constant input
                lstm_latents_scaled = torch.ones_like(lstm_latents) * np.pi
                data_min = torch.tensor(0.)
                data_max = torch.tensor(2*np.pi)
                data_range = torch.tensor(2*np.pi)

            if self.verbose and not self._rescale_warning_shown:
                print(f"🔧 Invertible MinMax scaling for quantum processing")
                print(f"   Original range: [{data_min:.3f}, {data_max:.3f}]")
                print(f"   Scaled range: [0.000, {2*np.pi:.3f}]")
                print(f"   Fully reversible transformation")
                self._rescale_warning_shown = True

            # Store scaling parameters for inverse transformation
            self.scaling_params = {
                'data_min': data_min,
                'data_max': data_max,
                'data_range': data_range,
                'scaled': True
            }

            lstm_latents = lstm_latents_scaled
        else:
            # Already in good range - no scaling needed
            self.scaling_params = {
                'data_min': torch.tensor(0.),
                'data_max': torch.tensor(2*np.pi),
                'data_range': torch.tensor(2*np.pi),
                'scaled': False
            }

        # Quantum processing
        quantum_features = self.quantum_enhancer(lstm_latents)
        return quantum_features

    def _inverse_scale_predictions(self, predictions):
        """Inverse MinMax scaling: [0, 2π] → original scale"""
        if hasattr(self, 'scaling_params') and self.scaling_params['scaled']:
            data_min = self.scaling_params['data_min']
            data_max = self.scaling_params['data_max']
            data_range = self.scaling_params['data_range']

            # Inverse: [0, 2π] → original range
            original_scale_predictions = predictions / (2 * np.pi) * data_range + data_min
            return original_scale_predictions
        else:
            # No scaling was applied
            return predictions

    def get_financial_breakdown(self, lstm_latents):
        """
        Get detailed breakdown of quantum + financial processing

        Returns:
            dict with quantum features, financial insights, predictions
        """
        with torch.no_grad():
            quantum_features = self._quantum_forward_with_rescaling(lstm_latents)
            financial_insights = self.financial_decoder.get_financial_insights(quantum_features)
            predictions = self.financial_decoder(quantum_features)

            return {
                'lstm_latents': lstm_latents,
                'quantum_features': quantum_features,
                'financial_insights': financial_insights,
                'predictions': predictions
            }


def augment_lstm_features(lstm_features):
    """
    Augment LSTM features with level information for better bias control.
    Reduced to 9D to avoid quantum mode limits.

    Args:
        lstm_features: (batch_size, 8) LSTM latent vectors

    Returns:
        augmented_features: (batch_size, 9) Enhanced features with level info
    """
    batch_size = lstm_features.shape[0]

    # Add only the most critical level feature to avoid quantum mode limits
    overall_level = lstm_features.mean(dim=1, keepdim=True)    # Overall price level (most important)

    # Concatenate: 8D LSTM + 1D level feature = 9D total
    augmented = torch.cat([
        lstm_features,      # Original 8D LSTM latent vectors
        overall_level       # Critical level indicator
    ], dim=1)

    return augmented


def multi_task_loss(predictions, targets, shape_weight=1.0, level_weight=1.0):
    """
    Multi-task loss: separates shape learning from level learning.

    Args:
        predictions: (batch_size, 14) Model predictions
        targets: (batch_size, 14) Ground truth values
        shape_weight: Weight for curve shape learning
        level_weight: Weight for overall level learning

    Returns:
        total_loss: Combined shape + level loss
        loss_breakdown: Dict with individual loss components
    """
    # Shape loss: focuses on relative patterns (centered data)
    pred_centered = predictions - predictions.mean(dim=1, keepdim=True)
    target_centered = targets - targets.mean(dim=1, keepdim=True)
    shape_loss = F.mse_loss(pred_centered, target_centered)

    # Level loss: focuses on absolute positioning (means only)
    pred_means = predictions.mean(dim=1)
    target_means = targets.mean(dim=1)
    level_loss = F.mse_loss(pred_means, target_means)

    # Combined loss
    total_loss = shape_weight * shape_loss + level_weight * level_loss

    return total_loss, {
        'total': total_loss.item(),
        'shape': shape_loss.item(),
        'level': level_loss.item(),
        'shape_weight': shape_weight,
        'level_weight': level_weight
    }


def load_quantum_data():
    """Load quantum-ready swaption data (simple version)"""
    print("\n📂 Loading quantum-ready swaption data...")

    try:
        # Quantum-scaled X data
        X_train_quantum = torch.load('X_train_quantum.pt', weights_only=False)
        X_val_quantum = torch.load('X_val_quantum.pt', weights_only=False)

        # Use NORMAL Y data (not quantum-scaled!) like original
        Y_train_raw = torch.load('y_train_lstm.pt', weights_only=False)
        Y_val_raw = torch.load('y_val_lstm.pt', weights_only=False)

        # Convert Y data to tensors if needed
        if not isinstance(Y_train_raw, torch.Tensor):
            Y_train = torch.FloatTensor(Y_train_raw)
            Y_val = torch.FloatTensor(Y_val_raw)
        else:
            Y_train = Y_train_raw
            Y_val = Y_val_raw

        # Skip augmentation - use original 8D LSTM features
        print(f"   🔧 Using original 8D LSTM features (no augmentation)...")

        print(f"   ✅ Data loaded successfully!")
        print(f"   📊 Training: {X_train_quantum.shape[0]:,} samples")
        print(f"   📊 Validation: {X_val_quantum.shape[0]:,} samples")
        print(f"   📊 LSTM latents: {X_train_quantum.shape[1]}D (quantum-scaled)")
        print(f"   📊 Swaption targets: {Y_train.shape[1]}D (14-day horizon)")
        print(f"   📊 Input range: [{X_train_quantum.min():.3f}, {X_train_quantum.max():.3f}]")

        return X_train_quantum, X_val_quantum, Y_train, Y_val

    except FileNotFoundError as e:
        print(f"   ❌ Error loading data: {e}")
        print("   📝 Run preprocessing notebook first!")
        return None, None, None, None


def plot_training_progress_live(epoch, model, val_loader, history, save_dir="training_plots"):
    """
    🎨 Live training visualization every 5 epochs
    Shows: Ground Truth vs Predictions vs LSTM inputs
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"🎨 Creating live training plots for epoch {epoch}...")

    model.eval()
    device = next(model.parameters()).device

    # Get sample batch for visualization (simple format)
    sample_batch = next(iter(val_loader))
    X_sample, Y_sample = sample_batch[0][:6].to(device), sample_batch[1][:6].to(device)

    with torch.no_grad():
        # Get predictions
        predictions = model(X_sample)

        # Get detailed breakdown
        breakdown = model.get_financial_breakdown(X_sample)
        financial_insights = breakdown['financial_insights']
        quantum_features = breakdown['quantum_features']

    # Convert to numpy
    X_np = X_sample.cpu().numpy()
    Y_np = Y_sample.cpu().numpy()
    pred_np = predictions.cpu().numpy()

    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

    fig.suptitle(f'🌌 QUANTUM SWAPTION TRAINING - EPOCH {epoch}', fontsize=16, fontweight='bold')

    # Plot 1-6: Individual prediction comparisons
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        days = np.arange(1, 15)

        # Plot LSTM features as trajectory (8D → 14D interpolation)
        lstm_features_original = X_np[i]  # Use 8D LSTM features

        # Map 252 quantum features to 14 future days via subsampling
        # Take every Nth feature to get 14 values
        step = max(1, len(lstm_features_original) // 14)
        lstm_interpolated = lstm_features_original[::step][:14]  # Subsample to 14 values

        # If we don't have enough, repeat the pattern
        if len(lstm_interpolated) < 14:
            lstm_interpolated = np.tile(lstm_interpolated, int(np.ceil(14/len(lstm_interpolated))))[:14]

        # Scale LSTM trajectory to be comparable with Y values (both represent price changes)
        # This shows how LSTM "expects" the price to evolve based on latent features
        # if np.std(lstm_interpolated) > 0:  # Avoid division by zero
        #     lstm_normalized = (lstm_interpolated - np.mean(lstm_interpolated)) / np.std(lstm_interpolated)
        #     lstm_normalized = lstm_normalized * np.std(Y_np[i]) + np.mean(Y_np[i])
        # else:
        #     lstm_normalized = np.full_like(lstm_interpolated, np.mean(Y_np[i]))

        # Simple level adjustment before plotting (same as simple_test_plot.py)
        pred_mean = np.mean(pred_np)
        gt_mean = np.mean(Y_np[i])

        # Adjust prediction to match ground truth level
        pred_adjusted = pred_np[i] + (gt_mean - pred_mean)

        # Plot standard quantum predictions with simple level shift
        ax.plot(days, Y_np[i], 'o-', label='Ground Truth', linewidth=2, markersize=6, color='black')
        ax.plot(days, pred_np[i], 's-', label='Original Prediction', linewidth=1.5, alpha=0.6, color='gray')
        ax.plot(days, pred_adjusted, 's-', label='Level-Adjusted Pred', linewidth=2, alpha=0.8, color='purple')

        # Calculate sample metrics (using level-adjusted predictions)
        mse = np.mean((Y_np[i] - pred_adjusted)**2)
        correlation = np.corrcoef(Y_np[i], pred_adjusted)[0, 1]

        ax.set_title(f'Sample {i+1}\nMSE: {mse:.4f} | Corr: {correlation:.3f}')
        ax.set_xlabel('Future Day')
        ax.set_ylabel('Normalized Price Change')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Color background based on performance
        if mse < 0.01:
            ax.set_facecolor('#f0fff0')  # Light green - excellent
        elif mse < 0.05:
            ax.set_facecolor('#fff8dc')  # Light yellow - good
        else:
            ax.set_facecolor('#ffe4e1')  # Light red - needs improvement

    # Plot 7: Training progress
    ax_progress = fig.add_subplot(gs[2, :3])
    epochs_so_far = range(1, len(history['train_loss']) + 1)

    ax_progress.plot(epochs_so_far, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax_progress.plot(epochs_so_far, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax_progress.set_title('🏋️ Training Progress')
    ax_progress.set_xlabel('Epoch')
    ax_progress.set_ylabel('MSE Loss')
    ax_progress.legend()
    ax_progress.grid(True, alpha=0.3)
    ax_progress.set_yscale('log')

    # Mark current epoch
    ax_progress.axvline(x=epoch, color='orange', linestyle='--', alpha=0.7, label=f'Current (Epoch {epoch})')
    ax_progress.legend()

    # Plot 8: LSTM Input Features Heatmap
    ax_lstm = fig.add_subplot(gs[3, :2])

    # Show LSTM latent features for the samples
    im = ax_lstm.imshow(X_np.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax_lstm.set_title('📊 LSTM Features (8D Quantum-scaled)')
    ax_lstm.set_xlabel('Sample Index')
    ax_lstm.set_ylabel('Feature Dimension')
    ax_lstm.set_xticks(range(6))
    ax_lstm.set_xticklabels([f'S{i+1}' for i in range(6)])
    ax_lstm.set_yticks(range(8))
    ax_lstm.set_yticklabels([f'F{i+1}' for i in range(8)])
    plt.colorbar(im, ax=ax_lstm, shrink=0.6)

    # Plot 9: Quantum Features Sample
    ax_quantum = fig.add_subplot(gs[3, 2:4])

    # Show first 50 quantum features for visualization
    quantum_sample = quantum_features[:6, :50].cpu().numpy()
    im2 = ax_quantum.imshow(quantum_sample.T, aspect='auto', cmap='plasma', interpolation='nearest')
    ax_quantum.set_title('🌌 Quantum Features (50 of 252)')
    ax_quantum.set_xlabel('Sample Index')
    ax_quantum.set_ylabel('Quantum Feature Index')
    ax_quantum.set_xticks(range(6))
    ax_quantum.set_xticklabels([f'S{i+1}' for i in range(6)])
    plt.colorbar(im2, ax=ax_quantum, shrink=0.6)

    # Plot 10: Financial Insights
    ax_insights = fig.add_subplot(gs[3, 4])

    # Extract financial metrics
    risk_levels = financial_insights['market_risk'].cpu().numpy()
    trend_signals = financial_insights['price_trend'].cpu().numpy()
    volatility_measures = financial_insights['volatility'].cpu().numpy()

    # Create financial insights bar plot
    x_pos = np.arange(6)
    width = 0.25

    ax_insights.bar(x_pos - width, risk_levels, width, label='Risk', alpha=0.8)
    ax_insights.bar(x_pos, trend_signals, width, label='Trend', alpha=0.8)
    ax_insights.bar(x_pos + width, volatility_measures, width, label='Volatility', alpha=0.8)

    ax_insights.set_title('🏦 Financial Insights')
    ax_insights.set_xlabel('Sample')
    ax_insights.set_ylabel('Signal Strength')
    ax_insights.set_xticks(x_pos)
    ax_insights.set_xticklabels([f'S{i+1}' for i in range(6)])
    ax_insights.legend(fontsize=8)
    ax_insights.grid(True, alpha=0.3)

    # Overall statistics
    current_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    improvement = f"{((history['val_loss'][0] - current_val_loss) / history['val_loss'][0] * 100):.1f}%" if len(history['val_loss']) > 0 else "N/A"

    # Add text summary
    fig.text(0.02, 0.02,
             f"📈 Epoch {epoch} Summary | Val Loss: {current_val_loss:.6f} | Best: {best_val_loss:.6f} | Improvement: {improvement}",
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

    # Save plot
    plot_filename = f"{save_dir}/epoch_{epoch:03d}_training_progress.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"   ✅ Live plot saved: {plot_filename}")

    model.train()  # Back to training mode


def train_unified_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    🏋️ Unified training pipeline for quantum-classical model

    Features:
    - End-to-end optimization of quantum + financial parameters
    - Live visualization every 5 epochs
    - Early stopping and learning rate scheduling
    - Comprehensive progress tracking
    """
    print(f"\n🏋️ Starting Unified Quantum-Classical Training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Unified optimizer (both quantum and classical parameters)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Multi-task loss configuration with enhanced level focus
    shape_weight = 1.0  # Maintain curve shape quality
    level_weight = 0.5  # INCREASED: Strong focus on level correction

    # Training history (extended for multi-task tracking)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_shape_loss': [],
        'train_level_loss': [],
        'val_shape_loss': [],
        'val_level_loss': [],
        'learning_rates': [],
        'quantum_scaling': [],
        'financial_bounds': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    print(f"   🎯 Target epochs: {epochs}")
    print(f"   🔧 Device: {device}")
    print(f"   ⚙️  Optimizer: AdamW (unified quantum + classical)")
    print(f"   📈 Live plots: Every 5 epochs")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.training_epoch = epoch
        epoch_start = time.time()

        # Training phase with weighted combination loss
        model.train()
        train_loss = 0.0
        train_shape_loss = 0.0
        train_level_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()

            # Standard forward pass (quantum only)
            predictions = model(batch_X)

            # Weighted combination loss: strong curve focus!
            loss, loss_breakdown = multi_task_loss(
                predictions, batch_Y, shape_weight=2.0, level_weight=0.2
            )

            # Unified backward pass (updates quantum + classical together)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track loss components
            train_loss += loss_breakdown['total']
            train_shape_loss += loss_breakdown['shape']
            train_level_loss += loss_breakdown['level']

            # Progress updates every 50 batches
            if (batch_idx + 1) % 50 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                current_loss = train_loss / (batch_idx + 1)
                print(f"     Batch {batch_idx+1:3d}/{num_batches} ({progress:5.1f}%) | Loss: {current_loss:.6f}", end='\r')

        avg_train_loss = train_loss / num_batches
        avg_train_shape_loss = train_shape_loss / num_batches
        avg_train_level_loss = train_level_loss / num_batches

        # Validation phase with weighted combination loss
        model.eval()
        val_loss = 0.0
        val_shape_loss = 0.0
        val_level_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                predictions = model(batch_X)

                # Weighted combination validation loss
                loss, loss_breakdown = multi_task_loss(
                    predictions, batch_Y, shape_weight=2.0, level_weight=0.2
                )

                val_loss += loss_breakdown['total']
                val_shape_loss += loss_breakdown['shape']
                val_level_loss += loss_breakdown['level']

        avg_val_loss = val_loss / len(val_loader)
        avg_val_shape_loss = val_shape_loss / len(val_loader)
        avg_val_level_loss = val_level_loss / len(val_loader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Track additional metrics
        financial_bound = model.financial_decoder.output_scaler.item()

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(current_lr)
        history['financial_bounds'].append(financial_bound)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_unified_quantum_model.pth')
        else:
            patience_counter += 1

        # Enhanced progress logging with multi-task breakdown
        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch <= 10 or epoch == epochs:
            print(f"   Epoch {epoch:3d}/{epochs}: "
                  f"Train={avg_train_loss:.6f} (S:{avg_train_shape_loss:.3f}+L:{avg_train_level_loss:.3f}*0.2), "
                  f"Val={avg_val_loss:.6f} (S:{avg_val_shape_loss:.3f}+L:{avg_val_level_loss:.3f}*0.2), "
                  f"LR={current_lr:.1e}, Time={epoch_time:.1f}s")

        # Live visualization EVERY epoch to debug curve issues
        if epoch % 1 == 0:
            plot_training_progress_live(epoch, model, val_loader, history)

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n   ⏰ Early stopping triggered after {epoch} epochs")
            break

    training_time = time.time() - start_time

    # Load best model
    model.load_state_dict(torch.load('best_unified_quantum_model.pth'))

    print(f"\n✅ Unified training completed!")
    print(f"   ⏱️  Total time: {training_time/60:.2f} minutes")
    print(f"   🎯 Best validation loss: {best_val_loss:.6f}")
    print(f"   🏆 Final financial output scaling: {history['financial_bounds'][-1]:.4f}")

    return model, history


def evaluate_unified_model(model, val_loader):
    """Comprehensive evaluation of unified quantum-classical model"""
    print(f"\n📊 Evaluating Unified Quantum-Classical Model...")

    device = next(model.parameters()).device
    model.eval()

    all_predictions = []
    all_targets = []
    all_lstm_inputs = []
    all_financial_insights = []

    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            # Get comprehensive breakdown
            breakdown = model.get_financial_breakdown(batch_X)

            all_predictions.append(breakdown['predictions'].cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
            all_lstm_inputs.append(breakdown['lstm_latents'].cpu().numpy())

            # Collect financial insights
            insights = breakdown['financial_insights']
            all_financial_insights.append({
                'risk': insights['market_risk'].cpu().numpy(),
                'trend': insights['price_trend'].cpu().numpy(),
                'volatility': insights['volatility'].cpu().numpy()
            })

    # Combine all results
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    lstm_inputs = np.vstack(all_lstm_inputs)

    # Calculate comprehensive metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)

    # Per-day metrics
    daily_mse = [mean_squared_error(targets[:, day], predictions[:, day]) for day in range(14)]
    daily_mae = [mean_absolute_error(targets[:, day], predictions[:, day]) for day in range(14)]

    # Overall correlation
    correlation = np.corrcoef(targets.flatten(), predictions.flatten())[0, 1]

    print(f"   📈 Overall Performance:")
    print(f"     MSE:         {mse:.6f}")
    print(f"     MAE:         {mae:.6f}")
    print(f"     RMSE:        {rmse:.6f}")
    print(f"     Correlation: {correlation:.4f}")

    print(f"\n   📅 Per-Day Performance (first 7 days):")
    for day in range(min(7, 14)):
        print(f"     Day {day+1:2d}: MSE={daily_mse[day]:.6f}, MAE={daily_mae[day]:.6f}")

    return {
        'mse': mse, 'mae': mae, 'rmse': rmse, 'correlation': correlation,
        'daily_mse': daily_mse, 'daily_mae': daily_mae,
        'predictions': predictions, 'targets': targets,
        'lstm_inputs': lstm_inputs, 'financial_insights': all_financial_insights
    }


def main():
    """
    🚀 Main training pipeline for unified quantum-classical swaption predictor
    """
    print("\n🌌✨ UNIFIED QUANTUM-CLASSICAL SWAPTION PREDICTOR ✨🌌")
    print("=" * 70)

    # Load data (simple quantum data)
    X_train, X_val, Y_train, Y_val = load_quantum_data()
    if X_train is None:
        return

    # Create simple data loaders
    batch_size = 128
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n📦 Data loaders created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    # Create unified model
    model = UnifiedQuantumSwaptionPredictor(verbose=True)

    # Train unified model
    trained_model, history = train_unified_model(
        model, train_loader, val_loader,
        epochs=50, lr=0.001
    )

    # Evaluate final model
    results = evaluate_unified_model(trained_model, val_loader)

    # Final summary
    print(f"\n🏆 FINAL UNIFIED QUANTUM-CLASSICAL RESULTS:")
    print(f"   🌌 Quantum Feature Extraction: 8D → {trained_model.quantum_output_dim}D")
    print(f"   🏦 Financial Decoding: {trained_model.quantum_output_dim}D → 14D")
    print(f"   📊 Best Performance: MSE={results['mse']:.6f}, Corr={results['correlation']:.4f}")
    print(f"   ⚙️  Total Parameters: {sum(p.numel() for p in trained_model.parameters()):,}")

    # Save final model and results
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'history': history,
        'results': results,
        'model_config': {
            'quantum_params': 90,
            'quantum_output_dim': trained_model.quantum_output_dim,
            'financial_decoder': 'multi_branch_financial'
        }
    }, 'unified_quantum_swaptions_final.pth')

    print(f"\n💾 Model and results saved to: unified_quantum_swaptions_final.pth")
    print(f"📊 Training plots available in: training_plots/")
    print(f"\n🌌✨ UNIFIED QUANTUM-CLASSICAL EXPERIMENT COMPLETE! ✨🌌")


if __name__ == "__main__":
    main()