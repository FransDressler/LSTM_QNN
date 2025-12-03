#!/usr/bin/env python3
"""
🎯 Robust Quantum-Compatible Scaler
===================================

Reduces error amplification by using percentile-based range reduction
while maintaining [0, 2π] compatibility for quantum processing.

Key insight: Instead of scaling full data range, use 5th-95th percentiles
to remove outliers and reduce amplification factor significantly.
"""

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class RobustQuantumScaler:
    """
    Quantum-compatible scaler that reduces error amplification by:
    1. Clipping outliers using percentiles (5th-95th percentile)
    2. Scaling clipped range to [0, 2π]
    3. Maintaining invertibility for proper evaluation
    """

    def __init__(self, percentile_low=5, percentile_high=95, target_range=(0, 2*np.pi)):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.target_range = target_range
        self.clip_min_ = None
        self.clip_max_ = None
        self.scaler = MinMaxScaler(feature_range=target_range)
        self.fitted = False

    def fit(self, data):
        """Fit the scaler to training data"""
        data_np = data.numpy() if torch.is_tensor(data) else data

        # Calculate percentile-based range
        self.clip_min_ = np.percentile(data_np, self.percentile_low)
        self.clip_max_ = np.percentile(data_np, self.percentile_high)

        print(f"🎯 RobustQuantumScaler fitted:")
        print(f"   Original range: [{data_np.min():.4f}, {data_np.max():.4f}]")
        print(f"   Percentile range: [{self.clip_min_:.4f}, {self.clip_max_:.4f}]")
        print(f"   Amplification factor: {(self.clip_max_ - self.clip_min_) / (self.target_range[1] - self.target_range[0]):.2f}")

        # Fit scaler on clipped data
        clipped_data = np.clip(data_np, self.clip_min_, self.clip_max_)
        self.scaler.fit(clipped_data.reshape(-1, 1))
        self.fitted = True

        return self

    def transform(self, data):
        """Transform data to [0, 2π] range"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        data_np = data.numpy() if torch.is_tensor(data) else data
        original_shape = data_np.shape

        # Clip outliers and scale
        clipped_data = np.clip(data_np, self.clip_min_, self.clip_max_)
        scaled_data = self.scaler.transform(clipped_data.reshape(-1, 1)).reshape(original_shape)

        return torch.FloatTensor(scaled_data) if torch.is_tensor(data) else scaled_data

    def inverse_transform(self, scaled_data):
        """Transform back from [0, 2π] to original space"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        scaled_np = scaled_data.numpy() if torch.is_tensor(scaled_data) else scaled_data
        original_shape = scaled_np.shape

        # Inverse transform
        original_data = self.scaler.inverse_transform(scaled_np.reshape(-1, 1)).reshape(original_shape)

        return torch.FloatTensor(original_data) if torch.is_tensor(scaled_data) else original_data

    def fit_transform(self, data):
        """Fit and transform in one step"""
        return self.fit(data).transform(data)

def compare_scaling_methods():
    """Compare amplification factors of different scaling approaches"""
    print("🔍 SCALING METHOD COMPARISON")
    print("=" * 50)

    # Load actual Y data to get realistic ranges
    try:
        Y_train = torch.load('Y_train_quantum.pt')
        if Y_train.dim() == 3:
            Y_train = Y_train[:, :, 0]  # Use first feature if 3D
        Y_flat = Y_train.flatten()

        print(f"📊 Loaded Y data: shape={Y_train.shape}")
        print(f"   Range: [{Y_flat.min():.4f}, {Y_flat.max():.4f}]")
        print(f"   Mean: {Y_flat.mean():.4f}, Std: {Y_flat.std():.4f}")

        # Test different scaling methods
        methods = {
            "Standard MinMax": {
                "min": Y_flat.min(),
                "max": Y_flat.max()
            },
            "5-95 Percentile": {
                "min": np.percentile(Y_flat.numpy(), 5),
                "max": np.percentile(Y_flat.numpy(), 95)
            },
            "10-90 Percentile": {
                "min": np.percentile(Y_flat.numpy(), 10),
                "max": np.percentile(Y_flat.numpy(), 90)
            },
            "1-99 Percentile": {
                "min": np.percentile(Y_flat.numpy(), 1),
                "max": np.percentile(Y_flat.numpy(), 99)
            }
        }

        target_range = 2 * np.pi

        print(f"\n📈 Amplification Factor Analysis:")
        print(f"   Target scaling range: [0, {target_range:.4f}]")
        print(f"   Lower amplification = better error control")
        print()

        for name, bounds in methods.items():
            original_range = bounds["max"] - bounds["min"]
            amplification = original_range / target_range

            print(f"   {name:18s}: range={original_range:.4f}, amplification={amplification:.2f}x")

    except FileNotFoundError:
        print("❌ Y_train_quantum.pt not found, using synthetic data")

        # Create synthetic data for demonstration
        Y_synthetic = torch.randn(1000, 14) * 0.05 + 0.02  # Realistic swaption-like data
        Y_flat = Y_synthetic.flatten()

        methods = {
            "Standard MinMax": {
                "min": Y_flat.min(),
                "max": Y_flat.max()
            },
            "5-95 Percentile": {
                "min": np.percentile(Y_flat.numpy(), 5),
                "max": np.percentile(Y_flat.numpy(), 95)
            }
        }

        for name, bounds in methods.items():
            original_range = bounds["max"] - bounds["min"]
            amplification = original_range / (2 * np.pi)
            print(f"   {name}: amplification={amplification:.2f}x")

def test_robust_scaler():
    """Test the RobustQuantumScaler with actual data"""
    print("\n🧪 TESTING ROBUST QUANTUM SCALER")
    print("=" * 50)

    try:
        # Load data
        Y_train = torch.load('Y_train_quantum.pt')
        if Y_train.dim() == 3:
            Y_train = Y_train[:, :, 0]  # Use price feature only

        print(f"📊 Testing with Y_train shape: {Y_train.shape}")

        # Test robust scaler
        robust_scaler = RobustQuantumScaler(percentile_low=5, percentile_high=95)

        # Fit and transform
        Y_scaled = robust_scaler.fit_transform(Y_train)
        print(f"   Scaled range: [{Y_scaled.min():.4f}, {Y_scaled.max():.4f}]")

        # Test inverse transform
        Y_recovered = robust_scaler.inverse_transform(Y_scaled)

        # Calculate reconstruction error
        reconstruction_error = torch.mean(torch.abs(Y_train - Y_recovered))
        print(f"   Reconstruction error: {reconstruction_error:.6f}")

        # Test error amplification with small perturbation
        small_error = torch.randn_like(Y_scaled) * 0.01  # Small error in scaled space
        Y_perturbed = Y_scaled + small_error
        Y_recovered_perturbed = robust_scaler.inverse_transform(Y_perturbed)

        amplified_error = torch.mean(torch.abs(Y_recovered - Y_recovered_perturbed))
        scaling_factor = amplified_error / torch.mean(torch.abs(small_error))

        print(f"   Error amplification test:")
        print(f"     Small error (scaled): {torch.mean(torch.abs(small_error)):.6f}")
        print(f"     Amplified error (orig): {amplified_error:.6f}")
        print(f"     Amplification factor: {scaling_factor:.2f}x")

        return robust_scaler

    except FileNotFoundError:
        print("❌ Cannot test - Y_train_quantum.pt not found")
        return None

if __name__ == "__main__":
    # Run comparison and tests
    compare_scaling_methods()
    robust_scaler = test_robust_scaler()

    if robust_scaler:
        print(f"\n✅ RobustQuantumScaler ready for use!")
        print(f"   Use: scaler = RobustQuantumScaler(percentile_low=5, percentile_high=95)")
        print(f"   Fit: Y_scaled = scaler.fit_transform(Y_train)")
        print(f"   Inverse: Y_orig = scaler.inverse_transform(Y_scaled)")