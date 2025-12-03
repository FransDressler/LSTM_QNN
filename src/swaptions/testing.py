#!/usr/bin/env python3
"""
🌌 Quantum-Enhanced Swaption Testing
===================================

Tests the complete quantum-enhanced swaption prediction pipeline.

Author: Claude & Frans
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths for imports
sys.path.append('../../src/quantum')
sys.path.append('../../src/utils')

def main():
    print("🌌 QUANTUM SWAPTION TESTING")
    print("=" * 40)

    # Load quantum model
    try:
        quantum_model = torch.load('../../models/best_quantum_swaption_model.pth',
                                  weights_only=False, map_location='cpu')
        quantum_model.eval()
        print("✅ Quantum model loaded")
    except:
        print("❌ No quantum model found. Run unified_quantum_swaptions.py first!")
        return

    # Load test data
    try:
        X_val = torch.load('../../data/X_val_lstm.pt', weights_only=True)
        y_val = torch.load('../../data/y_val_lstm.pt', weights_only=True)
        print(f"✅ Test data loaded: {X_val.shape}")
    except:
        print("❌ No test data found. Run preprocessing.py first!")
        return

    # Test on sample data
    with torch.no_grad():
        # Take first batch
        test_batch = X_val[:12]
        test_targets = y_val[:12]

        # Get predictions
        predictions = quantum_model(test_batch)

        # Plot results
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle('🌌 Quantum-Enhanced Swaption Predictions', fontsize=16)

        for i in range(12):
            row, col = i // 4, i % 4
            ax = axes[row, col]

            days = range(1, 15)

            # Plot actual vs predicted
            ax.plot(days, test_targets[i].numpy(), 'o-', label='Actual', color='black', linewidth=2)
            ax.plot(days, predictions[i].numpy(), 's-', label='Quantum Pred', color='blue', linewidth=2)

            # Calculate correlation
            corr = np.corrcoef(test_targets[i].numpy(), predictions[i].numpy())[0, 1]

            ax.set_title(f'Sample {i+1}\nCorrelation: {corr:.3f}')
            ax.set_xlabel('Future Days')
            ax.set_ylabel('Normalized Price')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../../models/quantum_swaption_test_results.png', dpi=200, bbox_inches='tight')
        plt.show()

        # Calculate overall metrics
        all_actual = test_targets.numpy().flatten()
        all_pred = predictions.numpy().flatten()
        overall_corr = np.corrcoef(all_actual, all_pred)[0, 1]
        mse = np.mean((all_actual - all_pred)**2)

        print(f"\n📊 QUANTUM MODEL PERFORMANCE:")
        print(f"   Overall Correlation: {overall_corr:.3f}")
        print(f"   MSE: {mse:.6f}")
        print(f"   Results saved to: ../../models/quantum_swaption_test_results.png")

if __name__ == "__main__":
    main()