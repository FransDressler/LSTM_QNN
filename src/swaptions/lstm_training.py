#!/usr/bin/env python3
"""
🧠 CNN-LSTM Model Training for Swaption Prediction
=================================================

Trains a CNN-LSTM-Residual model on preprocessed data and saves:
- Model weights
- Training history
- Performance plots
- 8D latent vectors for quantum processing

Author: Claude & Frans
Date: 2025-11-30
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
LATENT_DIM = 8
INPUT_DIM = 3
OUTPUT_DIM = 14
HIDDEN_DIM = 128
DROPOUT = 0.2

device = torch.device("cpu")
print(f"🧠 CNN-LSTM TRAINING")
print("=" * 50)
print(f"Running on: {device}")

# ==========================================
# LOAD PREPROCESSED DATA
# ==========================================
print("📂 Loading preprocessed data...")

try:
    X_train_tensor = torch.load('data_output/X_train_lstm.pt')
    y_train_tensor = torch.load('data_output/y_train_lstm.pt')
    X_val_tensor = torch.load('data_output/X_val_lstm.pt')
    y_val_tensor = torch.load('data_output/y_val_lstm.pt')

    print(f"✅ Data loaded successfully:")
    print(f"   X_train: {X_train_tensor.shape}")
    print(f"   y_train: {y_train_tensor.shape}")
    print(f"   X_val: {X_val_tensor.shape}")
    print(f"   y_val: {y_val_tensor.shape}")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("   Run preprocessing.py first!")
    exit(1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# CNN-LSTM-RESIDUAL MODEL
# ==========================================
class CNNLSTMResidual(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, dropout):
        super(CNNLSTMResidual, self).__init__()

        # A. 1D Convolution: "The Spike Detector"
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # B. LSTM: "The Trend Follower"
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=1, batch_first=True)

        # C. Bottleneck
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)

        # D. Decoder Head
        self.head_dropout = nn.Dropout(dropout)
        self.classical_head = nn.Linear(latent_dim, output_dim)  # 8 → 14

    def forward(self, x):
        # x shape: (Batch, 60, 3)

        # 1. CNN Pass (Needs channels first: Batch, 3, 60)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu(self.conv1(x_conv))
        x_conv = self.pool(x_conv)       # Reduces length from 60 -> 30
        x_conv = x_conv.permute(0, 2, 1) # Back to (Batch, 30, 32)

        # 2. LSTM Pass
        lstm_out, _ = self.lstm(x_conv)
        last_hidden = lstm_out[:, -1, :]  # (Batch, 128)

        # 3. Bottleneck
        latent = self.bottleneck(last_hidden)  # (Batch, 8)

        # 4. Direct prediction of 14 future prices
        delta = self.classical_head(self.head_dropout(latent))  # (Batch, 14)

        # Use last known price as baseline
        last_known_price = x[:, -1, 0].unsqueeze(1)  # (Batch, 1)

        # Residual connection: baseline + deltas
        return last_known_price + delta  # (Batch, 14)

    def get_latent(self, x):
        """Extract 8D latent representation for quantum processing."""
        with torch.no_grad():
            x_conv = x.permute(0, 2, 1)
            x_conv = self.relu(self.conv1(x_conv))
            x_conv = self.pool(x_conv)
            x_conv = x_conv.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x_conv)
            return self.bottleneck(lstm_out[:, -1, :])  # Returns (Batch, 8)

# ==========================================
# MODEL SETUP
# ==========================================
model = CNNLSTMResidual(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, OUTPUT_DIM, DROPOUT).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

print("✅ Model Initialized:")
print(f"   Input: (batch, 60, 3)")
print(f"   Output: (batch, 14)")
print(f"   Latent: (batch, 8)")

# ==========================================
# TRAINING LOOP
# ==========================================
print(f"\n🚀 Starting Training for {EPOCHS} epochs...")

history = {'train': [], 'val': []}
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        # Clip gradients to ensure stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    history['train'].append(avg_train)
    history['val'].append(avg_val)
    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), 'data_output/best_lstm_model.pth')
        saved_msg = "<- Best"
    else:
        saved_msg = ""

    print(f"Epoch {epoch+1:02d} | Train: {avg_train:.5f} | Val: {avg_val:.5f} {saved_msg}")

# Load best weights
model.load_state_dict(torch.load('data_output/best_lstm_model.pth'))
print("✅ Training Complete.")

# ==========================================
# SAVE TRAINING RESULTS
# ==========================================
print("\n💾 Saving training results...")

# Save training history
with open('data_output/lstm_training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

# Save final model (not just state_dict)
torch.save(model, 'data_output/lstm_model_complete.pth')

print(f"✅ Saved:")
print(f"   - best_lstm_model.pth: Best model weights")
print(f"   - lstm_model_complete.pth: Complete model")
print(f"   - lstm_training_history.json: Training history")

# ==========================================
# PERFORMANCE EVALUATION & PLOTS
# ==========================================
print("\n📊 Creating performance plots...")

# Create a 2x2 grid of plots
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# ---------------------------------------
# Plot 1: Training Dynamics (Log Scale)
# ---------------------------------------
axs[0, 0].plot(history['train'], label='Train Loss', linewidth=2)
axs[0, 0].plot(history['val'], label='Val Loss', linewidth=2)
axs[0, 0].set_title("1. Learning Curve (Log Scale)", fontsize=14, fontweight='bold')
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("MSE (Log)")
axs[0, 0].set_yscale('log')
axs[0, 0].legend()
axs[0, 0].grid(True, which="both", ls="-", alpha=0.3)

# Add final loss values as text
final_train = history['train'][-1]
final_val = history['val'][-1]
axs[0, 0].text(0.02, 0.98, f'Final Train: {final_train:.5f}\nFinal Val: {final_val:.5f}',
               transform=axs[0, 0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ---------------------------------------
# Plot 2: Time Series Forecast (Sample View)
# ---------------------------------------
model.eval()
x_val_sample, y_val_sample = next(iter(val_loader))
x_val_sample, y_val_sample = x_val_sample.to(device), y_val_sample.to(device)
with torch.no_grad():
    y_pred_sample = model(x_val_sample)

# Pick 3 random indices from the batch to plot
indices = [0, 5, 10]
colors = ['blue', 'red', 'green']
days = np.arange(1, 15)

for i, idx in enumerate(indices):
    if idx < len(y_val_sample):
        axs[0, 1].plot(days, y_val_sample[idx].cpu().numpy(), color=colors[i],
                      marker='o', alpha=0.7, label=f'Actual #{idx}', linewidth=2)
        axs[0, 1].plot(days, y_pred_sample[idx].cpu().numpy(), color=colors[i],
                      linestyle='--', marker='x', label=f'Pred #{idx}', linewidth=2)

axs[0, 1].set_title("2. Forecast vs Actual (3 Random Samples)", fontsize=14, fontweight='bold')
axs[0, 1].set_xlabel("Future Days (1-14)")
axs[0, 1].set_ylabel("Normalized Price")
axs[0, 1].legend(fontsize='small')
axs[0, 1].grid(True, alpha=0.3)

# ---------------------------------------
# Plot 3: Correlation Check (All Validation Data)
# ---------------------------------------
model.eval()
all_actual = []
all_pred = []

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred_batch = model(x_batch)
        all_actual.append(y_batch.cpu().numpy())
        all_pred.append(pred_batch.cpu().numpy())

all_actual = np.concatenate(all_actual, axis=0)
all_pred = np.concatenate(all_pred, axis=0)

# Use day 1 predictions for correlation
actual_day1 = all_actual[:, 0]
pred_day1 = all_pred[:, 0]

correlation = np.corrcoef(actual_day1, pred_day1)[0, 1]

axs[1, 0].scatter(actual_day1, pred_day1, alpha=0.6, color='purple', s=20)
axs[1, 0].plot([actual_day1.min(), actual_day1.max()], [actual_day1.min(), actual_day1.max()],
               'k--', lw=2, label='Perfect Correlation')
axs[1, 0].set_title(f"3. Correlation: Predicted vs Actual (Day 1)\nR = {correlation:.3f}",
                   fontsize=14, fontweight='bold')
axs[1, 0].set_xlabel("Actual Normalized Price")
axs[1, 0].set_ylabel("Predicted Normalized Price")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# ---------------------------------------
# Plot 4: Latent Space Distribution
# ---------------------------------------
latent_vectors = model.get_latent(x_val_sample).cpu().numpy()

axs[1, 1].hist(latent_vectors.flatten(), bins=30, color='orange',
               edgecolor='black', alpha=0.7, density=True)
axs[1, 1].set_title("4. Latent Features Distribution (8D Quantum Inputs)",
                   fontsize=14, fontweight='bold')
axs[1, 1].set_xlabel("Feature Value")
axs[1, 1].set_ylabel("Density")
axs[1, 1].grid(True, alpha=0.3)

# Add statistics
latent_mean = np.mean(latent_vectors)
latent_std = np.std(latent_vectors)
axs[1, 1].text(0.02, 0.98, f'Mean: {latent_mean:.3f}\nStd: {latent_std:.3f}',
               transform=axs[1, 1].transAxes, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.suptitle('🧠 CNN-LSTM Model Performance Analysis', fontsize=16, fontweight='bold')
plt.savefig('data_output/lstm_performance_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# FINAL SUMMARY
# ==========================================
print(f"\n🎯 LSTM TRAINING COMPLETE!")
print(f"   📊 Best Validation Loss: {best_val_loss:.6f}")
print(f"   📊 Final Train Loss: {final_train:.6f}")
print(f"   📊 Correlation (Day 1): {correlation:.3f}")
print(f"   📊 Latent Features: Mean={latent_mean:.3f}, Std={latent_std:.3f}")
print(f"\n💾 All outputs saved in: ./data_output/")
print(f"   🚀 Ready for quantum preprocessing!")

# Extract and save latent vectors for quantum processing
print(f"\n🔄 Extracting latent vectors for quantum processing...")
train_latents = []
val_latents = []

model.eval()
with torch.no_grad():
    for x_batch, _ in train_loader:
        latents = model.get_latent(x_batch.to(device))
        train_latents.append(latents.cpu())

    for x_batch, _ in val_loader:
        latents = model.get_latent(x_batch.to(device))
        val_latents.append(latents.cpu())

train_latents = torch.cat(train_latents, dim=0)
val_latents = torch.cat(val_latents, dim=0)

torch.save(train_latents, 'data_output/train_latents_8d.pt')
torch.save(val_latents, 'data_output/val_latents_8d.pt')

print(f"✅ 8D Latent vectors saved:")
print(f"   - train_latents_8d.pt: {train_latents.shape}")
print(f"   - val_latents_8d.pt: {val_latents.shape}")
print(f"   🌌 Ready for quantum scaling pipeline!")