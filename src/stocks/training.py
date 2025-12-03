#!/usr/bin/env python3
"""
🧠 Stock LSTM Training (Relative Features)
=========================================

Trains LSTM on relative stock features + company characteristics.

Author: Claude & Frans
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load configuration from JSON
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Extract configuration values
DATA_STORAGE = CONFIG['data_storage']

class XLSTMCell(nn.Module):
    """Extended LSTM Cell with exponential gating and memory mixing"""
    def __init__(self, input_size, hidden_size, use_exponential_gates=True):
        super(XLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_exponential_gates = use_exponential_gates

        # Standard LSTM gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Extended LSTM components
        # Memory mixing parameters
        self.memory_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Exponential gating normalization
        if use_exponential_gates:
            self.exp_gate_norm = nn.LayerNorm(hidden_size)

        # Enhanced memory retention
        self.retention_factor = nn.Parameter(torch.ones(hidden_size) * 0.9)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=-1)

        # Standard LSTM gates
        f_gate = torch.sigmoid(self.forget_gate(combined))
        i_gate = torch.sigmoid(self.input_gate(combined))
        o_gate = torch.sigmoid(self.output_gate(combined))
        c_candidate = torch.tanh(self.candidate_gate(combined))

        # Extended LSTM: Memory mixing gate
        m_gate = torch.sigmoid(self.memory_gate(combined))

        # Enhanced cell state computation with memory mixing
        c_new = f_gate * c_prev + i_gate * c_candidate

        # Memory retention mechanism
        c_new = c_new * self.retention_factor + (1 - self.retention_factor) * c_prev

        # Memory mixing: blend old and new cell states
        c_new = m_gate * c_new + (1 - m_gate) * c_prev

        # Exponential gating for output (optional)
        if self.use_exponential_gates:
            o_gate_exp = torch.exp(self.exp_gate_norm(o_gate)) / (1 + torch.exp(self.exp_gate_norm(o_gate)))
            h_new = o_gate_exp * torch.tanh(c_new)
        else:
            h_new = o_gate * torch.tanh(c_new)

        return h_new, c_new

class StockXLSTMTS(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, latent_dim=16, output_dim=14, dropout=0.2, num_layers=2):
        super(StockXLSTMTS, self).__init__()

        self.relative_dim = 9  # Returns, volume, volatility, VIX, fear/greed, etc.
        self.company_dim = 8   # Market cap, beta, etc.
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN for relative features preprocessing
        self.conv1 = nn.Conv1d(self.relative_dim, 24, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn1 = nn.BatchNorm1d(24)

        # Company embedding
        self.company_embedding = nn.Linear(self.company_dim, 20)

        # Combined feature projection
        combined_dim = 24 + 20  # 44 total after CNN + embedding
        self.feature_projection = nn.Linear(44, hidden_dim)

        # X-LSTM layers
        self.xlstm_layers = nn.ModuleList([
            XLSTMCell(hidden_dim if i > 0 else hidden_dim, hidden_dim, use_exponential_gates=True)
            for i in range(num_layers)
        ])

        # Layer normalization for each X-LSTM layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])

        # Attention mechanism for sequence aggregation
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)

        # Output projection layers
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(latent_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)

    def init_hidden(self, batch_size, device):
        """Initialize hidden states for X-LSTM layers"""
        hidden_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
            hidden_states.append((h, c))
        return hidden_states

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        device = x.device

        # Split features
        relative_features = x[:, :, :self.relative_dim]  # (batch, 60, 9)
        company_features = x[:, :, self.relative_dim:]   # (batch, 60, 8)

        # Process relative features with CNN
        rel_conv = relative_features.permute(0, 2, 1)    # (batch, 9, 60)
        rel_conv = self.relu(self.conv1(rel_conv))       # (batch, 24, 60)
        rel_conv = self.bn1(rel_conv)
        rel_conv = self.pool(rel_conv)                   # (batch, 24, 30)
        rel_conv = rel_conv.permute(0, 2, 1)             # (batch, 30, 24)

        # Process company features (downsample to match CNN output)
        company_emb = self.company_embedding(company_features[:, :30, :])  # (batch, 30, 20)
        company_emb = self.relu(company_emb)

        # Combine features
        combined = torch.cat([rel_conv, company_emb], dim=2)  # (batch, 30, 44)

        # Project to hidden dimension
        combined = self.feature_projection(combined)  # (batch, 30, hidden_dim)

        # Initialize hidden states for all X-LSTM layers
        hidden_states = self.init_hidden(batch_size, device)

        # Process through X-LSTM layers
        x_seq = combined
        all_outputs = []

        for t in range(x_seq.size(1)):  # Iterate through sequence
            x_t = x_seq[:, t, :]  # (batch, hidden_dim)

            for layer_idx, xlstm_layer in enumerate(self.xlstm_layers):
                # Apply X-LSTM cell
                h_new, c_new = xlstm_layer(x_t, hidden_states[layer_idx])

                # Apply layer normalization
                h_new = self.layer_norms[layer_idx](h_new)

                # Apply dropout
                h_new = self.dropout_layers[layer_idx](h_new)

                # Update hidden state for this layer
                hidden_states[layer_idx] = (h_new, c_new)

                # Use output as input for next layer
                x_t = h_new

            all_outputs.append(x_t.unsqueeze(1))  # (batch, 1, hidden_dim)

        # Concatenate all timestep outputs
        sequence_output = torch.cat(all_outputs, dim=1)  # (batch, 30, hidden_dim)

        # Apply self-attention for sequence aggregation
        attended_output, _ = self.attention(sequence_output, sequence_output, sequence_output)

        # Use the last timestep for prediction (or could use attention weights)
        final_hidden = attended_output[:, -1, :]  # (batch, hidden_dim)

        # Output projection
        latent = self.bottleneck(final_hidden)  # (batch, latent_dim)
        latent = self.bn2(latent)
        output = self.output_layer(self.dropout(latent))  # (batch, 14)

        return output

def main():
    print("🧠 STOCK X-LSTM-TS TRAINING (PERCENTAGE TARGETS)")
    print("=" * 52)

    device = torch.device("cpu")

    # Load data using config paths
    try:
        X_train = torch.load(DATA_STORAGE['preprocessed_data_dir'] + 'X_train_mixed.pt')
        y_train = torch.load(DATA_STORAGE['preprocessed_data_dir'] + 'y_train_mixed.pt')
        X_val = torch.load(DATA_STORAGE['preprocessed_data_dir'] + 'X_val_mixed.pt')
        y_val = torch.load(DATA_STORAGE['preprocessed_data_dir'] + 'y_val_mixed.pt')

        print(f"✅ Data loaded: {X_train.shape}, {X_val.shape}")

        # Validate feature dimensions
        expected_features = 17  # 9 relative + 8 asset
        if X_train.shape[-1] != expected_features:
            print(f"❌ Feature dimension mismatch! Expected {expected_features}, got {X_train.shape[-1]}")
            print("   Run preprocessing.py again to generate data with sentiment features")
            return
        else:
            print(f"✅ Feature validation passed: {expected_features} features (9 relative + 8 asset)")

        print(f"📊 Target stats: mean={y_train.mean():.2f}%, std={y_train.std():.2f}%")
        print(f"📊 Target range: [{y_train.min():.1f}%, {y_train.max():.1f}%]")
    except FileNotFoundError:
        print("❌ Run preprocessing.py first!")
        return

    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model with X-LSTM-TS architecture
    model = StockXLSTMTS(
        input_dim=17,
        hidden_dim=96,      # Increased for X-LSTM capability
        latent_dim=24,      # Increased bottleneck
        output_dim=14,
        dropout=0.3,        # Higher dropout for regularization
        num_layers=3        # More layers for complex patterns
    ).to(device)

    # X-LSTM optimized training parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=2e-4)  # Lower LR for stability
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, factor=0.6)

    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training with early stopping
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    patience = 12  # More patience for X-LSTM convergence
    no_improve = 0

    for epoch in range(100):  # More epochs for X-LSTM convergence
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # Save best and check for early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), '../../models/best_stock_xlstm.pth')
            torch.save(model, '../../models/stock_xlstm_complete.pth')
            saved_msg = " <- Best"
        else:
            no_improve += 1
            saved_msg = ""

        # Convert MSE to RMSE for better percentage interpretation
        rmse_train = np.sqrt(avg_train)
        rmse_val = np.sqrt(avg_val)
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.5f} (RMSE: {rmse_train:.2f}%) | Val: {avg_val:.5f} (RMSE: {rmse_val:.2f}%){saved_msg}")

        # Early stopping check
        if no_improve >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Save results
    with open('../../models/stock_xlstm_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Create training plot
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train']) + 1)
    plt.plot(epochs, [np.sqrt(loss) for loss in history['train']], 'b-', label='Train RMSE (%)')
    plt.plot(epochs, [np.sqrt(loss) for loss in history['val']], 'r-', label='Val RMSE (%)')
    plt.title('📈 X-LSTM-TS Training Progress (RMSE in Percentage Points)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../../models/xlstm_training_progress.png', dpi=200, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.5f} (RMSE: {np.sqrt(best_val_loss):.2f}%)")
    print(f"   Models saved to: ../../models/")
    print(f"   Training plot: ../../models/xlstm_training_progress.png")

if __name__ == "__main__":
    main()