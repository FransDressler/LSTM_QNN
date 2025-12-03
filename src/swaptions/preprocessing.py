import pandas as pd
import numpy as np
import re
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---



# m: How many past days the model sees to make a prediction
DAYS_BASIS = 60 

# n: How many future days the model tries to predict
DAYS_FOR_PREDICTION = 14 

# Stride: How many days to slide the window forward (1 = max data)
STRIDE = 1 

# File Path
FILE_PATH = '../Daten.csv'

# Load the dataset
df = pd.read_csv(FILE_PATH, sep=';', decimal=',', quotechar='"')
df = df.drop(columns=['Date'])

print(df.columns[:5])
print(f"\nDataset Shape: {df.shape}\n")

# We need to extract the `Tenor` and `Maturity` from the column headers so we can include them as 
# features in our input X.

def parse_column_name(col_name):
    """
    Extracts Tenor and Maturity values from string.
    Format expected: 'Tenor : 1; Maturity : 0.5'
    """
    # Regex to find numbers (integers or floats)
    tenor_match = re.search(r'Tenor\s*:\s*([0-9.]+)', col_name)
    maturity_match = re.search(r'Maturity\s*:\s*([0-9.]+)', col_name)
    
    if tenor_match and maturity_match:
        return float(tenor_match.group(1)), float(maturity_match.group(1))
    else:
        return None, None

# Dictionary to store metadata for each column index
col_metadata = {}

valid_columns = []
for col in df.columns:
    t, m = parse_column_name(col)
    if t is not None:
        col_metadata[col] = (t, m)
        valid_columns.append(col)
    else:
        print(f"Skipping column (could not parse): {col}")

# Filter df to only valid columns (just in case)
df = df[valid_columns]

print(f"Successfully parsed {len(col_metadata)} columns.")
print(f"Example Metadata for '{valid_columns[0]}': {col_metadata[valid_columns[0]]}")
print(df)

def create_grand_dataset(df, seed=42):
      """
      Create sliding window dataset for swaption prediction.

      Returns:
          X_train: (N, 60, 3) - Input sequences [price, tenor, maturity]
          y_train: (N, 14) - Target prices (only price values)
          X_val: (N, 60, 3) - Validation inputs
          y_val: (N, 14) - Validation targets (only price values)
          stats: Dict with normalization statistics for inverse transformation
      """

      # --- Part 1: Helper to Parse Headers ---
      def parse_column_name(col_name):
          """Extract tenor and maturity from column name."""
          tenor_match = re.search(r'Tenor\s*:\s*([0-9.]+)', col_name)
          maturity_match = re.search(r'Maturity\s*:\s*([0-9.]+)', col_name)
          if tenor_match and maturity_match:
              return float(tenor_match.group(1)), float(maturity_match.group(1))
          return None, None

      # Constants
      SEQ_TOTAL = DAYS_BASIS + DAYS_FOR_PREDICTION  # 60 + 14 = 74
      SEQ_X = DAYS_BASIS  # 60

      all_sequences = []
      print("Processing columns...")

      # --- Part 2: Create Sequences for Each Column ---
      for col_idx, col in enumerate(df.columns):
          # 1. Extract metadata
          tenor, maturity = parse_column_name(col)
          if tenor is None:
              print(f"Skipping column {col_idx}: Could not parse tenor/maturity")
              continue

          # 2. Get clean price data
          raw_values = df[col].dropna().values
          if len(raw_values) < SEQ_TOTAL:
              print(f"Skipping column {col_idx}: Not enough data ({len(raw_values)} < {SEQ_TOTAL})")
              continue

          # 3. Create feature matrix for this column
          t_len = len(raw_values)
          combined_features = np.column_stack([
              raw_values,                          # Feature 0: Price
              np.full(t_len, tenor),              # Feature 1: Tenor (constant)
              np.full(t_len, maturity)            # Feature 2: Maturity (constant)
          ])  # Shape: (t_len, 3)

          # 4. Sliding window over this column
          for i in range(len(combined_features) - SEQ_TOTAL + 1):
              window = combined_features[i : i + SEQ_TOTAL]  # Shape: (74, 3)
              all_sequences.append(window)

          if (col_idx + 1) % 50 == 0:
              print(f"   Processed {col_idx + 1}/{len(df.columns)} columns...")

      # --- Part 3: Convert to Array and Shuffle ---
      grand_dataset = np.array(all_sequences)  # Shape: (N, 74, 3)
      print(f"Grand Dataset Created. Shape: {grand_dataset.shape}")

      np.random.seed(seed)
      np.random.shuffle(grand_dataset)
      print("Dataset shuffled with fixed seed.")

      # --- Part 4: Train/Validation Split ---
      train_size = int(len(grand_dataset) * 0.8)
      train_set = grand_dataset[:train_size]
      val_set = grand_dataset[train_size:]
      print(f"Split: Train={train_size}, Val={len(val_set)}")

      # --- Part 5: Calculate Normalization Stats (Train Only) ---
      print("Calculating normalization statistics on training set only...")

      stats = {
          'price': {
              'mean': np.mean(train_set[:, :, 0]),  # All price values in training
              'std':  np.std(train_set[:, :, 0])
          },
          'tenor': {
              'mean': np.mean(train_set[:, :, 1]),  # All tenor values
              'std':  np.std(train_set[:, :, 1])
          },
          'maturity': {
              'mean': np.mean(train_set[:, :, 2]),  # All maturity values
              'std':  np.std(train_set[:, :, 2])
          }
      }

      print(f"   Price: mean={stats['price']['mean']:.6f}, std={stats['price']['std']:.6f}")
      print(f"   Tenor: mean={stats['tenor']['mean']:.2f}, std={stats['tenor']['std']:.2f}")
      print(f"   Maturity: mean={stats['maturity']['mean']:.2f}, std={stats['maturity']['std']:.2f}")

      # --- Part 6: Apply Normalization ---
      def normalize_data(data, stats):
          """Apply z-score normalization using provided stats."""
          data_norm = data.copy().astype(np.float32)

          # Normalize each feature
          data_norm[:, :, 0] = (data[:, :, 0] - stats['price']['mean']) / stats['price']['std']
          data_norm[:, :, 1] = (data[:, :, 1] - stats['tenor']['mean']) / stats['tenor']['std']
          data_norm[:, :, 2] = (data[:, :, 2] - stats['maturity']['mean']) / stats['maturity']['std']

          return data_norm

      # Apply same normalization to both train and validation
      train_set_norm = normalize_data(train_set, stats)
      val_set_norm = normalize_data(val_set, stats)

      print("Normalization applied to both train and validation sets.")

      # --- Part 7: Split X and Y (CORRECTED) ---
      # X: Input sequences (60 days, all 3 features)
      X_train = train_set_norm[:, :SEQ_X, :]        # Shape: (N_train, 60, 3)
      X_val = val_set_norm[:, :SEQ_X, :]            # Shape: (N_val, 60, 3)

      # Y: Target sequences (14 future days, ONLY PRICE!)
      y_train = train_set_norm[:, SEQ_X:, 0]        # Shape: (N_train, 14) ✅ Only prices!
      y_val = val_set_norm[:, SEQ_X:, 0]            # Shape: (N_val, 14) ✅ Only prices!

      print(f"\n✅ Final Dataset Shapes:")
      print(f"   X_train: {X_train.shape} (input sequences)")
      print(f"   y_train: {y_train.shape} (target prices)")
      print(f"   X_val: {X_val.shape} (validation inputs)")
      print(f"   y_val: {y_val.shape} (validation targets)")

      # Quick sanity checks
      print(f"\n🔍 Data Quality Checks:")
      print(f"   X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
      print(f"   y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")
      print(f"   Any NaN in X_train: {np.isnan(X_train).any()}")
      print(f"   Any NaN in y_train: {np.isnan(y_train).any()}")

      return X_train, y_train, X_val, y_val, stats

# --- Usage ---
X_train, y_train, X_val, y_val, normalization_stats = create_grand_dataset(df, seed=42)

# Convert to PyTorch tensors for model training
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)  # Now correct shape: (N, 14)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)      # Now correct shape: (N, 14)

print(f"\n🚀 Ready for PyTorch:")
print(f"   X_train_tensor: {X_train_tensor.shape}")
print(f"   y_train_tensor: {y_train_tensor.shape}")  # Should be (N, 14)
print(f"   Normalization stats saved for inverse transformation")

import matplotlib.pyplot as plt
import numpy as np

# Set up a plot with 3 columns (one for each feature)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Feature Distributions: Train vs Validation (Normalized)', fontsize=16)

# The names of your 3 features in order
feature_names = ['Rate (Z-Score)', 'Tenor (Normalized)', 'Maturity (Normalized)']

for i in range(3):
    # Flatten the arrays: We want to see the distribution of ALL values 
    # across ALL samples and ALL time steps.
    train_data = X_train[:, :, i].flatten()
    val_data = X_val[:, :, i].flatten()
    
    # Plot Histogram
    # density=True ensures we compare shapes, even though Train is bigger than Val
    axes[i].hist(train_data, bins=50, alpha=0.5, color='blue', label='Train', density=True)
    axes[i].hist(val_data, bins=50, alpha=0.5, color='orange', label='Validation', density=True)
    
    # Aesthetics
    axes[i].set_title(feature_names[i])
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Calculate simple stats to display
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    axes[i].set_xlabel(f"Mean: {train_mean:.2f} | Std: {train_std:.2f}")

# Create output directory for organized storage
import os
os.makedirs('data_output', exist_ok=True)
print(f"\n💾 Saving processed data to 'data_output/' folder...")

# Save processed PyTorch tensors
torch.save(X_train_tensor, 'data_output/X_train_lstm.pt')
torch.save(y_train_tensor, 'data_output/y_train_lstm.pt')
torch.save(X_val_tensor, 'data_output/X_val_lstm.pt')
torch.save(y_val_tensor, 'data_output/y_val_lstm.pt')

# Save normalization statistics for inverse transformation
with open('data_output/normalization_stats.json', 'w') as f:
    json.dump(normalization_stats, f, indent=2)

print(f"✅ Data saved:")
print(f"   - X_train_lstm.pt: {X_train_tensor.shape}")
print(f"   - y_train_lstm.pt: {y_train_tensor.shape}")
print(f"   - X_val_lstm.pt: {X_val_tensor.shape}")
print(f"   - y_val_lstm.pt: {y_val_tensor.shape}")
print(f"   - normalization_stats.json: Statistics for inverse transformation")

plt.tight_layout()

# Save the feature distribution plot
plt.savefig('data_output/feature_distributions.png', dpi=150, bbox_inches='tight')
print(f"   - feature_distributions.png: Feature distribution plots")

plt.show()

print(f"\n🎯 LSTM Preprocessing Complete!")
print(f"   📁 All outputs saved in: ./data_output/")
print(f"   🚀 Ready for LSTM training!")

