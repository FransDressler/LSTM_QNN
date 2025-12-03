import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Machine Learning Imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb # <--- NEU: XGBoost importiert
import pandas_datareader.data as web

# Deep Learning Imports (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# KONFIGURATION
# ==========================================
FILENAME = "deutsche_bank_hybrid_compare.csv"
MODEL_FILE_RF = "dbk_rf_model.pkl"
TICKER = "DBK.DE"

LOOKBACK = 60       # Input: 60 Tage
FORECAST_STEPS = 10 # 10 Tage Simulation
TEST_DAYS_TOTAL = 70 

# NN Hyperparameter
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. DATEN BESCHAFFEN
# ==========================================
def get_data_robust():
    min_needed = 200 
    if os.path.exists(FILENAME):
        try:
            df = pd.read_csv(FILENAME, index_col='Date', parse_dates=True)
            if len(df) < min_needed:
                os.remove(FILENAME)
                df = None
        except:
            os.remove(FILENAME)
            df = None
    else:
        df = None

    if df is None:
        print(f"⬇️ Lade Historie von Stooq...")
        start = datetime(2010, 1, 1)
        end = datetime.now()
        try:
            df = web.DataReader(TICKER, 'stooq', start, end)
            df = df.sort_index()[['Close']].dropna()
            df.to_csv(FILENAME)
        except Exception as e:
            print(f"❌ Fehler: {e}")
            exit()
    return df

df = get_data_robust()
print(f"✅ Daten geladen: {len(df)} Tage.")

# Skalieren
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df.values)

# ==========================================
# 2. DATEN VORBEREITUNG
# ==========================================
train_raw = data_scaled[:-TEST_DAYS_TOTAL]
test_raw = data_scaled[-TEST_DAYS_TOTAL:]

X_train, y_train = [], []
for i in range(len(train_raw) - LOOKBACK - 1):
    X_train.append(train_raw[i:(i + LOOKBACK)])
    y_train.append(train_raw[i + LOOKBACK]) 

X_train = np.array(X_train)
y_train = np.array(y_train)

# Flatten für RF
nsamples, nx, ny = X_train.shape
X_train_flat = X_train.reshape((nsamples, nx*ny))
y_train_flat = y_train.ravel()

# ==========================================
# 3. RANDOM FOREST (BASIS)
# ==========================================
print("\n=== 🌲 1. Trainiere Random Forest (Basis) ===")
start_rf = time.time()

rf = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_train_flat, y_train_flat)

print(f"✅ RF fertig in {time.time()-start_rf:.2f}s")

# Residuen berechnen (RF Fehler lernen)
rf_train_pred = rf.predict(X_train_flat)
residuals = y_train_flat - rf_train_pred 

# ==========================================
# 4. NEURONALES NETZ (KORREKTUR)
# ==========================================
print("\n=== 🧠 2. Trainiere Neuronales Netz (Korrektur) ===")

X_torch = torch.from_numpy(X_train_flat).float().to(DEVICE)
y_resid_torch = torch.from_numpy(residuals).float().unsqueeze(1).to(DEVICE)

train_ds = TensorDataset(X_torch, y_resid_torch)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

class ErrorCorrector(nn.Module):
    def __init__(self, input_dim):
        super(ErrorCorrector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

nn_model = ErrorCorrector(input_dim=LOOKBACK).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=LEARNING_RATE)

# Training Loop
loss_hist = []
for epoch in range(EPOCHS):
    nn_model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    loss_hist.append(avg_loss)

print("✅ NN fertig trainiert.")

# ==========================================
# 4.5. XGBOOST META-TRAINING (NEU!)
# ==========================================
print("\n=== 🚀 3. Trainiere XGBoost (Stacking/Meta-Learner) ===")

# Wir müssen die Trainingsdaten für den XGBoost erstellen.
# Input: [RF_Vorhersage, Hybrid_Vorhersage]
# Target: Echte Trainingsdaten

# 1. Hybrid-Vorhersagen für das Training-Set generieren
nn_model.eval()
with torch.no_grad():
    nn_train_correction = nn_model(X_torch).cpu().numpy().ravel()

hybrid_train_pred = rf_train_pred + nn_train_correction

# 2. Features für XGBoost stapeln (RF Preds + Hybrid Preds)
X_meta_train = np.column_stack((rf_train_pred, hybrid_train_pred))
y_meta_train = y_train_flat

# 3. XGBoost Modell definieren und trainieren
xgb_model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=3, 
    random_state=42, 
    n_jobs=-1
)
xgb_model.fit(X_meta_train, y_meta_train)
print("✅ XGBoost fertig trainiert.")

# ==========================================
# 5. DREIFACHE SIMULATION (RF vs. HYBRID vs. XGBOOST)
# ==========================================
print("\n=== 🔮 Simuliere Zukunft (Vergleich: 3 Wege) ===")

# Wir brauchen DREI separate Inputs, da sich die Pfade unterscheiden werden
input_rf_only = test_raw[:LOOKBACK].reshape(1, -1) 
input_hybrid  = test_raw[:LOOKBACK].reshape(1, -1)
input_xgb     = test_raw[:LOOKBACK].reshape(1, -1) # NEU: Eigener Pfad für XGB

preds_rf_only = []
preds_hybrid = []
preds_xgb = [] # NEU

nn_model.eval()

for i in range(FORECAST_STEPS):
    # --- PFAD A: Rein Random Forest ---
    pred_pure_rf = rf.predict(input_rf_only)[0]
    preds_rf_only.append(pred_pure_rf)
    
    # Update Pfad A
    new_win_rf = np.append(input_rf_only[0, 1:], pred_pure_rf)
    input_rf_only = new_win_rf.reshape(1, -1)
    
    # --- PFAD B: Hybrid (RF + NN) ---
    pred_base_rf_h = rf.predict(input_hybrid)[0]
    tens_h = torch.from_numpy(input_hybrid).float().to(DEVICE)
    with torch.no_grad():
        pred_err_h = nn_model(tens_h).item()
    final_hybrid = pred_base_rf_h + pred_err_h
    preds_hybrid.append(final_hybrid)
    
    # Update Pfad B
    new_win_hyb = np.append(input_hybrid[0, 1:], final_hybrid)
    input_hybrid = new_win_hyb.reshape(1, -1)

    # --- PFAD C: XGBoost (Stacking) --- (NEU!)
    # 1. Wir brauchen die RF-Meinung auf Basis der XGB-Historie
    rf_feat = rf.predict(input_xgb)[0]
    
    # 2. Wir brauchen die Hybrid-Meinung auf Basis der XGB-Historie
    tens_x = torch.from_numpy(input_xgb).float().to(DEVICE)
    with torch.no_grad():
        nn_feat = nn_model(tens_x).item()
    hyb_feat = rf_feat + nn_feat
    
    # 3. XGBoost entscheidet basierend auf beiden Meinungen
    meta_features = np.array([[rf_feat, hyb_feat]])
    final_xgb = xgb_model.predict(meta_features)[0]
    preds_xgb.append(final_xgb)
    
    # Update Pfad C (Autoregressiv mit XGB Prognose)
    new_win_xgb = np.append(input_xgb[0, 1:], final_xgb)
    input_xgb = new_win_xgb.reshape(1, -1)

# ==========================================
# 6. PLOT
# ==========================================
# Echte Daten
true_targets = test_raw[LOOKBACK:]
real_price = scaler.inverse_transform(true_targets)

# Prognosen Rückrechnen
price_rf = scaler.inverse_transform(np.array(preds_rf_only).reshape(-1, 1))
price_hyb = scaler.inverse_transform(np.array(preds_hybrid).reshape(-1, 1))
price_xgb = scaler.inverse_transform(np.array(preds_xgb).reshape(-1, 1)) # NEU
last_known = scaler.inverse_transform(test_raw[LOOKBACK-1].reshape(-1, 1))

plt.figure(figsize=(12, 6))

# Subplot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(loss_hist, color='orange', label='NN Error Loss')
plt.title('Training der Fehlerkorrektur')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Vergleich
plt.subplot(1, 2, 2)
days = range(1, 11)
all_days = [0] + list(days)

# Linien Startpunkte verbinden
plot_real = np.concatenate([last_known, real_price])
plot_rf   = np.concatenate([last_known, price_rf])
plot_hyb  = np.concatenate([last_known, price_hyb])
plot_xgb  = np.concatenate([last_known, price_xgb]) # NEU

plt.plot(all_days, plot_real, 'g-o', label='Echt', linewidth=3, alpha=0.6)
plt.plot(all_days, plot_rf,   'r--.', label='Nur Random Forest', linewidth=1.5)
plt.plot(all_days, plot_hyb,  'b-x',  label='Hybrid (RF + NN)', linewidth=1.5)
plt.plot(all_days, plot_xgb,  'green', marker='*', label='XGBoost (Stacking)', linewidth=2.0) # NEU

plt.title(f'10-Tage Prognose: {TICKER}\nVergleich: Basis vs. Hybrid vs. XGBoost')
plt.ylabel('Kurs (EUR)')
plt.xlabel('Tage (0 = Heute)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(all_days)

plt.tight_layout()
plt.show()

# Fazit im Terminal
print(f"\nFazit (Tag 10):")
print(f"Echt:       {real_price[-1][0]:.2f} €")
print(f"Nur RF:     {price_rf[-1][0]:.2f} €")
print(f"Hybrid:     {price_hyb[-1][0]:.2f} €")
print(f"XGBoost:    {price_xgb[-1][0]:.2f} €")