import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
from datetime import datetime

# ==========================================
# 1. KONFIGURATION
# ==========================================
CONFIG = {
    'ticker': 'DHL.DE',     
    'seq_length': 150,      
    'batch_size': 16,       
    'hidden_size': 64,      
    'output_size': 1,       
    'learning_rate': 0.0001,
    'epochs': 60,           
    'patience': 15,         
    'clip_norm': 1.0,       
    'test_split': 0.15,     
    'data_file': 'DHL.DE_data.csv' 
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # MPS für Mac möglich, hier CPU safe
print(f"🚀 System gestartet auf: {DEVICE}")

# ==========================================
# 2. DATA LOADING (MIT FIX FÜR MEHR DATEN)
# ==========================================

def load_data_with_cache(ticker, filename):
    if os.path.exists(filename):
        print(f"💾 Lade gespeicherte Daten aus '{filename}'...")
        df = pd.read_csv(filename, index_col='Date', parse_dates=True)
    else:
        print(f"🌐 Lade neue Daten von API (Stooq) für {ticker}...")
        try:
            # FIX: Startdatum explizit setzen, um alle historischen Daten zu holen
            start_date = datetime(2000, 1, 1)
            df = web.DataReader(ticker, 'stooq', start=start_date)
            df = df.sort_index(ascending=True)
            df.to_csv(filename)
            print(f"💾 Daten erfolgreich als '{filename}' gespeichert.")
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return None
            
    return df[['Close']]

def wavelet_denoising(data, wavelet='db4'):
    print("🌊 Wende Wavelet Denoising an...")
    signal = data.values.flatten()
    coeff = pywt.wavedec(signal, wavelet, mode="per")
    sigma = (1/0.6745) * np.mean(np.abs(coeff[-1] - np.mean(coeff[-1])))
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    denoised = pywt.waverec(coeff, wavelet, mode="per")
    
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    elif len(denoised) < len(signal):
        denoised = np.pad(denoised, (0, len(signal) - len(denoised)), 'edge')
        
    return pd.DataFrame(denoised, index=data.index, columns=['Close_Denoised'])

# ==========================================
# 3. PREPROCESSING & DATASET (MIT FIX FÜR OVERLAP)
# ==========================================

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        # Hier entstand der Fehler: Wenn len(data) < seq_length ist, kommt negativ raus.
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_length]
        y = self.data[index + self.seq_length]
        return x, y

# -- Ausführung --
# BITTE VORHER DIE ALTE .CSV LÖSCHEN!
df_raw = load_data_with_cache(CONFIG['ticker'], CONFIG['data_file'])
if df_raw is None or df_raw.empty:
    raise ValueError("Keine Daten geladen.")

df_clean = wavelet_denoising(df_raw)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_clean.values)

# --- DER FIX FÜR DEN SPLIT ---
split_idx = int(len(data_scaled) * (1 - CONFIG['test_split']))

# Train: Von Anfang bis Split
train_data = data_scaled[:split_idx]

# Test: Wir müssen seq_length Schritte ZURÜCK gehen, damit der erste Test-Tag
# auch 150 Tage Historie (aus den Trainingsdaten) hat.
overlap = CONFIG['seq_length']
test_data = data_scaled[split_idx - overlap :] 

print(f"📊 Training Samples (Netto): {len(train_data) - CONFIG['seq_length']} | Test Samples (Netto): {len(test_data) - CONFIG['seq_length']}")

# DataLoaders
train_dataset = StockDataset(train_data, CONFIG['seq_length'])
test_dataset = StockDataset(test_data, CONFIG['seq_length'])

# Sicherheitscheck
if len(train_dataset) <= 0 or len(test_dataset) <= 0:
    print("❌ FEHLER: Zu wenig Daten für die gewählte Sequenzlänge!")
    exit()

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# ==========================================
# 4. MODELL
# ==========================================

class xLSTM_TS_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(xLSTM_TS_Model, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm_stack = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=4, batch_first=True, dropout=0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.lstm_stack(x)
        last_step = out[:, -1, :]
        normed = self.layer_norm(last_step)
        return self.head(normed)

model = xLSTM_TS_Model(1, CONFIG['hidden_size'], CONFIG['output_size']).to(DEVICE)

# ==========================================
# 5. TRAINING
# ==========================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5) # Verbose entfernt (Warning Fix)

best_loss = float('inf')
no_improve = 0
save_path = "best_model_weights.pth"

print("\n🚀 Starte Training...")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=True)
    
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip_norm'])
        optimizer.step()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    
    avg_val_loss = val_loss / len(test_loader)
    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), save_path)
    else:
        no_improve += 1
        
    if no_improve >= CONFIG['patience']:
        print(f"🛑 Early Stopping in Epoch {epoch+1}!")
        break

print("✅ Training beendet.")

# ==========================================
# 6. EVALUIERUNG & PLOTS
# ==========================================

print("\n📈 Erstelle Analyse-Plots...")
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
else:
    print("Warnung: Kein gespeichertes Modell gefunden, nutze letztes.")
    
model.eval()

predictions = []
actuals = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        pred = model(x)
        predictions.append(pred.cpu().numpy())
        actuals.append(y.numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)
pred_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(actuals)

# BONUS: Vorhersage für Morgen
last_seq = torch.tensor(data_scaled[-CONFIG['seq_length']:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    future_val = model(last_seq)
    future_price = scaler.inverse_transform(future_val.cpu().numpy())[0][0]
    last_real = scaler.inverse_transform(data_scaled[-1].reshape(-1, 1))[0][0]

print(f"\n🔮 PREDICTION FÜR MORGEN:")
print(f"Letzter Schlusskurs: {last_real:.2f}€")
print(f"KI Prognose:         {future_price:.2f}€")
print(f"Tendenz:             {'📈 STEIGT' if future_price > last_real else '📉 FÄLLT'}")

# PLOTTING
fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'Stock Prediction Analysis: {CONFIG["ticker"]} (xLSTM-TS + Wavelet)', fontsize=16)
days = np.arange(len(actual_prices))

# 1. Gesamt
axs[0, 0].plot(actual_prices, label='Echt', color='black', alpha=0.6)
axs[0, 0].plot(pred_prices, label='KI', color='blue', alpha=0.7)
axs[0, 0].set_title('1. Gesamter Testzeitraum')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# 2. Zoom
zoom = 60
if len(days) > zoom:
    axs[0, 1].plot(days[-zoom:], actual_prices[-zoom:], label='Echt', color='black', marker='o', markersize=3)
    axs[0, 1].plot(days[-zoom:], pred_prices[-zoom:], label='KI', color='red', linestyle='--', linewidth=2)
else:
    axs[0, 1].plot(days, actual_prices, label='Echt')
    axs[0, 1].plot(days, pred_prices, label='KI')
axs[0, 1].set_title(f'2. Zoom: Letzte {zoom} Tage')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# 3. Scatter
axs[0, 2].scatter(actual_prices, pred_prices, alpha=0.5, color='purple')
min_val, max_val = actual_prices.min(), actual_prices.max()
axs[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--')
axs[0, 2].set_title('3. Korrelation')
axs[0, 2].set_xlabel('Echt')
axs[0, 2].set_ylabel('Vorhersage')

# 4. Fehler
errors = actual_prices - pred_prices
axs[1, 0].hist(errors, bins=30, color='orange', edgecolor='black')
axs[1, 0].set_title('4. Fehlerverteilung')

# 5. Confusion Matrix
act_dir = np.sign(np.diff(actual_prices.flatten()))
pred_dir = np.sign(np.diff(pred_prices.flatten()))
mask = (act_dir != 0) & (pred_dir != 0)
if np.any(mask):
    cm = confusion_matrix(act_dir[mask], pred_dir[mask], labels=[1, -1])
    im = axs[1, 1].matshow(cm, cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        axs[1, 1].text(j, i, f'{z}\n({z/cm.sum()*100:.1f}%)', ha='center', va='center', color='red', fontsize=12)
    axs[1, 1].set_title('5. Richtungs-Treffer')
    axs[1, 1].set_xticklabels(['', 'Steigt', 'Fällt'])
    axs[1, 1].set_yticklabels(['', 'Steigt', 'Fällt'])
else:
    axs[1, 1].text(0.5, 0.5, "Zu wenig Bewegung für Matrix", ha='center')

# 6. Trading
returns = np.diff(actual_prices.flatten()) / actual_prices[:-1].flatten()
signals = (np.diff(pred_prices.flatten()) > 0).astype(int)
strat_ret = signals[:-1] * returns[1:]
cum_strat = (1 + strat_ret).cumprod()
cum_hold = (1 + returns[1:]).cumprod()

axs[1, 2].plot(cum_hold, label='Buy & Hold', color='gray')
axs[1, 2].plot(cum_strat, label='KI Strategie', color='green')
axs[1, 2].set_title('6. 1€ Investment Simulation')
axs[1, 2].legend()
axs[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()