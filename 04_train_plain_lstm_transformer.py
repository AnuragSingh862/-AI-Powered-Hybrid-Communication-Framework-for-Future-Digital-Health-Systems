"""
================================================================================
 TWO MORE ARCHITECTURES: Plain-LSTM (no attention) + Transformer encoder
 (genuinely non-recurrent family - pure self-attention, no GRU/LSTM at all)
================================================================================
Same everything as train_architectures.py: full dual-receiver feature set,
HORIZON=1, dataset_gru.csv, same split/seed, same training budget.
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, math
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = 'dataset_gru.csv'
WINDOW, HORIZON = 80, 1
BATCH_SIZE, HIDDEN_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY = 64, 32, 35, 1e-3, 1e-4
HEAD_DROPOUT = 0.3

feature_cols = ['CSI_mag_mean', 'CSI_mag_std', 'CSI_phase_std', 'Noise_mean',
                 'r_pos', 'snr_db', 'SNR_wifi_dB', 'SNR_vlc_dB',
                 'LLR_WiFi_mean', 'LLR_VLC_mean',
                 'Reliability_WiFi', 'Reliability_VLC']

df = pd.read_csv(DATA_PATH)
df['sequence_id'] = (df['snr_db'] != df['snr_db'].shift()).cumsum()
df['time_step'] = df.groupby('sequence_id').cumcount()
mag_cols = [f'WiFi_CSI_mag_{i}' for i in range(1, 51)]
phase_cols = [f'WiFi_CSI_phase_{i}' for i in range(1, 51)]
noise_cols = [f'WiFi_noise_{i}' for i in range(1, 51)]
extra = pd.DataFrame({
    'CSI_mag_mean':  df[mag_cols].mean(axis=1),
    'CSI_mag_std':   df[mag_cols].std(axis=1),
    'CSI_phase_std': df[phase_cols].std(axis=1),
    'Noise_mean':    df[noise_cols].mean(axis=1),
})
df = pd.concat([df, extra], axis=1)

X, y_cls, y_reg, seq_ids = [], [], [], []
for sid, g in df.groupby('sequence_id'):
    g = g.sort_values('time_step').reset_index(drop=True)
    n = len(g)
    feats = g[feature_cols].values.astype(np.float32)
    lab = g['label'].values
    sw, sv = g['SNR_wifi_dB'].values, g['SNR_vlc_dB'].values
    for t in range(WINDOW - 1, n - HORIZON):
        X.append(feats[t - WINDOW + 1: t + 1])
        y_cls.append(lab[t + HORIZON])
        y_reg.append([sw[t + HORIZON], sv[t + HORIZON]])
        seq_ids.append(sid)
X = np.stack(X); y_cls = np.array(y_cls, dtype=np.int64)
y_reg = np.array(y_reg, dtype=np.float32); seq_ids = np.array(seq_ids)
print("windows:", X.shape)

uniq = np.unique(seq_ids)
rng = np.random.default_rng(0); rng.shuffle(uniq)
n_val, n_test = int(len(uniq) * 0.15), int(len(uniq) * 0.15)
test_s, val_s = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
train_mask = ~np.isin(seq_ids, list(test_s) + list(val_s))
val_mask = np.isin(seq_ids, list(val_s)); test_mask = np.isin(seq_ids, list(test_s))

feat_mean = X[train_mask].mean(axis=(0, 1), keepdims=True)
feat_std = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
reg_mean = y_reg[train_mask].mean(axis=0, keepdims=True)
reg_std = y_reg[train_mask].std(axis=0, keepdims=True) + 1e-6

class DS(Dataset):
    def __init__(self, mask):
        self.X = (X[mask] - feat_mean) / feat_std
        self.yc = y_cls[mask]
        self.yr = (y_reg[mask] - reg_mean) / reg_std
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), self.yc[i], torch.from_numpy(self.yr[i])

train_loader = DataLoader(DS(train_mask), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DS(val_mask), batch_size=BATCH_SIZE)
test_loader = DataLoader(DS(test_mask), batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Architecture 1: Plain LSTM, bidirectional, NO attention (last hidden state only)
# ---------------------------------------------------------------------------
class PlainBiLSTM(nn.Module):
    def __init__(self, num_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        d = hidden_size * 2
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d, 16),
                                       nn.ReLU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        h_last = torch.cat([h[-2], h[-1]], dim=1)
        return self.cls_head(h_last), self.reg_head(h_last)

# ---------------------------------------------------------------------------
# Architecture 2: Transformer encoder - genuinely non-recurrent, pure
# self-attention over the window (no GRU/LSTM anywhere in this model)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=WINDOW):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerForecaster(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=True, activation='relu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # learned query token (like a [CLS] token) to pool the sequence into one vector
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d_model, 16),
                                       nn.ReLU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        cls_tok = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls_tok, h], dim=1)  # prepend CLS token
        out = self.encoder(h)
        pooled = out[:, 0, :]  # CLS token's output = pooled representation
        return self.cls_head(pooled), self.reg_head(pooled)

def train_and_save(model, name, save_path):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_counts = np.bincount(y_cls[train_mask])
    inv_freq = len(y_cls[train_mask]) / (2 * class_counts)
    class_weights = torch.tensor(np.sqrt(inv_freq), dtype=torch.float32)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    mse = nn.MSELoss()
    best_val_f1 = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for xb, ycb, yrb in train_loader:
            opt.zero_grad()
            cls_out, reg_out = model(xb)
            loss = ce(cls_out, ycb) + 0.5 * mse(reg_out, yrb)
            loss.backward(); opt.step()
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, ycb, yrb in val_loader:
                cls_out, _ = model(xb)
                vp.append(cls_out.argmax(1).numpy()); vt.append(ycb.numpy())
        val_f1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{name}] epoch {epoch:3d} val_macroF1 {val_f1:.4f}")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    cp, ct = [], []
    with torch.no_grad():
        for xb, ycb, yrb in test_loader:
            cls_out, _ = model(xb)
            cp.append(cls_out.argmax(1).numpy()); ct.append(ycb.numpy())
    cp, ct = np.concatenate(cp), np.concatenate(ct)
    acc = accuracy_score(ct, cp); f1 = f1_score(ct, cp, average='macro')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] TEST acc={acc:.4f} macroF1={f1:.4f} params={n_params} -> saved to {save_path}")
    return acc, f1

print("\n=== Training Plain-LSTM (no attention) ===")
train_and_save(PlainBiLSTM(len(feature_cols)), "Plain-LSTM", "arch_plain_lstm.pt")

print("\n=== Training Transformer encoder (non-recurrent) ===")
train_and_save(TransformerForecaster(len(feature_cols)), "Transformer", "arch_transformer.pt")

print("\nBoth architectures trained and saved.")
