"""
================================================================================
 ARCHITECTURE COMPARISON: train 4 forecasters, identical everything except
 the recurrent/temporal core, for a fair latency comparison later.
================================================================================
Held constant across all 4 models: feature set (full dual-receiver, 12
features), HORIZON=1, training data (dataset_gru.csv), train/val/test split,
loss weighting, epochs, hidden size budget. Only the temporal architecture
changes:
  1. Attention-GRU   - bidirectional GRU + additive attention (current best)
  2. Attention-LSTM  - same attention mechanism, LSTM instead of GRU
  3. Plain BiGRU      - bidirectional GRU, NO attention (last-hidden-state only)
  4. TCN              - dilated causal 1D convolutions (non-recurrent family)
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
# Architectures
# ---------------------------------------------------------------------------
class AttentionRNN(nn.Module):
    """Bidirectional GRU or LSTM + additive attention over the window."""
    def __init__(self, num_features, hidden_size=HIDDEN_SIZE, cell='gru'):
        super().__init__()
        rnn_cls = nn.GRU if cell == 'gru' else nn.LSTM
        self.rnn = rnn_cls(num_features, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        d = hidden_size * 2
        self.attn_W = nn.Linear(d, d)
        self.attn_v = nn.Linear(d, 1, bias=False)
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d, 16),
                                       nn.ReLU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        h_seq, _ = self.rnn(x)
        scores = self.attn_v(torch.tanh(self.attn_W(h_seq))).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)
        return self.cls_head(context), self.reg_head(context)

class PlainBiGRU(nn.Module):
    """Bidirectional GRU, NO attention - last hidden state only."""
    def __init__(self, num_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(num_features, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        d = hidden_size * 2
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d, 16),
                                       nn.ReLU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        _, h = self.gru(x)
        h_last = torch.cat([h[-2], h[-1]], dim=1)
        return self.cls_head(h_last), self.reg_head(h_last)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(0.2))
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_features, hidden_size=HIDDEN_SIZE, levels=4, kernel_size=3):
        super().__init__()
        layers = []; ch_in = num_features
        for i in range(levels):
            layers.append(TemporalBlock(ch_in, hidden_size, kernel_size, 2 ** i))
            ch_in = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(hidden_size, 16),
                                       nn.ReLU(), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        out = self.tcn(x.transpose(1, 2))
        last = out[:, :, -1]
        return self.cls_head(last), self.reg_head(last)

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

print("\n=== Training Attention-GRU ===")
train_and_save(AttentionRNN(len(feature_cols), cell='gru'), "Attention-GRU", "arch_attn_gru.pt")

print("\n=== Training Attention-LSTM ===")
train_and_save(AttentionRNN(len(feature_cols), cell='lstm'), "Attention-LSTM", "arch_attn_lstm.pt")

print("\n=== Training Plain BiGRU (no attention) ===")
train_and_save(PlainBiGRU(len(feature_cols)), "Plain-BiGRU", "arch_plain_bigru.pt")

print("\n=== Training TCN ===")
train_and_save(TCN(len(feature_cols)), "TCN", "arch_tcn.pt")

print("\nAll 4 architectures trained and saved.")
