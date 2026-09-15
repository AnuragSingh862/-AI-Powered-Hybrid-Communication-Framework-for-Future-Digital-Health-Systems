"""
================================================================================
 CROSS-SCENARIO GENERALIZATION TEST
 Train: dataset_home.csv   |   Zero-shot test: dataset_home_1.csv
================================================================================
This is the strongest evidence of real-world deployment readiness we can
produce without a physical testbed: the model is trained (and internally
validated) ENTIRELY on dataset_home.csv, then evaluated on dataset_home_1.csv
- a separately-generated scenario it has never seen, sharing the same nominal
SNR-sweep design but with independent random channel/noise realizations.

If performance holds up close to the in-distribution (train-scenario) result,
that's real evidence the model learned generalizable WiFi<->VLC switching
dynamics rather than overfitting to one dataset's specific noise instance.

Uses the Attention-GRU architecture (best performer found in the earlier
architecture sweep on dataset_gru.csv) with the same WiFi-only, leak-free
feature set validated throughout this project.
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

TRAIN_PATH = 'dataset_home.csv'
CROSS_TEST_PATH = 'dataset_home_1.csv'
WINDOW, HORIZON = 80, 3
BATCH_SIZE, HIDDEN_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY = 64, 32, 40, 1e-3, 1e-4
HEAD_DROPOUT = 0.3

feature_cols = ['CSI_mag_mean', 'CSI_mag_std', 'CSI_phase_std', 'Noise_mean',
                 'snr_db', 'SNR_wifi_dB', 'LLR_WiFi_mean', 'Reliability_WiFi']

def build_windows(path):
    df = pd.read_csv(path)
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
    return X, y_cls, y_reg, seq_ids

# ---------------------------------------------------------------------------
# Load TRAIN scenario (dataset_home.csv) - split into train/val internally
# ---------------------------------------------------------------------------
X, y_cls, y_reg, seq_ids = build_windows(TRAIN_PATH)
print("Train-scenario windows:", X.shape, "label dist:", np.bincount(y_cls) / len(y_cls))

uniq = np.unique(seq_ids)
rng = np.random.default_rng(0); rng.shuffle(uniq)
n_val = int(len(uniq) * 0.15)
val_s = set(uniq[:n_val])
train_mask = ~np.isin(seq_ids, list(val_s))
val_mask = np.isin(seq_ids, list(val_s))

# ---------------------------------------------------------------------------
# Load CROSS-SCENARIO test set (dataset_home_1.csv) - entirely held out,
# never touched during training or normalization-statistic computation
# ---------------------------------------------------------------------------
X_cross, y_cross, y_reg_cross, seq_ids_cross = build_windows(CROSS_TEST_PATH)
print("Cross-scenario (home_1) windows:", X_cross.shape, "label dist:", np.bincount(y_cross) / len(y_cross))

# ALSO carve out an in-distribution test slice from home.csv for a fair
# apples-to-apples comparison against the cross-scenario number.
n_test_indist = int(len(uniq) * 0.15)
test_s_indist = set(uniq[n_val:n_val + n_test_indist])
test_mask_indist = np.isin(seq_ids, list(test_s_indist))
train_mask = train_mask & ~test_mask_indist  # exclude in-dist test windows from training

feat_mean = X[train_mask].mean(axis=(0, 1), keepdims=True)
feat_std = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
reg_mean = y_reg[train_mask].mean(axis=0, keepdims=True)
reg_std = y_reg[train_mask].std(axis=0, keepdims=True) + 1e-6

class DS(Dataset):
    def __init__(self, Xa, yc, yr):
        self.X = (Xa - feat_mean) / feat_std
        self.yc = yc
        self.yr = (yr - reg_mean) / reg_std
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), self.yc[i], torch.from_numpy(self.yr[i])

train_loader = DataLoader(DS(X[train_mask], y_cls[train_mask], y_reg[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DS(X[val_mask], y_cls[val_mask], y_reg[val_mask]), batch_size=BATCH_SIZE)
indist_test_loader = DataLoader(DS(X[test_mask_indist], y_cls[test_mask_indist], y_reg[test_mask_indist]), batch_size=BATCH_SIZE)
cross_test_loader = DataLoader(DS(X_cross, y_cross, y_reg_cross), batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Attention-GRU (same architecture as the dataset_gru.csv sweep winner)
# ---------------------------------------------------------------------------
class AttentionGRU(nn.Module):
    def __init__(self, num_features, hidden_size=HIDDEN_SIZE, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(num_features, hidden_size, num_layers=1,
                           batch_first=True, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        d = hidden_size * mult
        self.attn_W = nn.Linear(d, d)
        self.attn_v = nn.Linear(d, 1, bias=False)
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d, 16),
                                       nn.ReLU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        h_seq, _ = self.gru(x)
        scores = self.attn_v(torch.tanh(self.attn_W(h_seq))).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)
        return self.cls_head(context), self.reg_head(context)

model = AttentionGRU(X.shape[-1])
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
class_counts = np.bincount(y_cls[train_mask])
inv_freq = len(y_cls[train_mask]) / (2 * class_counts)
class_weights = torch.tensor(np.sqrt(inv_freq), dtype=torch.float32)
ce = nn.CrossEntropyLoss(weight=class_weights)
mse = nn.MSELoss()
REG_LOSS_WEIGHT = 0.5

best_val_f1 = -1.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    for xb, ycb, yrb in train_loader:
        opt.zero_grad()
        cls_out, reg_out = model(xb)
        loss = ce(cls_out, ycb) + REG_LOSS_WEIGHT * mse(reg_out, yrb)
        loss.backward(); opt.step()
    model.eval(); vp, vt = [], []
    with torch.no_grad():
        for xb, ycb, yrb in val_loader:
            cls_out, _ = model(xb)
            vp.append(cls_out.argmax(1).numpy()); vt.append(ycb.numpy())
    val_f1 = f1_score(np.concatenate(vt), np.concatenate(vp), average='macro')
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_attngru_crossscenario.pt")
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} val_macroF1 {val_f1:.4f}")

model.load_state_dict(torch.load("best_attngru_crossscenario.pt"))
model.eval()

def evaluate(loader, name):
    cp, ct, probs = [], [], []
    with torch.no_grad():
        for xb, ycb, yrb in loader:
            cls_out, _ = model(xb)
            p = torch.softmax(cls_out, dim=1)[:, 1]
            cp.append(cls_out.argmax(1).numpy()); ct.append(ycb.numpy()); probs.append(p.numpy())
    cp, ct, probs = np.concatenate(cp), np.concatenate(ct), np.concatenate(probs)
    acc = accuracy_score(ct, cp); f1 = f1_score(ct, cp, average='macro')
    cm = confusion_matrix(ct, cp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    try: auc = roc_auc_score(ct, probs)
    except Exception: auc = float('nan')
    print(f"\n--- {name} ---")
    print(f"acc={acc:.4f}  macroF1={f1:.4f}  AUC={auc:.4f}  VLC_recall={rec:.1f}%  VLC_prec={prec:.1f}%")
    return acc, f1, auc

acc_id, f1_id, auc_id = evaluate(indist_test_loader, "IN-DISTRIBUTION test (held-out slice of dataset_home.csv)")
acc_cs, f1_cs, auc_cs = evaluate(cross_test_loader, "CROSS-SCENARIO test (dataset_home_1.csv, never seen in training)")

print("\n=== Generalization gap ===")
print(f"macroF1 drop: {f1_id:.4f} -> {f1_cs:.4f}  ({(f1_id-f1_cs)/f1_id*100:+.1f}%)")
print(f"AUC drop:     {auc_id:.4f} -> {auc_cs:.4f}  ({(auc_id-auc_cs)/auc_id*100:+.1f}%)")
