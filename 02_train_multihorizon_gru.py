"""
================================================================================
 MULTI-HORIZON FORECASTER: horizons {1, 5, 10} minutes ahead, DIRECT output
================================================================================
Single Attention-GRU trunk, with SEPARATE classification/regression output
heads per horizon (direct multi-horizon prediction), rather than recursively
feeding 1-step predictions back in as fake "real" future inputs
(autoregressive chaining compounds errors badly - direct multi-horizon output
is the more robust approach per the literature reviewed).

Same legitimate dual-receiver feature set as the medical pipeline (includes
r_pos, SNR_vlc_dB, LLR_VLC_mean, Reliability_VLC - all live readings under
the confirmed dual-simultaneous-receiver hardware assumption).

Trained on dataset_gru.csv, evaluated in-dataset AND cross-scenario on
dataset_home.csv (never touched during training), at each horizon
separately, so you get a real horizon-vs-accuracy trade-off curve.

Decision layer cost ratio updated per request: C_FALSE_SWITCH=2,
C_MISSED_SWITCH=3 (1.5:1 - a milder safety bias than the earlier 4:1 medical
run), applied independently at each horizon.
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

TRAIN_PATH = 'dataset_gru.csv'
CROSS_TEST_PATH = 'dataset_home.csv'
WINDOW = 80
HORIZONS = [1, 5, 10]
BATCH_SIZE, HIDDEN_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY = 64, 32, 40, 1e-3, 1e-4
HEAD_DROPOUT = 0.3
MC_PASSES = 30
C_FALSE_SWITCH = 2.0
C_MISSED_SWITCH = 3.0

feature_cols = ['CSI_mag_mean', 'CSI_mag_std', 'CSI_phase_std', 'Noise_mean',
                 'r_pos', 'snr_db', 'SNR_wifi_dB', 'SNR_vlc_dB',
                 'LLR_WiFi_mean', 'LLR_VLC_mean',
                 'Reliability_WiFi', 'Reliability_VLC']
MAX_H = max(HORIZONS)

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
        for t in range(WINDOW - 1, n - MAX_H):
            X.append(feats[t - WINDOW + 1: t + 1])
            y_cls.append([lab[t + h] for h in HORIZONS])
            y_reg.append([[sw[t + h], sv[t + h]] for h in HORIZONS])
            seq_ids.append(sid)
    X = np.stack(X)
    y_cls = np.array(y_cls, dtype=np.int64)          # (N, num_horizons)
    y_reg = np.array(y_reg, dtype=np.float32)         # (N, num_horizons, 2)
    seq_ids = np.array(seq_ids)
    return X, y_cls, y_reg, seq_ids, df

X, y_cls, y_reg, seq_ids, df_train = build_windows(TRAIN_PATH)
print("Train windows:", X.shape, "per-horizon label rates:", y_cls.mean(axis=0))

uniq = np.unique(seq_ids)
rng = np.random.default_rng(0); rng.shuffle(uniq)
n_val, n_test = int(len(uniq) * 0.15), int(len(uniq) * 0.15)
test_s, val_s = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
train_mask = ~np.isin(seq_ids, list(test_s) + list(val_s))
val_mask = np.isin(seq_ids, list(val_s)); test_mask = np.isin(seq_ids, list(test_s))

feat_mean = X[train_mask].mean(axis=(0, 1), keepdims=True)
feat_std = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
reg_mean = y_reg[train_mask].mean(axis=(0, 1), keepdims=True)
reg_std = y_reg[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6

class DS(Dataset):
    def __init__(self, Xa, yc, yr):
        self.X = (Xa - feat_mean) / feat_std
        self.yc = yc
        self.yr = (yr - reg_mean) / reg_std
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), self.yc[i], torch.from_numpy(self.yr[i])

train_loader = DataLoader(DS(X[train_mask], y_cls[train_mask], y_reg[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DS(X[val_mask], y_cls[val_mask], y_reg[val_mask]), batch_size=BATCH_SIZE)
test_loader = DataLoader(DS(X[test_mask], y_cls[test_mask], y_reg[test_mask]), batch_size=BATCH_SIZE)

class MultiHorizonAttentionGRU(nn.Module):
    def __init__(self, num_features, num_horizons, hidden_size=HIDDEN_SIZE, bidirectional=True):
        super().__init__()
        self.num_horizons = num_horizons
        self.gru = nn.GRU(num_features, hidden_size, num_layers=1,
                           batch_first=True, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        d = hidden_size * mult
        self.attn_W = nn.Linear(d, d)
        self.attn_v = nn.Linear(d, 1, bias=False)
        # separate small head per horizon (direct multi-horizon output)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d, 16), nn.ReLU(),
                          nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
            for _ in range(num_horizons)])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
            for _ in range(num_horizons)])
    def forward(self, x):
        h_seq, _ = self.gru(x)
        scores = self.attn_v(torch.tanh(self.attn_W(h_seq))).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h_seq).squeeze(1)
        cls_outs = [head(context) for head in self.cls_heads]   # list of (batch, 2)
        reg_outs = [head(context) for head in self.reg_heads]
        return torch.stack(cls_outs, dim=1), torch.stack(reg_outs, dim=1)  # (batch, H, 2)

model = MultiHorizonAttentionGRU(X.shape[-1], len(HORIZONS))
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# per-horizon class weights (imbalance can differ slightly per horizon)
class_weights_per_h = []
for hi in range(len(HORIZONS)):
    cc = np.bincount(y_cls[train_mask][:, hi], minlength=2)
    inv_freq = len(y_cls[train_mask]) / (2 * cc)
    class_weights_per_h.append(torch.tensor(np.sqrt(inv_freq), dtype=torch.float32))

mse = nn.MSELoss()
REG_LOSS_WEIGHT = 0.5

best_val_f1 = -1.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    for xb, ycb, yrb in train_loader:
        opt.zero_grad()
        cls_out, reg_out = model(xb)  # (batch, H, 2), (batch, H, 2)
        loss = 0.0
        for hi in range(len(HORIZONS)):
            loss = loss + nn.functional.cross_entropy(cls_out[:, hi, :], ycb[:, hi], weight=class_weights_per_h[hi])
            loss = loss + REG_LOSS_WEIGHT * mse(reg_out[:, hi, :], yrb[:, hi, :])
        loss.backward(); opt.step()
    model.eval()
    vp_all, vt_all = [[] for _ in HORIZONS], [[] for _ in HORIZONS]
    with torch.no_grad():
        for xb, ycb, yrb in val_loader:
            cls_out, _ = model(xb)
            for hi in range(len(HORIZONS)):
                vp_all[hi].append(cls_out[:, hi, :].argmax(1).numpy())
                vt_all[hi].append(ycb[:, hi].numpy())
    val_f1s = [f1_score(np.concatenate(vt_all[hi]), np.concatenate(vp_all[hi]), average='macro') for hi in range(len(HORIZONS))]
    val_f1_mean = np.mean(val_f1s)
    if val_f1_mean > best_val_f1:
        best_val_f1 = val_f1_mean
        torch.save(model.state_dict(), "best_multihorizon_attngru.pt")
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} val_macroF1_per_h={[round(f,4) for f in val_f1s]} mean={val_f1_mean:.4f}")

model.load_state_dict(torch.load("best_multihorizon_attngru.pt"))
model.eval()

def evaluate(loader, name):
    cp_all, ct_all, probs_all = [[] for _ in HORIZONS], [[] for _ in HORIZONS], [[] for _ in HORIZONS]
    with torch.no_grad():
        for xb, ycb, yrb in loader:
            cls_out, _ = model(xb)
            for hi in range(len(HORIZONS)):
                probs = torch.softmax(cls_out[:, hi, :], dim=1)[:, 1]
                cp_all[hi].append(cls_out[:, hi, :].argmax(1).numpy())
                ct_all[hi].append(ycb[:, hi].numpy())
                probs_all[hi].append(probs.numpy())
    results = {}
    print(f"\n=== {name} ===")
    for hi, h in enumerate(HORIZONS):
        cp = np.concatenate(cp_all[hi]); ct = np.concatenate(ct_all[hi]); probs = np.concatenate(probs_all[hi])
        acc = accuracy_score(ct, cp); f1 = f1_score(ct, cp, average='macro')
        cm = confusion_matrix(ct, cp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
        rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
        try: auc = roc_auc_score(ct, probs)
        except Exception: auc = float('nan')
        print(f"H={h:2d}min: acc={acc:.4f} macroF1={f1:.4f} AUC={auc:.4f} VLC_recall={rec:.1f}% VLC_prec={prec:.1f}%")
        results[h] = dict(acc=acc, f1=f1, auc=auc, cp=cp, ct=ct, probs=probs)
    return results

res_indist = evaluate(test_loader, "IN-DATASET test (dataset_gru.csv)")

# naive persistence baseline per horizon
print("\n=== Naive persistence baseline (in-dataset) ===")
for hi, h in enumerate(HORIZONS):
    cur_label = []
    for sid, g in df_train.groupby('sequence_id'):
        g = g.sort_values('time_step').reset_index(drop=True); n = len(g)
        lab = g['label'].values
        for t in range(WINDOW - 1, n - MAX_H):
            cur_label.append(lab[t])
    cur_label = np.array(cur_label)
    acc = accuracy_score(y_cls[test_mask][:, hi], cur_label[test_mask])
    f1 = f1_score(y_cls[test_mask][:, hi], cur_label[test_mask], average='macro')
    print(f"H={h:2d}min naive persistence: acc={acc:.4f} macroF1={f1:.4f}")

# cross-scenario
Xc, yc_cls, yc_reg, seq_ids_c, df_cross = build_windows(CROSS_TEST_PATH)
print("\ncross-scenario windows:", Xc.shape)
cross_loader = DataLoader(DS(Xc, yc_cls, yc_reg), batch_size=BATCH_SIZE)
res_cross = evaluate(cross_loader, "CROSS-SCENARIO test (dataset_home.csv)")

print("\n=== Naive persistence baseline (cross-scenario) ===")
for hi, h in enumerate(HORIZONS):
    cur_label_cross = []
    for sid, g in df_cross.groupby('sequence_id'):
        g = g.sort_values('time_step').reset_index(drop=True); n = len(g)
        lab = g['label'].values
        for t in range(WINDOW - 1, n - MAX_H):
            cur_label_cross.append(lab[t])
    cur_label_cross = np.array(cur_label_cross)
    acc = accuracy_score(yc_cls[:, hi], cur_label_cross)
    f1 = f1_score(yc_cls[:, hi], cur_label_cross, average='macro')
    print(f"H={h:2d}min naive persistence: acc={acc:.4f} macroF1={f1:.4f}")

# ---------------------------------------------------------------------------
# Per-horizon MC-Dropout + cost-sensitive decision layer (3:2 ratio), cross-scenario
# ---------------------------------------------------------------------------
def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()

def mc_predict(loader, hi):
    model.eval(); enable_dropout(model)
    passes = []; y_true = None
    with torch.no_grad():
        for _ in range(MC_PASSES):
            pp, pt = [], []
            for xb, ycb, yrb in loader:
                cls_out, _ = model(xb)
                probs = torch.softmax(cls_out[:, hi, :], dim=1)[:, 1]
                pp.append(probs.numpy())
                if y_true is None: pt.append(ycb[:, hi].numpy())
            passes.append(np.concatenate(pp))
            if y_true is None: y_true = np.concatenate(pt)
    p_mc = np.stack(passes)
    return p_mc.mean(axis=0), p_mc.std(axis=0), y_true

def expected_cost(decisions, truth):
    fp = np.sum((decisions == 1) & (truth == 0))
    fn = np.sum((decisions == 0) & (truth == 1))
    total_cost = fp * C_FALSE_SWITCH + fn * C_MISSED_SWITCH
    return total_cost, total_cost / len(truth)

print(f"\n=== Cost-sensitive decision layer per horizon (C_missed:C_false = {C_MISSED_SWITCH}:{C_FALSE_SWITCH}) ===")
for hi, h in enumerate(HORIZONS):
    p_hat_val, p_sigma_val, y_val_true = mc_predict(val_loader, hi)
    p_hat_cross, p_sigma_cross, y_cross_true = mc_predict(cross_loader, hi)

    candidate_taus = np.linspace(0.01, 0.99, 197)
    val_costs, val_switch_rate = [], []
    for tau in candidate_taus:
        dec_val = (p_hat_val >= tau).astype(int)
        c, _ = expected_cost(dec_val, y_val_true)
        val_costs.append(c); val_switch_rate.append(dec_val.mean())
    val_costs = np.array(val_costs); val_switch_rate = np.array(val_switch_rate)
    valid = (val_switch_rate >= 0.05) & (val_switch_rate <= 0.95)
    if valid.sum() == 0: valid = np.ones_like(valid, dtype=bool)
    tau_star = candidate_taus[valid][int(np.argmin(val_costs[valid]))]

    decisions_naive = (p_hat_cross >= 0.5).astype(int)
    cost_naive, cost_per_naive = expected_cost(decisions_naive, y_cross_true)
    decisions_cost = (p_hat_cross >= tau_star).astype(int)
    cost_cost, cost_per_cost = expected_cost(decisions_cost, y_cross_true)

    acc_n = accuracy_score(y_cross_true, decisions_naive); f1_n = f1_score(y_cross_true, decisions_naive, average='macro')
    acc_c = accuracy_score(y_cross_true, decisions_cost); f1_c = f1_score(y_cross_true, decisions_cost, average='macro')
    imp = (cost_per_naive - cost_per_cost) / cost_per_naive * 100 if cost_per_naive > 0 else 0
    print(f"H={h:2d}min: tau*={tau_star:.3f}  naive(acc={acc_n:.3f},f1={f1_n:.3f},cost={cost_per_naive:.4f}) "
          f"-> cost-opt(acc={acc_c:.3f},f1={f1_c:.3f},cost={cost_per_cost:.4f})  cost_reduction={imp:.1f}%")

print("\n=== SUMMARY: accuracy vs horizon (cross-scenario) ===")
for h in HORIZONS:
    print(f"H={h:2d} min ahead: acc={res_cross[h]['acc']:.4f}  macroF1={res_cross[h]['f1']:.4f}  AUC={res_cross[h]['auc']:.4f}")
