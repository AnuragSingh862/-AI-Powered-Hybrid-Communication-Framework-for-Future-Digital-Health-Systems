"""
================================================================================
 MEDICAL-DEPLOYMENT WiFi<->VLC PROACTIVE SWITCHING FORECASTER
================================================================================
Hardware assumption (confirmed with user): DUAL SIMULTANEOUS RECEIVERS.
Both WiFi and VLC channels are continuously, actively monitored in real time,
even while only one carries data. Under this assumption, SNR_vlc_dB,
LLR_VLC_mean, Reliability_VLC, h_vlc are all legitimate LIVE inputs (not
oracle/future leakage) - they're just the VLC receiver's own current
readings, exactly like SNR_wifi_dB is WiFi's own current reading. r_pos is
also included since the system has a positioning subsystem available.

The only thing that must never be an input is information from AFTER the
window's end (t+1 .. t+HORIZON) - that's what's actually being forecast.

Forecast horizon: HORIZON=1 = 1 sample ahead. Data is sampled once per
minute (confirmed), so this is a genuine 1-MINUTE-AHEAD proactive forecast -
predicting which link will be optimal a minute from now, before it happens,
to enable make-before-break handover and avoid the latency/data-loss spike
of a reactive (break-before-make) switch. This matters specifically for
medical telemetry, where a dropped or degraded link during patient
monitoring has real safety consequences.

Architecture: Attention-GRU (bidirectional GRU + temporal attention), the
best-performing architecture found in the earlier sweep - also valuable here
for interpretability (attention weights over the window can be inspected /
audited, relevant for medical/regulatory review).

Decision layer: MC-Dropout uncertainty + empirically-tuned cost-sensitive
threshold, weighted 4:1 (missed-switch : false-switch) to reflect the
asymmetric safety cost in a medical context - failing to proactively move
off a degrading link is worse than an unnecessary handover.
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

TRAIN_PATH = 'dataset_gru.csv'
CROSS_TEST_PATH = 'dataset_home.csv'
WINDOW, HORIZON = 80, 1   # HORIZON=1 -> 1-minute-ahead forecast (confirmed sampling rate)
BATCH_SIZE, HIDDEN_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY = 64, 32, 40, 1e-3, 1e-4
HEAD_DROPOUT = 0.3
MC_PASSES = 30
C_FALSE_SWITCH = 1.0
C_MISSED_SWITCH = 4.0   # medical-safety-weighted: missing a proactive switch is 4x worse

# Full legitimate feature set for dual-simultaneous-receiver hardware:
# both channels' live readings + position (positioning subsystem available).
feature_cols = ['CSI_mag_mean', 'CSI_mag_std', 'CSI_phase_std', 'Noise_mean',
                 'r_pos', 'snr_db', 'SNR_wifi_dB', 'SNR_vlc_dB',
                 'LLR_WiFi_mean', 'LLR_VLC_mean',
                 'Reliability_WiFi', 'Reliability_VLC']

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
    return X, y_cls, y_reg, seq_ids, df

X, y_cls, y_reg, seq_ids, df_train = build_windows(TRAIN_PATH)
print("Train windows:", X.shape, "label dist:", np.bincount(y_cls) / len(y_cls))
print("features:", feature_cols)

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
    def __init__(self, Xa, yc, yr):
        self.X = (Xa - feat_mean) / feat_std
        self.yc = yc
        self.yr = (yr - reg_mean) / reg_std
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), self.yc[i], torch.from_numpy(self.yr[i])

train_loader = DataLoader(DS(X[train_mask], y_cls[train_mask], y_reg[train_mask]), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DS(X[val_mask], y_cls[val_mask], y_reg[val_mask]), batch_size=BATCH_SIZE)
test_loader = DataLoader(DS(X[test_mask], y_cls[test_mask], y_reg[test_mask]), batch_size=BATCH_SIZE)

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
        torch.save(model.state_dict(), "best_medical_attngru.pt")
    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d} val_macroF1 {val_f1:.4f}")

model.load_state_dict(torch.load("best_medical_attngru.pt"))
model.eval()

def evaluate_argmax(loader, name):
    cp, ct = [], []
    with torch.no_grad():
        for xb, ycb, yrb in loader:
            cls_out, _ = model(xb)
            cp.append(cls_out.argmax(1).numpy()); ct.append(ycb.numpy())
    cp, ct = np.concatenate(cp), np.concatenate(ct)
    acc = accuracy_score(ct, cp); f1 = f1_score(ct, cp, average='macro')
    cm = confusion_matrix(ct, cp, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    print(f"\n--- {name} ---")
    print(f"acc={acc:.4f}  macroF1={f1:.4f}  confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    if (tp+fn) > 0: print(f"VLC recall={tp/(tp+fn)*100:.1f}%")
    if (tp+fp) > 0: print(f"VLC precision={tp/(tp+fp)*100:.1f}%")
    return acc, f1

evaluate_argmax(test_loader, "In-dataset test (dataset_gru.csv, held-out), argmax")

# naive persistence baseline
cur_label = []
for sid, g in df_train.groupby('sequence_id'):
    g = g.sort_values('time_step').reset_index(drop=True); n = len(g)
    lab = g['label'].values
    for t in range(WINDOW - 1, n - HORIZON):
        cur_label.append(lab[t])
cur_label = np.array(cur_label)
print("\n--- In-dataset naive persistence baseline ---")
print("acc:", accuracy_score(y_cls[test_mask], cur_label[test_mask]))
print("macroF1:", f1_score(y_cls[test_mask], cur_label[test_mask], average='macro'))

# ---------------------------------------------------------------------------
# Cross-scenario test on dataset_home.csv
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"CROSS-SCENARIO TEST: trained on {TRAIN_PATH}, evaluated on {CROSS_TEST_PATH}")
print("=" * 70)
Xc, yc_cls, yc_reg, seq_ids_c, df_cross = build_windows(CROSS_TEST_PATH)
print("cross-scenario windows:", Xc.shape, "label dist:", np.bincount(yc_cls) / len(yc_cls))
cross_loader = DataLoader(DS(Xc, yc_cls, yc_reg), batch_size=BATCH_SIZE)
evaluate_argmax(cross_loader, "Cross-scenario test (dataset_home.csv), argmax")

cur_label_cross = []
for sid, g in df_cross.groupby('sequence_id'):
    g = g.sort_values('time_step').reset_index(drop=True); n = len(g)
    lab = g['label'].values
    for t in range(WINDOW - 1, n - HORIZON):
        cur_label_cross.append(lab[t])
cur_label_cross = np.array(cur_label_cross)
print("\n--- Cross-scenario naive persistence baseline ---")
print("acc:", accuracy_score(yc_cls, cur_label_cross))
print("macroF1:", f1_score(yc_cls, cur_label_cross, average='macro'))

# ---------------------------------------------------------------------------
# MC-Dropout + medically-weighted cost-sensitive decision layer
# (applied to the cross-scenario set - the realistic deployment test)
# ---------------------------------------------------------------------------
def enable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()

def mc_predict(loader):
    model.eval(); enable_dropout(model)
    passes = []; y_true = None
    with torch.no_grad():
        for _ in range(MC_PASSES):
            pp, pt = [], []
            for xb, ycb, yrb in loader:
                cls_out, _ = model(xb)
                probs = torch.softmax(cls_out, dim=1)[:, 1]
                pp.append(probs.numpy())
                if y_true is None: pt.append(ycb.numpy())
            passes.append(np.concatenate(pp))
            if y_true is None: y_true = np.concatenate(pt)
    p_mc = np.stack(passes)
    return p_mc.mean(axis=0), p_mc.std(axis=0), y_true

p_hat_val, p_sigma_val, y_val_true = mc_predict(val_loader)
p_hat_cross, p_sigma_cross, y_cross_true = mc_predict(cross_loader)

print(f"\n--- MC-Dropout uncertainty (T={MC_PASSES}) ---")
print("mean predictive std (cross-scenario):", p_sigma_cross.mean())
try:
    print("ROC-AUC (cross-scenario):", roc_auc_score(y_cross_true, p_hat_cross))
except Exception as e:
    print("AUC skipped:", e)

def expected_cost(decisions, truth):
    fp = np.sum((decisions == 1) & (truth == 0))
    fn = np.sum((decisions == 0) & (truth == 1))
    total_cost = fp * C_FALSE_SWITCH + fn * C_MISSED_SWITCH
    return total_cost, total_cost / len(truth)

candidate_taus = np.linspace(0.01, 0.99, 197)
val_costs, val_switch_rate = [], []
for tau in candidate_taus:
    dec_val = (p_hat_val >= tau).astype(int)
    c, _ = expected_cost(dec_val, y_val_true)
    val_costs.append(c); val_switch_rate.append(dec_val.mean())
val_costs = np.array(val_costs); val_switch_rate = np.array(val_switch_rate)
MIN_RATE, MAX_RATE = 0.05, 0.95
valid = (val_switch_rate >= MIN_RATE) & (val_switch_rate <= MAX_RATE)
if valid.sum() == 0: valid = np.ones_like(valid, dtype=bool)
tau_star = candidate_taus[valid][int(np.argmin(val_costs[valid]))]

print(f"\n--- Cost-sensitive decision layer (C_missed:C_false = {C_MISSED_SWITCH}:{C_FALSE_SWITCH}) ---")
print(f"Empirically cost-optimal threshold: tau*={tau_star:.3f}")

decisions_naive = (p_hat_cross >= 0.5).astype(int)
cost_naive, cost_per_naive = expected_cost(decisions_naive, y_cross_true)
decisions_cost = (p_hat_cross >= tau_star).astype(int)
cost_cost, cost_per_cost = expected_cost(decisions_cost, y_cross_true)
decisions_gated = ((p_hat_cross - 0.5 * p_sigma_cross) >= tau_star).astype(int)
cost_gated, cost_per_gated = expected_cost(decisions_gated, y_cross_true)

def report(name, decisions, cost, cost_per):
    acc = accuracy_score(y_cross_true, decisions)
    f1 = f1_score(y_cross_true, decisions, average='macro')
    cm = confusion_matrix(y_cross_true, decisions, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    rec = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    print(f"\n{name}")
    print(f"  acc={acc:.4f}  macroF1={f1:.4f}  VLC_recall={rec:.1f}%  VLC_prec={prec:.1f}%  switch_rate={decisions.mean()*100:.1f}%")
    print(f"  total_cost={cost:.1f}  avg_cost/decision={cost_per:.4f}  (FP={fp}, FN={fn})")

report("Policy A: naive argmax (0.5) [cross-scenario]", decisions_naive, cost_naive, cost_per_naive)
report(f"Policy B: medically-weighted cost-optimal threshold [cross-scenario]", decisions_cost, cost_cost, cost_per_cost)
report(f"Policy C: cost-optimal + uncertainty-gated [cross-scenario]", decisions_gated, cost_gated, cost_per_gated)

imp_B = (cost_per_naive - cost_per_cost) / cost_per_naive * 100 if cost_per_naive > 0 else 0
imp_C = (cost_per_naive - cost_per_gated) / cost_per_naive * 100 if cost_per_naive > 0 else 0
print(f"\nCost reduction vs naive argmax: Policy B={imp_B:.1f}%, Policy C={imp_C:.1f}%")
