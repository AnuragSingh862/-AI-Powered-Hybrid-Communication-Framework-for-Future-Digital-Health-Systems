"""
================================================================================
 LATENCY COMPARISON ACROSS ARCHITECTURES
 Attention-GRU vs Attention-LSTM vs Plain-BiGRU (no attention) vs TCN
================================================================================
Same everything - features, HORIZON=1, cross-scenario test set
(dataset_home.csv), hysteresis-margin decision layer (+/-0.15, the winning
policy from the earlier comparison) - only the temporal architecture differs.
This isolates the effect of architecture choice on real-world latency, not
just classification accuracy.
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0)

L_HO_MS = 60.0
PAYLOAD_BYTES = 1024
BANDWIDTH_HZ = 20e6
HYSTERESIS_MARGIN = 0.15

WINDOW, HORIZON = 80, 1
CROSS_TEST_PATH = 'dataset_home.csv'
HIDDEN_SIZE = 32
HEAD_DROPOUT = 0.3

feature_cols = ['CSI_mag_mean', 'CSI_mag_std', 'CSI_phase_std', 'Noise_mean',
                 'r_pos', 'snr_db', 'SNR_wifi_dB', 'SNR_vlc_dB',
                 'LLR_WiFi_mean', 'LLR_VLC_mean',
                 'Reliability_WiFi', 'Reliability_VLC']

# ---------------------------------------------------------------------------
# Architecture definitions (must match train_architectures.py exactly)
# ---------------------------------------------------------------------------
class AttentionRNN(nn.Module):
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

import math
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_head = nn.Sequential(nn.Dropout(HEAD_DROPOUT), nn.Linear(d_model, 16),
                                       nn.ReLU(), nn.Dropout(HEAD_DROPOUT), nn.Linear(16, 2))
        self.reg_head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        cls_tok = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls_tok, h], dim=1)
        out = self.encoder(h)
        pooled = out[:, 0, :]
        return self.cls_head(pooled), self.reg_head(pooled)

# --- Multi-Horizon Attention-GRU, from multihorizon_pipeline.py - included
# here via a wrapper that exposes ONLY the horizon=1 head, so it plugs into
# the same evaluate_architecture() function as every other model (fair,
# single-metric comparison at matched HORIZON=1). ---
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
        cls_outs = [head(context) for head in self.cls_heads]
        reg_outs = [head(context) for head in self.reg_heads]
        return torch.stack(cls_outs, dim=1), torch.stack(reg_outs, dim=1)  # (batch, H, 2)

class MultiHorizonWrapper(nn.Module):
    """Exposes only the horizon-index-0 (HORIZON=1) head so this model can be
    evaluated through the exact same evaluate_architecture() pipeline as the
    single-horizon models - same feature set, same forward(x) -> (cls, reg)
    signature."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, x):
        cls_outs, reg_outs = self.base(x)
        return cls_outs[:, 0, :], reg_outs[:, 0, :]  # horizon index 0 == HORIZON=1

# ---------------------------------------------------------------------------
# Data prep (shared across all architectures)
# ---------------------------------------------------------------------------
df_train_stats = pd.read_csv('dataset_gru.csv')
mag_cols = [f'WiFi_CSI_mag_{i}' for i in range(1, 51)]
phase_cols = [f'WiFi_CSI_phase_{i}' for i in range(1, 51)]
noise_cols = [f'WiFi_noise_{i}' for i in range(1, 51)]
df_train_stats['sequence_id'] = (df_train_stats['snr_db'] != df_train_stats['snr_db'].shift()).cumsum()
df_train_stats['time_step'] = df_train_stats.groupby('sequence_id').cumcount()
mag_t = df_train_stats[mag_cols].values; phase_t = df_train_stats[phase_cols].values; noise_t = df_train_stats[noise_cols].values
extra_t = pd.DataFrame({'CSI_mag_mean': mag_t.mean(1), 'CSI_mag_std': mag_t.std(1),
                         'CSI_phase_std': phase_t.std(1), 'Noise_mean': noise_t.mean(1)})
df_train_stats = pd.concat([df_train_stats, extra_t], axis=1)

# Reproduce the EXACT train split used during training (same seed, same
# windowing) - normalization stats must come from ONLY the train segments,
# not the full file, to match what the model was actually trained on.
X_ts, seq_ids_ts = [], []
for sid, g in df_train_stats.groupby('sequence_id'):
    g = g.sort_values('time_step').reset_index(drop=True)
    n = len(g)
    feats = g[feature_cols].values.astype(np.float32)
    for t in range(WINDOW - 1, n - HORIZON):
        X_ts.append(feats[t - WINDOW + 1: t + 1])
        seq_ids_ts.append(sid)
X_ts = np.stack(X_ts); seq_ids_ts = np.array(seq_ids_ts)
uniq_ts = np.unique(seq_ids_ts)
rng_ts = np.random.default_rng(0); rng_ts.shuffle(uniq_ts)
n_val_ts, n_test_ts = int(len(uniq_ts) * 0.15), int(len(uniq_ts) * 0.15)
test_s_ts, val_s_ts = set(uniq_ts[:n_test_ts]), set(uniq_ts[n_test_ts:n_test_ts + n_val_ts])
train_mask_ts = ~np.isin(seq_ids_ts, list(test_s_ts) + list(val_s_ts))
feat_mean = X_ts[train_mask_ts].mean(axis=(0, 1))
feat_std = X_ts[train_mask_ts].std(axis=(0, 1)) + 1e-6
print(f"Normalization stats from TRAIN split only ({train_mask_ts.sum()}/{len(train_mask_ts)} windows)")

df = pd.read_csv(CROSS_TEST_PATH)
df['sequence_id'] = (df['snr_db'] != df['snr_db'].shift()).cumsum()
df['time_step'] = df.groupby('sequence_id').cumcount()
mag_c = df[mag_cols].values; phase_c = df[phase_cols].values; noise_c = df[noise_cols].values
extra_c = pd.DataFrame({'CSI_mag_mean': mag_c.mean(1), 'CSI_mag_std': mag_c.std(1),
                         'CSI_phase_std': phase_c.std(1), 'Noise_mean': noise_c.mean(1)})
df = pd.concat([df, extra_c], axis=1)

def evaluate_architecture(model, name):
    model.eval()
    all_pred, all_prob, all_optimal = [], [], []
    all_c_active, all_c_optimal = [], []
    all_block_ids = []

    for sid, g in df.groupby('sequence_id'):
        g = g.sort_values('time_step').reset_index(drop=True)
        n = len(g)
        if n < WINDOW + HORIZON:
            continue
        feats = g[feature_cols].values.astype(np.float32)
        feats_norm = (feats - feat_mean) / feat_std
        lab = g['label'].values
        sw_db = g['SNR_wifi_dB'].values; sv_db = g['SNR_vlc_dB'].values
        sw_lin = 10 ** (sw_db / 10.0); sv_lin = 10 ** (sv_db / 10.0)

        block_pred = {}; block_prob = {}
        with torch.no_grad():
            for t in range(WINDOW - 1, n - HORIZON):
                window = feats_norm[t - WINDOW + 1: t + 1]
                xb = torch.from_numpy(window).unsqueeze(0)
                cls_out, _ = model(xb)
                probs = torch.softmax(cls_out, dim=1)[0]
                block_pred[t + HORIZON] = cls_out.argmax(1).item()
                block_prob[t + HORIZON] = probs[1].item()

        valid_idxs = sorted(block_pred.keys())
        if len(valid_idxs) < 2:
            continue

        # Hysteresis-margin decision policy (the winning policy from earlier)
        hyst_decisions = {}
        current_link = block_pred[valid_idxs[0]]
        hyst_decisions[valid_idxs[0]] = current_link
        for idx in valid_idxs[1:]:
            p_vlc = block_prob[idx]
            if current_link == 0 and p_vlc >= 0.5 + HYSTERESIS_MARGIN:
                current_link = 1
            elif current_link == 1 and p_vlc <= 0.5 - HYSTERESIS_MARGIN:
                current_link = 0
            hyst_decisions[idx] = current_link

        for idx in valid_idxs:
            dec = hyst_decisions[idx]
            optimal_dec = lab[idx]
            C_opt = BANDWIDTH_HZ * np.log2(1 + max(sw_lin[idx], sv_lin[idx]))
            C_active = BANDWIDTH_HZ * np.log2(1 + (sv_lin[idx] if dec == 1 else sw_lin[idx]))
            all_pred.append(dec); all_optimal.append(optimal_dec)
            all_c_active.append(C_active); all_c_optimal.append(C_opt)
            all_block_ids.append(sid)

    all_pred = np.array(all_pred); all_optimal = np.array(all_optimal)
    all_c_active = np.array(all_c_active); all_c_optimal = np.array(all_c_optimal)
    all_block_ids = np.array(all_block_ids)
    same_block = all_block_ids[1:] == all_block_ids[:-1]

    from sklearn.metrics import f1_score, accuracy_score
    acc = accuracy_score(all_optimal, all_pred)
    f1 = f1_score(all_optimal, all_pred, average='macro', zero_division=0)

    switches = np.sum((all_pred[1:] != all_pred[:-1]) & same_block)
    ho_latency_ms = switches * L_HO_MS
    payload_bits = PAYLOAD_BYTES * 8
    extra_time_s = np.where(all_c_active < all_c_optimal,
                             payload_bits * (1.0/np.maximum(all_c_active,1e-6) - 1.0/np.maximum(all_c_optimal,1e-6)),
                             0.0)
    throughput_penalty_ms = extra_time_s.sum() * 1000.0
    total_ms = ho_latency_ms + throughput_penalty_ms

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] acc={acc:.4f} macroF1={f1:.4f} switches={switches} "
          f"ho_latency={ho_latency_ms:.1f}ms throughput_penalty={throughput_penalty_ms:.1f}ms "
          f"TOTAL={total_ms:.1f}ms params={n_params}")
    return dict(name=name, acc=acc, f1=f1, switches=switches, total_ms=total_ms, n_params=n_params)

results = []

model = AttentionRNN(len(feature_cols), cell='gru')
model.load_state_dict(torch.load('arch_attn_gru.pt'))
results.append(evaluate_architecture(model, "Attention-GRU"))

model = AttentionRNN(len(feature_cols), cell='lstm')
model.load_state_dict(torch.load('arch_attn_lstm.pt'))
results.append(evaluate_architecture(model, "Attention-LSTM"))

model = PlainBiGRU(len(feature_cols))
model.load_state_dict(torch.load('arch_plain_bigru.pt'))
results.append(evaluate_architecture(model, "Plain-BiGRU"))

model = TCN(len(feature_cols))
model.load_state_dict(torch.load('arch_tcn.pt'))
results.append(evaluate_architecture(model, "TCN"))

model = PlainBiLSTM(len(feature_cols))
model.load_state_dict(torch.load('arch_plain_lstm.pt'))
results.append(evaluate_architecture(model, "Plain-LSTM"))

model = TransformerForecaster(len(feature_cols))
model.load_state_dict(torch.load('arch_transformer.pt'))
results.append(evaluate_architecture(model, "Transformer"))

base_mh = MultiHorizonAttentionGRU(len(feature_cols), num_horizons=3)  # HORIZONS=[1,5,10] in original training
base_mh.load_state_dict(torch.load('best_multihorizon_attngru.pt'))
model = MultiHorizonWrapper(base_mh)
mh_result = evaluate_architecture(model, "MultiHorizon-GRU (H=1 head)")
# correct the reported param count: only trunk + ONE head is actually used
# at inference here, not all 3 horizon heads in the full checkpoint.
trunk_params = sum(p.numel() for p in base_mh.gru.parameters()) + \
                sum(p.numel() for p in base_mh.attn_W.parameters()) + \
                sum(p.numel() for p in base_mh.attn_v.parameters())
one_head_params = sum(p.numel() for p in base_mh.cls_heads[0].parameters()) + \
                   sum(p.numel() for p in base_mh.reg_heads[0].parameters())
mh_result['n_params'] = trunk_params + one_head_params
print(f"  (effective params for H=1 head only: {mh_result['n_params']} - "
      f"full checkpoint has {sum(p.numel() for p in base_mh.parameters())} across all 3 horizon heads)")
results.append(mh_result)

print("\n=== SUMMARY (cross-scenario, dataset_home.csv, hysteresis-margin decision layer) ===")
print(f"{'Model':<16}{'Acc':>8}{'MacroF1':>10}{'Switches':>10}{'Latency(ms)':>14}{'Params':>10}")
for r in results:
    print(f"{r['name']:<16}{r['acc']:>8.4f}{r['f1']:>10.4f}{r['switches']:>10}{r['total_ms']:>14.1f}{r['n_params']:>10}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

names = [r['name'] for r in results]
latencies = [r['total_ms'] for r in results]
f1s = [r['f1'] for r in results]
switches_l = [r['switches'] for r in results]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
ax = axes[0]
bars = ax.bar(names, latencies, color=colors)
ax.set_ylabel('Total latency (ms)')
ax.set_title('Latency by architecture\n(hysteresis-margin policy, cross-scenario)')
for b, v in zip(bars, latencies):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
bars = ax.bar(names, f1s, color=colors)
ax.set_ylabel('Macro-F1')
ax.set_ylim(0, 1)
ax.set_title('Classification accuracy by architecture')
for b, v in zip(bars, f1s):
    ax.text(b.get_x()+b.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

ax = axes[2]
for name, s, t, c in zip(names, switches_l, latencies, colors):
    ax.scatter(s, t, s=150, color=c, zorder=3)
    ax.annotate(name, (s, t), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax.set_xlabel('Switches'); ax.set_ylabel('Total latency (ms)')
ax.set_title('Latency vs switch count by architecture')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('architecture_latency_comparison.png', dpi=130)
print("\nSaved plot to architecture_latency_comparison.png")
