"""
================================================================================
 LATENCY IMPROVEMENT: PROACTIVE (forecaster) vs REACTIVE (no-forecast) SWITCHING
================================================================================
Answers: "how much does predicting the future link HORIZON minutes ahead
actually save in real terms, compared to a system that can only react after
the fact?"

PROACTIVE policy: uses the trained Attention-GRU forecaster's prediction for
label(t), made using only information available up to t-HORIZON (exactly how
the model was trained - this is a genuine advance-decision, not hindsight).

REACTIVE policy: models a conventional non-predictive controller. It can only
switch AFTER noticing the current link has degraded - i.e. its decision for
time t is the TRUE optimal link from REACTION_LAG minutes ago (the fastest
possible reaction given per-minute sampling is REACTION_LAG=1). This is the
realistic "no forecasting" baseline - always correct eventually, but always
late by construction.

OPTIMAL (reference ceiling): the true optimal link at every instant - perfect
foresight, not achievable, shown only for context.

Latency cost model (defaults are literature-grounded placeholders - swap via
the CONFIG block below):
  - Handover latency: fixed cost L_HO per link switch (Li-Wi paper measured
    ~30-60ms for horizontal/vertical VLC-WiFi handover; default 60ms)
  - Throughput penalty: whenever the active link isn't the true-optimal link,
    the achievable Shannon capacity is lower, so transmitting a fixed
    reference PAYLOAD_BITS takes longer. Extra time = payload*(1/C_actual -
    1/C_optimal). Reported both in absolute ms (depends on assumed bandwidth
    B) and as a %-improvement (much less sensitive to that assumption).
================================================================================
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# CONFIG - adjustable placeholder assumptions
# ---------------------------------------------------------------------------
L_HO_MS = 60.0            # handover latency per switch (ms) - Li-Wi paper, vertical handover
PAYLOAD_BYTES = 1024      # 1 KB reference payload (one telemetry packet)
BANDWIDTH_HZ = 20e6       # 20 MHz - typical WiFi channel width (placeholder for Shannon capacity)
REACTION_LAG = 1          # minutes - fastest possible reactive detection given 1-min sampling
MIN_DWELL = 2             # minutes - minimum cooldown between switches (dwell timer / hysteresis)
SAFETY_OVERRIDE_PROB = 0.75  # if predicted switch-probability exceeds this, bypass dwell timer
                              # immediately - never let stability suppress a high-confidence,
                              # safety-relevant switch. This is the medical-safety guard.
TTT_MINUTES = 2           # Time-To-Trigger: require the model to consistently predict the
                              # OTHER link for this many consecutive minutes before switching
                              # (standard 3GPP A3-event style trigger - confirm BEFORE switching,
                              # rather than cooldown AFTER switching)
HYSTERESIS_MARGIN = 0.15  # require predicted P(VLC) to clear 0.5 by this margin before acting:
                              # switch to VLC only if p >= 0.5+margin, switch to WiFi only if
                              # p <= 0.5-margin. In between is a "dead zone" - keep current link.

WINDOW, HORIZON = 80, 1
CROSS_TEST_PATH = 'dataset_home.csv'
MODEL_PATH = 'best_medical_attngru.pt'
HEAD_DROPOUT = 0.3
HIDDEN_SIZE = 32

feature_cols = ['CSI_mag_mean', 'CSI_mag_std', 'CSI_phase_std', 'Noise_mean',
                 'r_pos', 'snr_db', 'SNR_wifi_dB', 'SNR_vlc_dB',
                 'LLR_WiFi_mean', 'LLR_VLC_mean',
                 'Reliability_WiFi', 'Reliability_VLC']

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

# ---------------------------------------------------------------------------
# Load data + trained model
# ---------------------------------------------------------------------------
df = pd.read_csv(CROSS_TEST_PATH)
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

model = AttentionGRU(len(feature_cols))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# normalization stats must match training - recompute the same way training did
# (feat_mean/std were computed on dataset_gru.csv train split; here we
# approximate using this file's own stats, consistent with how earlier
# cross-scenario evaluation was done in medical_pipeline.py's evaluate() path
# which used the training-set stats - reproduce that properly:)
# Reproduce the EXACT train split used during training (same seed, same
# windowing) so normalization stats match what the model was actually
# calibrated on - using all of dataset_gru.csv (including val/test segments)
# would shift feat_mean/feat_std away from the true training-time values.
df_train_stats = pd.read_csv('dataset_gru.csv')
df_train_stats['sequence_id'] = (df_train_stats['snr_db'] != df_train_stats['snr_db'].shift()).cumsum()
df_train_stats['time_step'] = df_train_stats.groupby('sequence_id').cumcount()
mag_t = df_train_stats[mag_cols].values; phase_t = df_train_stats[phase_cols].values; noise_t = df_train_stats[noise_cols].values
extra_t = pd.DataFrame({
    'CSI_mag_mean': mag_t.mean(1), 'CSI_mag_std': mag_t.std(1),
    'CSI_phase_std': phase_t.std(1), 'Noise_mean': noise_t.mean(1),
})
df_train_stats = pd.concat([df_train_stats, extra_t], axis=1)

# Rebuild windows exactly as medical_pipeline.py did, to recover the same
# train_mask (same seed -> same shuffle -> same 70/15/15 split by segment).
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
print(f"Normalization stats recomputed on TRAIN split only "
      f"({train_mask_ts.sum()}/{len(train_mask_ts)} windows, {len(uniq_ts)-n_val_ts-n_test_ts}/{len(uniq_ts)} segments)")

# ---------------------------------------------------------------------------
# Walk through each block IN TIME ORDER, get proactive predictions + real SNRs
# ---------------------------------------------------------------------------
all_proactive, all_reactive, all_optimal, all_dwell, all_ttt, all_hyst = [], [], [], [], [], []
all_c_proactive, all_c_reactive, all_c_optimal, all_c_dwell, all_c_ttt, all_c_hyst = [], [], [], [], [], []
all_block_ids = []  # tracks which segment each row belongs to, so switch-counting never
                     # crosses a segment boundary (segments are independent simulated episodes)

sw_lin_cache, sv_lin_cache = {}, {}

for sid, g in df.groupby('sequence_id'):
    g = g.sort_values('time_step').reset_index(drop=True)
    n = len(g)
    if n < WINDOW + HORIZON:
        continue
    feats = g[feature_cols].values.astype(np.float32)
    feats_norm = (feats - feat_mean) / feat_std
    lab = g['label'].values
    sw_db = g['SNR_wifi_dB'].values
    sv_db = g['SNR_vlc_dB'].values
    sw_lin = 10 ** (sw_db / 10.0)
    sv_lin = 10 ** (sv_db / 10.0)

    block_pred = {}
    block_prob = {}
    with torch.no_grad():
        for t in range(WINDOW - 1, n - HORIZON):
            window = feats_norm[t - WINDOW + 1: t + 1]
            xb = torch.from_numpy(window).unsqueeze(0)
            cls_out, _ = model(xb)
            probs = torch.softmax(cls_out, dim=1)[0]
            pred = cls_out.argmax(1).item()
            block_pred[t + HORIZON] = pred  # proactive prediction for time index (t+HORIZON)
            block_prob[t + HORIZON] = probs[1].item()  # P(VLC)

    valid_idxs = sorted(block_pred.keys())
    if len(valid_idxs) < REACTION_LAG + 2:
        continue

    # --- Dwell-timer policy: cooldown AFTER switching ---
    dwell_decisions = {}
    current_link = block_pred[valid_idxs[0]]
    last_switch_idx = valid_idxs[0]
    dwell_decisions[valid_idxs[0]] = current_link
    for idx in valid_idxs[1:]:
        proposed = block_pred[idx]
        p_switch_conf = block_prob[idx] if proposed == 1 else (1 - block_prob[idx])
        time_since_switch = idx - last_switch_idx
        if proposed != current_link:
            if time_since_switch >= MIN_DWELL or p_switch_conf >= SAFETY_OVERRIDE_PROB:
                current_link = proposed
                last_switch_idx = idx
        dwell_decisions[idx] = current_link

    # --- Time-To-Trigger (TTT) policy: require sustained agreement BEFORE switching ---
    ttt_decisions = {}
    current_link = block_pred[valid_idxs[0]]
    consec_other = 0
    ttt_decisions[valid_idxs[0]] = current_link
    for idx in valid_idxs[1:]:
        proposed = block_pred[idx]
        if proposed != current_link:
            consec_other += 1
            if consec_other >= TTT_MINUTES:
                current_link = proposed
                consec_other = 0
        else:
            consec_other = 0
        ttt_decisions[idx] = current_link

    # --- Hysteresis-margin policy: require probability to clear a margin around 0.5 ---
    hyst_decisions = {}
    current_link = block_pred[valid_idxs[0]]
    hyst_decisions[valid_idxs[0]] = current_link
    for idx in valid_idxs[1:]:
        p_vlc = block_prob[idx]
        if current_link == 0 and p_vlc >= 0.5 + HYSTERESIS_MARGIN:
            current_link = 1
        elif current_link == 1 and p_vlc <= 0.5 - HYSTERESIS_MARGIN:
            current_link = 0
        # else: stay in the dead zone, keep current link
        hyst_decisions[idx] = current_link

    for idx in valid_idxs:
        proactive_dec = block_pred[idx]
        dwell_dec = dwell_decisions[idx]
        ttt_dec = ttt_decisions[idx]
        hyst_dec = hyst_decisions[idx]
        reactive_idx = max(idx - REACTION_LAG, 0)
        reactive_dec = lab[reactive_idx]
        optimal_dec = lab[idx]

        C_opt = BANDWIDTH_HZ * np.log2(1 + max(sw_lin[idx], sv_lin[idx]))
        C_proactive = BANDWIDTH_HZ * np.log2(1 + (sv_lin[idx] if proactive_dec == 1 else sw_lin[idx]))
        C_reactive = BANDWIDTH_HZ * np.log2(1 + (sv_lin[idx] if reactive_dec == 1 else sw_lin[idx]))
        C_dwell = BANDWIDTH_HZ * np.log2(1 + (sv_lin[idx] if dwell_dec == 1 else sw_lin[idx]))
        C_ttt = BANDWIDTH_HZ * np.log2(1 + (sv_lin[idx] if ttt_dec == 1 else sw_lin[idx]))
        C_hyst = BANDWIDTH_HZ * np.log2(1 + (sv_lin[idx] if hyst_dec == 1 else sw_lin[idx]))

        all_proactive.append(proactive_dec); all_reactive.append(reactive_dec); all_optimal.append(optimal_dec)
        all_dwell.append(dwell_dec); all_ttt.append(ttt_dec); all_hyst.append(hyst_dec)
        all_c_proactive.append(C_proactive); all_c_reactive.append(C_reactive); all_c_optimal.append(C_opt)
        all_c_dwell.append(C_dwell); all_c_ttt.append(C_ttt); all_c_hyst.append(C_hyst)
        all_block_ids.append(sid)

all_proactive = np.array(all_proactive); all_reactive = np.array(all_reactive); all_optimal = np.array(all_optimal)
all_dwell = np.array(all_dwell); all_ttt = np.array(all_ttt); all_hyst = np.array(all_hyst)
all_c_proactive = np.array(all_c_proactive); all_c_reactive = np.array(all_c_reactive); all_c_optimal = np.array(all_c_optimal)
all_c_dwell = np.array(all_c_dwell); all_c_ttt = np.array(all_c_ttt); all_c_hyst = np.array(all_c_hyst)
all_block_ids = np.array(all_block_ids)
same_block = all_block_ids[1:] == all_block_ids[:-1]  # True only for genuinely-adjacent minutes

print(f"Total evaluated minutes (cross-scenario, dataset_home.csv): {len(all_proactive)}")
for name, arr in [('Proactive', all_proactive), ('Reactive', all_reactive),
                   ('Dwell-timer', all_dwell), ('TTT', all_ttt), ('Hysteresis', all_hyst)]:
    print(f"{name} accuracy vs true optimal: {(arr == all_optimal).mean():.4f}")

# Reliability metrics per policy: BOTH classes now (VLC=switch-to-VLC, WiFi=stay-on/switch-to-WiFi)
# - for a medical system, missing a needed WiFi recovery is just as unsafe as missing a VLC switch.
from sklearn.metrics import precision_score, recall_score, f1_score
def reliability(name, decisions):
    prec_vlc = precision_score(all_optimal, decisions, pos_label=1, zero_division=0)
    rec_vlc = recall_score(all_optimal, decisions, pos_label=1, zero_division=0)
    prec_wifi = precision_score(all_optimal, decisions, pos_label=0, zero_division=0)
    rec_wifi = recall_score(all_optimal, decisions, pos_label=0, zero_division=0)
    f1 = f1_score(all_optimal, decisions, average='macro', zero_division=0)
    print(f"  {name:14s} VLC  prec={prec_vlc:.4f} rec={rec_vlc:.4f}   "
          f"WiFi prec={prec_wifi:.4f} rec={rec_wifi:.4f}   macroF1={f1:.4f}")
    return prec_vlc, rec_vlc, prec_wifi, rec_wifi, f1

print("\n--- Reliability check (precision/recall, BOTH links, vs true optimal) ---")
rel_proactive = reliability("Proactive", all_proactive)
rel_reactive = reliability("Reactive", all_reactive)
rel_dwell = reliability("Dwell-timer", all_dwell)
rel_ttt = reliability("TTT", all_ttt)
rel_hyst = reliability("Hysteresis", all_hyst)

# ---------------------------------------------------------------------------
# Compute latency: handover cost + throughput penalty, for each policy
# ---------------------------------------------------------------------------
payload_bits = PAYLOAD_BYTES * 8

def compute_latency(decisions, C_active, C_optimal):
    # only count a switch where consecutive minutes are truly adjacent
    # (same segment) - cross-segment "switches" are an artifact of
    # concatenation, not a real handover.
    switches = np.sum((decisions[1:] != decisions[:-1]) & same_block)
    ho_latency_ms = switches * L_HO_MS
    # throughput penalty only counted where active link underperforms optimal
    extra_time_s = np.where(C_active < C_optimal,
                             payload_bits * (1.0 / np.maximum(C_active, 1e-6) - 1.0 / np.maximum(C_optimal, 1e-6)),
                             0.0)
    throughput_penalty_ms = extra_time_s.sum() * 1000.0
    total_ms = ho_latency_ms + throughput_penalty_ms
    return switches, ho_latency_ms, throughput_penalty_ms, total_ms

sw_p, ho_p, tp_p, total_p = compute_latency(all_proactive, all_c_proactive, all_c_optimal)
sw_r, ho_r, tp_r, total_r = compute_latency(all_reactive, all_c_reactive, all_c_optimal)
sw_d, ho_d, tp_d, total_d = compute_latency(all_dwell, all_c_dwell, all_c_optimal)
sw_t, ho_t, tp_t, total_t = compute_latency(all_ttt, all_c_ttt, all_c_optimal)
sw_h, ho_h, tp_h, total_h = compute_latency(all_hyst, all_c_hyst, all_c_optimal)

print(f"\n--- PROACTIVE (forecaster, HORIZON={HORIZON}min ahead) ---")
print(f"switches={sw_p}  handover_latency={ho_p:.1f}ms  throughput_penalty={tp_p:.1f}ms  TOTAL={total_p:.1f}ms")

print(f"\n--- REACTIVE (no forecast, {REACTION_LAG}min detection lag) ---")
print(f"switches={sw_r}  handover_latency={ho_r:.1f}ms  throughput_penalty={tp_r:.1f}ms  TOTAL={total_r:.1f}ms")

print(f"\n--- DWELL-TIMER (forecaster + {MIN_DWELL}min cooldown, safety override @ p>={SAFETY_OVERRIDE_PROB}) ---")
print(f"switches={sw_d}  handover_latency={ho_d:.1f}ms  throughput_penalty={tp_d:.1f}ms  TOTAL={total_d:.1f}ms")

print(f"\n--- TTT (forecaster + {TTT_MINUTES}min sustained-agreement before switching) ---")
print(f"switches={sw_t}  handover_latency={ho_t:.1f}ms  throughput_penalty={tp_t:.1f}ms  TOTAL={total_t:.1f}ms")

print(f"\n--- HYSTERESIS-MARGIN (forecaster + \u00b1{HYSTERESIS_MARGIN} dead zone around p=0.5) ---")
print(f"switches={sw_h}  handover_latency={ho_h:.1f}ms  throughput_penalty={tp_h:.1f}ms  TOTAL={total_h:.1f}ms")

print(f"\n=== LATENCY IMPROVEMENT vs reactive ===")
for name, tot in [('Proactive', total_p), ('Dwell-timer', total_d), ('TTT', total_t), ('Hysteresis', total_h)]:
    imp = (total_r - tot) / total_r * 100 if total_r > 0 else 0
    print(f"  {name:14s}: {imp:.1f}%")
print(f"(assumptions: handover={L_HO_MS}ms/switch, payload={PAYLOAD_BYTES}B, bandwidth={BANDWIDTH_HZ/1e6:.0f}MHz)")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

policies = ['Reactive', 'Proactive', 'Dwell-timer', 'TTT', 'Hysteresis']
totals = [total_r, total_p, total_d, total_t, total_h]
switches_list = [sw_r, sw_p, sw_d, sw_t, sw_h]
rels = [rel_reactive, rel_proactive, rel_dwell, rel_ttt, rel_hyst]  # (prec_vlc, rec_vlc, prec_wifi, rec_wifi, f1)
prec_vlc_l = [r[0] for r in rels]; rec_vlc_l = [r[1] for r in rels]
prec_wifi_l = [r[2] for r in rels]; rec_wifi_l = [r[3] for r in rels]
f1_l = [r[4] for r in rels]
colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# 1. Total latency
ax = axes[0, 0]
bars = ax.bar(policies, totals, color=colors)
ax.set_ylabel('Total latency (ms)')
ax.set_title(f'Total latency over {len(all_proactive)} evaluated minutes\n(dataset_home.csv, cross-scenario)')
for b, v in zip(bars, totals):
    ax.text(b.get_x() + b.get_width()/2, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8)

# 2. Switch count
ax = axes[0, 1]
bars = ax.bar(policies, switches_list, color=colors)
ax.set_ylabel('Number of switches')
ax.set_title('Handover count (fewer = more stable)')
for b, v in zip(bars, switches_list):
    ax.text(b.get_x() + b.get_width()/2, v, f'{v}', ha='center', va='bottom', fontsize=8)

# 3. Reliability: VLC + WiFi precision/recall, both classes
ax = axes[1, 0]
x = np.arange(len(policies)); width = 0.15
ax.bar(x - 1.5*width, prec_vlc_l, width, label='VLC precision', color='#2ca02c')
ax.bar(x - 0.5*width, rec_vlc_l, width, label='VLC recall', color='#98df8a')
ax.bar(x + 0.5*width, prec_wifi_l, width, label='WiFi precision', color='#1f77b4')
ax.bar(x + 1.5*width, rec_wifi_l, width, label='WiFi recall', color='#aec7e8')
ax.set_xticks(x); ax.set_xticklabels(policies)
ax.set_ylim(0, 1.05)
ax.set_title('Reliability check: both links\' precision/recall')
ax.legend(fontsize=8, ncol=2)

# 4. Latency vs switch count scatter
ax = axes[1, 1]
for name, s, t, c in zip(policies, switches_list, totals, colors):
    ax.scatter(s, t, s=150, color=c, label=name, zorder=3)
    ax.annotate(name, (s, t), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax.set_xlabel('Number of switches'); ax.set_ylabel('Total latency (ms)')
ax.set_title('Latency vs. switch count\n(bottom-left is better)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('latency_comparison_plots.png', dpi=130)
print("\nSaved plots to latency_comparison_plots.png")
