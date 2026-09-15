# AI-Powered-Hybrid-Communication-Framework-for-Future-Digital-Health-Systems
# Proactive WiFi 6 ↔ VLC Link Switching for Medical IoT

**B.Tech Project** — a complete, three-paradigm machine-learning study of intelligent link switching between WiFi 6 and Visible Light Communication (VLC), culminating in a forecasting-based controller that makes the switching decision *before* a link degrades instead of after.

---

## Motive

A wearable health-monitoring device (glucose, hydration, lactate, fall detection, gait, lung capacity, cough rate, tremor) streams live telemetry over a dual WiFi/VLC link. Each link's quality drifts constantly with motion, obstruction, and interference, and a late or dropped reading — a missed fall alert, an undetected lactate spike — has real consequences. The system needs to continuously pick the better link, and do it fast enough that the switch actually happens before the data is lost, not after.

The most accurate signal of link quality, Bit Error Rate, is only known *after* a packet has been LDPC-decoded — by definition too late to inform the decision for that same packet. This single constraint is the reason the project has two eras rather than one: an initial phase that establishes how well switching *can* work given full post-decode information, and a second phase that removes that dependency entirely.

## Objective

1. Build a physically-grounded WiFi 6 / VLC channel simulator (Rayleigh WiFi + Lambertian VLC, LDPC-coded 16-QAM, real interleaving) and use it to generate labeled switching datasets.
2. Establish an accuracy ceiling: how well can a model choose the optimal link when it has access to post-decode channel-quality metrics (BER, reliability)?
3. Remove the causal-availability problem: build a model that forecasts the optimal link ahead of time from *pre-decode* channel-state trends only, and pair it with a decision layer that converts raw predictions into a stable real-world switching policy.
4. Quantify the actual real-world payoff — not just classification accuracy, but real-world handover latency — of forecasting-based (proactive) switching versus reactive switching.
5. Report all of this honestly: completed results are reported as completed, negative/ablation findings and bugs found during development are documented rather than hidden.

## What We Did

### Phase 1 — Classifier & Reinforcement-Learning Models (post-decode)

Simulated in MATLAB (`WIFi_LiFI_Data_set_generation5_fixed.m`): WiFi 6 OFDMA + DCO-OFDM VLC (Lambertian channel), LDPC-coded, AMC, evaluated across SNR 0–40 dB → `AI_switching_dataset_NEW.csv` (20,500 packets, 15 columns: per-link BER, reliability, latency, throughput, channel statistics, and the optimal-link label).

| Model | Approach | Result |
|---|---|---|
| **Hybrid CNN + SVM** | 1D-CNN embedding (10 physics-meaningful features) → RBF-SVM, grid-searched (best: C=100, γ=0.01, balanced class weights) | **97.54%** hybrid accuracy (95.56% CNN-only); per-class accuracy 97.1% WiFi / 99.2% VLC / 95.3% MRC; ROC AUC 0.997–1.000 |
| **DQCNN** | Dueling Double DQN, 3-stream CNN Q-network (channel / performance / stats), behavioral-cloning warm start | **99.91%** agreement with the optimal label, near-diagonal confusion matrix |
| **Dueling Double DQN** | Same RL stack, plain MLP Q-network | **99.76%** agreement with the optimal label |

**Why we did this:** to confirm that the engineered features and reward/label design correctly capture the physics-optimal switching decision when full post-decode information is available, before addressing the deployability problem in Phase 2.

### Phase 2 — Forecasting-Based Proactive Switching (pre-decode)

A second, temporally-correlated MATLAB simulator (`generate_switching_dataset_timecorrelated_New.m`, AR(1)/random-walk channel fading, per-sample LDPC-decoded BER for WiFi/VLC/SC/MRC) produces `dataset_gru.csv` (~42,000 minute-level samples, training) and `dataset_home.csv` (an independently-generated scenario, used only for cross-scenario generalization testing).

Six forecasting architectures across three families were trained and compared on an identical pipeline (`src/01`–`src/04` in the repo): Attention-GRU, Attention-LSTM, Plain BiGRU, Plain BiLSTM, TCN, and a Transformer encoder.

**Results:**
- The primary forecaster (attention-augmented bidirectional GRU) predicts the better link 1 minute ahead with **~87–88% accuracy and AUC ≈ 0.95** on a completely unseen cross-scenario test set — confirming genuine generalization, not memorization.
- Across all six architectures, raw classification accuracy differs by **less than 0.5 points** once paired with the decision layer below — but real-world switching *latency* differs meaningfully by architecture (`src/07_architecture_latency_comparison.py`).
- A **hysteresis-margin decision layer** (requires the predicted probability to clear 0.5 by a margin before switching, evaluated against dwell-timer and 3GPP-style Time-To-Trigger alternatives — `src/06_latency_decision_policies.py`) cuts total real-world handover latency by up to **~49%** versus a reactive baseline, while keeping classification reliability within ~1 point of the raw forecaster.
- A secondary multi-horizon model (`src/02_train_multihorizon_gru.py`) forecasts {1, 5, 10} minutes ahead simultaneously via parallel output heads.

**Why we did this:** classification accuracy alone doesn't tell you whether a model is deployable — a forecaster with slightly lower raw accuracy but far fewer unnecessary switches can produce dramatically lower real-world latency. Separating "is the prediction right?" from "does this make a good switching policy?" is the actual contribution of this phase.

### Honesty in what's reported

Two real bugs were found and fixed during development, and are documented rather than silently corrected:
1. **Cross-segment switch counting** — early latency scripts counted the boundary between two unrelated test segments as a switch; fixed by tracking segment IDs.
2. **Normalization statistics** were briefly computed on the full dataset instead of the train split only; corrected to exactly reproduce training-time normalization.

Negative/ablation findings are also kept in, not hidden: naive "assume no change" persistence is a genuinely strong baseline at short horizons (80–93% depending on dataset/horizon), and forecasting features derived from the VLC receiver are only legitimate under a **dual-simultaneous-receiver** hardware assumption — under a single-radio assumption, forecasting accuracy drops sharply (macro-F1 ~0.52 vs ~0.87), since most of the apparent accuracy would otherwise come from being handed a near-answer rather than genuine predictive power. See `docs/METHODOLOGY.md` in the repo for the full detail.

## Why We Did This — the overall narrative

Reactive, BER-driven switching (Phase 1) and forecasting-based switching (Phase 2) are not a discarded approach followed by its replacement — they answer two different questions:

| | Phase 1 (Classifier / RL) | Phase 2 (Forecasting) |
|---|---|---|
| Decision basis | Post-decode BER, reliability | Pre-decode CSI / SNR trend |
| Causal at decision time? | No — needs the decode result | Yes — decision made before degradation |
| Best-known result | 99.91% (DQCNN) | ~87–88% accuracy, ~49% latency reduction |
| Best fit | Fast pilot-based BER estimation available | Hard real-time latency budget |

A natural next step is a **hybrid controller**: the forecaster makes the fast, proactive first decision, periodically corrected by the classifier/RL model once decode results become available — combining the forecaster's speed with the classifier's accuracy ceiling.

## Repository Structure

```
├── classifier-phase/
│   ├── WIFi_LiFI_Data_set_generation5_fixed.m   # Phase 1 MATLAB dataset generator
│   ├── AI_switching_dataset_NEW.csv              # Phase 1 dataset (20,500 packets)
│   ├── cnn_svm.ipynb                             # Hybrid CNN + SVM (97.54%)
│   ├── DQN_Model.ipynb                           # Dueling Double DQN (99.76%)
│   └── Hybrid_Model.ipynb                        # DQCNN (99.91%)
├── forecasting-phase/                            # (btp-github-repo)
│   ├── src/
│   │   ├── 01_train_medical_attention_gru.py           # Primary forecaster (HORIZON=1)
│   │   ├── 02_train_multihorizon_gru.py                # Multi-horizon forecaster (H=1,5,10)
│   │   ├── 03_train_architectures_gru_lstm_bigru_tcn.py # Architecture sweep, part 1
│   │   ├── 04_train_plain_lstm_transformer.py           # Architecture sweep, part 2
│   │   ├── 05_cross_scenario_generalization.py         # Train/test on independent scenarios
│   │   ├── 06_latency_decision_policies.py             # Reactive vs proactive vs dwell/TTT/hysteresis
│   │   └── 07_architecture_latency_comparison.py       # Unified latency comparison
│   ├── models/                                    # Trained checkpoints (.pt)
│   ├── results/
│   │   ├── architecture_latency_comparison.png
│   │   └── latency_comparison_plots.png
│   ├── docs/METHODOLOGY.md                        # Full methodology, caveats, bugs found
│   └── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r forecasting-phase/requirements.txt
python forecasting-phase/src/01_train_medical_attention_gru.py
```

Place `dataset_gru.csv` and `dataset_home.csv` (generated by the Phase 2 MATLAB script) in the working directory before running the forecasting scripts — they are not included in the repo due to size.

## Future Work

- Build the hybrid controller described above (forecaster + periodic classifier correction).
- Extend the latency cost model with measured (rather than literature-derived) handover timing.
- Re-run the Phase 1 classifiers on the time-correlated Phase 2 dataset for a fully apples-to-apples comparison.

## Acknowledgment

Built iteratively with extensive use of Claude (Anthropic) for implementation, debugging, and methodology review — including identifying and fixing the two real bugs listed above.
