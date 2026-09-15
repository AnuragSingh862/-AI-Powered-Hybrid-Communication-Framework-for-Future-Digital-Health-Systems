% =========================================================================
% WiFi 6 + VLC  —  LLR-level Diversity Combining
% Hard Switching Selection (HSS) + Maximum Ratio Combining (MRC)
% Interleaver added to BOTH chains (same seed = bit-aligned LLRs)
% AMC + LDPC on both links
% =========================================================================
clear; clc; close all;
%rng(42);
rng('shuffle');
% Run once — tells MATLAB which Python to use
addpath('D:\files');
%% =====================================================================
%  SECTION 1: LOAD DATA
%% =====================================================================
try
   % txt   = fileread('basic_health_data_with_anomalies_30000.csv');
   txt = fileread('Sdata.csv');
    bytes = unicode2native(txt, 'UTF-8');
    bits  = reshape(de2bi(bytes, 8, 'left-msb')', [], 1);
    bits  = double(bits);
    disp('CSV loaded successfully.');
catch
    disp('CSV not found - using random bits.');
    bits = randi([0 1], 64800, 1);
end
bits = bits(1:min(end, 64800));

%% =====================================================================
%  SECTION 2: SIMULATION PARAMETERS
%% =====================================================================
SNR_dB    = 0:1:40;
N_SNR     = length(SNR_dB);
N_packets = 100;

% Result arrays — 4 curves
BER_WiFi  = zeros(1,N_SNR);  BER_VLC   = zeros(1,N_SNR);
BER_HSS   = zeros(1,N_SNR);  BER_MRC   = zeros(1,N_SNR);
BER_AI   = zeros(1,N_SNR);
Tput_WiFi = zeros(1,N_SNR);  Tput_VLC  = zeros(1,N_SNR);
Tput_HSS  = zeros(1,N_SNR);  Tput_MRC  = zeros(1,N_SNR);
Tput_AI = zeros(1,N_SNR);
Lat_WiFi  = zeros(1,N_SNR);  Lat_VLC   = zeros(1,N_SNR);
Lat_HSS   = zeros(1,N_SNR);  Lat_MRC   = zeros(1,N_SNR);
Lat_AI   = zeros(1,N_SNR);

% AI DATASET
%% =====================================================
ML_dataset = [];

%% =====================================================================
%  SECTION 3: WiFi 6 OFDMA PARAMETERS  (106-tone RU, 20 MHz)
%% =====================================================================
Nfft_W    = 256;
cp_W      = 64;
sym_len_W = Nfft_W + cp_W;
Fs_W      = 20e6;

ru_tones    = [-54:-2, 2:54];
pilot_tones = [-22, -10, 10, 22];
data_tones  = setdiff(ru_tones, pilot_tones);

fft_shift_W  = Nfft_W/2 + 1;
ru_idx       = ru_tones    + fft_shift_W;
pilot_idx    = pilot_tones + fft_shift_W; %#ok<NASGU>
data_idx     = data_tones  + fft_shift_W;

[~, d_loc_W] = ismember(data_idx, ru_idx);

rng('shuffle');
HE_LTF_seq = randi([0 1], length(ru_idx), 1) * 2 - 1;
rng('shuffle');

%% =====================================================================
%  SECTION 4: VLC DCO-OFDM PARAMETERS
%% =====================================================================
Nfft_V         = 256;
cp_V           = 64;
Fs_V           = 20e7;
dataCarriers_V = Nfft_V/2 - 1;   % = 31

% Lambert optical channel constants
semiangle_deg = 50;
m_lambert     = -1 / log2(cos(deg2rad(semiangle_deg)));
%R_p  = 0.6;  U = 0.9;  G = 15;  A_p = 0.002;
R_p  = 0.9;  U = 8;  G = 25;  A_p = 0.01;
L_room = 1.8;  r_max = 2;
c_vlc = (m_lambert+1) * (L_room^(m_lambert+1)) * U * G * R_p * A_p / (2*pi);

%% =====================================================================
%  SECTION 5: AMC TABLE  (shared — same MCS on both links)
%% =====================================================================
MCS_table = [
      2,  1/2,  1;   % BPSK    1/2
      4,  1/2,  2;   % QPSK    1/2
      4,  3/4,  2;   % QPSK    3/4
     16,  1/2,  4;   % 16-QAM  1/2
     16,  3/4,  4;   % 16-QAM  3/4
     64,  2/3,  6;   % 64-QAM  2/3
     64,  3/4,  6;   % 64-QAM  3/4
    256,  3/4,  8;   % 256-QAM 3/4
    256,  5/6,  8;   % 256-QAM 5/6
];
MCS_thresh = [0, 5, 8, 10, 15, 20, 25, 30, 35];

%% =====================================================================
%  SECTION 6: MAIN SNR LOOP
%% =====================================================================
disp('=================================================================');
disp(' WiFi 6 + VLC  LLR Combining  (Interleaver on BOTH chains)');
disp(' HSS = select best LLR   |   MRC = add LLRs   |   AMC + LDPC');
disp('=================================================================');

rxBits_MRC = zeros(length(bits), 1);   % pre-declare for Section 7

% NOTE: rfModel is trained AFTER the SNR loop generates AI_switching_dataset.csv
% A placeholder is created here so the packet-loop prediction step is skipped
% until the real model is available on the second run.
rfModel = [];

for s = 1:N_SNR

    snr_db  = SNR_dB(s);
    snr_lin = 10^(snr_db/10);

    %% ---- AMC: pick MCS ----
    mcs_idx   = select_mcs(snr_db, MCS_thresh);
    M         = MCS_table(mcs_idx, 1);
    R         = MCS_table(mcs_idx, 2);
    k         = MCS_table(mcs_idx, 3);   % bits per symbol = log2(M)
    modLabel  = sprintf('%d-QAM', M);
    rateLabel = strtrim(rats(R));

    %% ---- LDPC setup (rate matches AMC) ----
    H          = dvbs2ldpc(R);
    cfgLDPCEnc = ldpcEncoderConfig(H);
    cfgLDPCDec = ldpcDecoderConfig(H);
    k_ldpc     = size(H,2) - size(H,1);   % info bits per codeword
    n_ldpc     = size(H,2);               % coded bits per codeword

    %% ---- Pad raw bits to multiple of k_ldpc ----
    pad_ldpc  = mod(-length(bits), k_ldpc);
    msg_bits  = [bits; zeros(pad_ldpc, 1)];
    numBlocks = length(msg_bits) / k_ldpc;
    msgMat    = reshape(msg_bits, k_ldpc, numBlocks);

    %% ---- LDPC encode ----
    encMat = zeros(n_ldpc, numBlocks);
    for b = 1:numBlocks
        encMat(:,b) = ldpcEncode(msgMat(:,b), cfgLDPCEnc);
    end
    encBits = encMat(:);   % length = n_ldpc * numBlocks

    %% ==============================================================
    %%  INTERLEAVER  (SHARED by both chains — identical permutation)
    %%  WHY SAME SEED: TX and RX must use the same randperm.
    %%  WHY SAME FOR BOTH LINKS: so llr_W(i) and llr_V(i) correspond
    %%  to the same coded bit, making HSS / MRC mathematically valid.
    %% ==============================================================
    pad_int  = mod(-length(encBits), k);
    txBits   = [encBits; zeros(pad_int, 1)];

    rng('shuffle');                          % <-- fixed seed, same every iteration
    intr     = randperm(length(txBits));
    deintr   = zeros(size(intr));
    deintr(intr) = 1:length(txBits); % inverse permutation
    txBits_int   = txBits(intr);     % interleaved coded bits

    %% ---- QAM modulation  (shared symbols fed to both OFDM frames) ----
    sym_int = bi2de(reshape(txBits_int, k, [])', 'left-msb');
    txMod   = qammod(sym_int, M, 'UnitAveragePower', true);

    %% ============================================================
    %%  WiFi 6 OFDM FRAME BUILD
    %% ============================================================
    numSym_W = ceil(length(txMod) / length(data_idx));
    txMod_W  = [txMod; zeros(numSym_W*length(data_idx)-length(txMod), 1)];
    txMat_W  = reshape(txMod_W, length(data_idx), numSym_W);

    ofdmGrid = zeros(Nfft_W, numSym_W + 2);
    ofdmGrid(ru_idx,    1)     = HE_LTF_seq;   % HE-LTF preamble #1
    ofdmGrid(ru_idx,    2)     = HE_LTF_seq;   % HE-LTF preamble #2
    ofdmGrid(data_idx,  3:end) = txMat_W;
    ofdmGrid(pilot_idx, 3:end) = 1;

    tx_ifft_W = ifft(ifftshift(ofdmGrid, 1), Nfft_W);
    tx_cp_W   = [tx_ifft_W(end-cp_W+1:end,:); tx_ifft_W];
    txSig_W   = tx_cp_W(:);
    T_pkt_W   = (numSym_W + 2) * sym_len_W / Fs_W;

    %% ============================================================
    %%  VLC DCO-OFDM FRAME BUILD  (now with interleaver)
    %%  NOTE: txMod is THE SAME interleaved QAM symbols as WiFi 6.
    %%  The difference is only in the OFDM framing and channel.
    %% ============================================================
    numSym_V = ceil(length(txMod) / dataCarriers_V);
    txMod_V  = [txMod; zeros(numSym_V*dataCarriers_V-length(txMod), 1)];
    txMat_V  = reshape(txMod_V, dataCarriers_V, numSym_V);

    % Hermitian symmetry forces real IFFT output (required for intensity mod)
    ofdmFrame_V = zeros(Nfft_V, numSym_V);
    ofdmFrame_V(2:dataCarriers_V+1, :)             = txMat_V;
    ofdmFrame_V(Nfft_V-dataCarriers_V+1:Nfft_V, :) = conj(flipud(txMat_V));
    ofdmFrame_V(Nfft_V/2+1, :)                      = 0;   % Nyquist = 0

    tx_time_V = ifft(ofdmFrame_V, Nfft_V, 'symmetric');   % real output
    DC_bias   = 2 * std(tx_time_V(:));                     % lift above zero
    tx_time_V = tx_time_V + DC_bias;
    tx_cp_V   = [tx_time_V(end-cp_V+1:end,:); tx_time_V];
    T_pkt_V   = (numSym_V + 2) * (Nfft_V + cp_V) / Fs_V;

    %% ============================================================
    %%  PACKET LOOP
    %% ============================================================
    err_W = 0;  err_V = 0;  err_HSS = 0;  err_MRC = 0; err_AI=0;

    for pkt = 1:N_packets

        %% ------ WiFi 6 chain ------

        % Rayleigh multipath fading (8-tap complex FIR)
        L_ch  = 8;
        h_ch  = (randn(L_ch,1) + 1i*randn(L_ch,1)) / sqrt(2*L_ch);
        rxFad = filter(h_ch, 1, txSig_W);

        % Complex AWGN
        sig_pw  = mean(abs(rxFad).^2);
        noise   = sqrt(sig_pw/(2*snr_lin)) * ...
                  (randn(size(rxFad)) + 1i*randn(size(rxFad)));
        rxSig_W = rxFad + noise;

        % OFDM demodulation
        rxMat_W = reshape(rxSig_W, sym_len_W, []);
        rxMat_W = rxMat_W(cp_W+1:end, :);
        rxFFT_W = fftshift(fft(rxMat_W, Nfft_W), 1);

        % HE-LTF preamble channel estimation
        rx_LTF  = mean(rxFFT_W(ru_idx, 1:2), 2);
        H_est_W = rx_LTF ./ HE_LTF_seq;

        % Zero-forcing equalisation on data subcarriers
        H_data_W = H_est_W(d_loc_W);
        rxData_W = rxFFT_W(data_idx, 3:end) ./ repmat(H_data_W, 1, numSym_W);
        rxSym_W  = rxData_W(:);
        rxSym_W  = rxSym_W(1:length(sym_int));

        % Per-subcarrier noise variance for accurate soft LLR
        nv_base_W = (length(ru_idx)/Nfft_W) / snr_lin;
        nv_W      = nv_base_W ./ (abs(H_data_W).^2);
        nv_W_flat = repmat(nv_W, 1, numSym_W);
        nv_W_flat = nv_W_flat(:);
        nv_W_flat = nv_W_flat(1:length(sym_int));

        % Soft LLRs
        llr_W_raw = qamdemod(rxSym_W, M, 'OutputType','approxllr', ...
                             'UnitAveragePower',true, 'NoiseVariance',nv_W_flat);
        llr_W_raw = llr_W_raw(:);
        llr_W_raw = llr_W_raw(1:length(txBits));

        % De-interleave  ->  WiFi LLR vector aligned to coded-bit order
        llr_W = llr_W_raw(deintr);
        llr_W = llr_W(1:length(encBits));

        %% ------ VLC chain (now WITH interleaver) ------
%%%%%%%%%%%%%%%%%%%%%% Without Blockage %%%%%%%%%%%%%%%%
        % Lambert LOS fading: random position in room
        r_pos = r_max * sqrt(rand(1, numSym_V));
      %  h_vlc = c_vlc * ((r_pos.^2 + L_room^2)).^(-0.5*m_lambert - 1.5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Blockage modeling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%%%%%%
%% VLC Blockage: Stochastic Geometry (Poisson Point Process)
lambda = 0.2;       % Blocker density (persons per m^2)
r_B = 0.15;         % Human radius (m)
h_B = 1.7;          % Human height (m)
h_tx = 3.0;         % TX height
h_rx = 0.8;         % RX height
%r_pos = 1.5;        % Horizontal Tx-Rx distance

% Effective shadow area (A_shadow)
% A blocker's center falling in this area on the floor blocks the link
A_shadow = 2 * r_B .* r_pos* (h_B / (h_tx - h_rx)); 

% Blockage Probability (P_b) based on PPP
P_b = 1 - exp(-lambda * A_shadow); %

% Determine current state
is_blocked_stochastic = rand < P_b;
%h_vlc = (is_blocked_stochastic * 0.1 + ~is_blocked_stochastic * 1.0).*c_vlc .* ((r_pos.^2 + L_room^2)).^(-0.5*m_lambert - 1.5);
       
%% VLC Blockage: Cylindrical Geometry Model
tx_pos = [2.5, 2.5, 3.0]; % [x, y, z]
rx_pos = [1.0, 1.5, 0.8]; 
human_pos = [2.0, 2.1];   % [x, y] center of human on floor
r_human = 0.15; 
h_human = 1.7;

% 1. 2D projection intersection check (Distance to line segment)
v = rx_pos(1:2) - tx_pos(1:2);
w = human_pos - tx_pos(1:2);
proj = dot(w, v) / dot(v, v);
proj = max(0, min(1, proj)); % Closest point on line segment
closest_pt = tx_pos(1:2) + proj * v;
dist_to_path = norm(human_pos - closest_pt);

% 2. Height check (Is the beam above the human at intersection?)
beam_height_at_blocker = tx_pos(3) + proj * (rx_pos(3) - tx_pos(3));

% Result
is_blocked_cyl = (dist_to_path <= r_human) && (h_human > beam_height_at_blocker);
h_vlc = (is_blocked_cyl * 0.1 + ~is_blocked_cyl * 1.0) .*c_vlc .* ((r_pos.^2 + L_room^2)).^(-0.5*m_lambert - 1.5);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply VLC channel: scalar gain per OFDM symbol
        rx_time_V  = tx_cp_V .* h_vlc;
        tx_ser_V   = rx_time_V(:);

        % Real AWGN (VLC is an intensity channel — noise is real)
        sigPow_dBW = 10*log10(var(tx_ser_V));
        rxSig_V    = awgn(tx_ser_V, snr_db, sigPow_dBW);

        % OFDM demodulation: strip CP, equalise channel, remove DC bias
        rxMat_V = reshape(rxSig_V, Nfft_V+cp_V, numSym_V);
        rxMat_V = rxMat_V(cp_V+1:end, :);
        %rxMat_V = rxMat_V ./ h_vlc - DC_bias;   % undo gain + bias
        % FIX 1: VLC receiver order
        rxMat_V = rxMat_V - h_vlc .* DC_bias;
         rxMat_V = rxMat_V ./ h_vlc;
        rxFFT_V = fft(rxMat_V, Nfft_V);
        rxSym_V = rxFFT_V(2:dataCarriers_V+1, :);
        rxSym_V = rxSym_V(:);
        rxSym_V = rxSym_V(1:length(sym_int));

        % Noise variance for VLC: AWGN-only after perfect scalar equalisation
      %  nv_V = 1 ./ (2 * snr_lin);
      % FIX 2: VLC channel-aware noise variance
%nv_V = 1 ./ (2 .* snr_lin .* (abs(h_vlc)).^2);
nv_V = 1 ./ (2 .* snr_lin);
%nv_V_flat = repmat(nv_V, dataCarriers_V, 1);
%nv_V_flat = nv_V_flat(:);
%nv_V_flat = nv_V_flat(1:length(sym_int));

        % Soft LLRs
        llr_V_raw = qamdemod(rxSym_V, M, 'OutputType','approxllr', ...
                             'UnitAveragePower',true, 'NoiseVariance',nv_V);
        llr_V_raw = llr_V_raw(:);
        llr_V_raw = llr_V_raw(1:length(txBits));

        % De-interleave  ->  VLC LLR vector  (same deintr as WiFi!)
        %
        % KEY POINT: because both chains used the same txBits_int
        % (same interleaver, same rng(42) seed), llr_W(i) and llr_V(i)
        % both carry soft information about the SAME coded bit i.
        % This is what makes HSS and MRC mathematically meaningful.
        llr_V = llr_V_raw(deintr);
        llr_V = llr_V(1:length(encBits));

        %% ==============================================================
        %%  LLR-LEVEL COMBINING
        %% ==============================================================

        % --- Hard Switching Selection (HSS) ---
        % Measure reliability of each link by mean |LLR| magnitude.
        % Higher |LLR| = more confident soft decisions = better link.
        % Select the entire LLR vector from the winning link.
        reliability_W = mean(abs(llr_W));
        reliability_V = mean(abs(llr_V));
%% =====================================================
% FEATURE EXTRACTION FOR AI
% %% =====================================================
% 
% % WiFi channel features
% mean_H_wifi = mean(abs(H_data_W));
% var_H_wifi  = var(abs(H_data_W));
% 
% % VLC channel features
% mean_h_vlc = mean(h_vlc);
% var_h_vlc  = var(h_vlc);
% 
% % Packet latency estimate
% lat_wifi_pkt = T_pkt_W * 1e3;
% lat_vlc_pkt  = T_pkt_V * 1e3;
% lat_mrc_pkt  = (T_pkt_W + T_pkt_V)/2 * 1e3;
% 
% % Packet throughput estimate
% tput_wifi_pkt = (length(bits)/T_pkt_W)/1e6;
% tput_vlc_pkt  = (length(bits)/T_pkt_V)/1e6;
% tput_mrc_pkt  = length(bits)/((T_pkt_W+T_pkt_V)/2)/1e6;
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if reliability_W >= reliability_V
            llr_HSS      = llr_W;
            selected_link = 'W';
        else
            llr_HSS      = llr_V;
            selected_link = 'V'; %#ok<NASGU>
        end

        % --- Maximum Ratio Combining (MRC) ---
        % Under independent Gaussian noise on both links, the optimal
        % soft combiner simply adds the LLR vectors element-by-element.
        % Proof: LLR_combined(i) = log[P(b=1|y_W,y_V)/P(b=0|y_W,y_V)]
        %                        = LLR_W(i) + LLR_V(i)   (independence)
        % The LDPC decoder then sees reinforced soft information from
        % both channels simultaneously — always >= HSS in BER.
        llr_MRC = llr_W + llr_V;

        %% ==============================================================
        %%  LDPC DECODING — four independent streams
        %% ==============================================================
        rxBits_W   = ldpc_decode(llr_W,   cfgLDPCDec, n_ldpc, numBlocks, k_ldpc, length(bits));
        rxBits_V   = ldpc_decode(llr_V,   cfgLDPCDec, n_ldpc, numBlocks, k_ldpc, length(bits));
        rxBits_HSS = ldpc_decode(llr_HSS, cfgLDPCDec, n_ldpc, numBlocks, k_ldpc, length(bits));
        rxBits_MRC = ldpc_decode(llr_MRC, cfgLDPCDec, n_ldpc, numBlocks, k_ldpc, length(bits));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Packet BERs AI
ber_pkt_wifi = sum(bits ~= rxBits_W)   / length(bits);
ber_pkt_vlc  = sum(bits ~= rxBits_V)   / length(bits);
ber_pkt_hss  = sum(bits ~= rxBits_HSS) / length(bits);
ber_pkt_mrc  = sum(bits ~= rxBits_MRC) / length(bits);

%% =====================================================
% FEATURE EXTRACTION
%% =====================================================
%% =====================================================
% ACTUAL CHANNEL FEATURES
%% =====================================================

% VLC absolute channel values
%h_vlc_feat = abs(h_vlc(:))';

% WiFi absolute channel matrix
%H_wifi_feat = abs(H_data_W(:))';
mean_H_wifi = mean(abs(H_data_W));
var_H_wifi  = var(abs(H_data_W));

mean_h_vlc = mean(h_vlc);
var_h_vlc  = var(h_vlc);

lat_wifi_pkt = T_pkt_W * 1e3;
lat_vlc_pkt  = T_pkt_V * 1e3;
lat_mrc_pkt  = (T_pkt_W + T_pkt_V)/2 * 1e3;

tput_wifi_pkt = (length(bits)/T_pkt_W)/1e6;
tput_vlc_pkt  = (length(bits)/T_pkt_V)/1e6;
tput_mrc_pkt  = length(bits)/((T_pkt_W+T_pkt_V)/2)/1e6;

%% =====================================================
% LABEL GENERATION
%% =====================================================

metric_V = 0.7*ber_pkt_vlc + ...
           0.3*(lat_vlc_pkt/100);

metric_W = 0.7*ber_pkt_wifi + ...
           0.3*(lat_wifi_pkt/100);

metric_M = 0.7*ber_pkt_mrc + ...
           0.3*(lat_mrc_pkt/100);

[~, label] = min([metric_V metric_W metric_M]);

%% =====================================================
% STORE DATASET SAMPLE
%% =====================================================

 sample = [
 snr_db,...
 mean_h_vlc,...
 var_h_vlc,...
 mean_H_wifi,...
 var_H_wifi,...
 reliability_V,...
 reliability_W,...
 ber_pkt_vlc,...
 ber_pkt_wifi,...
 ber_pkt_mrc,...
 lat_vlc_pkt,...
 lat_wifi_pkt,...
 tput_vlc_pkt,...
 tput_wifi_pkt,...
 label
 ];

% sample = [
% snr_db,...
% h_vlc_feat,...
% H_wifi_feat,...
% reliability_V,...
% reliability_W,...
% ber_pkt_vlc,...
% ber_pkt_wifi,...
% ber_pkt_mrc,...
% lat_vlc_pkt,...
% lat_wifi_pkt,...
% tput_vlc_pkt,...
% tput_wifi_pkt,...
% label
% ];

ML_dataset = [ML_dataset;
sample];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% INSIDE PACKET LOOP — after computing features "Random forest"
feat = [snr_db, mean_h_vlc, var_h_vlc, mean_H_wifi, var_H_wifi, ...
        reliability_V, reliability_W, ber_pkt_vlc, ber_pkt_wifi, ...
        ber_pkt_mrc, lat_vlc_pkt, lat_wifi_pkt, tput_vlc_pkt, tput_wifi_pkt];

% rfModel is only available on the second run (after dataset CSV is generated).
% Fall back to MRC when no trained model exists yet.
if ~isempty(rfModel)
    label_pred = str2double(predict(rfModel, feat));
else
    label_pred = 3;   % default to MRC on first run
end
% label_pred = 1 → use VLC, 2 → use WiFi, 3 → use MRC

if label_pred == 1
    llr_AI = llr_V;
elseif label_pred == 2
    llr_AI = llr_W;
else
    llr_AI = llr_W + llr_V;   % MRC
end

rxBits_AI = ldpc_decode(llr_AI, cfgLDPCDec, n_ldpc, numBlocks, k_ldpc, length(bits));


%% =====================================================
% ACCUMULATE ERRORS
%% =====================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Accumulate errors
        err_W   = err_W   + sum(bits ~= rxBits_W);
        err_V   = err_V   + sum(bits ~= rxBits_V);
        err_HSS = err_HSS + sum(bits ~= rxBits_HSS);
        err_MRC = err_MRC + sum(bits ~= rxBits_MRC);
err_AI  = err_AI   + sum(bits ~= rxBits_AI);
    end  % --- end packet loop ---

    %% ---- BER ----
    nb = length(bits) * N_packets;
    BER_WiFi(s) = err_W   / nb;
    BER_VLC(s)  = err_V   / nb;
    BER_HSS(s)  = err_HSS / nb;
    BER_MRC(s)  = err_MRC / nb;
    BER_AI(s)   = err_AI / nb;   
    %% ---- Throughput & Latency ----
    T_avg = (T_pkt_W + T_pkt_V) / 2;
    [Tput_WiFi(s), Lat_WiFi(s)] = tput_lat(BER_WiFi(s), length(bits), T_pkt_W);
    [Tput_VLC(s),  Lat_VLC(s)]  = tput_lat(BER_VLC(s),  length(bits), T_pkt_V);
    [Tput_HSS(s),  Lat_HSS(s)]  = tput_lat(BER_HSS(s),  length(bits), T_avg);
    [Tput_MRC(s),  Lat_MRC(s)]  = tput_lat(BER_MRC(s),  length(bits), T_avg);
    [Tput_MRC(s),  Lat_MRC(s)]  = tput_lat(BER_MRC(s),  length(bits), T_avg);
    [Tput_AI(s),  Lat_AI(s)]  = tput_lat(BER_AI(s),  length(bits), T_avg);

    fprintf('SNR=%2d dB | %s R=%s | WiFi=%.2e | VLC=%.2e | HSS=%.2e | MRC=%.2e | AI=%.2e\n', ...
        snr_db, modLabel, rateLabel, ...
        BER_WiFi(s), BER_VLC(s), BER_HSS(s), BER_MRC(s), BER_AI(s));

end  % --- end SNR loop ---

%% =====================================================
% SAVE AI TRAINING DATASET
%% =====================================================

feature_names = {
'SNR_dB',...
'Mean_hVLC',...
'Var_hVLC',...
'Mean_HWiFi',...
'Var_HWiFi',...
'Reliability_VLC',...
'Reliability_WiFi',...
'BER_VLC',...
'BER_WiFi',...
'BER_MRC',...
'Latency_VLC_ms',...
'Latency_WiFi_ms',...
'Throughput_VLC_Mbps',...
'Throughput_WiFi_Mbps',...
'Label'
};

T_AI = array2table(ML_dataset,...
'VariableNames',feature_names);

% Save in the same folder as this .m file (no hardcoded path needed)
save_path = fullfile(fileparts(mfilename('fullpath')), 'AI_switching_dataset.csv');
% If that returns empty (run from Command Window), fall back to current dir
if strcmp(save_path, 'AI_switching_dataset.csv') || isempty(fileparts(save_path))
    save_path = fullfile(pwd, 'AI_switching_dataset.csv');
end

writetable(T_AI, save_path);

disp(['Dataset saved at: ', save_path])

disp('===================================')
disp('AI DATASET GENERATED SUCCESSFULLY')
disp('File: AI_switching_dataset.csv')
disp(['Total Samples = ',...
num2str(size(ML_dataset,1))])
disp('===================================')

%% =====================================================================
%  TRAIN RANDOM FOREST MODEL  (now that the dataset CSV exists)
%% =====================================================================
disp(' ');
disp('Training Random Forest model on generated dataset...');
T_rf    = readtable(save_path);
X_train = T_rf{:, 1:14};   % feature columns
y_train = T_rf{:, 15};     % Label column (1, 2, or 3)

rfModel = TreeBagger(400, X_train, y_train, ...
    'Method', 'classification', ...
    'NumPredictorsToSample', 'all');

rf_model_path = fullfile(fileparts(save_path), 'rfModel.mat');
save(rf_model_path, 'rfModel');
disp(['RF model saved at: ', rf_model_path]);

%% ==========================================================
% AI MODEL EVALUATION
%% ==========================================================
disp(' ');
disp('==================================================');
disp(' AI MODEL PERFORMANCE');
disp('==================================================');

% Load dataset (use the same save_path defined above)
T = readtable(save_path);

X = T{:,1:14};
Y = categorical(T{:,15});

% Train/Test split
cv = cvpartition(Y,'HoldOut',0.2);

idxTrain = training(cv);
idxTest  = test(cv);

X_train = X(idxTrain,:);
Y_train = Y(idxTrain);

X_test = X(idxTest,:);
Y_test = Y(idxTest);

% Prediction
Y_pred = predict(rfModel,X_test);
Y_pred = categorical(str2double(Y_pred));

%% ==========================================================
% ACCURACY
%% ==========================================================
accuracy = mean(Y_pred == Y_test)*100;

fprintf('\nAI Accuracy = %.2f %%\n',accuracy);

%% ==========================================================
% CONFUSION MATRIX
%% ==========================================================
figure('Name','Confusion Matrix');

confusionchart(Y_test,Y_pred);

title('AI Switching Confusion Matrix');

%% ==========================================================
% PRECISION / RECALL / F1 SCORE
%% ==========================================================
C = confusionmat(Y_test,Y_pred);

precision = zeros(3,1);
recall    = zeros(3,1);
f1_score  = zeros(3,1);

for i = 1:3

    TP = C(i,i);

    FP = sum(C(:,i)) - TP;

    FN = sum(C(i,:)) - TP;

    precision(i) = TP/(TP+FP+eps);

    recall(i) = TP/(TP+FN+eps);

    f1_score(i) = ...
        2*(precision(i)*recall(i))/...
        (precision(i)+recall(i)+eps);
end

Results = table(...
precision,...
recall,...
f1_score,...
'VariableNames',...
{'Precision','Recall','F1Score'},...
'RowNames',...
{'VLC','WiFi','MRC'});

disp(Results)

%% ==========================================================
% FEATURE IMPORTANCE
%% ==========================================================
% importance = predictorImportance(rfModel);
% 
% figure('Name','Feature Importance');
% 
% bar(importance)
% 
% xticks(1:14)
% 
% xticklabels({...
% 'SNR',...
% 'hVLC mean',...
% 'hVLC var',...
% 'HWiFi mean',...
% 'HWiFi var',...
% 'Rel VLC',...
% 'Rel WiFi',...
% 'BER VLC',...
% 'BER WiFi',...
% 'BER MRC',...
% 'Lat VLC',...
% 'Lat WiFi',...
% 'Tput VLC',...
% 'Tput WiFi'});
% 
% xtickangle(45)
% 
% ylabel('Importance')
% 
% title('Random Forest Feature Importance')
% 
% grid on

%% ==========================================================
% SWITCHING DECISION HISTOGRAM
%% ==========================================================
figure('Name','AI Decision Histogram');

histogram(double(Y_pred))

xticks([1 2 3])

xticklabels({'VLC','WiFi','MRC'})

xlabel('Selected Link')

ylabel('Count')

title('AI Switching Distribution')

grid on

%% ==========================================================
% AI DECISION VS SNR
%% ==========================================================
figure('Name','Decision vs SNR');

scatter(X_test(:,1),...
        double(Y_pred),...
        40,...
        'filled')

yticks([1 2 3])

yticklabels({'VLC','WiFi','MRC'})

xlabel('SNR (dB)')

ylabel('AI Decision')

title('AI Switching Behaviour')

grid on

%% ==========================================================
% RELIABILITY PLOT
%% ==========================================================
figure('Name','Reliability');

plot(X_test(:,1),...
     X_test(:,6),...
     'o',...
     'LineWidth',2)

hold on

plot(X_test(:,1),...
     X_test(:,7),...
     's',...
     'LineWidth',2)

xlabel('SNR (dB)')

ylabel('Reliability')

legend('VLC','WiFi')

title('Reliability vs SNR')

grid on
hold on;

%% =====================================================================
%  SECTION 7: DATA RECONSTRUCTION  (MRC at highest SNR)
%% =====================================================================
disp(' ');
disp('==================================================');
disp('  DATA RECONSTRUCTION  -  MRC at peak SNR');
disp('==================================================');

rx_bytes = bi2de(reshape(rxBits_AI, 8, [])', 'left-msb');
rx_txt   = native2unicode(uint8(rx_bytes), 'UTF-8');
rx_txt   = rx_txt(:)';
nl_idx   = find(rx_txt == char(10), 1, 'last');
if ~isempty(nl_idx), rx_txt_clean = rx_txt(1:nl_idx);
else,                rx_txt_clean = rx_txt;
end

fname = 'recovered_health_data_MRC.csv';
fid   = fopen(fname, 'w');
fprintf(fid, '%s', rx_txt_clean);
fclose(fid);
fprintf('SUCCESS: AI recovered data written to "%s"\n', fname);

warning('off','MATLAB:table:ModifiedAndSavedVarnames');
try
    rec_table = readtable(fname, 'Delimiter', ',');
    disp('--- Received data (first 12 rows, MRC) ---');
    disp(head(rec_table, 12));
catch
    disp('Notice: Recovered file may be corrupted at this SNR.');
end
warning('on','MATLAB:table:ModifiedAndSavedVarnames');

%% =====================================================================
%  SECTION 8: MEDICAL ANOMALY DETECTION
%% =====================================================================
disp(' ');
disp('==================================================');
disp('  MEDICAL ANOMALY DETECTION  (CLINICAL BOUNDS)');
disp('==================================================');

if exist('rec_table','var') && istable(rec_table) && height(rec_table) > 0
    cn     = rec_table.Properties.VariableNames;
    hr_i   = find(~cellfun(@isempty,regexpi(cn,'heart_rate|hr|pulse')),1);
    hrv_i  = find(~cellfun(@isempty,regexpi(cn,'hrv')),1);
    spo2_i = find(~cellfun(@isempty,regexpi(cn,'spo2|oxygen|o2')),1);
    sys_i  = find(~cellfun(@isempty,regexpi(cn,'systolic')),1);
    dia_i  = find(~cellfun(@isempty,regexpi(cn,'diastolic')),1);
    resp_i = find(~cellfun(@isempty,regexpi(cn,'respiratory|resp')),1);
    tmp_i  = find(~cellfun(@isempty,regexpi(cn,'temperature|temp')),1);

    n_chk  = height(rec_table);
    fprintf('Scanning all %d records...\n', n_chk);
    n_anom = 0;

    for i = 1:n_chk
        flagged = false; msg = '';
        if ~isempty(hr_i),   v=rec_table{i,hr_i};   if isnumeric(v)&&~isnan(v); if v>120, msg=[msg,sprintf('[HR:%d->TACHYCARDIA] ',v)];    flagged=true; elseif v<50,  msg=[msg,sprintf('[HR:%d->BRADYCARDIA] ',v)];   flagged=true; end; end; end
        if ~isempty(hrv_i),  v=rec_table{i,hrv_i};  if isnumeric(v)&&~isnan(v)&&v<20,  msg=[msg,sprintf('[HRV:%d->STRESS] ',v)];           flagged=true; end; end
        if ~isempty(spo2_i), v=rec_table{i,spo2_i}; if isnumeric(v)&&~isnan(v)&&v<90,  msg=[msg,sprintf('[SpO2:%.1f%%->HYPOXIA] ',v)];     flagged=true; end; end
        if ~isempty(sys_i),  v=rec_table{i,sys_i};  if isnumeric(v)&&~isnan(v); if v>160,msg=[msg,sprintf('[SysBP:%d->HYPERT.] ',v)];      flagged=true; elseif v<90,  msg=[msg,sprintf('[SysBP:%d->HYPOT.] ',v)];    flagged=true; end; end; end
        if ~isempty(dia_i),  v=rec_table{i,dia_i};  if isnumeric(v)&&~isnan(v); if v>100,msg=[msg,sprintf('[DiaBP:%d->HYPERT.] ',v)];      flagged=true; elseif v<60,  msg=[msg,sprintf('[DiaBP:%d->HYPOT.] ',v)];    flagged=true; end; end; end
        if ~isempty(resp_i), v=rec_table{i,resp_i}; if isnumeric(v)&&~isnan(v); if v>25, msg=[msg,sprintf('[Resp:%d->TACHYPNEA] ',v)];     flagged=true; elseif v<10, msg=[msg,sprintf('[Resp:%d->BRADYPNEA] ',v)];  flagged=true; end; end; end
        if ~isempty(tmp_i),  v=rec_table{i,tmp_i};  if isnumeric(v)&&~isnan(v); if v>38, msg=[msg,sprintf('[Temp:%.1fC->FEVER] ',v)];      flagged=true; elseif v<35, msg=[msg,sprintf('[Temp:%.1fC->HYPO] ',v)];    flagged=true; end; end; end
        if flagged, n_anom=n_anom+1; fprintf(2,'ALERT | Record %03d: %s\n',i,msg); end
    end

    if n_anom==0, disp('Scan complete: All vitals normal.');
    else, fprintf('Scan complete: %d anomalies flagged.\n',n_anom); end
else
    disp('Notice: Valid table not available for anomaly scan.');
end

%% =====================================================================
%  SECTION 9: PLOTS
%% =====================================================================
pW   = max(BER_WiFi,1e-6);
pV   = max(BER_VLC, 1e-6);
pHSS = max(BER_HSS, 1e-6);
pMRC = max(BER_MRC, 1e-6);
pAI  = max(BER_AI, 1e-6);
cW='#D95319'; cV='#0072BD'; cH='#EDB120'; cM='#77AC30'; cAI = '#7E2F8E';   % AI (purple)

figure('Name','BER','Position',[50 80 720 520]);
semilogy(SNR_dB,pW,'o-','LineWidth',2,'Color',cW,'MarkerSize',7,'DisplayName','WiFi 6 (interleaved)');
hold on;
semilogy(SNR_dB,pV,'s-','LineWidth',2,'Color',cV,'MarkerSize',7,'DisplayName','VLC (interleaved)');
semilogy(SNR_dB,pHSS,'d--','LineWidth',2,'Color',cH,'MarkerSize',7,'DisplayName','HSS combining');
semilogy(SNR_dB,pMRC,'^-','LineWidth',2.5,'Color',cM,'MarkerSize',8,'DisplayName','MRC combining');
semilogy(SNR_dB,pAI,'p-','LineWidth',3,'Color',cAI,'MarkerSize',9,'DisplayName','AI-Based Switching');
yline(0.01,'k--','1% limit','LineWidth',1.5,'LabelVerticalAlignment','bottom');
grid on; grid minor; ylim([1e-6,1]);
xlabel('SNR (dB)'); ylabel('BER');
title('BER — WiFi 6 | VLC | HSS | MRC  (interleaver on both chains)');
legend('Location','southwest'); set(gca,'FontSize',11);

figure('Name','Throughput','Position',[800 80 720 520]);
plot(SNR_dB,Tput_WiFi,'o-','LineWidth',2,'Color',cW,'MarkerSize',7,'DisplayName','WiFi 6');
hold on;
plot(SNR_dB,Tput_VLC,'s-','LineWidth',2,'Color',cV,'MarkerSize',7,'DisplayName','VLC');
plot(SNR_dB,Tput_HSS,'d--','LineWidth',2,'Color',cH,'MarkerSize',7,'DisplayName','HSS');
plot(SNR_dB,Tput_MRC,'^-','LineWidth',2.5,'Color',cM,'MarkerSize',8,'DisplayName','MRC');
plot(SNR_dB,Tput_AI,'^-','LineWidth',2.5,'Color',cAI,'MarkerSize',8,'DisplayName','AI');
grid on; xlabel('SNR (dB)'); ylabel('Throughput (Mbps)');
title('Throughput comparison'); legend('Location','northwest'); set(gca,'FontSize',11);
ylim([-1, max([Tput_WiFi Tput_VLC Tput_HSS Tput_MRC])*1.15]);

figure('Name','Latency','Position',[50 650 720 420]);
plot(SNR_dB,Lat_WiFi,'o-','LineWidth',2,'Color',cW,'MarkerSize',7,'DisplayName','WiFi 6');
hold on;
plot(SNR_dB,Lat_VLC,'s-','LineWidth',2,'Color',cV,'MarkerSize',7,'DisplayName','VLC');
plot(SNR_dB,Lat_HSS,'d--','LineWidth',2,'Color',cH,'MarkerSize',7,'DisplayName','HSS');
plot(SNR_dB,Lat_MRC,'^-','LineWidth',2.5,'Color',cM,'MarkerSize',8,'DisplayName','MRC');
plot(SNR_dB,Lat_AI,'^-','LineWidth',2.5,'Color',cAI,'MarkerSize',8,'DisplayName','AI');
grid on; xlabel('SNR (dB)'); ylabel('Latency (ms)');
title('Latency comparison'); legend('Location','northeast'); set(gca,'FontSize',11);
ylim([0 55]);

figure('Name','Diversity Gain','Position',[800 650 720 420]);
plot(SNR_dB,10*log10(pW./pHSS),'d--','LineWidth',2,'Color',cH,'DisplayName','HSS gain over WiFi');
hold on;
plot(SNR_dB,10*log10(pV./pHSS),'d:','LineWidth',2,'Color',cH,'DisplayName','HSS gain over VLC');
plot(SNR_dB,10*log10(pW./pMRC),'^-','LineWidth',2.5,'Color',cM,'DisplayName','MRC gain over WiFi');
plot(SNR_dB,10*log10(pV./pMRC),'^:','LineWidth',2.5,'Color',cM,'DisplayName','MRC gain over VLC');
plot(SNR_dB,10*log10(pAI./pMRC),'^:','LineWidth',2.5,'Color',cAI,'DisplayName','AI gain over VLC LIFI_MRC');
yline(0,'k--','No gain','LineWidth',1.2);
grid on; xlabel('SNR (dB)'); ylabel('BER reduction (dB)');
title('Diversity gain of AI, HSS, and MRC vs individual links');
legend('Location','northwest'); set(gca,'FontSize',11);

disp(' '); disp('Done. 4 figures generated.');
