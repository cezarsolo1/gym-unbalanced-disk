import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
BASE = os.path.join(REPO_ROOT, 'disc-benchmark-files')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
raw    = np.load(os.path.join(BASE, 'training-val-test-data.npz'))
th_raw = raw['th'].astype(np.float32)
u_raw  = raw['u'].astype(np.float32)

# ── Normalize (zero-mean, unit variance) ──────────────────────────────────────
u_mean,  u_std  = float(u_raw.mean()),  float(u_raw.std())
th_mean, th_std = float(th_raw.mean()), float(th_raw.std())

u_n  = (u_raw  - u_mean)  / u_std
th_n = (th_raw - th_mean) / th_std

# ── Build overlapping training windows ────────────────────────────────────────
# At each step k: input = [u_k, th_{k-1}], target = th_k  (teacher-forcing)
SEQ_LEN = 64
STRIDE  = 32

th_prev_full = np.concatenate([[0.], th_n[:-1]])     # th_prev[k] = th[k-1]
inp_full     = np.stack([u_n, th_prev_full], axis=1) # (N, 2)

N = len(u_n)
Xs, Ys = [], []
for start in range(0, N - SEQ_LEN, STRIDE):
    Xs.append(inp_full[start : start + SEQ_LEN])
    Ys.append(th_n[start : start + SEQ_LEN])

X = np.array(Xs, dtype=np.float32)
Y = np.array(Ys, dtype=np.float32)

split = int(0.8 * len(X))
X_tr, Y_tr   = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

# ── LSTM model ────────────────────────────────────────────────────────────────
# Input:  [u_k, th_{k-1}]  (voltage + previous angle)
# Hidden: LSTM state approximates physical state [theta, omega]
# Output: th_k  (predicted current angle)
class LSTMSysID(nn.Module):
    def __init__(self, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.head(out).squeeze(-1), state  # (B, T), state

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

model = LSTMSysID().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-6)
crit  = nn.MSELoss()

ds_tr  = TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(Y_tr))
ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
dl_tr  = DataLoader(ds_tr,  batch_size=64, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=64)

EPOCHS     = 500
CKPT_PATH  = os.path.join(HERE, 'lstm_sysid_best.pt')
best_val   = float('inf')

if os.path.exists(CKPT_PATH):
    print('Checkpoint found — skipping training.')
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
else:
    print(f'Training for {EPOCHS} epochs...')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                crit(model(xb.to(DEVICE))[0], yb.to(DEVICE)).item() * len(xb)
                for xb, yb in dl_val
            ) / len(ds_val)

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CKPT_PATH)

        if epoch % 50 == 0:
            print(f'  Epoch {epoch:3d}  val_loss={val_loss:.6f}  (best={best_val:.6f})')

    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    print(f'Done (best val_loss={best_val:.6f})\n')

model.eval()

# ── 1-step prediction on full training set ────────────────────────────────────
inp_t = torch.from_numpy(inp_full[None].astype(np.float32)).to(DEVICE)
with torch.no_grad():
    th_pred_n, _ = model(inp_t)
th_pred = th_pred_n.squeeze().cpu().numpy() * th_std + th_mean

rmse_pred = np.sqrt(np.mean((th_pred[1:] - th_raw[1:])**2))
print(f'=== 1-step prediction (train) ===')
print(f'RMSE : {rmse_pred:.5f} rad  ({rmse_pred/(2*np.pi)*360:.3f}°)  NRMS: {rmse_pred/th_raw.std()*100:.2f}%')
print(f'Linear ARX baseline: 0.00665 rad\n')

# ── Plot: 1-step prediction (zoomed window for readability) ───────────────────
zoom = slice(2000, 3000)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(th_raw[zoom], label='measured', color='steelblue')
ax.plot(th_pred[zoom], label='LSTM prediction', color='crimson', ls='--')
ax.set_xlabel('sample'); ax.set_ylabel(r'$\theta$ [rad]')
ax.set_title('LSTM one-step-ahead prediction (zoom)')
ax.legend(); ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'LSTM_prediction.png'), dpi=200)
plt.close()
print(f'Saved {os.path.join(FIG_DIR, "LSTM_prediction.png")}')

# ── Simulation (autoregressive from step 50) ──────────────────────────────────
def simulate_lstm(model, u_n, th_n_init, skip=50, device='cpu'):
    model.eval()
    th_sim_n = list(th_n_init[:skip])

    # Warm-up: build hidden state from first `skip` true steps
    th_prev_init = np.concatenate([[0.], th_n_init[:skip - 1]])
    warm_inp = torch.tensor(
        np.stack([u_n[:skip], th_prev_init], axis=1)[None],
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        _, state = model(warm_inp)

    # Autoregressive rollout: only voltage inputs from here on
    th_k = float(th_n_init[skip - 1])
    with torch.no_grad():
        for k in range(skip, len(u_n)):
            inp = torch.tensor([[[u_n[k], th_k]]], dtype=torch.float32, device=device)
            out, state = model(inp, state)
            th_k = float(out.squeeze())
            th_sim_n.append(th_k)

    return np.array(th_sim_n, dtype=np.float32)

SKIP = 50
th_sim = simulate_lstm(model, u_n, th_n, skip=SKIP, device=DEVICE) * th_std + th_mean

rmse_sim = np.sqrt(np.mean((th_sim[SKIP:] - th_raw[SKIP:])**2))
print(f'=== Simulation (train, autoregressive from step {SKIP}) ===')
print(f'RMSE : {rmse_sim:.5f} rad  ({rmse_sim/(2*np.pi)*360:.3f}°)  NRMS: {rmse_sim/th_raw.std()*100:.2f}%')
print(f'Linear ARX baseline: 0.255 rad')
print(f'Good NN target:      0.02710 rad\n')

# ── Plot: free-running simulation vs measured (zoom), plus error (full range) ──
th_raw_sim = th_raw[SKIP:]
err_full   = th_sim[SKIP:] - th_raw_sim
sim_zoom   = slice(28950, 29350)  # window around the peak simulation error

fig, axes = plt.subplots(2, 1, figsize=(6, 5))
axes[0].plot(th_raw_sim[sim_zoom], label='measured', color='steelblue', linewidth=1.4)
axes[0].plot(th_sim[SKIP:][sim_zoom], label='LSTM simulation', color='crimson', ls='--', linewidth=1.4)
axes[0].set_ylabel(r'$\theta$ [rad]'); axes[0].set_xlabel('sample (zoom)')
axes[0].legend(); axes[0].grid(True, alpha=0.4)
axes[0].set_title('LSTM free-running simulation vs measured (zoom)')

axes[1].plot(err_full, color='darkorange', linewidth=0.6)
axes[1].axhline(0, color='k', linewidth=0.5)
axes[1].axvspan(sim_zoom.start, sim_zoom.stop, color='red', alpha=0.15)
axes[1].set_ylabel('error [rad]'); axes[1].set_xlabel('sample (full run)')
axes[1].set_title('Simulation error over the full dataset (zoom window shaded)')
axes[1].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'LSTM_simulation.png'), dpi=200)
plt.close()
print(f'Saved {os.path.join(FIG_DIR, "LSTM_simulation.png")}')

# ── Prediction submission ─────────────────────────────────────────────────────
pred_data   = np.load(os.path.join(BASE, 'hidden-test-prediction-submission-file.npz'))
upast_test  = pred_data['upast'].astype(np.float32)   # (N, 15)
thpast_test = pred_data['thpast'].astype(np.float32)  # (N, 15)

upast_n  = (upast_test  - u_mean) / u_std
thpast_n = (thpast_test - th_mean) / th_std

N_test     = len(upast_test)
th_shifted = np.concatenate([np.zeros((N_test, 1), dtype=np.float32), thpast_n[:, :-1]], axis=1)
inp_test   = np.stack([upast_n, th_shifted], axis=2)  # (N, 15, 2)

with torch.no_grad():
    pred_test_n, _ = model(torch.from_numpy(inp_test).to(DEVICE))
    th_pred_test   = pred_test_n[:, -1].cpu().numpy() * th_std + th_mean

out_pred = os.path.join(BASE, 'hidden-test-prediction-lstm-submission-file.npz')
np.savez(out_pred, upast=upast_test, thpast=thpast_test, thnow=th_pred_test)
print(f'Prediction submission saved: {os.path.basename(out_pred)}  ({len(th_pred_test)} samples)')

# ── Simulation submission ─────────────────────────────────────────────────────
sim_data    = np.load(os.path.join(BASE, 'hidden-test-simulation-submission-file.npz'))
u_test_raw  = sim_data['u'].astype(np.float32)
th_test_raw = sim_data['th'].astype(np.float32)   # only [:50] valid

u_test_n  = (u_test_raw  - u_mean) / u_std
th_test_n = (th_test_raw - th_mean) / th_std

th_test_sim = simulate_lstm(model, u_test_n, th_test_n, skip=50, device=DEVICE) * th_std + th_mean

assert len(th_test_sim) == len(u_test_raw), 'length mismatch in simulation output'
out_sim = os.path.join(BASE, 'hidden-test-simulation-lstm-submission-file.npz')
np.savez(out_sim, th=th_test_sim, u=u_test_raw)
print(f'Simulation submission saved:  {os.path.basename(out_sim)}  ({len(th_test_sim)} samples)')
