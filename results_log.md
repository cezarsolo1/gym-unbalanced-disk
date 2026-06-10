# LSTM System ID — Results Log

Benchmarks (from assignment):

| Type | Pred RMSE (rad) | Sim RMSE (rad) |
|------|----------------|----------------|
| Lower bound | — | 0.0195 |
| Good NN | 0.00382 | 0.0271 |
| Linear ARX | 0.00665 | 0.255 |

---

## Run 1 — Baseline LSTM (teacher-forcing only)
**Config:** hidden=64, layers=2, seq_len=64, epochs=300, Adam lr=1e-3, no simulation fine-tuning

| Task | RMSE (rad) | RMSE (deg) | vs Good NN |
|------|-----------|-----------|-----------|
| Prediction | 0.00470 | 0.269 | +23% worse |
| Simulation | 0.02783 | 1.594 | +2.7% worse |

Notes: First attempt, default hyperparameters. Simulation very close to good NN target already.
Beats linear ARX by 5.7× on simulation. Val loss still slowly decreasing at epoch 300.

---

## Run 2 — NOE fine-tuning + longer training
**Config:** hidden=64, layers=2, pre-train 500 epochs (min_lr=1e-6) + NOE fine-tune 100 epochs lr=1e-4, seq=80, warmup=10

| Task | RMSE (rad) | RMSE (deg) | vs Good NN |
|------|-----------|-----------|-----------|
| Prediction | 0.00491 | 0.281 | +28% worse |
| Simulation | 0.02776 | 1.591 | +2.4% worse |

Notes: Barely any improvement. NOE fine-tuning val_loss converged at epoch 1 (0.005089) then bounced
around 0.007-0.009 for 100 epochs — never improved further. Prediction slightly degraded (fine-tuning
hurt the teacher-forcing quality). Extra 200 pre-train epochs helped a little (0.02783 → 0.02776).
Fine-tuning approach: not effective here.

---

## Run 3 — Larger hidden size (hidden=128) + NOE fine-tuning
**Config:** hidden=128, layers=2, pre-train 500 epochs + NOE fine-tune 100 epochs

| Task | RMSE (rad) | RMSE (deg) | vs Good NN |
|------|-----------|-----------|-----------|
| Prediction | 0.00444 | 0.254 | +16% worse |
| Simulation | 0.03053 | 1.749 | +12.7% worse |

Notes: Prediction improved (best so far at 0.93% NRMS) but simulation got WORSE vs Run 1.
NOE fine-tuning again converged only at epoch ~1, then bounced. Larger model + NOE fine-tuning
= worse simulation, likely overfitting during autoregressive rollout. NOE fine-tuning consistently
hurts or barely helps — drop it entirely going forward.

---

## Run 4 — hidden=128, teacher-forcing only (no NOE) ✓ BEST
**Config:** hidden=128, layers=2, pre-train 500 epochs, min_lr=1e-6, NO fine-tuning

| Task | RMSE (rad) | RMSE (deg) | vs Good NN |
|------|-----------|-----------|-----------|
| Prediction | 0.00388 | 0.222 | +1.6% worse |
| Simulation | 0.02456 | 1.407 | **-9.4% BEATS target** |

Notes: Best result. Larger hidden size (128) without NOE fine-tuning beats the good NN benchmark
on simulation and nearly matches it on prediction. NOE fine-tuning confirmed harmful — dropped entirely.
Submission files generated from this run.
