# Swing-Up + Reference Tracking Policy (Kayden)

PPO-based actor-critic policy for Section 4.2.2 (single policy for swing-up and
±15° reference tracking around the top). See report Section IV.B.

- `ppo_reference_tracking.ipynb` — training notebook (env wrapper, reward, PPO
  training, hardware deployment loop).
- `ppo_reftrack.zip` — trained model (discretized 7-level bang-bang voltages,
  dt matched to measured real hardware control-loop timing).
- `hardware_log_working.npy` — best real hardware run obtained (swing-up
  reliable, ~29% of run time within 15° of target).
- `figures/` — report figures: training curve, simulation results
  (continuous-action variant), and the real hardware run.

Note: there is a separate `AC-ppo/` folder from a teammate's own PPO
reference-tracking work — this folder is Kayden's independent implementation.
