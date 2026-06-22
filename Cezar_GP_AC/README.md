# GP System Identification + PPO Swing-Up Policy (Cezar)

Cezar's contributions covering GP-based system identification and
PPO-based actor-critic swing-up policies. 

## `GP/` — Gaussian Process System Identification

- `main.ipynb` / `GP-sklearn.ipynb` — GP-NARX training notebook: data loading,
  lag-order tuning (na=3, nb=3), Matérn (ν=2.5) kernel with sklearn, simulation
  and one-step-ahead evaluation, submission file generation.
- `gp-prediction-submission.npz` — one-step-ahead prediction results on hidden test set.
- `gp-simulation-submission.npz` — free-run simulation results on hidden test set.

## `AC/` — PPO Swing-Up Policy

- `ppo.ipynb` — training notebook (reward shaping, domain randomisation, PPO
  training, evaluation).
- `ppo_swingup_domrand.zip` — base model trained with domain randomisation.
- `ppo_swingup_finetune.zip` — fine-tuned variant.
- `ppo_swingup_max.zip` — highest-reward model obtained during training.
- `ppo_swingup_robust.zip` — robustness-optimised variant.
- `run_on_hardware.py` — hardware deployment script.
