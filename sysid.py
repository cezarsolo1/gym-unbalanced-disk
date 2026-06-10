"""
System Identification & Diagnostic Script — Unbalanced Disc Hardware

Tests:
  1. step_response  — identifies motor gain (Ku) and damping
  2. free_decay     — identifies natural frequency and friction (no motor)
  3. prbs           — broadband excitation for full sysid dataset
  4. omega_check    — compares firmware omega vs numerical derivative
"""

import sys, os, time
import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import gym_unbalanced_disk

# ── Configuration ─────────────────────────────────────────────────────────────

UMAX     = 3.0
DT       = 0.025   # 40 Hz
N_STEPS  = 800     # steps per test (800 = 20 seconds)
SAVE_DATA = True   # save .npz after each test

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_theta(o):
    return np.arctan2(o[0], o[1])

def numerical_omega(obs_prev, obs_curr, dt):
    dth = get_theta(obs_curr) - get_theta(obs_prev)
    if dth >  np.pi: dth -= 2 * np.pi
    if dth < -np.pi: dth += 2 * np.pi
    return dth / dt

def connect():
    env = gym_unbalanced_disk.UnbalancedDisk_exp_sincos(umax=UMAX, dt=DT)
    print("Connecting to hardware and waiting for disc to settle...")
    env.reset()
    return env

def record_step(env, obs_prev, u):
    obs_raw, _, _, _, _ = env.step(np.float32(u))
    th    = get_theta(obs_raw)
    o_fw  = obs_raw[2]
    o_num = numerical_omega(obs_prev, obs_raw, DT)
    return obs_raw, th, o_fw, o_num

# ── Test 1: Step Response ─────────────────────────────────────────────────────

def test_step_response(env):
    """
    Applies a sequence of voltage steps and records the angle response.
    Useful for estimating Ku (motor gain) and checking sign conventions.
    Steps: 0 -> +1V -> 0 -> -1V -> 0 -> +2V -> 0 -> -2V -> 0
    Each held for 2 seconds (80 steps).
    """
    step_sequence = (
        [0.0] * 40 +
        [1.0] * 80 + [0.0] * 40 +
        [-1.0] * 80 + [0.0] * 40 +
        [2.0] * 80 + [0.0] * 40 +
        [-2.0] * 80 + [0.0] * 40
    )
    print(f"  Running step_response — {len(step_sequence)} steps ({len(step_sequence)*DT:.1f}s)")

    log_th, log_omega_fw, log_omega_num, log_u = [], [], [], []
    obs_prev = env.get_obs()
    time.sleep(DT)

    for i, u in enumerate(step_sequence):
        obs_raw, th, o_fw, o_num = record_step(env, obs_prev, u)
        log_th.append(th)
        log_omega_fw.append(o_fw)
        log_omega_num.append(o_num)
        log_u.append(u)
        obs_prev = obs_raw

        if i % 40 == 0:
            print(f"    step {i:4d} | θ={np.degrees(th):+6.1f}° | ω_num={o_num:+.3f} | u={u:+.1f}V")

    return np.array(log_th), np.array(log_omega_fw), np.array(log_omega_num), np.array(log_u)

# ── Test 2: Free Decay ────────────────────────────────────────────────────────

def test_free_decay(env):
    """
    Gives the disc a short kick then cuts motor to zero.
    Records natural oscillation — used to identify omega0 and damping (gamma, Fc).
    """
    print(f"  Running free_decay — {N_STEPS} steps ({N_STEPS*DT:.1f}s)")

    log_th, log_omega_fw, log_omega_num, log_u = [], [], [], []
    obs_prev = env.get_obs()
    time.sleep(DT)

    for i in range(N_STEPS):
        u = 2.0 if i < 20 else 0.0   # 0.5s kick then free swing
        obs_raw, th, o_fw, o_num = record_step(env, obs_prev, u)
        log_th.append(th)
        log_omega_fw.append(o_fw)
        log_omega_num.append(o_num)
        log_u.append(u)
        obs_prev = obs_raw

        if i % 40 == 0:
            print(f"    step {i:4d} | θ={np.degrees(th):+6.1f}° | ω_num={o_num:+.3f} | u={u:+.1f}V")

    return np.array(log_th), np.array(log_omega_fw), np.array(log_omega_num), np.array(log_u)

# ── Test 3: PRBS Excitation ───────────────────────────────────────────────────

def test_prbs(env):
    """
    Applies a Pseudo-Random Binary Sequence (PRBS) input.
    Standard sysid dataset — covers all frequencies evenly.
    Use this data to fit ARX or neural-network models.
    PRBS switches between +umax and -umax at random intervals (4-16 steps).
    """
    print(f"  Running prbs — {N_STEPS} steps ({N_STEPS*DT:.1f}s)")

    rng  = np.random.default_rng(42)
    log_th, log_omega_fw, log_omega_num, log_u = [], [], [], []
    obs_prev = env.get_obs()
    time.sleep(DT)

    u      = UMAX
    hold   = int(rng.integers(4, 16))
    hold_i = 0

    for i in range(N_STEPS):
        obs_raw, th, o_fw, o_num = record_step(env, obs_prev, u)
        log_th.append(th)
        log_omega_fw.append(o_fw)
        log_omega_num.append(o_num)
        log_u.append(u)
        obs_prev = obs_raw

        hold_i += 1
        if hold_i >= hold:
            u      = -u
            hold   = int(rng.integers(4, 16))
            hold_i = 0

        if i % 80 == 0:
            print(f"    step {i:4d} | θ={np.degrees(th):+6.1f}° | ω_num={o_num:+.3f} | u={u:+.1f}V")

    return np.array(log_th), np.array(log_omega_fw), np.array(log_omega_num), np.array(log_u)

# ── Test 4: Omega Check ───────────────────────────────────────────────────────

def test_omega_check(env):
    """
    Applies a slow sine-wave voltage and compares:
      - Firmware omega  (from USB packet — known to be unreliable)
      - Numerical omega (Δθ/Δt from encoder position)
    Directly shows the magnitude and sign error in the firmware omega.
    """
    print(f"  Running omega_check — {N_STEPS} steps ({N_STEPS*DT:.1f}s)")

    t_arr = np.arange(N_STEPS) * DT
    u_seq = 1.5 * np.sin(2 * np.pi * 0.4 * t_arr)   # 0.4 Hz sine, 1.5V amplitude

    log_th, log_omega_fw, log_omega_num, log_u = [], [], [], []
    obs_prev = env.get_obs()
    time.sleep(DT)

    for i, u in enumerate(u_seq):
        obs_raw, th, o_fw, o_num = record_step(env, obs_prev, u)
        log_th.append(th)
        log_omega_fw.append(o_fw)
        log_omega_num.append(o_num)
        log_u.append(u)
        obs_prev = obs_raw

        if i % 40 == 0:
            print(f"    step {i:4d} | θ={np.degrees(th):+6.1f}° | ω_fw={o_fw:+.3f} | ω_num={o_num:+.3f}")

    return np.array(log_th), np.array(log_omega_fw), np.array(log_omega_num), np.array(log_u)

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(th, omega_fw, omega_num, u, title):
    t      = np.arange(len(u)) * DT
    th_deg = np.degrees(th)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=13)

    axes[0].plot(t, th_deg, color='steelblue', label='θ (deg)')
    axes[0].axhline( 180, color='gray', linestyle='--', linewidth=0.8, label='±180° (top)')
    axes[0].axhline(-180, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Angle (°)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    axes[1].plot(t, omega_num, color='darkorange', label='ω numerical (Δθ/Δt)', linewidth=1.2)
    axes[1].plot(t, omega_fw,  color='red',        label='ω firmware (raw USB)', alpha=0.5, linestyle='--')
    axes[1].set_ylabel('ω (rad/s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    axes[2].plot(t, u, color='green', label='u (V)')
    axes[2].axhline( UMAX, color='red', linestyle='--', linewidth=0.8, label='±umax')
    axes[2].axhline(-UMAX, color='red', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f'sysid_{title}.png', dpi=120)
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    env = connect()
    results = {}

    tests = [
        ('step_response', test_step_response),
        ('free_decay',    test_free_decay),
        ('prbs',          test_prbs),
        ('omega_check',   test_omega_check),
    ]

    try:
        for name, fn in tests:
            print(f"\n{'='*50}")
            print(f"  TEST: {name.upper()}")
            print(f"{'='*50}")
            input("  Press ENTER when disc is at rest at the bottom...")
            th, omega_fw, omega_num, u = fn(env)
            results[name] = (th, omega_fw, omega_num, u)
            print(f"  Done. {len(u)} steps recorded.")

    except KeyboardInterrupt:
        print("\nStopped early — saving and plotting whatever was collected.")

    finally:
        env.close()

    for name, (th, omega_fw, omega_num, u) in results.items():
        if len(u) == 0:
            continue

        th        = np.array(th)
        omega_fw  = np.array(omega_fw)
        omega_num = np.array(omega_num)
        u         = np.array(u)

        if SAVE_DATA:
            fname = f'sysid_{name}.npz'
            np.savez(fname, th=th, omega_fw=omega_fw, omega_num=omega_num, u=u, dt=DT)
            print(f"Saved {fname}")

        plot_results(th, omega_fw, omega_num, u, title=name)
