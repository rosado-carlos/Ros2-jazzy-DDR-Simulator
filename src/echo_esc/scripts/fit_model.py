#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# ── Load data ────────────────────────────────────────────────────────────────
df_v   = pd.read_csv("vicon_velocity.csv")
df_cmd = pd.read_csv("command_raw.csv")

t_vicon = df_v["t"].values
v_vicon = df_v["v"].values

# yaw rate desde VICON
yaw_rate_meas = df_v["omega"].values

t_cmd = df_cmd["t"].values
u_raw = df_cmd["throttle"].values
steer_raw = df_cmd["steering"].values  # YA en radianes

# ── Interpolación ────────────────────────────────────────────────────────────
u_interp_fn = interp1d(t_cmd, u_raw, kind="zero", fill_value="extrapolate")
u = u_interp_fn(t_vicon)

steer_interp_fn = interp1d(t_cmd, steer_raw, kind="zero", fill_value="extrapolate")
delta = steer_interp_fn(t_vicon)  # ya en rad

dt = np.median(np.diff(t_vicon))

# ── Checks ───────────────────────────────────────────────────────────────────
print("len u:", len(u))
print("len delta:", len(delta))
print("len v:", len(v_vicon))
print("len yaw:", len(yaw_rate_meas))
print("delta range:", np.min(delta), np.max(delta))


# ── Parámetros físicos ───────────────────────────────────────────────────────
m = 5.0
lf = 0.1875
lr = 0.1875
Iz = 0.065

Cf = 250.0
Cr = 250.0

mu = 0.9
g = 9.81


# ── Deadzone ─────────────────────────────────────────────────────────────────
def apply_deadzone(u_arr, u_dead):
    u_eff = np.zeros_like(u_arr)
    mask_pos = u_arr >  u_dead
    mask_neg = u_arr < -u_dead
    u_eff[mask_pos] = u_arr[mask_pos] - u_dead
    u_eff[mask_neg] = u_arr[mask_neg] + u_dead
    return u_eff


# ── Dinámica STD-lite ────────────────────────────────────────────────────────
def vehicle_dynamics(x, u, params):
    v, beta, yaw_rate = x
    throttle, delta = u

    K, tau, u_dead = params

    # deadzone
    if abs(throttle) < u_dead:
        u_eff = 0.0
    else:
        u_eff = throttle - np.sign(throttle) * u_dead

    v = max(v, 0.1)

    alpha_f = np.arctan2(
        v*np.sin(beta) + lf*yaw_rate,
        v*np.cos(beta)
    ) - delta

    alpha_r = np.arctan2(
        v*np.sin(beta) - lr*yaw_rate,
        v*np.cos(beta)
    )

    Fyf = Cf * alpha_f
    Fyr = Cr * alpha_r

    # saturación
    Fz = m * g / 2
    Fyf = np.clip(Fyf, -mu * Fz, mu * Fz)
    Fyr = np.clip(Fyr, -mu * Fz, mu * Fz)

    beta_dot = (
        -yaw_rate +
        (Fyf*np.cos(delta - beta) + Fyr*np.cos(beta)) / (m*v)
    )

    yaw_rate_dot = (
        lf * Fyf * np.cos(delta) -
        lr * Fyr
    ) / Iz

    v_dot = (
        (K * u_eff - v) / tau
        - (Fyf * np.sin(delta - beta)) / m
    )

    return np.array([v_dot, beta_dot, yaw_rate_dot])


# ── Simulación ───────────────────────────────────────────────────────────────
def simulate_model(params, u_arr, delta_arr, dt):
    N = len(u_arr)

    v = 0.0
    beta = 0.0
    yaw_rate = 0.0

    v_hist = np.zeros(N)
    yaw_hist = np.zeros(N)

    for i in range(1, N):
        x = np.array([v, beta, yaw_rate])
        u_vec = [u_arr[i-1], delta_arr[i-1]]

        dx = vehicle_dynamics(x, u_vec, params)
        x = x + dt * dx

        v, beta, yaw_rate = x
        beta = np.clip(beta, -np.pi/4, np.pi/4)

        v_hist[i] = v
        yaw_hist[i] = yaw_rate

    return v_hist, yaw_hist


# ── Cost function ────────────────────────────────────────────────────────────
def cost(params):
    K, tau, u_dead = params

    if K <= 0 or tau <= 0 or u_dead < 0 or u_dead > 0.5:
        return 1e9

    v_sim, yaw_sim = simulate_model(params, u, delta, dt)

    error_v = v_sim - v_vicon
    error_yaw = yaw_sim - yaw_rate_meas

    # pesos automáticos
    w_v = 1.0 / np.var(v_vicon)
    w_y = 1.0 / np.var(yaw_rate_meas)

    return np.mean(w_v * error_v**2 + w_y * error_yaw**2)


# ── Optimización ─────────────────────────────────────────────────────────────
x0 = [1.0, 0.3, 0.05]
bounds = [(0.1, 5.0), (0.01, 2.0), (0.0, 0.4)]

result = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                  options={"maxiter": 500, "ftol": 1e-9})

K_fit, tau_fit, u_dead_fit = result.x

print("\n── Fitted Parameters ────────────────────────────────")
print(f"K      ={K_fit:.4f}")
print(f"tau    ={tau_fit:.4f}")
print(f"u_dead ={u_dead_fit:.4f}")
print(f"RMSE_v ={np.sqrt(np.mean((simulate_model(result.x,u,delta,dt)[0]-v_vicon)**2)):.4f}")


# ── Plot ─────────────────────────────────────────────────────────────────────
v_fit, yaw_fit = simulate_model(result.x, u, delta, dt)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

ax1.plot(t_vicon, v_vicon, label="VICON")
ax1.plot(t_vicon, v_fit, '--', label="Model")
ax1.set_ylabel("Speed [m/s]")
ax1.legend()
ax1.grid(True)

ax2.plot(t_vicon, yaw_rate_meas, label="Yaw rate meas")
ax2.plot(t_vicon, yaw_fit, '--', label="Yaw rate model")
ax2.set_ylabel("Yaw rate [rad/s]")
ax2.legend()
ax2.grid(True)

ax3.step(t_vicon, u, where="post", label="Throttle")
ax3.step(t_vicon, delta, where="post", label="Steering")
ax3.set_ylabel("Inputs")
ax3.set_xlabel("Time [s]")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("model_fit_std.png", dpi=150)
plt.show()

print("Saved: model_fit_std.png")