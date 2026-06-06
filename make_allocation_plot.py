"""
Allocation plot: cómo asigna capital el agente a lo largo del tiempo (eval 2024-25).
Genera results/allocation_eval_2025.png  (figura requerida por la Sección 9).
"""
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data import load_prices, split
import agent as A
from agent import TradingEnv, Agent, N_ACTIONS, _ACTION_WEIGHTS, _HOLD_ACTION

random.seed(0); np.random.seed(0); torch.manual_seed(0)

data = load_prices("1h")
train_df, eval_df = split(data, "2023-12-31", "2025-12-31")

# entrenar una semilla
tenv = TradingEnv(train_df, transaction_cost_bps=10.0)
obs, _ = tenv.reset()
ag = Agent(obs_dim=obs.shape[0], n_actions=N_ACTIONS)
ag.train(tenv, n_steps=40000)

# rollout en eval, registrando los pesos efectivos en cada paso
env = TradingEnv(eval_df, transaction_cost_bps=10.0)
obs, _ = env.reset()
done = False
weights_hist, vals = [], []
while not done:
    a = ag.act(obs)
    obs, _, term, trunc, info = env.step(a)
    done = term or trunc
    weights_hist.append(env._weights.copy())
    vals.append(info["portfolio_value"])

W = np.array(weights_hist)            # (T, 4)
idx = eval_df.index[-len(W):]

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})

labels = ["asset_0", "asset_1", "asset_2", "cash"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#999999"]
axes[0].stackplot(idx, [W[:, i] for i in range(4)], labels=labels, colors=colors, alpha=0.85)
axes[0].set_title("Asignación de capital del agente en el tiempo (eval 2024-2025)")
axes[0].set_ylabel("peso del portafolio")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="upper left", ncol=4, fontsize=8)

axes[1].plot(idx, vals, "k-", lw=0.9)
axes[1].axhline(10000, color="gray", ls="--", lw=0.7)
axes[1].set_title("Valor del portafolio")
axes[1].set_ylabel("valor")
axes[1].set_yscale("log")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))

plt.tight_layout()
plt.savefig("results/allocation_eval_2025.png", dpi=130)
print("OK results/allocation_eval_2025.png")

# resumen de exposición media
print("\nExposición media del agente en eval:")
for i, lab in enumerate(labels):
    print(f"  {lab}: {W[:, i].mean():.1%}")
