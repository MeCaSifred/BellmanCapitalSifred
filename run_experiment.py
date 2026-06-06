"""
BELLMAN CAPITAL — evaluación walk-forward del agente contra los baselines.

Produce:
  - tabla de métricas por ventana (cumret, sortino, sharpe, maxdd, fees) a 0/10/25 bps
  - equity curves agente vs baselines (con spread de semillas en el último split)
  - allocation plot del agente en el tiempo
  - todo guardado en results/

Uso:
    python run_experiment.py --interval 1h --steps 200000 --seeds 3
    python run_experiment.py --quick          # corrida corta de prueba

No modifica src/ ni agent.py. Importa de ambos.
"""
import os
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data import load_prices, split
from src.metrics import compute_metrics
from src.baselines import RandomPolicy, HoldCash, HoldAsset0, EqualWeight, SMA
import agent as A
from agent import TradingEnv, Agent, N_ACTIONS

# ventanas walk-forward (de configs/default.yaml — NO alterar)
SPLITS = [
    ("2019-12-31", "2020-12-31"),
    ("2020-12-31", "2021-12-31"),
    ("2021-12-31", "2022-12-31"),
    ("2022-12-31", "2023-12-31"),
    ("2023-12-31", "2025-12-31"),
]
FREQ = {"1h": 24 * 365, "30m": 48 * 365, "15m": 96 * 365}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_policy(policy, prices, tc_bps, sma_lookback=None):
    """Corre una política sobre un período y devuelve (values, turnovers)."""
    env = TradingEnv(prices, transaction_cost_bps=tc_bps)
    obs, _ = env.reset()
    done = False
    vals, turns = [], []
    while not done:
        if sma_lookback is not None:                  # SMA lee retornos crudos
            a = policy.act(obs[:sma_lookback * 3])
        else:
            a = policy.act(obs)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        vals.append(info["portfolio_value"])
        turns.append(info["turnover"])
    return np.array(vals), np.array(turns)


def metrics_row(name, vals, turns, tc_bps, freq, initial=10_000.0):
    m = compute_metrics(vals, freq=freq)
    # fees aproximados: turnover * tc * valor (acumulado)
    fees = float(np.sum(turns) * (tc_bps / 10_000) * initial)
    return {
        "policy": name, "tc_bps": tc_bps,
        "cum_ret": m["cum_ret"], "sortino": m["sortino"],
        "sharpe": m["sharpe"], "max_dd": m["max_dd"],
        "total_turnover": float(np.sum(turns)),
        "approx_fees": fees,
    }


def evaluate_window(train_df, eval_df, steps, seeds, tc_list, freq):
    """Entrena el agente (varias semillas) y evalúa agente + baselines."""
    rows = []
    agent_curves = {}   # tc -> list of value arrays (una por semilla)

    # --- baselines (no dependen de entrenamiento ni de semilla del agente) ---
    L = TradingEnv.LOOKBACK
    baseline_defs = [
        ("Random",      lambda: RandomPolicy(N_ACTIONS), None),
        ("HoldCash",    lambda: HoldCash(),              None),
        ("HoldAsset0",  lambda: HoldAsset0(),            None),
        ("EqualWeight", lambda: EqualWeight(),           None),
        ("SMA",         lambda: SMA(risky_action=4, safe_action=0), L),
    ]
    baseline_curves = {}
    for tc in tc_list:
        for name, ctor, sma_L in baseline_defs:
            set_seed(0)
            vals, turns = run_policy(ctor(), eval_df, tc, sma_lookback=sma_L)
            rows.append(metrics_row(name, vals, turns, tc, freq))
            if tc == 10:
                baseline_curves[name] = vals

    # --- agente (entrenar por semilla, evaluar a cada costo) -----------------
    for seed in range(seeds):
        set_seed(seed)
        tenv = TradingEnv(train_df, transaction_cost_bps=10.0)
        obs, _ = tenv.reset()
        ag = Agent(obs_dim=obs.shape[0], n_actions=N_ACTIONS)
        ag.train(tenv, n_steps=steps)
        for tc in tc_list:
            vals, turns = run_policy(ag, eval_df, tc)
            rows.append(metrics_row(f"DQN_seed{seed}", vals, turns, tc, freq))
            agent_curves.setdefault(tc, []).append(vals)

    return rows, agent_curves, baseline_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h", choices=["1h", "30m", "15m"])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tc", default="0,10,25")
    ap.add_argument("--only-last", action="store_true",
                    help="evaluar solo el último split (rápido)")
    ap.add_argument("--quick", action="store_true",
                    help="corrida corta de humo: 1 split, pocos pasos")
    args = ap.parse_args()

    if args.quick:
        args.steps, args.seeds, args.only_last = 8_000, 1, True

    tc_list = [int(x) for x in args.tc.split(",")]
    freq = FREQ[args.interval]
    os.makedirs("results", exist_ok=True)

    data = load_prices(args.interval)
    splits = SPLITS[-1:] if args.only_last else SPLITS

    all_rows = []
    for i, (train_end, eval_end) in enumerate(splits):
        train_df, eval_df = split(data, train_end, eval_end)
        wname = f"eval≤{eval_end[:4]}"
        print(f"\n=== Ventana {wname}: train {len(train_df)} | eval {len(eval_df)} ===")
        rows, agent_curves, baseline_curves = evaluate_window(
            train_df, eval_df, args.steps, args.seeds, tc_list, freq
        )
        for r in rows:
            r["window"] = wname
        all_rows.extend(rows)

        # --- figura de equity curves (a 10 bps) para esta ventana -----------
        plt.figure(figsize=(12, 5))
        idx = eval_df.index[-len(next(iter(baseline_curves.values()))):]
        for name, vals in baseline_curves.items():
            plt.plot(idx[:len(vals)], vals, lw=0.9, alpha=0.7, label=name)
        if 10 in agent_curves:
            curves = agent_curves[10]
            arr = np.array([c[:min(map(len, curves))] for c in curves])
            mean = arr.mean(0)
            plt.plot(idx[:len(mean)], mean, "k-", lw=1.6, label="DQN (media)")
            if len(curves) > 1:
                plt.fill_between(idx[:len(mean)], arr.min(0), arr.max(0),
                                 color="k", alpha=0.15, label="DQN (rango semillas)")
        plt.yscale("log")
        plt.title(f"Equity curves — {wname} (10 bps)")
        plt.ylabel("valor del portafolio")
        plt.legend(ncol=3, fontsize=8)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        plt.tight_layout()
        plt.savefig(f"results/equity_{wname.replace('≤','_')}.png", dpi=130)
        plt.close()

    # --- guardar tabla de métricas -----------------------------------------
    df = pd.DataFrame(all_rows)
    df.to_csv("results/metrics.csv", index=False)

    # resumen legible: pivote de sortino y cum_ret a 10 bps
    print("\n" + "=" * 80)
    print("RESUMEN — métricas a 10 bps")
    print("=" * 80)
    sub = df[df.tc_bps == 10]
    for wname in sub.window.unique():
        print(f"\n{wname}")
        w = sub[sub.window == wname][["policy", "cum_ret", "sortino", "max_dd", "total_turnover"]]
        print(w.to_string(index=False,
              formatters={"cum_ret": "{:+.1%}".format,
                          "sortino": "{:.2f}".format,
                          "max_dd": "{:.1%}".format,
                          "total_turnover": "{:.0f}".format}))

    with open("results/summary.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\nResultados guardados en results/  (metrics.csv, summary.json, equity_*.png)")


if __name__ == "__main__":
    main()
