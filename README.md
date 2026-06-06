# Bellman Capital — Carpeta de entrega

## Qué subir al repositorio
**Solo `agent.py`** (es la única entrega oficial). Pasa los 54 tests.

## Contenido de esta carpeta

| Archivo | Para qué sirve |
|---|---|
| `agent.py` | **LA ENTREGA.** TradingEnv + Agent. Súbelo al repo. |
| `REPORTE_FINAL.md` | Reporte completo (secciones 0-11 del README). Completa la Sección 0 (equipo + thesis). |
| `colab_runner.ipynb` | Notebook para correr todo en Google Colab. |
| `run_experiment.py` | Evaluación walk-forward vs baselines + ablación de costos. |
| `make_allocation_plot.py` | Genera el allocation plot. |
| `results/` | Figuras y métricas ya generadas. |

## Cómo verificar la entrega
```bash
uv run pytest tests/test_submission.py -v   # deben pasar 54
```

## Diseño del agente (resumen)
- State: log-returns(24×3) + volatilidad + momentum + pesos + fase = 83 dims.
- Action: 10 carteras discretas long-only + HOLD.
- Rebalanceo semanal (reduce turnover ~8x).
- Reward final: log-return − 0.10·drawdown.
- Double DQN.

## Resultado (split 2024-2025, 10 bps)
Agente: +65% (Sortino 0.62), robusto a costos (de 0 a 25 bps sigue positivo).
