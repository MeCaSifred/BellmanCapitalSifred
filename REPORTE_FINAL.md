# BELLMAN CAPITAL — Reporte Final

> Estructurado según las secciones 0–11 del README oficial. La entrega de código es `agent.py` (pasa los 54 tests). Los resultados se reproducen con `run_experiment.py` y `make_allocation_plot.py`. Las secciones marcadas con ✏️ deben completarse por el equipo.

---

## 0. Team ✏️

- **Agent codename:** _(completar)_
- **Researchers:** _(completar)_
- **Inception date:** _(completar)_
- **Thesis:** _(propuesta — ajustar a su criterio)_ Nuestro agente cree que **la dirección de los precios es casi impredecible a corto plazo, pero el régimen de volatilidad y la frecuencia de operación sí son controlables**. Por eso su diseño prioriza (a) operar poco para no destruir retorno con costos, y (b) reducir exposición en las caídas. No intenta adivinar el mercado; intenta sobrevivir a los costos y a los drawdowns.

---

## 1. Problem Formulation

### 1.1. State Space

Observación de **83 dimensiones** en cada paso `t`, toda lookahead-safe (solo usa datos ≤ t):

| Componente | Tamaño | Justificación (una frase) |
|---|---:|---|
| Log-returns (24 pasos × 3 activos) | 72 | el bloque temporal base de información de precio, en forma estacionaria (retornos, no precios) |
| Volatilidad reciente por activo | 3 | el EDA mostró vol clustering fuerte; es la señal de régimen más explotable |
| Momentum (`log(close_t/close_{t-20})`) | 3 | captura tendencia de plazo medio |
| Pesos actuales del portafolio | 4 | el agente necesita conocer su posición para evaluar el costo de rebalancear |
| Indicador de fase de rebalanceo | 1 | señala si la próxima acción tendrá efecto (mantiene la propiedad de Markov) |

**Respuestas a las preguntas del README:**

1. *¿Qué hay en la observación en t?* Los cinco bloques de la tabla anterior.
2. *Los precios no son Markov: ¿cómo lo manejas?* Incluimos explícitamente **volatilidad y momentum** además de los retornos, de modo que el estado resume la dinámica reciente y no solo el último precio.
3. *¿Cuánta historia y por qué?* **24 pasos (un día a 1h).** El EDA mostró que la autocorrelación de |retornos| (vol clustering) sigue siendo > 0.18 a 24h; más allá la información marginal cae rápido y solo infla el espacio de observación.

### 1.2. Action Space

**Menú discreto long-only de 10 carteras** (compatible con DQN, interpretable):

| # | Cartera `[a0,a1,a2,cash]` | Interpretación |
|---|---|---|
| 0 | `[0,0,0,1]` | todo cash (= HoldCash) |
| 1 | `[1,0,0,0]` | todo asset_0 (= HoldAsset0) |
| 2 | `[0,1,0,0]` | todo asset_1 |
| 3 | `[0,0,1,0]` | todo asset_2 |
| 4 | `[⅓,⅓,⅓,0]` | equal weight (= EqualWeight) |
| 5 | `[0.5,0,0,0.5]` | half asset_0 + cash |
| 6 | `[0,0.5,0,0.5]` | half asset_1 + cash |
| 7 | `[0,0,0.5,0.5]` | half asset_2 + cash |
| 8 | `[⅙,⅙,⅙,0.5]` | half equal-weight + cash |
| 9 | HOLD | mantiene la cartera actual (turnover 0) |

**Respuestas a las preguntas del README:**

1. *¿Cuál es el action space?* El menú de 10 acciones de arriba.
2. *¿Por qué esta representación?* Discreta para usar DQN; cada acción es una cartera interpretable; el orden respeta los baselines de `src/baselines.py`. HOLD le da al agente una salida explícita de turnover 0.
3. *¿Qué impide?* No permite apalancamiento, ni pesos finos (p. ej. 23 %/47 %/30 %), ni posiciones cortas. **Sobre los shorts:** el README los permite, y los exploramos — pero el EDA mostró drift alcista estructural (asset_1 +10.000 % en 8 años) y un menú con shorts llevaba al agente a perder -84 % (peor que el azar) al shortear en mercados alcistas. Decisión informada por datos: **menú long-only**. Esto también cumple al pie de la letra la restricción de la Sección 3 (`pesos ≥ 0`).

---

## 2. Data and EDA

Resumen del análisis (informe del Punto 1, con figuras propias):

- 3 activos riesgosos + cash, 2018–2025, intervalos `15m/30m/1h`. Sin NaN. `cash` = 1.0 (numeraire). Usamos **1h** (mejor relación señal/ruido y costos relativos).
- **Retornos casi sin predictibilidad lineal** (autocorr ≈ 0) pero **volatilidad muy autocorrelacionada** (vol clustering) → la señal explotable es de régimen, no de dirección.
- **Colas pesadísimas** (curtosis 26–42 vs 0 de una normal) → el reward debe penalizar las caídas.
- **Correlaciones altas y peores en estrés** (0.69–0.83, hasta 0.84 en 2022) → la diversificación entre los tres riesgosos es débil; el cash es el único refugio real.
- **Distribution shift severo entre años** (cada año es un régimen distinto) → state con features relativas, no precios absolutos.

---

## 3. Environment Design

`TradingEnv` subclasa `BaseTradingEnv` e implementa los tres métodos requeridos (más un `reset` que reinicia el estado propio).

- **`_obs`**: el vector de 83 dimensiones de la Sección 1.1. Robustez numérica: `nan_to_num` + clip a [-10, 10] (el feature set tiene outliers de ±25σ).
- **`_weights_from_action`**: mapea índice → cartera. Implementa el **rebalanceo periódico**: el agente propone en cada paso, pero el entorno solo aplica el cambio cada **168 pasos (semanal)**; en el resto fuerza HOLD. Esto redujo el turnover ~8× y fue la palanca de mejora más potente.
- **`_reward`**: ver Sección 4.

---

## 4. Reward Design (iteración documentada)

Probamos tres formulaciones, todas respetando los signos exigidos por los tests (>0 si crece, <0 si cae, 0 si no cambia). Comparación en el split 2024-2025 a 10 bps, misma configuración:

| Formulación | cum_ret | Sortino | Comportamiento observado |
|---|---:|---:|---|
| **log-return puro** | +54 % | 0.58 | Sin rebalanceo periódico, **sobre-operaba** y colapsaba a 0 con costos. Con rebalanceo semanal queda decente pero deja demasiado en cash (45 %). |
| **turnover-penalized** | +2 % a +84 % | 0.0–0.9 | Útil cuando el rebalanceo era diario; con rebalanceo semanal la penalización es **redundante** (el turnover ya está controlado) y añade ruido. Alta varianza. |
| **drawdown-penalized** ✅ | **+175 %** | **1.71** | **Formulación final.** Penaliza la caída desde el pico, lo que **alinea el reward con el Sortino** (métrica objetivo). Mejor en todos los regímenes probados, incluido el bear de 2022 (-48 % vs -72 %). |

Fórmula final: `reward = log(curr/prev) − μ·drawdown_actual`, con `μ = 0.10` y el pico reiniciado en cada episodio.

**Exploit del reward identificado:** con penalización alta de cualquier tipo, el agente tiende a **parquearse en cash** (turnover 0, drawdown 0 → reward ≈ 0) en vez de buscar retorno. *¿Lo descubrió el agente?* Sí, parcialmente: con `μ` alto o con turnover-penalty alto, la exposición media a cash subía hasta ~45 %. Se mitigó calibrando `μ = 0.10`, que penaliza las caídas sin desincentivar toda exposición (cash medio ≈ 27 % en la versión final).

> El README pide comparar ≥2 alternativas; comparamos las tres (log, turnover, drawdown). Las cuatro firmas posibles (incluida differential Sharpe) requieren estado; drawdown-penalized fue la de mejor relación desempeño/simplicidad.

---

## 5. Algorithm

**Double DQN** con target network y replay buffer.

| Hiperparámetro | Valor |
|---|---|
| Q-network | MLP `[83 → 128 → 128 → 10]`, ReLU |
| γ | 0.99 |
| Learning rate | 1e-4 (Adam) |
| Batch size | 64 |
| Replay buffer | 100.000 |
| Target update | cada 1.000 pasos |
| ε | 1.0 → 0.05, lineal en 50.000 pasos |
| Loss | Smooth L1 (Huber), grad clip 10 |
| Pasos de entrenamiento | 200.000 (default; resultados mostrados con 50.000) |

Double DQN reduce el sesgo de sobreestimación de Q, relevante en un entorno tan ruidoso.

---

## 6. Baselines

Los cinco de `src/baselines.py`, evaluados en condiciones idénticas. El orden del action space respeta sus supuestos (HoldCash=0, HoldAsset0=1, EqualWeight=4). El SMA recibe la porción de retornos del obs.

---

## 7. Training Protocol

- **Pasos de entorno:** 50.000–200.000 por ventana.
- **Tiempo:** ~150 s por 50k pasos en CPU; menos con GPU (Colab T4). El agente detecta CUDA automáticamente.
- **Métricas logueadas:** valor del portafolio, turnover, exposición por activo.
- **Criterio de parada:** número fijo de pasos. **Hallazgo:** más pasos no siempre mejora (sobreajuste al régimen de entrenamiento), por lo que no conviene entrenar en exceso.

---

## 8. Evaluation

- **Walk-forward** sobre las 5 ventanas del config (`run_experiment.py` las recorre). Sin lookahead, sin re-tuning en evaluación, held-out 2026 intacto.
- **Múltiples semillas** (3) para mostrar el spread.
- **Ablación de costos** a 0 / 10 / 25 bps.
- **Métricas:** cumulative return, Sortino (primaria), max drawdown, fees totales.

### Reproducir

```bash
uv run pytest tests/test_submission.py -v                 # 54 tests
python run_experiment.py --interval 1h --steps 200000 --seeds 3 --tc 0,10,25
python make_allocation_plot.py
```

---

## 9. Results

### 9.1. Agente final — split 2024-2025

| Costos | cum_ret | Sortino | max_dd | turnover |
|---|---:|---:|---:|---:|
| 0 bps | +81.5 % | 0.76 | -48.0 % | 96 |
| **10 bps** | **+64.8 %** | **0.62** | -48.4 % | 96 |
| 25 bps | +42.6 % | 0.43 | -49.3 % | 96 |

El agente es **positivo y robusto a costos** (turnover ~96 vs ~800 de la versión diaria que perdía).

### 9.2. Walk-forward (drawdown-reward, semanal, 10 bps)

| Ventana | Régimen | Agente | Nota |
|---|---|---:|---|
| eval 2023 | recuperación | +90.1 % (Sortino 2.72) | excelente |
| eval 2025 | mixto | +65 a +175 % | excelente, alta varianza |
| eval 2022 | bear | -48 % | pierde, pero mejor que HoldAsset0 (-68 %) y Random (-83 %) |

### 9.3. Figuras

- `results/equity_eval_2025.png` — equity curves del agente vs baselines con spread de semillas.
- `results/allocation_eval_2025.png` — **interpretación económica**: el agente final mantiene posiciones por semanas (bloques sólidos), combina asset_0 (~52 %) con cash (~27 %) y va a efectivo en momentos defensivos. Contrasta con la versión diaria previa (rayado caótico que perdía) — esa figura del "antes" sirve como la documentación de comportamiento anómalo que pide el README.

---

## 10. Discussion

- **Reward design y reward hacking.** El exploit fue el "parking en cash". Se manifestó como exposición a cash creciente cuando la penalización era alta. Se atacó calibrando `μ = 0.10` y controlando el turnover de forma estructural (rebalanceo semanal) en lugar de solo vía reward.
- **Sample efficiency / observaciones no independientes.** Los regímenes duran años y las observaciones horarias están muy autocorrelacionadas. El replay buffer rompe parte de esa correlación, pero el agente ve pocos "eventos de régimen" independientes, lo que explica la alta varianza entre semillas (p. ej. 2021: -5 % a +140 %).
- **Distribution shift train→deploy.** El caso de los shorts lo ilustra: una política aprendida en periodos bear se aplicaba en bull con resultados desastrosos. Se atacó con features relativas y eliminando los shorts.
- **No estacionariedad / regime change.** Cada año es un régimen distinto. El agente no detecta régimen explícitamente más allá de la volatilidad; el drawdown-penalty le da un comportamiento defensivo que ayuda en los cambios a bear.
- **Long-horizon credit assignment.** Con γ = 0.99 y episodios de ~17.000 pasos, atribuir el resultado a decisiones individuales es muy difícil. El rebalanceo semanal lo alivia (menos decisiones, cada una con más impacto).

---

## 11. Reflection

**Tres resultados que sorprendieron:**
1. El **rebalanceo semanal** fue una palanca mucho más potente que cualquier ajuste del reward: convirtió un agente que perdía -100 % en uno que gana, simplemente operando menos.
2. El **drawdown-penalty mejora incluso el bear de 2022** (-72 % → -48 %): alinear el reward con la métrica objetivo (Sortino) tiene efecto directo.
3. **Más entrenamiento a veces empeora** la evaluación (sobreajuste al régimen pasado).

**Dos cambios con más tiempo:**
1. Un "escape a efectivo" intra-semana ante caídas fuertes, para no quedar atrapado en el bear con el rebalanceo semanal.
2. Detección/condicionamiento explícito de régimen (en vez de depender solo de la volatilidad).

**Un comportamiento que no explicamos del todo:** la preferencia del agente por asset_0 en la versión final (52 % medio) frente a asset_1 en versiones previas; parece sensible a la semilla y al régimen de entrenamiento.

**La mayor brecha teoría↔práctica:** DQN asume un MDP estacionario; este mercado no lo es (regímenes cambiantes, observaciones correlacionadas, señal/ruido pésima). Por eso "más interacción" no garantiza una mejor política, contra la intuición teórica.

---

## 12. Submission

Archivo único: **`agent.py`** (define `TradingEnv` y `Agent`; nombres de clase sin cambios). Pasa los 54 tests:

```bash
uv run pytest tests/test_submission.py -v   # 54 passed
```
