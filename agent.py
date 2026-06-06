# =============================================================================
# BELLMAN CAPITAL — agent.py
# Único archivo de entrega. Define TradingEnv y Agent. No modifica nada en src/.
#
# Resumen de diseño (ver informe del Punto 1 para la justificación completa):
#   - Intervalo objetivo: 1h. Lookback de 24 (un día de historia).
#   - State: log-returns (24 x 3 activos) + régimen (vol, momentum) + pesos
#            actuales + indicador de rebalanceo. Todo lookahead-safe.
#   - Action space: 10 acciones discretas long-only (9 carteras + HOLD).
#            El orden respeta los baselines en src/baselines.py.
#   - Rebalanceo periódico: el agente propone cada paso, el entorno aplica el
#            cambio cada 168 pasos (semanal) -> preserva el edge bajo costos.
#   - Reward: log-return del portafolio (neto de costos) menos penalización de
#            turnover. Respeta los signos exigidos por los tests.
#   - Algoritmo: Double DQN con target network y replay buffer.
# =============================================================================

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.env import BaseTradingEnv
from src.base import BaseAgent


# ── Action space ────────────────────────────────────────────────────────────
# Pesos por accion: [asset_0, asset_1, asset_2, cash]. Cada fila suma 1,
# cash >= 0, pesos riesgosos en [0, 1]. El ORDEN respeta los baselines:
#   accion 0 = HoldCash, accion 1 = HoldAsset0, accion 4 = EqualWeight.
# Menu LONG-ONLY: el EDA del Punto 1 mostro drift alcista estructural fuerte;
# las posiciones cortas resultaron -EV y amplificaban el distribution shift
# (un agente entrenado en periodos bear shorteaba en periodos bull). Se exploro
# con shorts (acciones [-0.5,..,1.5]) y empeoraban el desempeno, asi que el menu
# final es long-only + cash.
# La ultima accion (HOLD) mantiene los pesos actuales (turnover 0), dando al
# agente una forma explicita de no pagar costos. Su fila es nominal (all-cash);
# _weights_from_action la intercepta.
_ACTION_WEIGHTS = np.array(
    [
        [0.0,      0.0,      0.0,      1.0],   # 0  todo cash
        [1.0,      0.0,      0.0,      0.0],   # 1  todo asset_0
        [0.0,      1.0,      0.0,      0.0],   # 2  todo asset_1
        [0.0,      0.0,      1.0,      0.0],   # 3  todo asset_2
        [1/3,      1/3,      1/3,      0.0],   # 4  equal weight riesgoso
        [0.5,      0.0,      0.0,      0.5],   # 5  half-bet asset_0 + cash
        [0.0,      0.5,      0.0,      0.5],   # 6  half-bet asset_1 + cash
        [0.0,      0.0,      0.5,      0.5],   # 7  half-bet asset_2 + cash
        [1/6,      1/6,      1/6,      0.5],   # 8  half equal-weight + cash
        [0.0,      0.0,      0.0,      1.0],   # 9  HOLD (mantiene pesos actuales)
    ],
    dtype=np.float32,
)
N_ACTIONS = len(_ACTION_WEIGHTS)
_HOLD_ACTION = N_ACTIONS - 1   # indice de la accion HOLD


# ── Environment ───────────────────────────────────────────────────────────────

class TradingEnv(BaseTradingEnv):

    # cuantos pasos de historia ve el agente (un dia a 1h)
    LOOKBACK = 24

    # penalizacion extra por turnover en el reward (amplifica la senal de costo
    # para que supere el ruido del retorno horario). 0.0 = solo log-return.
    TURNOVER_PENALTY = 5e-4

    # frecuencia de rebalanceo: el agente propone cada paso, pero el entorno solo
    # aplica el cambio cada REBALANCE_EVERY pasos. A 1h, 168 = una semana.
    # El EDA mostro que el agente tiene senal direccional (gana a 0 bps) pero su
    # edge moria por costos al rebalancear a diario; el rebalanceo semanal reduce
    # el turnover ~8x y preserva el desempeno bajo costos realistas.
    REBALANCE_EVERY = 168

    # formulacion del reward (ver Seccion 4 del informe):
    #   "log"      -> log-return puro
    #   "turnover" -> log-return - lambda*turnover
    #   "drawdown" -> log-return - mu*drawdown_actual   (FINAL; el que se entrena)
    # Con rebalanceo semanal el turnover ya esta controlado estructuralmente, asi
    # que penalizar el drawdown (alineado con el Sortino, la metrica objetivo)
    # rinde mejor en todos los regimenes probados, incluido el bear de 2022.
    REWARD_MODE = "drawdown"
    DRAWDOWN_PENALTY = 0.10

    def __init__(self, prices, transaction_cost_bps=10.0, initial_cash=10_000.0):
        super().__init__(prices, transaction_cost_bps, initial_cash)

        self._lookback = self.LOOKBACK
        self._initial_cash = float(initial_cash)

        # turnover del ultimo rebalanceo (0 hasta el primer step real; mantener
        # en 0 hace que los tests aislados de _reward devuelvan log-return puro).
        self._last_turnover = 0.0
        # pico del valor del portafolio, para el reward drawdown-penalized
        self._peak_value = float(initial_cash)

        self.action_space = spaces.Discrete(N_ACTIONS)

        # log-returns: LOOKBACK pasos x 3 activos riesgosos  -> 3 * LOOKBACK
        # regimen:     vol (3) + momentum (3)                 -> 6
        # posicion:    pesos actuales del portafolio          -> 4
        # fase:        indicador de "toca rebalancear"         -> 1
        obs_dim = 3 * self._lookback + 6 + 4 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # precios de los 3 activos riesgosos (sin la columna cash, que es constante)
        self._risky = self.prices[:, :3]

    def _is_rebalance_step(self) -> bool:
        # self._t aun no se incrementa cuando se evalua dentro de step()
        return (self._t % self.REBALANCE_EVERY) == 0

    # -- reset: reinicia el estado propio al empezar cada episodio ------------
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self._peak_value = float(getattr(self, "_value", self._initial_cash))
        self._last_turnover = 0.0
        return obs, info

    # -- observacion ----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        # en el paso terminal BaseTradingEnv llama _obs con self._t == len(prices);
        # acotamos al ultimo indice valido (ese obs no se usa para decidir).
        t = min(self._t, len(self._risky) - 1)
        L = self._lookback

        # --- bloque 1: ultimos L log-returns de cada activo riesgoso ---------
        # ventana de precios [t-L, ..., t]  -> L returns, lookahead-safe
        window = self._risky[t - L:t + 1]                        # (L+1, 3)
        rets = np.log((window[1:] + 1e-8) / (window[:-1] + 1e-8))  # (L, 3)
        rets = rets * 100.0                                      # llevar a orden ~1
        ret_block = rets.reshape(-1)                             # (L*3,) tiempo-mayor

        # --- bloque 2: features de regimen (volatilidad y momentum) ----------
        vol = rets.std(axis=0)                                   # (3,)
        m = min(20, t)
        mom = np.log((self._risky[t] + 1e-8) / (self._risky[t - m] + 1e-8))  # (3,)

        # --- bloque 3: pesos actuales del portafolio -------------------------
        weights = self._weights.astype(np.float32)               # (4,)

        # --- bloque 4: indicador de fase (1 si la proxima accion rebalancea) --
        phase = np.array([1.0 if (t % self.REBALANCE_EVERY) == 0 else 0.0],
                         dtype=np.float32)

        obs = np.concatenate([ret_block, vol, mom, weights, phase]).astype(np.float32)
        # robustez numerica: sin NaN/Inf y acotado para gradientes estables
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0).astype(np.float32)
        return obs

    # -- mapeo accion -> pesos ------------------------------------------------
    def _weights_from_action(self, action: int) -> np.ndarray:
        action = int(action)
        if not self._is_rebalance_step() or action == _HOLD_ACTION:
            # fuera de la ventana de rebalanceo, o accion HOLD:
            # mantener la cartera actual -> turnover 0, sin costo.
            w = self._weights.astype(np.float32).copy()
        else:
            w = _ACTION_WEIGHTS[action].copy()
        # self._weights todavia son los pesos previos en este punto del step,
        # asi que capturamos el turnover del rebalanceo para usarlo en _reward.
        self._last_turnover = float(np.abs(w - self._weights).sum())
        return w

    # -- reward ---------------------------------------------------------------
    def _reward(self, prev_value: float, curr_value: float) -> float:
        # En todos los modos: > 0 si el valor crece, < 0 si cae, y exactamente 0
        # si no cambia (con turnover 0 y sin drawdown), respetando los tests.
        prev = max(float(prev_value), 1e-8)
        curr = max(float(curr_value), 1e-8)
        log_ret = float(np.log(curr / prev))

        if self.REWARD_MODE == "log":
            return log_ret

        if self.REWARD_MODE == "drawdown":
            self._peak_value = max(self._peak_value, curr)
            dd = (self._peak_value - curr) / self._peak_value   # >= 0
            return log_ret - self.DRAWDOWN_PENALTY * dd

        # "turnover" (final): el valor ya viene neto de costos; la penalizacion
        # explicita amplifica la senal de costo sobre el ruido del retorno.
        return log_ret - self.TURNOVER_PENALTY * self._last_turnover


# ── DQN network ───────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Replay buffer ───────────────────────────────────────────────────────────

class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent(BaseAgent):

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__(obs_dim, n_actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hiperparametros (ver configs/default.yaml)
        self.gamma = 0.99
        self.lr = 1e-4
        self.batch_size = 64
        self.target_update_freq = 1000
        self.learning_starts = 200
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 50_000

        # exploracion
        self.epsilon = self.epsilon_start
        self._steps_done = 0

        # redes
        self.q = _QNetwork(obs_dim, n_actions).to(self.device)
        self.target = _QNetwork(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.buffer = _ReplayBuffer(100_000)

    # -- schedule de epsilon --------------------------------------------------
    def _update_epsilon(self):
        frac = min(1.0, self._steps_done / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # -- politica epsilon-greedy (entrenamiento) ------------------------------
    def _epsilon_greedy(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self.act(obs)

    # -- paso de aprendizaje (Double DQN) -------------------------------------
    def _learn(self):
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s,  device=self.device)
        a  = torch.as_tensor(a,  device=self.device).unsqueeze(1)
        r  = torch.as_tensor(r,  device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(s2, device=self.device)
        d  = torch.as_tensor(d,  device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            # Double DQN: la red online elige la accion, la target la evalua
            next_actions = self.q(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, next_actions)
            target = r + self.gamma * q_next * (1.0 - d)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

    # -- bucle de entrenamiento -----------------------------------------------
    def train(self, env, n_steps: int = 200_000) -> None:
        self.q.train()
        obs, _ = env.reset()
        for _ in range(n_steps):
            self._update_epsilon()
            action = self._epsilon_greedy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.buffer.push(obs, action, reward, next_obs, float(done))

            if len(self.buffer) >= max(self.batch_size, self.learning_starts):
                self._learn()

            self._steps_done += 1
            if self._steps_done % self.target_update_freq == 0:
                self.target.load_state_dict(self.q.state_dict())

            obs = next_obs if not done else env.reset()[0]
        self.q.eval()

    # -- politica greedy (evaluacion, deterministica) -------------------------
    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                device=self.device).unsqueeze(0)
            q = self.q(x)
            return int(q.argmax(dim=1).item())
