"""RL control: a Gaussian policy trained with an average-reward actor-critic.

This is the controller PID cannot match: it learns to pump energy in over several swings and
then catch and hold the pendulum upright. Everything here is hand-written NumPy -- no Gym, no
autograd -- so the learning rule is fully visible.

The pieces
----------
* `TileCoder`   -- turns the continuous state (theta, theta_dot) into a sparse set of binary
                  features via tile coding. The hashing is a *pure function* of the tile
                  coordinates, so indices are identical across processes: weights saved by
                  ``train.py`` line up with the same tiles when ``app.py`` loads them.
* `ActorCritic` -- linear policy and value function over those features. The policy is
                  Gaussian: mean ``mu(s) = w_mu . phi(s)`` and std ``sigma(s) = exp(w_sig . phi(s))``.
                  Learning is differential (average-reward) TD(0): no discount factor, a running
                  average-reward estimate ``r_bar`` instead.
* `RLController`-- adapts a trained `ActorCritic` to the `Controller` interface for the app.
* `swingup_reward` -- the task the agent optimises (only training uses it; the plant does not).
"""

from __future__ import annotations

import os

import numpy as np

from env import State

from .base import Controller


# ─────────────────────────────────────────────────────────────────────────────
#  Tile coding
# ─────────────────────────────────────────────────────────────────────────────
class TileCoder:
    """Maps (theta, theta_dot) to ``num_tilings`` active feature indices.

    Each tiling is a grid of ``num_tiles`` cells per axis, offset from the next. A state
    activates exactly one cell per tiling, giving a sparse binary feature vector whose active
    indices this returns.
    """

    def __init__(self, num_tilings: int = 8, num_tiles: int = 8,
                 theta_range: tuple[float, float] = (-np.pi, np.pi),
                 theta_dot_range: tuple[float, float] = (-2 * np.pi, 2 * np.pi)) -> None:
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.theta_range = theta_range
        self.theta_dot_range = theta_dot_range
        self._bins = num_tiles + 1                    # offsets push a coord up to num_tiles
        self.size = num_tilings * self._bins ** 2     # total number of features

    def __call__(self, theta: float, theta_dot: float) -> np.ndarray:
        a = self._scale(theta, self.theta_range)
        v = self._scale(theta_dot, self.theta_dot_range)
        tilings = np.arange(self.num_tilings)
        offset = tilings / self.num_tilings
        ta = np.clip(np.floor(a + offset).astype(int), 0, self.num_tiles)
        tv = np.clip(np.floor(v + offset).astype(int), 0, self.num_tiles)
        return tilings * self._bins ** 2 + ta * self._bins + tv

    def _scale(self, x: float, rng: tuple[float, float]) -> float:
        lo, hi = rng
        return (np.clip(x, lo, hi) - lo) / (hi - lo) * self.num_tiles


# ─────────────────────────────────────────────────────────────────────────────
#  Average-reward actor-critic
# ─────────────────────────────────────────────────────────────────────────────
class ActorCritic:
    """Linear Gaussian actor with a linear critic, trained by differential TD(0)."""

    def __init__(self, coder: TileCoder, tau_max: float = 3.0,
                 alpha_mu: float = 0.25, alpha_sigma: float = 0.01,
                 alpha_critic: float = 2.0, alpha_r: float = 2 ** -6) -> None:
        self.coder = coder
        self.tau_max = tau_max
        # Per-tiling learning: divide by the number of active features so the effective
        # step size is independent of how many tilings are used.
        self.alpha_mu = alpha_mu / coder.num_tilings
        self.alpha_sigma = alpha_sigma / coder.num_tilings
        self.alpha_critic = alpha_critic / coder.num_tilings
        self.alpha_r = alpha_r

        self.w_mu = np.zeros(coder.size)     # policy mean weights
        self.w_sigma = np.zeros(coder.size)  # policy log-std weights
        self.w_critic = np.zeros(coder.size) # value-function weights
        self.r_bar = 0.0                     # running average-reward estimate

    # ---- policy / value ----
    def policy(self, features: np.ndarray) -> tuple[float, float]:
        mu = float(self.w_mu[features].sum())
        log_sigma = np.clip(self.w_sigma[features].sum(), -5.0, 2.0)
        return mu, float(np.exp(log_sigma))

    def value(self, features: np.ndarray) -> float:
        return float(self.w_critic[features].sum())

    def act(self, features: np.ndarray, explore: bool = True) -> tuple[float, float, float]:
        """Return ``(tau, mu, sigma)``. With ``explore`` off, ``tau`` is the mean action."""
        mu, sigma = self.policy(features)
        tau = np.random.normal(mu, sigma) if explore else mu
        return float(np.clip(tau, -self.tau_max, self.tau_max)), mu, sigma

    # ---- learning ----
    def update(self, features: np.ndarray, features_next: np.ndarray,
               tau: float, mu: float, sigma: float, reward: float) -> float:
        """One differential TD(0) step. Returns the TD error (for logging)."""
        delta = np.clip(
            reward - self.r_bar + self.value(features_next) - self.value(features),
            -5.0, 5.0)
        self.r_bar += self.alpha_r * delta
        self.w_critic[features] += self.alpha_critic * delta

        grad_mu = np.clip((tau - mu) / sigma ** 2, -5.0, 5.0)
        grad_sigma = np.clip((tau - mu) ** 2 / sigma ** 2 - 1.0, -5.0, 5.0)
        self.w_mu[features] += self.alpha_mu * delta * grad_mu
        self.w_sigma[features] += self.alpha_sigma * delta * grad_sigma
        return float(delta)

    # ---- persistence ----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, w_mu=self.w_mu, w_sigma=self.w_sigma,
                 w_critic=self.w_critic, r_bar=self.r_bar,
                 num_tilings=self.coder.num_tilings, num_tiles=self.coder.num_tiles)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.w_mu = data["w_mu"]
        self.w_sigma = data["w_sigma"]
        self.w_critic = data["w_critic"]
        self.r_bar = float(data["r_bar"])
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Reward (the task the agent optimises)
# ─────────────────────────────────────────────────────────────────────────────
def swingup_reward(state: State, prev_theta: float, tau: float) -> float:
    """Two-phase reward: pump energy while far from upright, hold still once near it.

    ``w`` blends smoothly from 0 (hanging) to 1 (upright). Far away the agent is rewarded for
    *reducing* the pendulum's height error swing-by-swing; near upright it is rewarded for a
    small angle, low velocity and low torque.
    """
    w = np.cos(state.theta / 2) ** 4                       # 1 upright, 0 hanging
    swing = np.cos(state.theta) - np.cos(prev_theta)       # progress toward the top
    balance = 1.0 - (state.theta ** 2 + 0.1 * state.theta_dot ** 2 + 0.001 * tau ** 2)
    return float((1 - w) * swing + w * balance)


# ─────────────────────────────────────────────────────────────────────────────
#  Controller adapter
# ─────────────────────────────────────────────────────────────────────────────
class RLController(Controller):
    name = "RL agent"

    def __init__(self, agent: ActorCritic, deterministic: bool = True) -> None:
        self.agent = agent
        self.deterministic = deterministic
        # Last-step diagnostics, surfaced by the app.
        self.mu = 0.0
        self.sigma = 0.0
        self.value = 0.0

    @classmethod
    def load(cls, path: str, tau_max: float = 3.0) -> "RLController":
        """Build an RL controller from saved weights. Falls back to a zero policy if the
        file is missing (the app still runs; the agent just does nothing)."""
        coder = TileCoder()
        agent = ActorCritic(coder, tau_max=tau_max)
        agent.loaded = agent.load(path)
        return cls(agent)

    def action(self, state: State) -> float:
        features = self.agent.coder(state.theta, state.theta_dot)
        tau, mu, sigma = self.agent.act(features, explore=not self.deterministic)
        self.mu, self.sigma = mu, sigma
        self.value = self.agent.value(features)
        return tau
