"""The inverted-pendulum plant.

This module is the single source of truth for the physics. Every controller in the
comparison drives *this* plant and nothing else, so the only thing that differs between
open-loop, PID and RL is the control law -- never the dynamics. Do not reimplement the
integrator anywhere else.

Convention
----------
`theta` is measured from the upright:

        theta = 0      -> balanced, straight up
        theta = +-pi   -> hanging straight down

so the rendered bob sits at ``(L*sin(theta), L*cos(theta))`` and "swing-up" means driving
theta from +-pi to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class State(NamedTuple):
    """The full state a controller is allowed to see."""

    theta: float      # angle from upright [rad], wrapped to (-pi, pi]
    theta_dot: float  # angular velocity [rad/s]


@dataclass
class Pendulum:
    """A torque-driven pendulum integrated with semi-implicit Euler.

    The equation of motion is

        theta_ddot = (g / L) * sin(theta) + tau / (m * L**2) - c * theta_dot

    The applied torque is saturated to ``[-tau_max, tau_max]``; that limit is what makes
    the task interesting, since no controller can simply overpower gravity in one move.
    """

    g: float = 9.81       # gravity [m/s^2]
    L: float = 1.0        # pole length [m]
    m: float = 1.0        # bob mass [kg]
    dt: float = 0.02      # integration step [s]
    c: float = 0.001      # viscous damping at the pivot
    tau_max: float = 3.0  # torque saturation [N*m]

    theta_dot_max: float = 15.0  # angular-velocity clamp, keeps the sim well-behaved

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, theta: float = np.pi, theta_dot: float = 0.0) -> State:
        """Reset the plant. Defaults to hanging straight down (the swing-up start)."""
        self._theta = self._wrap(theta)
        self._theta_dot = float(theta_dot)
        return self.state

    def step(self, tau: float) -> State:
        """Advance one timestep under applied torque ``tau`` and return the new state."""
        tau = float(np.clip(tau, -self.tau_max, self.tau_max))

        theta_ddot = (
            (self.g / self.L) * np.sin(self._theta)
            + tau / (self.m * self.L**2)
            - self.c * self._theta_dot
        )
        self._theta_dot = float(
            np.clip(self._theta_dot + theta_ddot * self.dt,
                    -self.theta_dot_max, self.theta_dot_max)
        )
        self._theta = self._wrap(self._theta + self._theta_dot * self.dt)
        return self.state

    @property
    def state(self) -> State:
        return State(self._theta, self._theta_dot)

    @staticmethod
    def _wrap(theta: float) -> float:
        """Wrap an angle into (-pi, pi]."""
        return float((theta + np.pi) % (2 * np.pi) - np.pi)
