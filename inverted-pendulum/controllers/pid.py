"""PID control: classical feedback that regulates the angle to upright.

The law drives the error ``e = 0 - theta`` to zero:

    tau = Kp * e + Ki * integral(e) + Kd * d(e)/dt

The derivative term uses *rate feedback* -- ``d(e)/dt = -d(theta)/dt`` for a fixed setpoint,
read straight from the plant's angular velocity. That avoids the "derivative kick" a finite
difference of the error produces at reset, so the torque stays smooth from the first step.

Given enough torque authority and high gains, PID drives the pole from hanging straight to
upright and holds it. Under a *tight* torque limit it cannot (it saturates and stalls, having
no notion of energy) -- which is the contrast we will draw against the RL agent.
"""

from __future__ import annotations

from env import State

from .base import Controller


class PIDController(Controller):
    name = "PID"

    def __init__(self, kp: float = 40.0, ki: float = 1.0, kd: float = 12.0,
                 dt: float = 0.02, setpoint: float = 0.0) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.setpoint = setpoint  # target angle [rad]; 0 = upright
        self.reset()

    def action(self, state: State) -> float:
        error = self.setpoint - state.theta
        self._integral += error * self.dt
        # Rate feedback: d(error)/dt = -d(theta)/dt for a constant setpoint.
        derivative = -state.theta_dot
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self) -> None:
        self._integral = 0.0
