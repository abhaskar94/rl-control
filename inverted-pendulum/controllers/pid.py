"""PID control: classical feedback that regulates the angle to upright.

The law is the textbook one, driving the error ``e = 0 - theta`` to zero:

    tau = Kp * e + Ki * integral(e) + Kd * d(e)/dt

Near upright PID is fast and stable. But it has no notion of *energy*: started from
hanging down it sees a huge error, saturates the torque limit, and stalls -- it cannot pump
the pendulum up over several swings. That failure, on the very same plant the RL agent
succeeds on, is the point of the comparison.
"""

from __future__ import annotations

from env import State

from .base import Controller


class PIDController(Controller):
    name = "PID"

    def __init__(self, kp: float = 20.0, ki: float = 5.0, kd: float = 2.0,
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
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
