"""Open-loop control: a human driving the pendulum with the arrow keys.

There is no feedback here -- the torque depends only on which key is held, not on the state
of the pendulum. It is the baseline that makes the other two controllers look clever: try to
swing the pendulum up by hand and you feel exactly how hard the task is.

The controller itself stays pure. The app owns the keyboard and simply sets `command` on
key-press / key-release events; `action()` just reports it back.
"""

from __future__ import annotations

from env import State

from .base import Controller


class OpenLoopController(Controller):
    name = "Open-loop"

    def __init__(self, torque: float = 1.5) -> None:
        self.torque = torque   # magnitude applied while a key is held
        self.command = 0.0     # set by the app from key events: -torque, 0, or +torque

    def press(self, direction: int) -> None:
        """Hold a key. ``direction`` is -1 (left) or +1 (right)."""
        self.command = direction * self.torque

    def release(self) -> None:
        """Release the key -- no torque."""
        self.command = 0.0

    def action(self, state: State) -> float:
        return self.command

    def reset(self) -> None:
        self.command = 0.0
