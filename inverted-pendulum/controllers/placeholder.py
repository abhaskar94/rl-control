"""A stand-in for a control mode that isn't built yet.

It satisfies the `Controller` contract but applies no torque, so a placeholder mode simply lets
the pendulum swing freely. The app labels these modes as "coming soon". As each real controller
lands, its placeholder is swapped out for the genuine article — nothing else in the app changes.
"""

from __future__ import annotations

from env import State

from .base import Controller


class PlaceholderController(Controller):
    def __init__(self, name: str) -> None:
        self.name = name

    def action(self, state: State) -> float:
        return 0.0
