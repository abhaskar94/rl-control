"""The one interface every controller implements.

This tiny contract is what makes the comparison fair and the app simple: the app holds a
single `Controller`, and every frame it calls ``tau = controller.action(state)`` and feeds
that torque to the plant. Switching between open-loop, PID and RL is nothing more than
swapping which `Controller` object is active -- the plant never changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from env import State


class Controller(ABC):
    """Maps the pendulum state to an applied torque."""

    #: Short human-readable name, shown in the UI.
    name: str = "controller"

    @abstractmethod
    def action(self, state: State) -> float:
        """Return the torque to apply this timestep, in N*m."""

    def reset(self) -> None:
        """Clear any internal state (integrators, key commands, ...).

        Stateless controllers can rely on this no-op default.
        """
