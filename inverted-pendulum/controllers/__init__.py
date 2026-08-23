"""The controllers, all speaking the same `Controller` interface.

    from controllers import OpenLoopController, PlaceholderController

Each maps ``state -> torque``; the app swaps between them on the same plant. PID and the RL
agent land in later milestones; until then they run as `PlaceholderController`s.
"""

from .base import Controller
from .open_loop import OpenLoopController
from .pid import PIDController
from .placeholder import PlaceholderController

__all__ = [
    "Controller",
    "OpenLoopController",
    "PIDController",
    "PlaceholderController",
]
