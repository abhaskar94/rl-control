"""Inverted-pendulum control comparison — the single runnable app.

    python app.py

Milestone 1: the open-loop mode is live (drive the pendulum with the ← / → arrow keys). PID and
the RL agent are registered as placeholders and land in later milestones. The plant lives in
`env/`, the control laws in `controllers/`; this file is only UI + the animation loop:

    tau   = controller.action(state)
    state = env.step(tau)

Switching mode just swaps which `controller` is active — the plant never changes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from env import Pendulum
from controllers import OpenLoopController, PIDController, PlaceholderController

# ── palette (kept here because it's presentation, not physics) ──────────────
BG, PANEL = "#f7f8fa", "#ffffff"
INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, GREEN, RED, AMBER = "#2563eb", "#059669", "#dc2626", "#d97706"

MODE_COLOR = {"Open-loop": GREEN, "PID": BLUE, "RL agent": RED}
TRACE_LEN = 300  # frames of angle history to show


class App:
    """Owns the plant, the controllers, and all the drawing."""

    def __init__(self) -> None:
        # Torque limit raised well above the ±3 swing-up-challenge value so high-gain PID has
        # the authority to drive straight to upright without saturating. NOTE: this makes
        # swing-up easy for *every* controller — revisit when the RL agent lands (milestone 3).
        self.env = Pendulum(tau_max=150.0)
        # One controller per mode. Open-loop and PID are real; RL is a placeholder for now.
        self.open_loop = OpenLoopController(torque=1.5)
        self.controllers = {
            "Open-loop": self.open_loop,
            "PID": PIDController(kp=40.0, ki=1.0, kd=12.0, dt=self.env.dt),
            "RL agent": PlaceholderController("RL agent"),
        }
        # A one-line explanation shown under the stage for each mode.
        self.hints = {
            "Open-loop": "Hold  ←  /  →  to apply torque",
            "PID": "High-gain PID drives straight to upright and holds it",
            "RL agent": "placeholder — coming soon",
        }
        self.mode = "Open-loop"
        self.state = self.env.reset()
        self.theta_history: list[float] = []
        self._build_figure()

    # ── controller for the active mode ──────────────────────────────────────
    @property
    def controller(self):
        return self.controllers[self.mode]

    @property
    def is_placeholder(self) -> bool:
        return isinstance(self.controller, PlaceholderController)

    # ── figure ──────────────────────────────────────────────────────────────
    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(11, 6), facecolor=BG)
        self.fig.canvas.manager.set_window_title("Inverted Pendulum · Control Comparison")

        self.fig.text(0.06, 0.93, "Inverted Pendulum", color=INK,
                      fontsize=17, fontweight="bold")
        self.fig.text(0.06, 0.89, "Control comparison — open-loop / PID / RL",
                      color=MUTED, fontsize=10)

        # Pendulum stage (left).
        self.ax = self.fig.add_axes([0.06, 0.10, 0.52, 0.72], facecolor=PANEL)
        self.ax.set_xlim(-1.35, 1.35)
        self.ax.set_ylim(-1.35, 1.35)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_edgecolor(GRID)
        self.ax.axhline(0, color=GRID, lw=1, ls="--", zorder=0)
        self.ax.plot(0, 0, "o", color=INK, ms=9, zorder=5)  # pivot

        self.arm, = self.ax.plot([], [], lw=5, color=BLUE, solid_capstyle="round", zorder=3)
        self.bob, = self.ax.plot([], [], "o", ms=20, color=AMBER, zorder=4,
                                 markeredgecolor="#b45309")
        self.torque_arrow = None
        self.hint = self.ax.text(0, -1.24, self.hints[self.mode],
                                 color=MUTED, fontsize=9, ha="center")

        # Angle trace (bottom-right).
        self.ax_theta = self.fig.add_axes([0.66, 0.10, 0.30, 0.30], facecolor=PANEL)
        self.ax_theta.set_title("Angle θ (rad)", color=INK, fontsize=10, loc="left")
        self.ax_theta.set_xlim(0, TRACE_LEN)
        self.ax_theta.set_ylim(-3.3, 3.3)
        self.ax_theta.tick_params(colors=MUTED, labelsize=8)
        for sp in self.ax_theta.spines.values():
            sp.set_edgecolor(GRID)
        self.ax_theta.axhline(0, color=GREEN, lw=0.8, ls=":")  # upright
        self.theta_line, = self.ax_theta.plot([], [], lw=1.6, color=BLUE)

        # Mode buttons (top-right): three across the right column, evenly spaced.
        self.buttons: dict[str, Button] = {}
        for i, name in enumerate(self.controllers):
            ax_btn = self.fig.add_axes([0.66 + i * 0.10, 0.80, 0.093, 0.075])
            btn = Button(ax_btn, name, color=PANEL, hovercolor="#eef2ff")
            btn.label.set_fontsize(9)
            btn.on_clicked(lambda _e, n=name: self.set_mode(n))
            self.buttons[name] = btn

        # Reset button (own row, below the mode buttons).
        ax_reset = self.fig.add_axes([0.66, 0.71, 0.093, 0.055])
        self.reset_btn = Button(ax_reset, "Reset", color=PANEL, hovercolor="#fee2e2")
        self.reset_btn.label.set_fontsize(9)
        self.reset_btn.on_clicked(lambda _e: self.reset())

        # Status panel.
        self.status = self.fig.text(0.66, 0.70, "", color=INK, fontsize=10,
                                     family="monospace", va="top", linespacing=1.6)

        self._refresh_buttons()

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)

    # ── interaction ─────────────────────────────────────────────────────────
    def set_mode(self, name: str) -> None:
        self.mode = name
        self.controller.reset()
        self.hint.set_text(self.hints[name])
        self._refresh_buttons()

    def reset(self) -> None:
        self.state = self.env.reset()
        self.controller.reset()
        self.theta_history.clear()

    def _on_key_press(self, event) -> None:
        if event.key == "left":
            self.open_loop.press(-1)
        elif event.key == "right":
            self.open_loop.press(+1)

    def _on_key_release(self, event) -> None:
        if event.key in ("left", "right"):
            self.open_loop.release()

    def _refresh_buttons(self) -> None:
        for name, btn in self.buttons.items():
            active = name == self.mode
            btn.ax.set_facecolor(MODE_COLOR[name] if active else PANEL)
            btn.label.set_color("white" if active else INK)
            btn.label.set_fontweight("bold" if active else "normal")

    # ── animation ───────────────────────────────────────────────────────────
    def update(self, _frame):
        tau = self.controller.action(self.state)
        # The plant saturates torque to [-tau_max, tau_max]; show what is actually applied.
        applied = float(np.clip(tau, -self.env.tau_max, self.env.tau_max))
        saturated = abs(tau) > self.env.tau_max
        self.state = self.env.step(tau)
        theta, theta_dot = self.state

        # Pendulum geometry (θ=0 is up): bob at (L·sinθ, L·cosθ).
        x = self.env.L * np.sin(theta)
        y = self.env.L * np.cos(theta)
        self.arm.set_data([0, x], [0, y])
        self.bob.set_data([x], [y])

        if self.torque_arrow is not None:
            self.torque_arrow.remove()
        length = 0.35 * (applied / self.env.tau_max)
        self.torque_arrow = self.ax.arrow(0, 0, length, 0, width=0.02,
                                          head_width=0.07, head_length=0.05,
                                          color=RED, alpha=0.85, zorder=2)

        self.theta_history.append(theta)
        if len(self.theta_history) > TRACE_LEN:
            self.theta_history.pop(0)
        self.theta_line.set_data(range(len(self.theta_history)), self.theta_history)

        note = "  (placeholder — coming soon)" if self.is_placeholder else ""
        tau_flag = "  (saturated)" if saturated else ""
        # Show the PID gains when PID is the active mode.
        if isinstance(self.controller, PIDController):
            c = self.controller
            gains = f"P {c.kp:g}    I {c.ki:g}    D {c.kd:g}\n"
        else:
            gains = ""
        self.status.set_text(
            f"mode   {self.mode}{note}\n\n"
            f"{gains}"
            f"θ      {theta:+.3f} rad\n"
            f"θ̇      {theta_dot:+.3f} rad/s\n"
            f"τ      {applied:+.3f} N·m{tau_flag}")
        self.status.set_color(MUTED if self.is_placeholder else INK)

        return self.arm, self.bob, self.torque_arrow, self.theta_line

    def run(self) -> None:
        self._ani = FuncAnimation(self.fig, self.update, interval=20, blit=False)
        plt.show()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
