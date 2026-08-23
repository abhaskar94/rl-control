# CLAUDE.md — inverted-pendulum

Instructions for working in this folder. Read this before editing anything here.

## What this project is

One inverted-pendulum plant, controlled three ways and compared in a single app:

1. **Open-loop** — a human driving it with the arrow keys (no feedback).
2. **PID** — classical feedback control.
3. **RL agent** — a learned policy (average-reward actor-critic, written from scratch).

The whole point is a *fair* comparison: same plant, same torque limit, three control laws.
The deliverable is one runnable script that toggles between the three.

## The invariant — read this first

**All three controllers drive the exact same plant. Physics lives in exactly one place:
`env/pendulum.py`.** Never reimplement or copy the dynamics into a controller, the app, or a
training script. If the plant changes, it changes there and every controller inherits it —
that is the only thing that keeps the comparison honest.

Canonical dynamics (defined once, in `env/pendulum.py`):
- `theta_ddot = (g/L)·sin(theta) + tau/(m·L²) − c·theta_dot`
- constants: `g=9.81, L=1, m=1, dt=0.02, c=0.001`.
- **torque limit is a plant parameter** (`tau_max`): `env` defaults to a tight `3.0` (the hard
  swing-up challenge), but `app.py` currently raises it to `150` so high-gain PID has the
  authority to drive straight to upright without saturating. This makes swing-up easy for every
  controller — **revisit the shared limit when the RL agent lands** so the comparison stays
  meaningful.
- **convention: `theta=0` is upright, `theta=±pi` is hanging down**; the task is swing-up from
  hanging.

## The controller contract

Every controller implements the same tiny interface so the app can swap them freely:

```python
class Controller:
    name: str
    def action(self, state) -> float: ...   # state -> torque
    def reset(self) -> None: ...             # clear integrators / internal state
```

The app holds one active `Controller` and each frame runs
`tau = controller.action(state); state = env.step(tau)`. **Adding a new control method =
writing one new `Controller` and registering it in the app.** Nothing else should need to change.

## Hard constraints

- **NumPy + Matplotlib by default.** Everything so far is hand-written from scratch, and that
  visibility is the point of this repo (see [../README.md](../README.md)). Advanced libraries
  (Gym, PyTorch, stable-baselines, …) are **not forbidden**, but **ask the user first, every
  time, before adding any new dependency** — never pull one in unprompted.
- **Cite your sources.** Whenever an implementation follows a paper, course, or other external
  reference, note it properly — a `References` section in the relevant file and/or the README,
  with enough to find it again (author, title, year, link).
- Keep the **control/RL math out of the UI**: numerical logic lives in `env/` and
  `controllers/`; all rendering/drawing lives in the app. Don't mix plotting into a controller
  or physics into the app.
- Each script should be runnable and the app should launch with a single command.

## Layout

```
inverted-pendulum/
  app.py               # THE deliverable: UI, mode toggle, animation loop
  env/pendulum.py      # the one plant (the invariant)
  controllers/         # open_loop.py, pid.py, rl_agent.py — all share the contract
  policies/            # pre-trained RL weights so the RL toggle works on launch
  train.py             # headless RL training -> writes policies/
```

## Status

Being rebuilt to the structure above (clean-slate rewrite). Until this note is removed, assume
the layout is aspirational and check what actually exists before describing it as working.
