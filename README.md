# rl-control

Explorations in control — classical and reinforcement-learning — built from scratch
with NumPy and Matplotlib. No Gym, no PyTorch, no stable-baselines. Each piece of the
control loop is written out by hand so its behaviour, and its failure modes, are visible.

This repository is a growing collection of self-contained case studies. Each one takes a
single system and compares different ways of controlling it under identical dynamics.

---

## Explorations

### 1 · Inverted pendulum

A torque-limited inverted pendulum, controlled three ways and compared on the same plant:

- **Manual** — drive it yourself with the `←` / `→` keys. The human baseline, and the
  quickest way to feel how hard the swing-up really is.
- **PID** — classical feedback control. Fast and simple at balancing near upright, but it
  cannot swing up: with no notion of energy, it just saturates the torque limit and stalls.
- **RL agent** — a learned policy that pumps energy in over several swings and then catches
  and holds the pendulum upright — the thing PID can't do.

The point is the comparison: same physics, same limits, three controllers side by side.

*More explorations will be added here over time.*

---

## Getting started

Requires Python 3.11+, with only `numpy` and `matplotlib`.

```bash
git clone https://github.com/abhaskar94/rl-control
cd rl-control
```

Each exploration lives in its own folder with its own runnable scripts.
