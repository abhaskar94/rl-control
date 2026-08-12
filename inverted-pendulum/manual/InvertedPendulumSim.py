import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physics parameters
g = 9.81
L = 1.0
m = 1.0
theta = 0.1       # small initial angle
theta_dot = 0.0
tau = 0.0
dt = 0.02
c = 0.01

# Key state
keys_down = {"left": False, "right": False}

def on_key_press(event):
    if event.key == "left":
        keys_down["left"] = True
    if event.key == "right":
        keys_down["right"] = True

def on_key_release(event):
    if event.key == "left":
        keys_down["left"] = False
    if event.key == "right":
        keys_down["right"] = False

def PID_Control():
    print("test")

def step():
    global theta, theta_dot, tau

    # Determine torque from keys
    if keys_down["left"]:
        tau = -1.5
    elif keys_down["right"]:
        tau = 1.5
    else:
        tau = 0.0

    # Physics update
    theta_ddot = (g/L)*np.sin(theta) + tau/(m*L*L) - c*theta_dot
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
plt.title("Manual Inverted Pendulum Control (← → keys)")

line, = ax.plot([], [], lw=3)
arrow = ax.arrow(0, 0, 0, 0, color="red")

def update(frame):
    global arrow
    step()

    # Inverted pendulum coordinates (0 = UP)
    x =  L * np.sin(theta)
    y =  L * np.cos(theta)
    line.set_data([0, x], [0, y])

    arrow.remove()
    arrow = ax.arrow(0, 0, 0.3*np.sign(tau), 0,
                     width=0.03, color="red")

    return line, arrow


# Connect keyboard events
fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)

ani = FuncAnimation(fig, update, interval=20)
plt.show()
