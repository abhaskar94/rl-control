import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox
import random
from matplotlib.gridspec import GridSpec

# Physics parameters
g = 9.81
L = 1.0
m = 1.0
theta = 0.1
theta_dot = 0.0
tau = 0.0
dt = 0.02
c = 0.001
use_pid = False
P_gain = 20
I_gain = 5
D_gain = 2


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

def toggle_mode(event):
    global use_pid
    use_pid = not use_pid

integral = 0
prev_error = 0

def PID_Control(theta):
    global integral, prev_error, P_gain, I_gain, D_gain

    SP = 0
    error = SP - theta

    integral += error * dt
    derivative = (error - prev_error) / dt
    prev_error = error

    tau = P_gain*error + I_gain*integral + D_gain*derivative
    return tau



def step():
    global theta, theta_dot, tau

    if use_pid:
        tau = PID_Control(theta)
    else:
        if keys_down["left"]:
            tau = -1.5
        elif keys_down["right"]:
            tau = 1.5
        else:
            tau = 0.0

    theta_ddot = (g/L)*np.sin(theta) + tau/(m*L*L) - c*theta_dot
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

def update_P(text):
    global P_gain
    try:
        P_gain = float(text)
    except ValueError:
        pass

def update_I(text):
    global I_gain
    try:
        I_gain = float(text)
    except ValueError:
        pass

def update_D(text):
    global D_gain
    try:
        D_gain = float(text)
    except ValueError:
        pass

# ---------------------------
#   FIGURE LAYOUT
# ---------------------------
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

# Pendulum panel
ax_pend = fig.add_subplot(gs[:, 0])
ax_pend.set_xlim(-1.2, 1.2)
ax_pend.set_ylim(-1.2, 1.2)
ax_pend.set_aspect("equal")
ax_pend.set_title("Inverted Pendulum (Manual / PID)")
mode_text = ax_pend.text(-1.1, 1.1, "", fontsize=12)

line, = ax_pend.plot([], [], lw=3)
arrow = ax_pend.arrow(0, 0, 0, 0, color="red")

# Button panel
ax_button = fig.add_subplot(gs[0, 1])
ax_button.set_xticks([])
ax_button.set_yticks([])

# --- CLEAN RIGHT PANEL LAYOUT ---

# Toggle button at the top
button_ax = plt.axes([0.72, 0.83, 0.18, 0.08])
button = Button(button_ax, "Toggle PID")
button.on_clicked(toggle_mode)

# PID Gains title (centered)
ax_pid_title = plt.axes([0.72, 0.76, 0.18, 0.05])
ax_pid_title.set_xticks([])
ax_pid_title.set_yticks([])
ax_pid_title.text(0.5, 0.5, "PID Gains", fontsize=11, fontweight="bold",
                  ha="center", va="center")

# --- Labels ABOVE each box (centered) ---
ax_P_label = plt.axes([0.72, 0.70, 0.055, 0.04])
ax_I_label = plt.axes([0.78, 0.70, 0.055, 0.04])
ax_D_label = plt.axes([0.84, 0.70, 0.055, 0.04])

for ax, text in [(ax_P_label, "P"), (ax_I_label, "I"), (ax_D_label, "D")]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10)

# --- Input boxes BELOW labels (centered text) ---
ax_P = plt.axes([0.72, 0.64, 0.055, 0.05])
ax_I = plt.axes([0.78, 0.64, 0.055, 0.05])
ax_D = plt.axes([0.84, 0.64, 0.055, 0.05])

tb_P = TextBox(ax_P, "", initial=str(P_gain))
tb_I = TextBox(ax_I, "", initial=str(I_gain))
tb_D = TextBox(ax_D, "", initial=str(D_gain))

# --- Force text to stay centered INSIDE the boxes ---
for tb in [tb_P, tb_I, tb_D]:
    tb.text_disp.set_ha("center")
    tb.text_disp.set_va("center")
    tb.text_disp.set_position((0.5, 0.5))
    tb.text_disp.set_fontsize(9)




# Tau time-series panel
ax_tau = fig.add_subplot(gs[1, 1])
ax_tau.set_title("Torque τ over Time")
ax_tau.set_xlim(0, 200)
ax_tau.set_ylim(-8, 8)
tau_line, = ax_tau.plot([], [], lw=2)
tau_history = []

def update(frame):
    global arrow

    step()
    mode_text.set_text("PID" if use_pid else "Manual")

    # Pendulum geometry
    x = L * np.sin(theta)
    y = L * np.cos(theta)
    line.set_data([0, x], [0, y])

    arrow.remove()
    arrow = ax_pend.arrow(0, 0, 0.3*np.sign(tau), 0,
                          width=0.03, color="red")

    # Update tau history
    tau_history.append(tau)
    if len(tau_history) > 200:
        tau_history.pop(0)

    tau_line.set_data(range(len(tau_history)), tau_history)

    return line, arrow, tau_line

tb_P.on_submit(update_P)
tb_I.on_submit(update_I)
tb_D.on_submit(update_D)

# Keyboard events
fig.canvas.mpl_connect("key_press_event", on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)

ani = FuncAnimation(fig, update, interval=20)
plt.tight_layout()
plt.show()
