"""
Inverted Pendulum — Average Reward Actor-Critic with Gaussian Policy
=====================================================================
Based exactly on the course formulation (Sutton & Barto / Coursera).

State:    theta (angle), theta_dot (angular velocity)
Action:   tau  (torque), continuous, sampled from N(mu, sigma^2)
Features: tile coding — maps (theta, theta_dot) to sparse binary vector

Actor:
    mu(s)    = w_mu  . phi(s)          linear in tile features
    sigma(s) = exp(w_sigma . phi(s))   exp keeps sigma > 0

Critic:
    V(s)     = w_v . phi(s)            linear in tile features

Update every step (differential TD — no gamma, tracks average reward):
    delta   = r - r_bar + V(s') - V(s)     TD error
    r_bar  += alpha_r * delta              average reward update
    w_v    += alpha_c * delta * phi(s)     critic update
    w_mu   += alpha_a * delta * grad_mu    actor mean update
    w_sigma+= alpha_a * delta * grad_sig   actor std update

Policy gradients (derived in course):
    grad_mu  = (tau - mu) / sigma^2  *  TAU_MAX*(1-tanh^2(raw_mu))
    grad_sig = (tau - mu)^2 / sigma^2  - 1

Why tanh on mu?
    mu = TAU_MAX * tanh(w_mu . phi)
    Keeps mu bounded in [-TAU_MAX, TAU_MAX] so weights can't diverge.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import os

# ──────────────────────────────────────────────
#  PHYSICS
# ──────────────────────────────────────────────
g       = 9.81   # gravity (m/s^2)
L       = 1.0    # pendulum length (m)
m       = 1.0    # bob mass (kg)
dt      = 0.02   # timestep (s)
c       = 0.001  # damping
TAU_MAX = 3.0    # max torque (Nm) — course uses [-3, 3]


# ──────────────────────────────────────────────
#  TILE CODING
#
#  Divides the state space into overlapping grids.
#  Each grid is called a "tiling". Each cell in a
#  tiling is a "tile". For any state, exactly one
#  tile fires in each tiling.
#
#  With 32 tilings of 8x8 tiles:
#    - 32 tiles active at once (one per tiling)
#    - 4096 total possible tiles (IHT_SIZE)
#    - Each weight only updates when its tile is active
#    - No weight sharing between distant states -> stable!
#
#  Angle is WRAPPED — so theta=-pi and theta=+pi
#  are treated as the same state (they are physically).
# ──────────────────────────────────────────────
IHT_SIZE    = 4096   # total number of tiles
NUM_TILINGS = 32     # number of overlapping grids
NUM_TILES   = 8      # tiles per dimension per tiling

class IHT:
    """
    Index Hash Table — maps (tiling_id, tile_coords) to a unique index.
    Collision-free up to IHT_SIZE entries.
    """
    def __init__(self, size):
        self.size  = size
        self.table = {}   # maps key -> index

    def get(self, tiling, coords):
        key = (tiling, *coords)
        if key not in self.table:
            self.table[key] = len(self.table) % self.size
        return self.table[key]

_iht = IHT(IHT_SIZE)

def get_tiles(theta, theta_dot):
    """
    Returns array of NUM_TILINGS active tile indices for state (theta, theta_dot).

    How it works:
    1. Scale inputs to [0, NUM_TILES] range
    2. For each tiling, offset slightly (by t/NUM_TILINGS of a tile width)
    3. Take the floor to get which tile we're in
    4. Look up (or create) a unique index for this (tiling, tile) pair
    """
    # Scale angle to [0, NUM_TILES] — wrapping means -pi == +pi
    scale_angle = ((theta + np.pi) / (2 * np.pi)) * NUM_TILES

    # Scale velocity to [0, NUM_TILES]
    vel_clipped = np.clip(theta_dot, -2*np.pi, 2*np.pi)
    scale_vel   = ((vel_clipped + 2*np.pi) / (4*np.pi)) * NUM_TILES

    tiles = np.empty(NUM_TILINGS, dtype=np.int32)
    for t in range(NUM_TILINGS):
        offset = t / NUM_TILINGS   # each tiling offset slightly

        # angle wraps: modulo NUM_TILES so -pi and +pi map to same tile
        tile_angle = int(np.floor(scale_angle + offset)) % NUM_TILES
        tile_vel   = int(np.floor(scale_vel   + offset))

        tiles[t] = _iht.get(t, (tile_angle, tile_vel))

    return tiles


# ──────────────────────────────────────────────
#  ACTOR-CRITIC AGENT
# ──────────────────────────────────────────────
class ActorCritic:
    def __init__(self,
                 alpha_mu    = 0.25,    # actor step size for mean
                 alpha_sigma = 0.01,    # actor step size for std  (smaller = slower sigma changes)
                 alpha_critic= 2.0,     # critic step size
                 alpha_r     = 2**-6):  # average reward step size

        # Divide by NUM_TILINGS — standard practice for tile coding
        # (because NUM_TILINGS tiles fire at once, so the effective
        #  learning rate is already multiplied by NUM_TILINGS)
        self.aa  = alpha_mu     / NUM_TILINGS
        self.as_ = alpha_sigma  / NUM_TILINGS
        self.ac  = alpha_critic / NUM_TILINGS
        self.ar  = alpha_r

        # Average reward estimate (replaces gamma*V(s') in standard TD)
        self.r_bar = 0.0

        # Weight vectors — one weight per tile
        self.w_mu    = np.zeros(IHT_SIZE)   # actor: mean weights
        self.w_sigma = np.zeros(IHT_SIZE)   # actor: log-std weights
        self.w_v     = np.zeros(IHT_SIZE)   # critic: value weights

        # Tracking
        self.total_steps       = 0
        self.cumulative_reward = 0.0

    # ── compute mu and sigma from current state ───
    def mu_sigma(self, tiles):
        """
        mu    = TAU_MAX * tanh(w_mu . phi)     bounded in [-TAU_MAX, TAU_MAX]
        sigma = exp(w_sigma . phi)             always positive
        """
        raw_mu = self.w_mu[tiles].sum()
        mu     = TAU_MAX * np.tanh(raw_mu)

        raw_sigma = np.clip(self.w_sigma[tiles].sum(), -5.0, 2.0)
        sigma     = np.exp(raw_sigma)

        return mu, sigma, raw_mu, raw_sigma

    # ── critic value ──────────────────────────────
    def value(self, tiles):
        return float(self.w_v[tiles].sum())

    # ── select action ─────────────────────────────
    def select_action(self, theta, theta_dot, deterministic=False):
        """
        Get tiles for current state, compute mu/sigma, sample torque.
        deterministic=True uses mu directly (no exploration noise).
        """
        tiles         = get_tiles(theta, theta_dot)
        mu, sigma, raw_mu, _ = self.mu_sigma(tiles)

        if deterministic:
            tau = float(mu)
        else:
            tau = float(np.clip(np.random.normal(mu, sigma), -TAU_MAX, TAU_MAX))

        return tau, mu, sigma, tiles

    # ── reward ────────────────────────────────────
    @staticmethod
    def reward(theta, theta_prev, theta_dot, tau):
        """
        Two-phase reward that works for both swing-up and balance:

        w = cos(theta/2)^4
            0 when hanging (theta=pi), 1 when upright (theta=0)
            smoothly blends between swing-up and balance rewards

        swing   = cos(theta) - cos(theta_prev)
            rewards gaining height — teaches the swing-up strategy

        balance = 1 - (theta^2 + 0.1*theta_dot^2 + 0.001*tau^2)
            rewards being upright, slow, and using small torque
        """
        w       = np.cos(theta / 2) ** 4
        swing   = np.cos(theta) - np.cos(theta_prev)
        balance = 1.0 - (3.0*theta**2 + 0.1*theta_dot**2 + 0.001*tau**2)
        return float((1 - w) * swing + w * balance)

    # ── main update ───────────────────────────────
    def update(self, tiles, tau, reward, tiles_next):
        """
        One step of average-reward actor-critic.

        tiles       — active tiles for current state s
        tau         — action taken
        reward      — reward received
        tiles_next  — active tiles for next state s'
        """

        # ── TD error (differential version, no gamma) ──
        #
        # In average reward AC we don't discount future rewards.
        # Instead we track r_bar (the average reward we've been getting).
        # delta measures: was this step better or worse than average?
        #
        delta = reward - self.r_bar + self.value(tiles_next) - self.value(tiles)

        # ── Update average reward estimate ─────────────
        self.r_bar += self.ar * delta

        # ── Update critic ───────────────────────────────
        # Move V(s) toward r + V(s') - r_bar
        # Only the tiles that were active get updated
        self.w_v[tiles] += self.ac * delta

        # ── Update actor ────────────────────────────────
        # Gradient of log pi w.r.t. weights:
        #
        #   For mu:    d log pi / d raw_mu
        #              = (tau - mu) / sigma^2  *  d(tanh)/d(raw_mu)
        #              = (tau - mu) / sigma^2  *  TAU_MAX*(1 - tanh^2(raw_mu))
        #
        #   For sigma: d log pi / d raw_sigma
        #              = (tau - mu)^2 / sigma^2  - 1
        #              (chain rule through exp cancels sigma)
        #
        # Multiply by delta: reinforce if surprise was positive

        mu, sigma, raw_mu, _ = self.mu_sigma(tiles)

        d_tanh    = TAU_MAX * (1.0 - np.tanh(raw_mu)**2)
        grad_mu   = np.clip(((tau - mu) / sigma**2) * d_tanh, -5, 5)
        grad_sigma= np.clip((tau - mu)**2 / sigma**2 - 1.0,   -5, 5)

        self.w_mu[tiles]    += self.aa  * delta * grad_mu
        self.w_sigma[tiles] += self.as_ * delta * grad_sigma

        self.total_steps       += 1
        self.cumulative_reward += reward

    def save(self, path="policy.npz"):
        np.savez(path,
                 w_mu=self.w_mu, w_sigma=self.w_sigma, w_v=self.w_v,
                 r_bar=np.array([self.r_bar]))
        print(f"Saved → {path}")

    def load(self, path="policy.npz"):
        if not os.path.exists(path):
            print(f"Not found: {path}"); return
        d = np.load(path)
        self.w_mu    = d["w_mu"]
        self.w_sigma = d["w_sigma"]
        self.w_v     = d["w_v"]
        self.r_bar   = float(d["r_bar"][0])
        print(f"Loaded ← {path}")


# ──────────────────────────────────────────────
#  SIMULATION
# ──────────────────────────────────────────────
theta     = np.pi    # start hanging down
theta_dot = 0.0
tau       = 0.0
mode      = "manual"
learning  = True
fast_fwd  = False
keys      = {"left": False, "right": False}
agent     = ActorCritic()

def on_key_press(e):
    if e.key == "left":  keys["left"]  = True
    if e.key == "right": keys["right"] = True
def on_key_release(e):
    if e.key == "left":  keys["left"]  = False
    if e.key == "right": keys["right"] = False

def physics(torque):
    global theta, theta_dot
    tdd        = (g/L)*np.sin(theta) + torque/(m*L**2) - c*theta_dot
    theta_dot  = np.clip(theta_dot + tdd*dt, -15, 15)
    theta      = (theta + theta_dot*dt + np.pi) % (2*np.pi) - np.pi

def sim_step():
    global tau
    if mode == "manual":
        tau = -1.5 if keys["left"] else (1.5 if keys["right"] else 0.0)
        physics(tau)

    elif mode == "ac":
        # Deterministic when not learning (exploit: use mu, no noise)
        tau, mu, sigma, tiles = agent.select_action(
            theta, theta_dot, deterministic=not learning)

        theta_prev = theta
        physics(tau)

        if learning:
            tiles_next = get_tiles(theta, theta_dot)
            r = agent.reward(theta, theta_prev, theta_dot, tau)
            agent.update(tiles, tau, r, tiles_next)


# ──────────────────────────────────────────────
#  FIGURE
# ──────────────────────────────────────────────
BG    = "#f8f9fa"; PANEL = "#ffffff"; CTRL = "#f0f2f5"
GREY  = "#6c757d"; BLK   = "#212529"
BLUE  = "#1565C0"; RED   = "#C62828"; GRN  = "#2E7D32"; PRP = "#6A1B9A"

fig = plt.figure(figsize=(16, 9), facecolor=BG)
fig.suptitle("Inverted Pendulum  ·  Avg-Reward Actor-Critic  ·  Gaussian Policy  ·  Tile Coding",
             fontsize=12, fontweight="bold", color=BLK, y=0.97)

# ── Pendulum ──────────────────────────────────
ax_p = fig.add_axes([0.03, 0.08, 0.33, 0.83], facecolor=PANEL)
for sp in ax_p.spines.values(): sp.set_edgecolor("#dee2e6")
ax_p.set_xlim(-1.3,1.3); ax_p.set_ylim(-1.3,1.3); ax_p.set_aspect("equal")
ax_p.set_title("Pendulum", color=BLK, fontsize=11, fontweight="bold", pad=6)
ax_p.tick_params(colors=GREY, labelsize=8)
ax_p.plot(0,0,"o",color=RED,ms=8,zorder=5)
ax_p.axhline(0,color="#dee2e6",lw=1,ls="--")
mode_lbl = ax_p.text(-1.2,1.15,"",fontsize=10,color=BLUE,fontweight="bold",clip_on=False)
info_lbl = ax_p.text(-1.2,1.04,"",fontsize=7.5,color=GREY,clip_on=False)
rod,  = ax_p.plot([],[],lw=4,color=BLUE,solid_capstyle="round")
bob,  = ax_p.plot([],[],"o",ms=14,color="#F9A825",zorder=6)
arrow = ax_p.arrow(0,0,0,0,color=RED,width=0.0)

# ── Signal charts ─────────────────────────────
def make_chart(left, bot, h, title, tc, ylim, ylabel):
    ax = fig.add_axes([left, bot, 0.27, h], facecolor=PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#dee2e6")
    ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlim(0,300); ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=tc, fontsize=8, labelpad=3)
    ax.tick_params(axis="y",colors=tc,labelsize=7)
    ax.tick_params(axis="x",colors=GREY,labelsize=7)
    ax.axhline(0,color="#dee2e6",lw=0.8)
    return ax

ax_th = make_chart(0.39,0.60,0.31,"Angle  θ",      BLUE,(-np.pi,np.pi),"θ (rad)")
ax_tc = make_chart(0.39,0.25,0.28,"Torque  τ",     RED, (-3.2,3.2),    "τ (Nm)")
ax_rw = make_chart(0.39,0.08,0.13,"Reward",        GRN, (-1.5,1.5),    "r")

th_line, = ax_th.plot([],[],lw=1.8,color=BLUE)
tc_line, = ax_tc.plot([],[],lw=1.8,color=RED)
rw_line, = ax_rw.plot([],[],lw=1.5,color=GRN)

HIST=300
th_hist=[]; tc_hist=[]; rw_hist=[]

# ── Controls ──────────────────────────────────
PL=0.71; PW=0.27
fig.add_axes([PL,0.02,PW,0.95],facecolor=CTRL).set_axis_off()
WL=PL+0.012; WW=PW-0.024

def btn(rect,label,tc=BLK):
    ax=fig.add_axes(rect,facecolor=PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#ced4da")
    b=Button(ax,label,color=PANEL,hovercolor="#e8eaf6")
    b.label.set_color(tc); b.label.set_fontsize(10)
    return b

def lbl(rect,text,fs=10,fw="normal",tc=BLK,align="center"):
    ax=fig.add_axes(rect,facecolor=CTRL)
    ax.set_axis_off()
    ax.text(0.5 if align=="center" else 0.04,0.5,text,
            ha=align,va="center",fontsize=fs,fontweight=fw,color=tc)

lbl([WL,0.920,WW,0.048],"Controls",fs=14,fw="bold")

BW=WW/2-0.005
btn_manual = btn([WL,       0.860,BW,0.050],"Manual",tc=GRN)
btn_ac     = btn([WL+WW/2,  0.860,BW,0.050],"Actor-Critic",tc=RED)

# separator
ax_sep=fig.add_axes([WL,0.852,WW,0.005],facecolor=CTRL); ax_sep.set_axis_off()
ax_sep.axhline(0.5,color="#ced4da",lw=1.2)

btn_learn = btn([WL,0.798,WW,0.046],"● Learn: ON",tc=GRN)
btn_ff    = btn([WL,0.744,WW,0.046],"⚡ Fast Forward (50x)",tc=GREY)

BH=WW/2-0.005
btn_save  = btn([WL,       0.690,BH,0.046],"💾 Save",tc=BLUE)
btn_load  = btn([WL+WW/2,  0.690,BH,0.046],"📂 Load",tc=BLUE)

# stats box
ax_st=fig.add_axes([WL,0.430,WW,0.252],facecolor=PANEL)
ax_st.set_xticks([]); ax_st.set_yticks([])
for sp in ax_st.spines.values(): sp.set_edgecolor("#ced4da"); sp.set_visible(True)
stats=ax_st.text(0.06,0.96,"",fontsize=9.5,color=GREY,va="top",
                  transform=ax_st.transAxes,family="monospace",linespacing=1.9)

# sigma chart
sig_hist=[]
ax_sig=fig.add_axes([WL,0.270,WW,0.148],facecolor=PANEL)
ax_sig.set_xticks([]); ax_sig.set_ylim(0,3.5)
ax_sig.set_ylabel("σ",color=RED,fontsize=9); ax_sig.tick_params(axis="y",colors=RED,labelsize=8)
for sp in ax_sig.spines.values(): sp.set_edgecolor("#dee2e6")
ax_sig.set_title("σ  (exploration — should shrink as policy improves)",color=RED,fontsize=8,pad=4)
sig_ln,=ax_sig.plot([],[],lw=1.5,color=RED)
ax_sig.axhline(0.2,color=GREY,lw=0.5,ls=":",alpha=0.5)

# r_bar chart
rbar_hist=[]
ax_rb=fig.add_axes([WL,0.100,WW,0.148],facecolor=PANEL)
ax_rb.set_xticks([]); ax_rb.set_ylim(-1.5,1.5)
ax_rb.set_ylabel("r̄",color=PRP,fontsize=9); ax_rb.tick_params(axis="y",colors=PRP,labelsize=8)
for sp in ax_rb.spines.values(): sp.set_edgecolor("#dee2e6")
ax_rb.set_title("r̄  (avg reward — should rise toward +1)",color=PRP,fontsize=8,pad=4)
rb_ln,=ax_rb.plot([],[],lw=1.5,color=PRP)
ax_rb.axhline(0,color="#dee2e6",lw=0.8)
ax_rb.axhline(1,color=PRP,lw=0.5,ls=":",alpha=0.4)

# ── button callbacks ───────────────────────────
def set_manual(e):
    global mode
    mode="manual"
    btn_manual.ax.set_facecolor("#e8f5e9"); btn_manual.label.set_fontweight("bold")
    btn_ac.ax.set_facecolor(PANEL);         btn_ac.label.set_fontweight("normal")
    fig.canvas.draw_idle()

def set_ac(e):
    global mode
    mode="ac"
    btn_ac.ax.set_facecolor("#fce4ec"); btn_ac.label.set_fontweight("bold")
    btn_manual.ax.set_facecolor(PANEL); btn_manual.label.set_fontweight("normal")
    fig.canvas.draw_idle()

def toggle_learn(e):
    global learning
    learning=not learning
    btn_learn.label.set_text("● Learn: ON" if learning else "○ Exploit only")
    btn_learn.label.set_color(GRN if learning else GREY)
    fig.canvas.draw_idle()

def toggle_ff(e):
    global fast_fwd
    fast_fwd=not fast_fwd
    btn_ff.label.set_text("⚡ Fast Forward  ON" if fast_fwd else "⚡ Fast Forward (50x)")
    btn_ff.label.set_color(RED if fast_fwd else GREY)
    btn_ff.ax.set_facecolor("#fce4ec" if fast_fwd else PANEL)
    fig.canvas.draw_idle()

btn_manual.on_clicked(set_manual)
btn_ac.on_clicked(set_ac)
btn_learn.on_clicked(toggle_learn)
btn_ff.on_clicked(toggle_ff)
btn_save.on_clicked(lambda e: agent.save())
btn_load.on_clicked(lambda e: agent.load())

set_manual(None)   # highlight manual at start


# ──────────────────────────────────────────────
#  ANIMATION
# ──────────────────────────────────────────────
def update(frame):
    global arrow

    n = 50 if (fast_fwd and mode=="ac" and learning) else 1
    for _ in range(n):
        sim_step()

    # draw pendulum
    x=L*np.sin(theta); y=L*np.cos(theta)
    rod.set_data([0,x],[0,y]); bob.set_data([x],[y])
    arrow.remove()
    alen=np.clip(tau/TAU_MAX,-1,1)*0.3
    arrow=ax_p.arrow(0,0,alen,0,width=0.022,head_width=0.055,
                     head_length=0.04,color=RED,alpha=0.8,zorder=4)

    # info text
    mode_lbl.set_text(f"Mode: {mode.upper()}")
    tiles=get_tiles(theta,theta_dot)
    mu,sigma,_,_=agent.mu_sigma(tiles)
    r_now=agent.reward(theta,theta,theta_dot,tau)   # approx current reward
    info_lbl.set_text(f"θ={theta:.2f}  θ̇={theta_dot:.2f}  "
                      f"τ={tau:.2f}  μ={mu:.2f}  σ={sigma:.2f}")

    # histories
    th_hist.append(theta);  tc_hist.append(np.clip(tau,-3,3))
    rw_hist.append(r_now);  sig_hist.append(sigma)
    rbar_hist.append(agent.r_bar)
    for h in [th_hist,tc_hist,rw_hist,sig_hist,rbar_hist]:
        if len(h)>HIST: h.pop(0)

    xs=list(range(len(th_hist)))
    th_line.set_data(xs,th_hist)
    tc_line.set_data(xs,tc_hist)
    rw_line.set_data(xs,rw_hist)
    sig_ln.set_data(xs,sig_hist)
    rb_ln.set_data(xs,rbar_hist)
    for ax in [ax_th,ax_tc,ax_rw,ax_sig,ax_rb]:
        ax.set_xlim(0,max(HIST,len(th_hist)))

    # stats
    if mode=="ac":
        v_now=agent.value(tiles)
        upright="✓ UPRIGHT" if abs(theta)<0.2 else ""
        spd="50x ⚡" if fast_fwd else "1x"
        stats.set_text(
            f"V(s)   = {v_now:+.3f}\n"
            f"μ(s)   = {mu:+.3f}\n"
            f"σ(s)   =  {sigma:.4f}\n"
            f"r̄     = {agent.r_bar:.4f}\n"
            f"\nsteps  = {agent.total_steps:,}\n"
            f"speed  = {spd}\n"
            f"{upright}"
        )
    else:
        stats.set_text("")

    return rod,bob,arrow,th_line,tc_line,rw_line,sig_ln,rb_ln

fig.canvas.mpl_connect("key_press_event",   on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)
ani = FuncAnimation(fig, update, interval=20, blit=False)
plt.show()