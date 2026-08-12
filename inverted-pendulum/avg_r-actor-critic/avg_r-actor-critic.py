import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider
import os

# ─────────────────────────────────────────────
#  Physics
# ─────────────────────────────────────────────
g = 9.81; L = 1.0; m = 1.0; dt = 0.02; c = 0.001
TAU_MAX = 3.0   # continuous action range [-3, 3] as per course

# ─────────────────────────────────────────────
#  Tile Coding  (pure NumPy — no extra libraries)
# ─────────────────────────────────────────────
IHT_SIZE = 4096; NUM_TILINGS = 32; NUM_TILES = 8

class IHT:
    def __init__(self, size): self.size = size; self.table = {}
    def get(self, t, c):
        k = (t, *c)
        if k not in self.table: self.table[k] = len(self.table) % self.size
        return self.table[k]

_iht = IHT(IHT_SIZE)

def get_tiles(theta, tdot):
    sa = ((theta + np.pi) / (2*np.pi)) * NUM_TILES
    sv = ((np.clip(tdot, -2*np.pi, 2*np.pi) + 2*np.pi) / (4*np.pi)) * NUM_TILES
    tiles = np.empty(NUM_TILINGS, dtype=np.int32)
    for t in range(NUM_TILINGS):
        o = t / NUM_TILINGS
        tiles[t] = _iht.get(t, (int(np.floor(sa+o)) % NUM_TILES,
                                  int(np.floor(sv+o))))
    return tiles

# ─────────────────────────────────────────────
#  Average-Reward Actor-Critic — Gaussian policy
#
#  Exactly as derived in the course transcript:
#    μ(s)  = w_μ  · φ(s)          linear, tile-coded
#    σ(s)  = exp(w_σ · φ(s))      exp enforces σ > 0
#    τ     ~ N(μ, σ²), clipped to [-TAU_MAX, TAU_MAX]
#
#  Policy gradients:
#    ∂ log π / ∂w_μ  =  (τ - μ) / σ²          · φ(s)
#    ∂ log π / ∂w_σ  =  ((τ-μ)²/σ²  - 1)      · φ(s)
#      (chain rule through exp means d/d(w_σ·φ) cancels σ)
#
#  Critic + avg reward: differential TD(0), no γ
# ─────────────────────────────────────────────
class ActorCritic:
    def __init__(self, alpha_mu=0.25, alpha_sig=0.01,
                 alpha_critic=2.0, alpha_r=2**-6):
        self.alpha_mu  = alpha_mu  / NUM_TILINGS
        self.alpha_sig = alpha_sig / NUM_TILINGS
        self.alpha_c   = alpha_critic / NUM_TILINGS
        self.alpha_r   = alpha_r
        self.r_bar     = 0.0
        self.w_mu      = np.zeros(IHT_SIZE)   # mean weights
        self.w_sig     = np.zeros(IHT_SIZE)   # log-std weights (σ = exp(w_σ·φ))
        self.w_c       = np.zeros(IHT_SIZE)   # critic weights
        self.cumulative_reward = 0.0
        self.total_steps       = 0

    def mu_sigma(self, tiles):
        mu  = self.w_mu[tiles].sum()
        raw = np.clip(self.w_sig[tiles].sum(), -5.0, 2.0)
        sig = np.exp(raw)
        return mu, sig

    def select_action(self, theta, tdot):
        tiles = get_tiles(theta, tdot)
        mu, sig = self.mu_sigma(tiles)
        tau = np.clip(np.random.normal(mu, sig), -TAU_MAX, TAU_MAX)
        return tau, mu, sig, tiles

    def value(self, tiles):
        return float(self.w_c[tiles].sum())

    @staticmethod
    def reward(theta, theta_prev, tdot, tau):
        w       = np.cos(theta / 2) ** 4
        swing   = np.cos(theta) - np.cos(theta_prev)
        balance = 1.0 - (theta**2 + 0.1*tdot**2 + 0.001*tau**2)
        return (1 - w) * swing + w * balance

    def update(self, tiles, tau, mu, sig, reward, tiles_next):
        delta = np.clip(
            reward - self.r_bar + self.value(tiles_next) - self.value(tiles),
            -5, 5)
        self.r_bar      += self.alpha_r * delta
        self.w_c[tiles] += self.alpha_c * delta
        grad_mu  = np.clip((tau - mu) / sig**2,           -5, 5)
        grad_sig = np.clip((tau - mu)**2 / sig**2 - 1.0,  -5, 5)
        self.w_mu[tiles]  += self.alpha_mu  * delta * grad_mu
        self.w_sig[tiles] += self.alpha_sig * delta * grad_sig
        self.cumulative_reward += reward
        self.total_steps       += 1

    def save(self, path="ac_policy.npz"):
        np.savez(path, w_mu=self.w_mu, w_sig=self.w_sig, w_c=self.w_c,
                 alpha_mu=[self.alpha_mu * NUM_TILINGS],
                 alpha_sig=[self.alpha_sig * NUM_TILINGS],
                 alpha_r=[self.alpha_r], r_bar=[self.r_bar])
        print(f"[AC] Saved → {path}")

    def load(self, path="ac_policy.npz"):
        if not os.path.exists(path): print(f"[AC] Not found: {path}"); return False
        d = np.load(path)
        self.w_mu = d["w_mu"]; self.w_sig = d["w_sig"]; self.w_c = d["w_c"]
        if "r_bar" in d: self.r_bar = float(d["r_bar"][0])
        print(f"[AC] Loaded ← {path}"); return True

# ─────────────────────────────────────────────
#  Sim state
# ─────────────────────────────────────────────
theta_state = np.pi; theta_dot_state = 0.0; tau = 0.0
mode = "manual"; keys_down = {"left": False, "right": False}
P_gain, I_gain, D_gain = 20.0, 5.0, 2.0
integral_pid = 0.0; prev_error_pid = 0.0
agent = ActorCritic(); ac_training = True

def on_key_press(e):
    if e.key == "left":  keys_down["left"]  = True
    if e.key == "right": keys_down["right"] = True
def on_key_release(e):
    if e.key == "left":  keys_down["left"]  = False
    if e.key == "right": keys_down["right"] = False

def pid_control(theta):
    global integral_pid, prev_error_pid
    err = -theta; integral_pid += err * dt
    d = (err - prev_error_pid) / dt; prev_error_pid = err
    return P_gain*err + I_gain*integral_pid + D_gain*d

def step():
    global theta_state, theta_dot_state, tau
    if mode == "manual":
        tau = -1.5 if keys_down["left"] else (1.5 if keys_down["right"] else 0.0)
    elif mode == "pid":
        tau = pid_control(theta_state)
    elif mode == "ac":
        tp = theta_state
        tau, mu, sig, tiles = agent.select_action(theta_state, theta_dot_state)
        tdd = (g/L)*np.sin(theta_state) + tau/(m*L*L) - c*theta_dot_state
        theta_dot_state = np.clip(theta_dot_state + tdd*dt, -15, 15)
        theta_state = (theta_state + theta_dot_state*dt + np.pi) % (2*np.pi) - np.pi
        if ac_training:
            tiles_next = get_tiles(theta_state, theta_dot_state)
            agent.update(tiles, tau, mu, sig,
                         agent.reward(theta_state, tp, theta_dot_state, tau),
                         tiles_next)
        return
    tdd = (g/L)*np.sin(theta_state) + tau/(m*L*L) - c*theta_dot_state
    theta_dot_state = np.clip(theta_dot_state + tdd*dt, -15, 15)
    theta_state = (theta_state + theta_dot_state*dt + np.pi) % (2*np.pi) - np.pi


# ═════════════════════════════════════════════
#  LAYOUT  —  1920 × 1080
#  2×2 grid of square plots + controls panel
#
#  Square geometry:
#    QH = 0.400,  QW = QH × (10.8/19.2) = 0.225
#  Rows: B0=0.07, B1=B0+QH+GAP=0.495
#  Cols: L0=0.13, L1=L0+QW+GAP=0.38
#  Title at y=0.955  (top of plots = 0.895 ✓)
#  Controls: 0.70 → 0.99
# ═════════════════════════════════════════════
BG    = "#f8f9fa"; PANEL = "#ffffff"; CTRL = "#f0f2f5"
GREY  = "#6c757d"; BLK   = "#212529"
BLUE  = "#1565C0"; RED   = "#C62828"; GRN  = "#2E7D32"; PRP = "#6A1B9A"

fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)

QW=0.225; QH=0.400; GAP=0.025
L0=0.13; L1=L0+QW+GAP
BOT=0.07; TOP_ROW=BOT+QH+GAP

fig.text(0.38, 0.955,
         "Inverted Pendulum  ·  Manual / PID / Gaussian Actor-Critic",
         ha="center", va="center", fontsize=13, fontweight="bold", color=BLK)

def make_plot_ax(left, bottom, title, tc):
    ax = fig.add_axes([left, bottom, QW, QH], facecolor=PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#dee2e6")
    ax.set_title(title, color=tc, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors=GREY, labelsize=8)
    return ax

# ── TOP-LEFT: Pendulum ────────────────────────
ax_pend = make_plot_ax(L0, TOP_ROW, "Inverted Pendulum", BLK)
ax_pend.set_xlim(-1.3,1.3); ax_pend.set_ylim(-1.3,1.3); ax_pend.set_aspect("equal")
ax_pend.plot(0,0,"o",color=RED,ms=8,zorder=5)
ax_pend.axhline(0,color="#dee2e6",lw=1,ls="--")
mode_text = ax_pend.text(-1.22, 1.15, "", fontsize=10, color=BLUE,
                          fontweight="bold", clip_on=False)
info_text = ax_pend.text(-1.22, 1.04, "", fontsize=7, color=GREY, clip_on=False)
line, = ax_pend.plot([],[],lw=4,color=BLUE,solid_capstyle="round")
bob,  = ax_pend.plot([],[],"o",ms=13,color="#F9A825",zorder=6)
arrow = ax_pend.arrow(0,0,0,0,color=RED,width=0.0)

# ── TOP-RIGHT: Angle θ ────────────────────────
ax_theta = make_plot_ax(L1, TOP_ROW, "Angle  θ", BLUE)
ax_theta.set_xlim(0,300); ax_theta.set_ylim(-np.pi,np.pi)
ax_theta.set_ylabel("θ (rad)", color=BLUE, fontsize=8, labelpad=3)
ax_theta.tick_params(axis="y",colors=BLUE,labelsize=7)
ax_theta.tick_params(axis="x",colors=GREY,labelsize=7)
ax_theta.axhline(0,color="#dee2e6",lw=0.8)
theta_line, = ax_theta.plot([],[],lw=1.8,color=BLUE)

# ── BOTTOM-LEFT: Torque τ ─────────────────────
ax_tau = make_plot_ax(L0, BOT, "Torque  τ", RED)
ax_tau.set_xlim(0,300); ax_tau.set_ylim(-TAU_MAX,TAU_MAX)
ax_tau.set_ylabel("τ (Nm)", color=RED, fontsize=8, labelpad=3)
ax_tau.tick_params(axis="y",colors=RED,labelsize=7)
ax_tau.tick_params(axis="x",colors=GREY,labelsize=7)
ax_tau.axhline(0,color="#dee2e6",lw=0.8)
tau_line, = ax_tau.plot([],[],lw=1.8,color=RED)

# ── BOTTOM-RIGHT: Critic V̂ / r̄ ───────────────
ax_val = make_plot_ax(L1, BOT, "Critic  V̂  /  r̄", PRP)
ax_val.set_xlim(0,300); ax_val.set_ylim(-3.0,2.0)
ax_val.set_ylabel("V̂  /  r̄", color=PRP, fontsize=8, labelpad=3)
ax_val.tick_params(axis="y",colors=PRP,labelsize=7)
ax_val.tick_params(axis="x",colors=GREY,labelsize=7)
ax_val.axhline(0,color="#dee2e6",lw=0.8)
val_line,  = ax_val.plot([],[],lw=1.8,color=PRP,label="V̂")
rbar_line, = ax_val.plot([],[],lw=1.8,color=GRN,label="r̄",ls="--")
ax_val.legend(fontsize=7,framealpha=0.9,edgecolor="#dee2e6",loc="upper left")

HIST=300
tau_history=[]; theta_history=[]; val_history=[]; rbar_history=[]


# ═════════════════════════════════════════════
#  CONTROLS PANEL  — right 29%
# ═════════════════════════════════════════════
PL=0.700; PW=0.290
ctrl_bg = fig.add_axes([PL,0.02,PW,0.96], facecolor=CTRL, zorder=0)
ctrl_bg.set_xticks([]); ctrl_bg.set_yticks([])
for sp in ctrl_bg.spines.values(): sp.set_edgecolor("#ced4da"); sp.set_linewidth(1.2)

WL=PL+0.012; WW=PW-0.024

def make_btn(rect, label, fc=PANEL, tc=BLK):
    ax = fig.add_axes(rect, facecolor=fc)
    for sp in ax.spines.values(): sp.set_edgecolor("#ced4da")
    btn = Button(ax, label, color=fc, hovercolor="#dde3ff")
    btn.label.set_color(tc); btn.label.set_fontsize(10)
    return btn

def silent_ax(rect, fc=CTRL):
    ax = fig.add_axes(rect, facecolor=fc)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    return ax

# Header
ax_hdr = silent_ax([WL, 0.922, WW, 0.046])
ax_hdr.text(0.5,0.5,"Controls",ha="center",va="center",
            fontsize=14,fontweight="bold",color=BLK)
ax_hdiv = silent_ax([WL,0.916,WW,0.005])
ax_hdiv.axhline(0.5,color="#ced4da",lw=1.5)

# Mode buttons
BW = WW/3 - 0.003
btn_manual = make_btn([WL,           0.858, BW, 0.050], "Manual", tc=GRN)
btn_pid    = make_btn([WL+WW/3,      0.858, BW, 0.050], "PID",    tc=BLUE)
btn_ac     = make_btn([WL+2*WW/3,    0.858, BW, 0.050], "AC",     tc=RED)
ax_div = silent_ax([WL,0.848,WW,0.006])
ax_div.axhline(0.5,color="#ced4da",lw=1)

# ── PID panel ─────────────────────────────────
pid_widgets = []
ax_ptitle = silent_ax([WL,0.818,WW,0.026])
ax_ptitle.text(0.02,0.5,"PID Gains",fontsize=10,fontweight="bold",color=BLUE,va="center")
pid_widgets.append(ax_ptitle)
TW=WW/3-0.004
for col,txt in enumerate(["P","I","D"]):
    lax=silent_ax([WL+col*(WW/3),0.796,TW,0.020])
    lax.text(0.5,0.5,txt,ha="center",va="center",fontsize=9,color=GREY)
    pid_widgets.append(lax)
ax_P=fig.add_axes([WL,          0.752,TW,0.040])
ax_I=fig.add_axes([WL+WW/3,     0.752,TW,0.040])
ax_D=fig.add_axes([WL+2*WW/3,   0.752,TW,0.040])
tb_P=TextBox(ax_P,"",initial=str(int(P_gain)))
tb_I=TextBox(ax_I,"",initial=str(int(I_gain)))
tb_D=TextBox(ax_D,"",initial=str(int(D_gain)))
for tb in [tb_P,tb_I,tb_D]:
    tb.text_disp.set_ha("center"); tb.text_disp.set_va("center")
    tb.text_disp.set_position((0.5,0.5)); tb.text_disp.set_fontsize(10)
    tb.text_disp.set_color(BLK)
pid_widgets+=[ax_P,ax_I,ax_D]
def update_P(t):
    global P_gain
    try: P_gain=float(t)
    except: pass
def update_I(t):
    global I_gain
    try: I_gain=float(t)
    except: pass
def update_D(t):
    global D_gain
    try: D_gain=float(t)
    except: pass
tb_P.on_submit(update_P); tb_I.on_submit(update_I); tb_D.on_submit(update_D)

# ── AC panel ──────────────────────────────────
ac_widgets = []
ax_actitle=silent_ax([WL,0.820,WW,0.026])
ax_actitle.text(0.02,0.5,"Gaussian Actor-Critic",fontsize=10,
                fontweight="bold",color=RED,va="center")
ac_widgets.append(ax_actitle)

ax_αmu  = fig.add_axes([WL,0.786,WW,0.026], facecolor=PANEL)
ax_αsig = fig.add_axes([WL,0.754,WW,0.026], facecolor=PANEL)
ax_αc   = fig.add_axes([WL,0.722,WW,0.026], facecolor=PANEL)
ax_αr   = fig.add_axes([WL,0.690,WW,0.026], facecolor=PANEL)
sl_αmu  = Slider(ax_αmu,  "α μ",    0.01, 0.5,   valinit=0.25,    color=BLUE)
sl_αsig = Slider(ax_αsig, "α σ",    0.001,0.1,   valinit=0.01,    color=RED)
sl_αc   = Slider(ax_αc,   "α crit", 0.5,  5.0,   valinit=2.0,     color=PRP)
sl_αr   = Slider(ax_αr,   "α r̄",   0.001,0.1,   valinit=2**-6,   color=GRN)
for sl in [sl_αmu,sl_αsig,sl_αc,sl_αr]:
    sl.label.set_color(GREY); sl.label.set_fontsize(8)
    sl.valtext.set_color(BLK); sl.valtext.set_fontsize(8)
def upd_mu(v):  agent.alpha_mu  = v/NUM_TILINGS
def upd_sig(v): agent.alpha_sig = v/NUM_TILINGS
def upd_ac(v):  agent.alpha_c   = v/NUM_TILINGS
def upd_ar(v):  agent.alpha_r   = v
sl_αmu.on_changed(upd_mu); sl_αsig.on_changed(upd_sig)
sl_αc.on_changed(upd_ac);  sl_αr.on_changed(upd_ar)
ac_widgets+=[ax_αmu,ax_αsig,ax_αc,ax_αr]

BL=WW*0.60
btn_train=make_btn([WL,          0.650,BL,     0.034],"Learn: ON",tc=GRN)
btn_save =make_btn([WL+BL+0.004, 0.650,WW*0.17,0.034],"💾",        tc=BLUE)
btn_load =make_btn([WL+BL+WW*0.20,0.650,WW*0.17,0.034],"📂",       tc=BLUE)
ac_widgets+=[btn_train.ax,btn_save.ax,btn_load.ax]

# Stats box
ax_stats=fig.add_axes([WL,0.500,WW,0.142],facecolor=PANEL)
ax_stats.set_xticks([]); ax_stats.set_yticks([])
for sp in ax_stats.spines.values(): sp.set_edgecolor("#ced4da"); sp.set_visible(True)
stats_text=ax_stats.text(0.05,0.96,"",fontsize=9,color=GREY,
                          va="top",transform=ax_stats.transAxes,
                          family="monospace",linespacing=1.7)
ac_widgets.append(ax_stats)

# ── σ trend — small chart showing σ over time ─
sig_history=[]
ax_sig=fig.add_axes([WL,0.350,WW,0.130],facecolor=PANEL)
ax_sig.set_xticks([]); ax_sig.set_ylim(0,3)
ax_sig.set_ylabel("σ(s)", color=RED, fontsize=8, labelpad=2)
ax_sig.tick_params(axis="y",colors=RED,labelsize=7)
for sp in ax_sig.spines.values(): sp.set_edgecolor("#dee2e6")
ax_sig.set_title("Exploration σ", color=RED, fontsize=9, pad=4)
sig_line, = ax_sig.plot([],[],lw=1.5,color=RED)
ax_sig.axhline(0.1,color=GREY,lw=0.5,ls=":",alpha=0.5)
ac_widgets.append(ax_sig)

# ── show/hide ──────────────────────────────────
def set_vis(wlist,v):
    for ax in wlist: ax.set_visible(v)

set_vis(pid_widgets,False); set_vis(ac_widgets,False)

def highlight(active):
    cfg={"manual":(btn_manual,GRN,"#e8f5e9"),
         "pid":   (btn_pid,   BLUE,"#e3f2fd"),
         "ac":    (btn_ac,    RED, "#fce4ec")}
    for k,(btn,tc,hc) in cfg.items():
        if k==active: btn.ax.set_facecolor(hc); btn.label.set_fontweight("bold")
        else:         btn.ax.set_facecolor(PANEL); btn.label.set_fontweight("normal")

highlight("manual")

def set_manual(e):
    global mode,integral_pid,prev_error_pid
    mode="manual"; integral_pid=0.0; prev_error_pid=0.0
    set_vis(pid_widgets,False); set_vis(ac_widgets,False)
    highlight("manual"); fig.canvas.draw_idle()

def set_pid(e):
    global mode,integral_pid,prev_error_pid
    mode="pid"; integral_pid=0.0; prev_error_pid=0.0
    set_vis(pid_widgets,True); set_vis(ac_widgets,False)
    highlight("pid"); fig.canvas.draw_idle()

def set_ac(e):
    global mode
    mode="ac"
    set_vis(pid_widgets,False); set_vis(ac_widgets,True)
    highlight("ac"); fig.canvas.draw_idle()

def toggle_train(e):
    global ac_training
    ac_training = not ac_training
    btn_train.label.set_text("Learn: ON" if ac_training else "Exploit")
    btn_train.label.set_color(GRN if ac_training else GREY)
    fig.canvas.draw_idle()

def save_pol(e): agent.save("ac_policy.npz")
def load_pol(e): agent.load("ac_policy.npz")

btn_manual.on_clicked(set_manual); btn_pid.on_clicked(set_pid)
btn_ac.on_clicked(set_ac);         btn_train.on_clicked(toggle_train)
btn_save.on_clicked(save_pol);     btn_load.on_clicked(load_pol)


# ─────────────────────────────────────────────
#  Animation
# ─────────────────────────────────────────────
def update(frame):
    global arrow
    step()

    x=L*np.sin(theta_state); y=L*np.cos(theta_state)
    line.set_data([0,x],[0,y]); bob.set_data([x],[y])
    arrow.remove()
    alen=np.clip(tau/TAU_MAX,-1,1)*0.35
    arrow=ax_pend.arrow(0,0,alen,0,width=0.025,
                        head_width=0.06,head_length=0.05,
                        color=RED,alpha=0.8,zorder=4)

    mode_text.set_text(f"Mode: {mode.upper()}")
    tiles=get_tiles(theta_state,theta_dot_state)
    v_hat=agent.value(tiles)
    mu,sig=agent.mu_sigma(tiles)
    info_text.set_text(f"θ={theta_state:.2f}  θ̇={theta_dot_state:.2f}  "
                       f"τ={tau:.2f}  μ={mu:.2f}  σ={sig:.2f}")

    tau_history.append(np.clip(tau,-TAU_MAX,TAU_MAX))
    theta_history.append(theta_state)
    val_history.append(v_hat)
    rbar_history.append(agent.r_bar)
    sig_history.append(sig)
    for h in [tau_history,theta_history,val_history,rbar_history,sig_history]:
        if len(h)>HIST: h.pop(0)

    xs=list(range(len(tau_history)))
    theta_line.set_data(xs,theta_history)
    tau_line.set_data(xs,tau_history)
    val_line.set_data(xs,val_history)
    rbar_line.set_data(xs,rbar_history)
    sig_line.set_data(xs,sig_history)
    for ax in [ax_theta,ax_tau,ax_val,ax_sig]:
        ax.set_xlim(0,max(HIST,len(tau_history)))

    if mode=="ac":
        stats_text.set_text(
            f"μ(s)   = {mu:+.3f}\n"
            f"σ(s)   =  {sig:.4f}\n"
            f"τ sampled = {tau:+.3f}\n"
            f"\nr̄     = {agent.r_bar:.4f}\n"
            f"steps  = {agent.total_steps}\n"
            f"Σr     = {agent.cumulative_reward:.1f}"
        )
    else:
        stats_text.set_text("")

    return line,bob,arrow,theta_line,tau_line,val_line,rbar_line,sig_line

fig.canvas.mpl_connect("key_press_event",   on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)
ani = FuncAnimation(fig, update, interval=20, blit=False)
plt.show()