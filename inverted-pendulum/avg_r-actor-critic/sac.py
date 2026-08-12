import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import os

# ─────────────────────────────────────────────
#  Physics
# ─────────────────────────────────────────────
g = 9.81; L = 1.0; m = 1.0; dt = 0.02; c = 0.001
TAU_MAX = 3.0

# ─────────────────────────────────────────────
#  Neural network — 2-hidden-layer MLP
#  Pure NumPy, no frameworks needed
# ─────────────────────────────────────────────
def relu(x):  return np.maximum(0, x)
def drelu(x): return (x > 0).astype(np.float64)

class MLP:
    """Simple 2-hidden-layer MLP with He initialisation."""
    def __init__(self, layer_dims):
        self.dims = layer_dims
        self.W, self.b = [], []
        for i in range(len(layer_dims)-1):
            fan_in = layer_dims[i]
            scale  = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(layer_dims[i+1], layer_dims[i]) * scale)
            self.b.append(np.zeros(layer_dims[i+1]))

    def forward(self, x):
        """Returns output and all pre-activations/activations for backprop."""
        h = x; cache = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ h + b
            h = relu(z) if i < len(self.W)-1 else z
            cache.append(z); cache.append(h)
        return h, cache   # h = final output, cache for backprop

    def backward(self, cache, grad_out, lr, grad_clip=1.0):
        """Backprop. cache is from forward(). Returns grad w.r.t. input."""
        g = np.clip(grad_out, -grad_clip, grad_clip)
        x = cache[0]
        n = len(self.W)
        # walk backward through layers
        for i in range(n-1, -1, -1):
            z    = cache[1 + i*2]
            h_in = cache[i*2]      # input to layer i
            if i < n-1:            # apply relu grad for hidden layers
                g = g * drelu(z)
            dW = np.clip(np.outer(g, h_in), -grad_clip, grad_clip)
            self.W[i] += lr * dW
            self.b[i]  += lr * np.clip(g, -grad_clip, grad_clip)
            g = self.W[i].T @ g    # propagate gradient
        return g   # grad w.r.t. input (used for Q->actor chain)

    def clone(self):
        n = MLP(self.dims)
        n.W = [w.copy() for w in self.W]
        n.b = [b.copy() for b in self.b]
        return n

    def soft_update(self, source, tau=0.005):
        """Polyak averaging: self = tau*source + (1-tau)*self"""
        for sw, tw in zip(self.W, source.W):
            sw[:] = tau * tw + (1 - tau) * sw
        for sb, tb in zip(self.b, source.b):
            sb[:] = tau * tb + (1 - tau) * sb

# ─────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000, state_dim=3):
        self.cap = capacity; self.n = 0; self.pos = 0
        self.S  = np.zeros((capacity, state_dim))
        self.A  = np.zeros(capacity)
        self.R  = np.zeros(capacity)
        self.S2 = np.zeros((capacity, state_dim))

    def push(self, s, a, r, s2):
        self.S[self.pos]  = s;  self.A[self.pos] = a
        self.R[self.pos]  = r;  self.S2[self.pos] = s2
        self.pos = (self.pos + 1) % self.cap
        self.n   = min(self.n + 1, self.cap)

    def sample(self, n):
        idx = np.random.choice(self.n, n, replace=False)
        return self.S[idx], self.A[idx], self.R[idx], self.S2[idx]

    def __len__(self): return self.n

# ─────────────────────────────────────────────
#  SAC Agent
#
#  Actor:    state (3) -> [raw_mu, log_sigma]
#  Critics:  (state+action) (4) -> Q value  x2
#  Targets:  slow copies of critics
#
#  Key SAC equations:
#    Target Q:  y = r + γ * (min(Q1t,Q2t)(s',a') - α*log π(a'|s'))
#    Critic:    min (Q(s,a) - y)²
#    Actor:     max  Q(s,a~π) - α*log π(a~π|s)
#    Alpha:     auto-tuned toward target entropy = -|A| = -1
# ─────────────────────────────────────────────
class SACAgent:
    def __init__(self,
                 hidden      = 64,
                 lr_actor    = 3e-4,
                 lr_critic   = 3e-4,
                 lr_alpha    = 3e-4,
                 gamma       = 0.99,
                 tau         = 0.005,
                 alpha_init  = 0.2,
                 batch_size  = 256,
                 warmup      = 1000):

        self.gamma   = gamma
        self.tau     = tau
        self.batch   = batch_size
        self.warmup  = warmup
        self.lr_a    = lr_actor
        self.lr_c    = lr_critic
        self.lr_al   = lr_alpha

        # log alpha (learned temperature)
        self.log_alpha   = np.log(alpha_init)
        self.target_ent  = -1.0   # target entropy = -|action_dim|

        # Networks
        S = 3; A = 1
        self.actor  = MLP([S, hidden, hidden, 2])      # -> [raw_mu, log_sig]
        self.q1     = MLP([S+A, hidden, hidden, 1])    # Q1(s,a)
        self.q2     = MLP([S+A, hidden, hidden, 1])    # Q2(s,a)
        self.q1t    = self.q1.clone()                  # target Q1
        self.q2t    = self.q2.clone()                  # target Q2

        self.buf    = ReplayBuffer()
        self.steps  = 0
        self.cumr   = 0.0

    @property
    def alpha(self): return np.exp(self.log_alpha)

    def _state(self, theta, tdot):
        return np.array([np.sin(theta), np.cos(theta),
                         np.clip(tdot, -8, 8) / 8.0])

    def _actor_out(self, s):
        """Returns mu, sigma, raw_mu, log_sigma and forward cache."""
        out, cache = self.actor.forward(s)
        raw_mu = out[0]
        log_sig = np.clip(out[1], -4, 1)
        mu  = np.tanh(raw_mu) * TAU_MAX
        sig = np.exp(log_sig)
        return mu, sig, raw_mu, log_sig, cache

    def _sample_action(self, mu, sig, deterministic=False):
        if deterministic:
            return mu
        return float(np.clip(np.random.normal(mu, sig), -TAU_MAX, TAU_MAX))

    def _log_prob(self, a, mu, sig, log_sig):
        return -0.5 * ((a - mu) / sig)**2 - log_sig - 0.9189

    def select_action(self, theta, tdot, deterministic=False):
        s = self._state(theta, tdot)
        mu, sig, raw_mu, log_sig, _ = self._actor_out(s)
        a = self._sample_action(mu, sig, deterministic)
        return a, mu, sig, s

    def q_value(self, qnet, s, a):
        sa = np.append(s, a / TAU_MAX)
        out, _ = qnet.forward(sa)
        return float(out[0])

    @staticmethod
    def reward(theta, theta_prev, tdot, tau):
        w       = np.cos(theta / 2) ** 4
        swing   = np.cos(theta) - np.cos(theta_prev)
        balance = 1.0 - (theta**2 + 0.1*tdot**2 + 0.001*tau**2)
        return float((1 - w) * swing + w * balance)

    def push(self, s, a, r, s2):
        self.buf.push(s, a, r, s2)

    def update(self):
        if len(self.buf) < self.warmup:
            return

        S, A, R, S2 = self.buf.sample(self.batch)

        # ── 1. Compute target Q values ────────────────────────────
        # Sample next actions from current policy
        A2   = np.zeros(self.batch)
        LP2  = np.zeros(self.batch)
        for i in range(self.batch):
            mu2, sig2, _, lsig2, _ = self._actor_out(S2[i])
            a2 = self._sample_action(mu2, sig2)
            A2[i]  = a2
            LP2[i] = self._log_prob(a2, mu2, sig2, lsig2)

        # Bellman targets
        TGT = np.zeros(self.batch)
        for i in range(self.batch):
            q1t = self.q_value(self.q1t, S2[i], A2[i])
            q2t = self.q_value(self.q2t, S2[i], A2[i])
            TGT[i] = R[i] + self.gamma * (min(q1t, q2t) - self.alpha * LP2[i])

        # ── 2. Update critics ─────────────────────────────────────
        for qnet, qtnet in [(self.q1, self.q1t), (self.q2, self.q2t)]:
            for i in range(self.batch):
                sa = np.append(S[i], A[i] / TAU_MAX)
                qv, cache = qnet.forward(sa)
                td_err = np.clip(TGT[i] - qv[0], -10, 10)
                qnet.backward(cache, np.array([td_err]), self.lr_c, grad_clip=1.0)
            qtnet.soft_update(qnet, self.tau)

        # ── 3. Update actor ───────────────────────────────────────
        log_probs = np.zeros(self.batch)
        for i in range(self.batch):
            mu_, sig_, raw_, lsig_, cache = self._actor_out(S[i])
            a_ = self._sample_action(mu_, sig_)
            lp_ = self._log_prob(a_, mu_, sig_, lsig_)
            log_probs[i] = lp_

            q1v = self.q_value(self.q1, S[i], a_)
            q2v = self.q_value(self.q2, S[i], a_)
            qv_ = min(q1v, q2v)

            # gradient of (Q - alpha * log_pi) w.r.t actor output
            # d log_pi / d raw_mu:  (a-mu)/sig^2 * TAU_MAX*(1-tanh^2(raw_mu))
            # d log_pi / d log_sig: (a-mu)^2/sig^2 - 1
            dtanh   = TAU_MAX * (1 - np.tanh(raw_)**2)
            d_logpi_raw  = np.clip((a_ - mu_) / sig_**2, -5, 5) * dtanh
            d_logpi_lsig = np.clip((a_ - mu_)**2 / sig_**2 - 1, -5, 5)

            # actor objective gradient (ascent = positive direction)
            obj = qv_ - self.alpha * lp_
            g_raw  = d_logpi_raw  * obj
            g_lsig = d_logpi_lsig * obj

            self.actor.backward(cache, np.array([g_raw, g_lsig]),
                                self.lr_a, grad_clip=1.0)

        # ── 4. Update alpha (automatic entropy tuning) ────────────
        mean_lp = np.mean(log_probs)
        d_alpha = -self.lr_al * (mean_lp + self.target_ent)
        self.log_alpha = np.clip(self.log_alpha + d_alpha, -5, 2)

        self.steps += 1

    def save(self, path="sac_policy.npz"):
        data = {}
        for name, net in [('actor',self.actor),('q1',self.q1),('q2',self.q2)]:
            for i,(w,b) in enumerate(zip(net.W,net.b)):
                data[f'{name}_W{i}']=w; data[f'{name}_b{i}']=b
        data['log_alpha'] = np.array([self.log_alpha])
        np.savez(path, **data)
        print(f"[SAC] Saved → {path}")

    def load(self, path="sac_policy.npz"):
        if not os.path.exists(path): print(f"[SAC] Not found: {path}"); return False
        d = np.load(path)
        for name, net in [('actor',self.actor),('q1',self.q1),('q2',self.q2)]:
            for i in range(len(net.W)):
                net.W[i] = d[f'{name}_W{i}']; net.b[i] = d[f'{name}_b{i}']
        if 'log_alpha' in d: self.log_alpha = float(d['log_alpha'][0])
        print(f"[SAC] Loaded ← {path}"); return True

# ─────────────────────────────────────────────
#  Sim state
# ─────────────────────────────────────────────
theta_state = np.pi; theta_dot_state = 0.0; tau = 0.0
mode = "manual"; keys_down = {"left": False, "right": False}
P_gain, I_gain, D_gain = 20.0, 5.0, 2.0
integral_pid = 0.0; prev_error_pid = 0.0
agent = SACAgent(); sac_learning = True; fast_forward = False

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
    elif mode == "sac":
        tp = theta_state
        det = not sac_learning   # deterministic when exploiting
        tau, mu, sig, s = agent.select_action(theta_state, theta_dot_state, det)
        tdd = (g/L)*np.sin(theta_state) + tau/(m*L*L) - c*theta_dot_state
        theta_dot_state = np.clip(theta_dot_state + tdd*dt, -15, 15)
        theta_state = (theta_state + theta_dot_state*dt + np.pi) % (2*np.pi) - np.pi
        s2 = agent._state(theta_state, theta_dot_state)
        r  = agent.reward(theta_state, tp, theta_dot_state, tau)
        agent.push(s, tau, r, s2)
        agent.cumr += r
        agent.total_steps = agent.steps
        if sac_learning:
            agent.update()
        return
    tdd = (g/L)*np.sin(theta_state) + tau/(m*L*L) - c*theta_dot_state
    theta_dot_state = np.clip(theta_dot_state + tdd*dt, -15, 15)
    theta_state = (theta_state + theta_dot_state*dt + np.pi) % (2*np.pi) - np.pi


# ═════════════════════════════════════════════
#  LAYOUT  —  1920 × 1080, 2×2 grid + controls
# ═════════════════════════════════════════════
BG   = "#f8f9fa"; PANEL = "#ffffff"; CTRL = "#f0f2f5"
GREY = "#6c757d"; BLK   = "#212529"
BLUE = "#1565C0"; RED   = "#C62828"; GRN  = "#2E7D32"; PRP = "#6A1B9A"

fig = plt.figure(figsize=(19.2, 10.8), facecolor=BG)

QW=0.225; QH=0.400; GAP=0.025
L0=0.13; L1=L0+QW+GAP
BOT=0.07; TOP_ROW=BOT+QH+GAP

fig.text(0.38, 0.955,
         "Inverted Pendulum  ·  Manual / PID / SAC",
         ha="center", va="center", fontsize=13, fontweight="bold", color=BLK)

def make_ax(left, bottom, title, tc):
    ax = fig.add_axes([left, bottom, QW, QH], facecolor=PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor("#dee2e6")
    ax.set_title(title, color=tc, fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(colors=GREY, labelsize=8)
    return ax

# ── TOP-LEFT: Pendulum ────────────────────────
ax_pend = make_ax(L0, TOP_ROW, "Inverted Pendulum", BLK)
ax_pend.set_xlim(-1.3,1.3); ax_pend.set_ylim(-1.3,1.3); ax_pend.set_aspect("equal")
ax_pend.plot(0,0,"o",color=RED,ms=8,zorder=5)
ax_pend.axhline(0,color="#dee2e6",lw=1,ls="--")
mode_text = ax_pend.text(-1.22,1.15,"",fontsize=10,color=BLUE,fontweight="bold",clip_on=False)
info_text = ax_pend.text(-1.22,1.04,"",fontsize=7,color=GREY,clip_on=False)
line, = ax_pend.plot([],[],lw=4,color=BLUE,solid_capstyle="round")
bob,  = ax_pend.plot([],[],"o",ms=13,color="#F9A825",zorder=6)
arrow = ax_pend.arrow(0,0,0,0,color=RED,width=0.0)

# ── TOP-RIGHT: Angle θ ────────────────────────
ax_theta = make_ax(L1, TOP_ROW, "Angle  θ", BLUE)
ax_theta.set_xlim(0,300); ax_theta.set_ylim(-np.pi,np.pi)
ax_theta.set_ylabel("θ (rad)",color=BLUE,fontsize=8,labelpad=3)
ax_theta.tick_params(axis="y",colors=BLUE,labelsize=7)
ax_theta.tick_params(axis="x",colors=GREY,labelsize=7)
ax_theta.axhline(0,color="#dee2e6",lw=0.8)
theta_line, = ax_theta.plot([],[],lw=1.8,color=BLUE)

# ── BOTTOM-LEFT: Torque τ ─────────────────────
ax_tau = make_ax(L0, BOT, "Torque  τ", RED)
ax_tau.set_xlim(0,300); ax_tau.set_ylim(-TAU_MAX,TAU_MAX)
ax_tau.set_ylabel("τ (Nm)",color=RED,fontsize=8,labelpad=3)
ax_tau.tick_params(axis="y",colors=RED,labelsize=7)
ax_tau.tick_params(axis="x",colors=GREY,labelsize=7)
ax_tau.axhline(0,color="#dee2e6",lw=0.8)
tau_line, = ax_tau.plot([],[],lw=1.8,color=RED)

# ── BOTTOM-RIGHT: Q value ─────────────────────
ax_q = make_ax(L1, BOT, "Q-value  (critic)", PRP)
ax_q.set_xlim(0,300); ax_q.set_ylim(-5,5)
ax_q.set_ylabel("Q(s,a)",color=PRP,fontsize=8,labelpad=3)
ax_q.tick_params(axis="y",colors=PRP,labelsize=7)
ax_q.tick_params(axis="x",colors=GREY,labelsize=7)
ax_q.axhline(0,color="#dee2e6",lw=0.8)
q1_line, = ax_q.plot([],[],lw=1.8,color=PRP,label="Q1")
q2_line, = ax_q.plot([],[],lw=1.8,color=GRN,label="Q2",ls="--")
ax_q.legend(fontsize=7,framealpha=0.9,edgecolor="#dee2e6",loc="upper left")

HIST=300
tau_hist=[]; theta_hist=[]; q1_hist=[]; q2_hist=[]

# ═════════════════════════════════════════════
#  CONTROLS PANEL
# ═════════════════════════════════════════════
PL=0.700; PW=0.290
ctrl_bg=fig.add_axes([PL,0.02,PW,0.96],facecolor=CTRL,zorder=0)
ctrl_bg.set_xticks([]); ctrl_bg.set_yticks([])
for sp in ctrl_bg.spines.values(): sp.set_edgecolor("#ced4da"); sp.set_linewidth(1.2)

WL=PL+0.012; WW=PW-0.024

def make_btn(rect,label,fc=PANEL,tc=BLK):
    ax=fig.add_axes(rect,facecolor=fc)
    for sp in ax.spines.values(): sp.set_edgecolor("#ced4da")
    btn=Button(ax,label,color=fc,hovercolor="#dde3ff")
    btn.label.set_color(tc); btn.label.set_fontsize(10)
    return btn

def silent_ax(rect,fc=CTRL):
    ax=fig.add_axes(rect,facecolor=fc)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    return ax

# Header
ax_hdr=silent_ax([WL,0.922,WW,0.046])
ax_hdr.text(0.5,0.5,"Controls",ha="center",va="center",
            fontsize=14,fontweight="bold",color=BLK)
silent_ax([WL,0.916,WW,0.005]).axhline(0.5,color="#ced4da",lw=1.5)

# Mode buttons
BW=WW/3-0.003
btn_manual=make_btn([WL,          0.858,BW,0.050],"Manual",tc=GRN)
btn_pid   =make_btn([WL+WW/3,     0.858,BW,0.050],"PID",   tc=BLUE)
btn_sac   =make_btn([WL+2*WW/3,   0.858,BW,0.050],"SAC",   tc=RED)
silent_ax([WL,0.848,WW,0.006]).axhline(0.5,color="#ced4da",lw=1)

# SAC panel
sac_widgets=[]
ax_st=silent_ax([WL,0.820,WW,0.024])
ax_st.text(0.02,0.5,"Soft Actor-Critic",fontsize=10,fontweight="bold",color=RED,va="center")
sac_widgets.append(ax_st)

# SAC hyperparameter sliders
ax_slr_a=fig.add_axes([WL,0.790,WW,0.024],facecolor=PANEL)
ax_slr_c=fig.add_axes([WL,0.760,WW,0.024],facecolor=PANEL)
ax_slr_al=fig.add_axes([WL,0.730,WW,0.024],facecolor=PANEL)
sl_lr_a =Slider(ax_slr_a, "lr actor",  1e-5,1e-3,valinit=3e-4,color=RED)
sl_lr_c =Slider(ax_slr_c, "lr critic", 1e-5,1e-3,valinit=3e-4,color=PRP)
sl_alpha=Slider(ax_slr_al,"α entropy", 0.01, 0.5, valinit=0.2, color=GRN)
for sl in [sl_lr_a,sl_lr_c,sl_alpha]:
    sl.label.set_color(GREY); sl.label.set_fontsize(8)
    sl.valtext.set_color(BLK); sl.valtext.set_fontsize(8)
def upd_lra(v):  agent.lr_a=v
def upd_lrc(v):  agent.lr_c=v
def upd_alp(v):  agent.log_alpha=np.log(max(v,1e-6))
sl_lr_a.on_changed(upd_lra); sl_lr_c.on_changed(upd_lrc); sl_alpha.on_changed(upd_alp)
sac_widgets+=[ax_slr_a,ax_slr_c,ax_slr_al]

# Buttons
BL=WW*0.60
btn_learn=make_btn([WL,           0.690,BL,      0.034],"Learn: ON",tc=GRN)
btn_ff   =make_btn([WL,           0.648,WW,      0.034],"⚡ Fast Forward (50x)",tc=GREY)
btn_save =make_btn([WL+BL+0.004,  0.690,WW*0.17, 0.034],"💾",tc=BLUE)
btn_load =make_btn([WL+BL+WW*0.20,0.690,WW*0.17, 0.034],"📂",tc=BLUE)
sac_widgets+=[btn_learn.ax,btn_ff.ax,btn_save.ax,btn_load.ax]

# Stats box
ax_stats=fig.add_axes([WL,0.490,WW,0.150],facecolor=PANEL)
ax_stats.set_xticks([]); ax_stats.set_yticks([])
for sp in ax_stats.spines.values(): sp.set_edgecolor("#ced4da"); sp.set_visible(True)
stats_text=ax_stats.text(0.05,0.96,"",fontsize=9,color=GREY,
                          va="top",transform=ax_stats.transAxes,
                          family="monospace",linespacing=1.7)
sac_widgets.append(ax_stats)

# σ trend
sig_hist=[]
ax_sig=fig.add_axes([WL,0.340,WW,0.130],facecolor=PANEL)
ax_sig.set_xticks([]); ax_sig.set_ylim(0,3)
ax_sig.set_ylabel("σ(s)",color=RED,fontsize=8,labelpad=2)
ax_sig.tick_params(axis="y",colors=RED,labelsize=7)
for sp in ax_sig.spines.values(): sp.set_edgecolor("#dee2e6")
ax_sig.set_title("Exploration  σ",color=RED,fontsize=9,pad=4)
sig_line,=ax_sig.plot([],[],lw=1.5,color=RED)
ax_sig.axhline(0.1,color=GREY,lw=0.5,ls=":",alpha=0.5)
sac_widgets.append(ax_sig)

# PID panel
pid_widgets=[]
ax_ptitle=silent_ax([WL,0.820,WW,0.026])
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
from matplotlib.widgets import TextBox
tb_P=TextBox(ax_P,"",initial=str(int(P_gain)))
tb_I=TextBox(ax_I,"",initial=str(int(I_gain)))
tb_D=TextBox(ax_D,"",initial=str(int(D_gain)))
for tb in [tb_P,tb_I,tb_D]:
    tb.text_disp.set_ha("center"); tb.text_disp.set_va("center")
    tb.text_disp.set_position((0.5,0.5)); tb.text_disp.set_fontsize(10)
    tb.text_disp.set_color(BLK)
pid_widgets+=[ax_P,ax_I,ax_D]
def upd_P(t):
    global P_gain
    try: P_gain=float(t)
    except: pass
def upd_I(t):
    global I_gain
    try: I_gain=float(t)
    except: pass
def upd_D(t):
    global D_gain
    try: D_gain=float(t)
    except: pass
tb_P.on_submit(upd_P); tb_I.on_submit(upd_I); tb_D.on_submit(upd_D)

def set_vis(wlist,v):
    for ax in wlist: ax.set_visible(v)

set_vis(pid_widgets,False); set_vis(sac_widgets,False)

def highlight(active):
    cfg={"manual":(btn_manual,GRN,"#e8f5e9"),
         "pid":   (btn_pid,   BLUE,"#e3f2fd"),
         "sac":   (btn_sac,   RED, "#fce4ec")}
    for k,(btn,tc,hc) in cfg.items():
        if k==active: btn.ax.set_facecolor(hc); btn.label.set_fontweight("bold")
        else:         btn.ax.set_facecolor(PANEL); btn.label.set_fontweight("normal")

highlight("manual")

def set_manual(e):
    global mode,integral_pid,prev_error_pid
    mode="manual"; integral_pid=0.0; prev_error_pid=0.0
    set_vis(pid_widgets,False); set_vis(sac_widgets,False)
    highlight("manual"); fig.canvas.draw_idle()

def set_pid(e):
    global mode,integral_pid,prev_error_pid
    mode="pid"; integral_pid=0.0; prev_error_pid=0.0
    set_vis(pid_widgets,True); set_vis(sac_widgets,False)
    highlight("pid"); fig.canvas.draw_idle()

def set_sac(e):
    global mode
    mode="sac"
    set_vis(pid_widgets,False); set_vis(sac_widgets,True)
    highlight("sac"); fig.canvas.draw_idle()

def toggle_learn(e):
    global sac_learning
    sac_learning=not sac_learning
    btn_learn.label.set_text("Learn: ON" if sac_learning else "Exploit")
    btn_learn.label.set_color(GRN if sac_learning else GREY)
    fig.canvas.draw_idle()

def toggle_ff(e):
    global fast_forward
    fast_forward=not fast_forward
    btn_ff.label.set_text("⚡ Fast Forward (50x)  ON" if fast_forward else "⚡ Fast Forward (50x)")
    btn_ff.label.set_color(RED if fast_forward else GREY)
    btn_ff.ax.set_facecolor("#fce4ec" if fast_forward else PANEL)
    fig.canvas.draw_idle()

def save_pol(e): agent.save()
def load_pol(e): agent.load()

btn_manual.on_clicked(set_manual); btn_pid.on_clicked(set_pid)
btn_sac.on_clicked(set_sac);       btn_learn.on_clicked(toggle_learn)
btn_ff.on_clicked(toggle_ff);      btn_save.on_clicked(save_pol)
btn_load.on_clicked(load_pol)

# ─────────────────────────────────────────────
#  Animation
# ─────────────────────────────────────────────
def update(frame):
    global arrow
    n = 50 if (fast_forward and mode=="sac" and sac_learning) else 1
    for _ in range(n):
        step()

    x=L*np.sin(theta_state); y=L*np.cos(theta_state)
    line.set_data([0,x],[0,y]); bob.set_data([x],[y])
    arrow.remove()
    alen=np.clip(tau/TAU_MAX,-1,1)*0.35
    arrow=ax_pend.arrow(0,0,alen,0,width=0.025,
                        head_width=0.06,head_length=0.05,
                        color=RED,alpha=0.8,zorder=4)

    mode_text.set_text(f"Mode: {mode.upper()}")
    s=agent._state(theta_state,theta_dot_state)
    mu,sig,_,_,_=agent._actor_out(s)
    q1v=agent.q_value(agent.q1,s,tau)
    q2v=agent.q_value(agent.q2,s,tau)
    info_text.set_text(f"θ={theta_state:.2f}  θ̇={theta_dot_state:.2f}  "
                       f"τ={tau:.2f}  μ={mu:.2f}  σ={sig:.2f}")

    tau_hist.append(np.clip(tau,-TAU_MAX,TAU_MAX))
    theta_hist.append(theta_state)
    q1_hist.append(np.clip(q1v,-5,5))
    q2_hist.append(np.clip(q2v,-5,5))
    sig_hist.append(sig)
    for h in [tau_hist,theta_hist,q1_hist,q2_hist,sig_hist]:
        if len(h)>HIST: h.pop(0)

    xs=list(range(len(tau_hist)))
    theta_line.set_data(xs,theta_hist)
    tau_line.set_data(xs,tau_hist)
    q1_line.set_data(xs,q1_hist)
    q2_line.set_data(xs,q2_hist)
    sig_line.set_data(xs,sig_hist)
    for ax in [ax_theta,ax_tau,ax_q,ax_sig]:
        ax.set_xlim(0,max(HIST,len(tau_hist)))

    if mode=="sac":
        buf_pct=len(agent.buf)/agent.buf.cap*100
        bal="✓ BALANCING" if abs(theta_state)<0.2 else "learning..."
        spd="50x ⚡" if fast_forward else "1x"
        stats_text.set_text(
            f"μ(s)    = {mu:+.3f}\n"
            f"σ(s)    =  {sig:.4f}\n"
            f"α(ent)  =  {agent.alpha:.4f}\n"
            f"τ       = {tau:+.3f}\n"
            f"\nbuffer  = {len(agent.buf):,} ({buf_pct:.0f}%)\n"
            f"updates = {agent.steps:,}\n"
            f"speed   = {spd}\n"
            f"status  = {bal}"
        )
    else:
        stats_text.set_text("")

    return line,bob,arrow,theta_line,tau_line,q1_line,q2_line,sig_line

fig.canvas.mpl_connect("key_press_event",   on_key_press)
fig.canvas.mpl_connect("key_release_event", on_key_release)
ani=FuncAnimation(fig,update,interval=20,blit=False)
plt.show()