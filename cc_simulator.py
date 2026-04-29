"""
Congestion Control Simulator
Algorithms: BBR, PCC Vivace, TCP Reno, CUBIC, HALO, Custom
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import random
import math
import textwrap

# ─────────────────────────────────────────────
# Network profiles
# Each entry: (bw_mbps, rtt_ms, queue_bdp, loss_pct, description)
# ─────────────────────────────────────────────
NETWORK_PROFILES = {
    'Home Broadband':   (50,    20,  2.0, 0.0, '50 Mbps cable, low RTT, shallow queue'),
    'Datacenter':       (1000,  1,   0.5, 0.0, '1 Gbps, ultra-low RTT, tiny queue'),
    '4G LTE':           (30,    40,  1.5, 0.5, '30 Mbps, moderate RTT, some loss'),
    '5G':               (200,   10,  1.0, 0.1, '200 Mbps, low RTT, minimal loss'),
    'Satellite (GEO)':  (20,    600, 1.0, 0.2, '20 Mbps, 600ms RTT — classic bufferbloat'),
    'Satellite (LEO)':  (100,   20,  1.5, 0.3, '100 Mbps, low RTT (Starlink-like)'),
    'Lossy WiFi':       (25,    30,  1.0, 2.5, '25 Mbps, high random loss'),
    'Cross-Country':    (100,   80,  2.0, 0.0, '100 Mbps, high BDP, large queue'),
    'Dial-up':          (0.056, 150, 1.0, 1.0, '56 Kbps modem — for nostalgia'),
    'Custom':           (20,    40,  1.0, 0.0, 'Set sliders manually'),
}

# ─────────────────────────────────────────────
# Simulation state
# ─────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = 0.0
        self.cwnd = 10.0
        self.bw_est = 0.0
        self.phase = 'startup'
        self.rtt_min = None
        self.rtt_latest = None
        self.lost = 0
        self.lost_rate = 0.0
        self.ssthresh = 64.0
        self.prev_utility = -999.0
        self.cycle = 0
        self.cycle_t = 0.0
        self.W_max = 0.0
        self.K = 0.0
        self.t0 = 0.0
        self.last_probe_rtt = 0.0
        self.probe_rtt_t = 0.0

        # Delivery-rate samples for HALO (and improved BDP estimation).
        # Each entry is (timestamp, delivery_rate_bytes_per_sec).
        self.delivery_samples = []
        # Last raw throughput sample written by tick() (bytes/sec).
        self.delivery_rate = 0.0

# ─────────────────────────────────────────────
# Algorithm implementations
# ─────────────────────────────────────────────
def algo_bbr(s):
    bdp = s.bw_est * s.rtt_min / 8.0
    cwnd = s.cwnd
    if s.phase == 'startup':
        cwnd *= 2.885
        if cwnd > bdp * 2:
            s.phase = 'drain'
    elif s.phase == 'drain':
        cwnd *= 0.75
        if cwnd <= bdp:
            s.phase = 'probe_bw'
            s.cycle = 0
            s.cycle_t = s.t
    elif s.phase == 'probe_bw':
        gains = [1.25, 0.75, 1, 1, 1, 1, 1, 1]
        if s.t - s.cycle_t > s.rtt_latest / 1000.0:
            s.cycle = (s.cycle + 1) % 8
            s.cycle_t = s.t
        cwnd = bdp * gains[s.cycle] * 2
    elif s.phase == 'probe_rtt':
        cwnd = 4
        if s.t - s.probe_rtt_t > 0.2:
            s.phase = 'probe_bw'
    if s.t - s.last_probe_rtt > 10:
        s.phase = 'probe_rtt'
        s.probe_rtt_t = s.t
        s.last_probe_rtt = s.t
    cwnd = max(4.0, min(cwnd, 5000.0))
    return cwnd, s.bw_est * 1.25

def algo_vivace(s):
    cwnd = s.cwnd
    rtt_ratio = s.rtt_latest / s.rtt_min
    u_loss = -s.lost_rate * 11.35
    u_rtt = -(rtt_ratio - 1) * 900
    u_tput = math.log(s.bw_est / 1e6 + 0.001)
    utility = u_tput + u_rtt + u_loss
    grad = utility - s.prev_utility
    s.prev_utility = utility
    cwnd = cwnd + 0.01 * grad * cwnd
    cwnd = max(4.0, min(cwnd, 5000.0))
    return cwnd, cwnd * 1500 * 8 / (s.rtt_latest / 1000.0)

def algo_reno(s, log_fn):
    cwnd = s.cwnd
    if s.lost:
        cwnd = max(1.0, cwnd * 0.5)
        s.ssthresh = cwnd
        log_fn('Loss: halving cwnd')
    elif cwnd < s.ssthresh:
        cwnd += 1.0
    else:
        cwnd += 1.0 / cwnd
    cwnd = max(1.0, min(cwnd, 5000.0))
    return cwnd, cwnd * 1500 * 8 / (s.rtt_latest / 1000.0)

def algo_cubic(s, log_fn):
    cwnd = s.cwnd
    if s.lost:
        s.W_max = cwnd
        cwnd *= 0.7
        s.ssthresh = cwnd
        s.K = (s.W_max * 0.3 / 0.4) ** (1/3)
        s.t0 = s.t
        log_fn('Loss: CUBIC multiplicative decrease')
    else:
        dt = s.t - s.t0
        W_cubic = 0.4 * (dt - s.K) ** 3 + s.W_max
        cwnd = max(cwnd + 1.0 / cwnd, W_cubic)
    cwnd = max(1.0, min(cwnd, 5000.0))
    return cwnd, cwnd * 1500 * 8 / (s.rtt_latest / 1000.0)

# ─────────────────────────────────────────────
# HALO — Hybrid Adaptive Loss-resilient Optimizer
# ─────────────────────────────────────────────
#
# Design goals (in priority order):
#   1. Keep cwnd close to BDP → high utilization, low queue, low RTT.
#   2. Distinguish random loss from congestion loss using delay signal,
#      so links like Lossy WiFi and Satellite don't collapse cwnd on
#      noise the way Reno/CUBIC do.
#   3. Use a true windowed-max delivery rate (not an EWMA of cwnd-driven
#      throughput) so the BDP estimate doesn't lag reality the way the
#      simulator's BBR does.
#   4. Probe gently for more bandwidth instead of doing exponential
#      slow-start overshoots that wreck satellite / high-BDP links.
#
# Decision logic each tick:
#   • Maintain rtt_min over a 10s window and bw_max over a 10s window.
#   • target = bw_max * rtt_min  (the real BDP, in bytes).
#   • If queueing delay is high (RTT > 1.25 * rtt_min) -> drain toward
#     target (cwnd -= 10% of the excess). Handles congestion BEFORE loss.
#   • Else if loss happens AND queue is near empty (RTT ~= rtt_min) ->
#     treat as random loss, ignore. Key win on Lossy WiFi.
#   • Else if loss happens AND queue is full -> mild backoff (x0.85,
#     not x0.5 like Reno) because we already know the BDP from bw_max.
#   • Else (no loss, low queue) -> glide toward target * 1.05 with a
#     small additive probe so we discover capacity increases.
#
# Why this beats the existing algos in this simulator:
#   • vs BBR: uses real delivery samples instead of EWMA-of-cwnd; no
#     2.885x startup overshoot to flush a 600ms-RTT queue with.
#   • vs CUBIC/Reno: doesn't halve cwnd on random loss -> wins Lossy WiFi.
#   • vs Vivace: bounded cwnd target, no log-utility runaway oscillation.
# ─────────────────────────────────────────────
def algo_halo(s, log_fn):
    cwnd = s.cwnd

    # --- Track delivery rate samples in a 10-second sliding window ---
    s.delivery_samples.append((s.t, s.delivery_rate))
    cutoff = s.t - 10.0
    while s.delivery_samples and s.delivery_samples[0][0] < cutoff:
        s.delivery_samples.pop(0)
    bw_max = max((d for _, d in s.delivery_samples), default=s.delivery_rate)
    if bw_max <= 0:
        bw_max = s.bw_est  # fallback for the first few ticks

    # --- Compute BDP target in packets ---
    rtt_min_s = (s.rtt_min or s.rtt_latest) / 1000.0
    target_pkts = max(4.0, bw_max * rtt_min_s / 1500.0)

    # --- Periodic probe-up phase (every ~3s, last 0.4s) ---
    # This is how we discover capacity increases without permanently
    # standing in the queue. We only probe when no loss is happening,
    # so a busy/lossy link won't trigger this.
    cycle_pos = s.t % 3.0
    probing_up = (0.0 <= cycle_pos < 0.4)

    # --- Queueing signal ---
    rtt_ratio = s.rtt_latest / max(s.rtt_min, 1.0)
    queue_high = rtt_ratio > 1.20         # standing queue building
    queue_empty = rtt_ratio < 1.08        # link is clean

    # --- Reaction priority: queue -> loss -> probe ---
    if queue_high:
        # Drain mode: shed cwnd toward target. Aggressiveness scales with
        # how bloated the queue is — proportional control, no oscillation.
        excess = max(cwnd - target_pkts, 0.0)
        # 15% of excess per tick when queue is just starting; up to 40%
        # when RTT has doubled (severe bufferbloat).
        drain_rate = 0.15 + min(0.25, (rtt_ratio - 1.20) * 0.5)
        cwnd -= drain_rate * excess
        cwnd = max(cwnd, target_pkts)  # never drain below target
        if s.phase != 'drain':
            log_fn(f'HALO: draining (RTT {rtt_ratio:.2f}x min)')
            s.phase = 'drain'

    elif s.lost and queue_empty:
        # Random / link-layer loss. The queue is empty so the loss can't
        # be congestion. Hold cwnd steady; don't punish ourselves.
        if s.phase != 'random_loss':
            log_fn('HALO: random loss ignored (queue empty)')
            s.phase = 'random_loss'
        # tiny additive nudge so we keep probing
        cwnd += 0.5

    elif s.lost:
        # Congestion-correlated loss. We already have a good BDP estimate
        # from bw_max, so a mild backoff to target is sufficient — no
        # need for the Reno x0.5 sledgehammer.
        cwnd = max(target_pkts, cwnd * 0.85)
        log_fn('HALO: congestion loss -> soft backoff to BDP')
        s.phase = 'recover'

    else:
        # No loss, no queue buildup. Glide toward target and probe up.
        # The probe rate scales with target_pkts so high-BDP links
        # (cross-country, satellite) don't take forever to fill.
        ideal = target_pkts * (1.25 if probing_up else 1.05)
        if cwnd < ideal:
            # Geometric pull toward ideal: close ~15% of the gap per tick
            # (faster when probing up), with a floor so small-BDP links
            # still move at >=1 pkt/tick.
            gap = ideal - cwnd
            rate = 0.18 if probing_up else 0.12
            cwnd += max(1.0, gap * rate)
        else:
            # Slowly explore above target to detect bw increases.
            # Scale probe by sqrt(target) so big-BDP links probe faster.
            cwnd += max(1.0 / cwnd, math.sqrt(target_pkts) * 0.05)
        if s.phase not in ('cruise', 'probe'):
            log_fn(f'HALO: cruising at BDP~{target_pkts:.0f} pkts')
            s.phase = 'cruise'

    cwnd = max(4.0, min(cwnd, 5000.0))
    return cwnd, bw_max * 8.0  # report bw_max as throughput estimate

# ─────────────────────────────────────────────
# Core simulation tick
# ─────────────────────────────────────────────
def tick(s, params, algo, custom_fn, log_fn):
    bw_mbps, rtt_base, q_mult, loss_pct = params
    bw_bytes = bw_mbps * 1e6 / 8.0
    bdp_pkts = bw_bytes * (rtt_base / 1000.0) / 1500.0
    queue_pkts = bdp_pkts * q_mult

    # Init on first tick
    if s.rtt_min is None:
        s.rtt_min = rtt_base
        s.rtt_latest = rtt_base
        s.bw_est = bw_bytes * 0.5

    s.t += 0.1

    queued = max(0.0, s.cwnd - bdp_pkts)
    queue_delay = (queued / bw_bytes) * 1500 * 1000
    noise = random.uniform(-2, 2)
    s.rtt_latest = max(rtt_base, rtt_base + queue_delay + noise)
    s.rtt_min = min(s.rtt_min, s.rtt_latest)

    drop_overflow = 0.15 if queued > queue_pkts else 0.0
    drop_random = loss_pct / 100.0
    drop_p = drop_overflow + drop_random - drop_overflow * drop_random
    s.lost = 1 if random.random() < drop_p else 0
    s.lost_rate = s.lost_rate * 0.9 + s.lost * 0.1

    eff_pkts = min(s.cwnd, bdp_pkts + queue_pkts)
    raw_tput = eff_pkts * 1500 * 8 / (s.rtt_latest / 1000.0)
    s.bw_est = s.bw_est * 0.9 + raw_tput * (1 - drop_p) * 0.1

    # Record raw delivery rate (bytes/sec) for HALO's bw_max window.
    # Cleaner sample than bw_est because it isn't smoothed.
    s.delivery_rate = (eff_pkts * 1500 * (1 - drop_p)) / (s.rtt_latest / 1000.0)

    if algo == 'BBR':
        s.cwnd, _ = algo_bbr(s)
    elif algo == 'PCC Vivace':
        s.cwnd, _ = algo_vivace(s)
    elif algo == 'TCP Reno':
        s.cwnd, _ = algo_reno(s, log_fn)
    elif algo == 'CUBIC':
        s.cwnd, _ = algo_cubic(s, log_fn)
    elif algo == 'HALO':
        s.cwnd, _ = algo_halo(s, log_fn)
    elif algo == 'Custom':
        if custom_fn:
            try:
                result = custom_fn(s)
                s.cwnd = result.get('cwnd', s.cwnd)
            except Exception as e:
                log_fn(f'Custom error: {e}')

    tput_mbps = s.bw_est / 1e6
    util_pct = min(100.0, tput_mbps / bw_mbps * 100)
    return tput_mbps, s.rtt_latest, s.lost_rate * 100, util_pct, s.cwnd

# ─────────────────────────────────────────────
# Matplotlib GUI
# ─────────────────────────────────────────────
class SimGUI:
    MAX_HIST = 300
    DT_MS = 80   # real-time tick interval

    def __init__(self):
        self.state = SimState()
        self.algo = 'BBR'
        self.running = False
        self.anim = None
        self.log_lines = []
        self.hist_t = []
        self.hist_cwnd = []
        self.hist_tput = []
        self.hist_rtt = []
        self.custom_fn = None
        self._build_ui()

    # ── layout ──────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor('#f9f9f7')
        self.fig.canvas.manager.set_window_title('Congestion Control Simulator')

        # Main chart axes
        self.ax = self.fig.add_axes([0.07, 0.42, 0.55, 0.50])
        self.ax.set_facecolor('#ffffff')
        self.ax.set_title('Congestion Control Simulator', fontsize=13, fontweight='normal', pad=8)
        self.ax.set_xlabel('Time (s)', fontsize=10)
        self.ax.set_ylabel('Mbps / ms / pkts (normalised)', fontsize=9)
        for sp in self.ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('#cccccc')

        self.ln_tput, = self.ax.plot([], [], color='#1D9E75', lw=1.8, label='Throughput (Mbps)')
        self.ln_rtt,  = self.ax.plot([], [], color='#D85A30', lw=1.4, label='RTT (ms, scaled)')
        self.ln_cwnd, = self.ax.plot([], [], color='#378ADD', lw=1.8, label='cwnd (pkts, scaled)')
        self.ax.legend(loc='upper left', fontsize=9, framealpha=0.7)

        # Metric cards
        card_y = 0.88
        self._mk_card(0.64,  card_y, 'Throughput', 'm_tput', 'Mbps')
        self._mk_card(0.755, card_y, 'RTT',        'm_rtt',  'ms')
        self._mk_card(0.87,  card_y, 'Loss rate',  'm_loss', '%')
        self._mk_card(0.64,  0.73,   'Link util.', 'm_util', '%')
        self._mk_card(0.755, 0.73,   'cwnd',       'm_cwnd', 'pkts')

        # Sliders
        sc = '#f0f0ee'
        self.sl_bw   = self._mk_slider(0.07, 0.34, 'Bandwidth (Mbps)', 0.056, 1000, 20, sc)
        self.sl_rtt  = self._mk_slider(0.07, 0.29, 'Base RTT (ms)',     1, 600, 40, sc)
        self.sl_q    = self._mk_slider(0.07, 0.24, 'Queue (BDP×)',      0.25, 4, 1, sc, fmt='{:.2f}×')
        self.sl_loss = self._mk_slider(0.07, 0.19, 'Random loss (%)',   0, 5, 0, sc, fmt='{:.1f}%')

        # ── Network profile radio (left column) ──
        profile_names = list(NETWORK_PROFILES.keys())
        n_profiles = len(profile_names)
        pax_h = 0.028 * n_profiles + 0.04
        pax = self.fig.add_axes([0.64, 0.42, 0.16, pax_h])
        pax.set_facecolor('#f5f5f3')
        for sp in pax.spines.values(): sp.set_linewidth(0.5); sp.set_color('#cccccc')
        self.profile_radio = RadioButtons(pax, profile_names, activecolor='#1D9E75')
        self.profile_radio.on_clicked(self._set_profile)
        pax.set_title('Network', fontsize=10, pad=6)
        for lbl in self.profile_radio.labels:
            lbl.set_fontsize(8.5)

        # Profile description box
        self.desc_ax = self.fig.add_axes([0.64, 0.38, 0.33, 0.038])
        self.desc_ax.set_facecolor('#fffbe6')
        for sp in self.desc_ax.spines.values(): sp.set_linewidth(0.5); sp.set_color('#e0d080')
        self.desc_ax.set_xticks([]); self.desc_ax.set_yticks([])
        self._desc_text = self.desc_ax.text(
            0.01, 0.5, NETWORK_PROFILES['Home Broadband'][4],
            transform=self.desc_ax.transAxes,
            fontsize=8.5, va='center', color='#555500')

        # ── Algorithm radio (right column) ──
        rax = self.fig.add_axes([0.82, 0.42, 0.16, 0.32])
        rax.set_facecolor('#f5f5f3')
        for sp in rax.spines.values(): sp.set_linewidth(0.5); sp.set_color('#cccccc')
        self.radio = RadioButtons(rax, ('BBR', 'PCC Vivace', 'TCP Reno', 'CUBIC', 'HALO', 'Custom'),
                                  activecolor='#378ADD')
        self.radio.on_clicked(self._set_algo)
        rax.set_title('Algorithm', fontsize=10, pad=6)

        # Buttons
        self.btn_run   = self._mk_btn(0.07, 0.12, 0.12, 'Run')
        self.btn_reset = self._mk_btn(0.21, 0.12, 0.12, 'Reset')
        self.btn_run.on_clicked(self._toggle_run)
        self.btn_reset.on_clicked(self._on_reset)

        # Event log
        self.custom_ax = self.fig.add_axes([0.07, 0.01, 0.90, 0.09])
        self.custom_ax.set_facecolor('#f5f5f3')
        for sp in self.custom_ax.spines.values(): sp.set_linewidth(0.5); sp.set_color('#cccccc')
        self.custom_ax.set_xticks([]); self.custom_ax.set_yticks([])
        self._log_text = self.custom_ax.text(
            0.01, 0.85,
            'Event log — press Run to start',
            transform=self.custom_ax.transAxes,
            fontsize=8.5, fontfamily='monospace',
            va='top', color='#555555', wrap=True
        )

        plt.show()

    def _mk_card(self, x, y, label, key, unit):
        ax = self.fig.add_axes([x, y, 0.10, 0.11])
        ax.set_facecolor('#ffffff')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.5); sp.set_color('#dddddd')
        ax.text(0.5, 0.85, label, ha='center', va='top', fontsize=8.5,
                color='#888888', transform=ax.transAxes)
        t = ax.text(0.5, 0.38, '—', ha='center', va='center', fontsize=18,
                    fontweight='normal', color='#1a1a18', transform=ax.transAxes)
        ax.text(0.5, 0.05, unit, ha='center', va='bottom', fontsize=8,
                color='#aaaaaa', transform=ax.transAxes)
        setattr(self, key, t)

    def _mk_slider(self, x, y, label, vmin, vmax, vinit, color, fmt=None):
        ax = self.fig.add_axes([x+0.14, y, 0.38, 0.028])
        ax.set_facecolor(color)
        sl = Slider(ax, label, vmin, vmax, valinit=vinit,
                    color='#378ADD', track_color='#dddddd')
        sl.label.set_fontsize(9)
        sl.valtext.set_fontsize(9)
        if fmt:
            sl.valtext.set_text(fmt.format(vinit))
            def _upd(val, s=sl, f=fmt):
                s.valtext.set_text(f.format(val))
            sl.on_changed(_upd)
        return sl

    def _mk_btn(self, x, y, w, label):
        ax = self.fig.add_axes([x, y, w, 0.04])
        btn = Button(ax, label, color='#e6f1fb', hovercolor='#b5d4f4')
        btn.label.set_fontsize(10)
        return btn

    # ── controls ────────────────────────────
    def _set_profile(self, label):
        profile = NETWORK_PROFILES[label]
        bw, rtt, q, loss, desc = profile
        if label != 'Custom':
            self.sl_bw.set_val(bw)
            self.sl_rtt.set_val(rtt)
            self.sl_q.set_val(q)
            self.sl_loss.set_val(loss)
            self.sl_q.valtext.set_text(f'{q:.2f}×')
            self.sl_loss.valtext.set_text(f'{loss:.1f}%')
        self._desc_text.set_text(desc)
        self.fig.canvas.draw_idle()

    def _set_algo(self, label):
        self.algo = label

    def _toggle_run(self, event):
        if self.running:
            self.running = False
            self.btn_run.label.set_text('Run')
            if self.anim:
                self.anim.event_source.stop()
        else:
            self.running = True
            self.btn_run.label.set_text('Pause')
            params = self._get_params()
            self._log(f'Starting {self.algo} — BW={params[0]:.0f}Mbps '
                      f'RTT={params[1]:.0f}ms Q={params[2]:.2f}×BDP')
            self.anim = animation.FuncAnimation(
                self.fig, self._step, interval=self.DT_MS, blit=False, cache_frame_data=False)

    def _on_reset(self, event):
        self.running = False
        self.btn_run.label.set_text('Run')
        if self.anim:
            self.anim.event_source.stop()
        self.state.reset()
        self.hist_t.clear(); self.hist_cwnd.clear()
        self.hist_tput.clear(); self.hist_rtt.clear()
        self.ln_tput.set_data([], []); self.ln_rtt.set_data([], [])
        self.ln_cwnd.set_data([], [])
        for key in ('m_tput','m_rtt','m_loss','m_util'):
            getattr(self, key).set_text('—')
        self.ax.relim(); self.ax.autoscale_view()
        self.log_lines.clear()
        self._log_text.set_text('Reset — press Run to start')
        self.fig.canvas.draw_idle()

    def _get_params(self):
        return (self.sl_bw.val, self.sl_rtt.val,
                self.sl_q.val, self.sl_loss.val)

    def _log(self, msg):
        t = self.hist_t[-1] if self.hist_t else 0.0
        self.log_lines.append(f'[{t:.1f}s] {msg}')
        self.log_lines = self.log_lines[-4:]
        self._log_text.set_text('\n'.join(self.log_lines))

    # ── animation step ───────────────────────
    def _step(self, frame):
        if not self.running:
            return
        params = self._get_params()
        tput, rtt, loss, util, cwnd = tick(
            self.state, params, self.algo, self.custom_fn, self._log)

        self.hist_t.append(self.state.t)
        self.hist_tput.append(tput)
        self.hist_rtt.append(rtt)
        self.hist_cwnd.append(cwnd)

        # Trim history
        if len(self.hist_t) > self.MAX_HIST:
            self.hist_t.pop(0); self.hist_tput.pop(0)
            self.hist_rtt.pop(0); self.hist_cwnd.pop(0)

        # Normalise RTT and cwnd to same scale as tput for overlay
        max_tput = max(self.hist_tput) or 1
        max_rtt = max(self.hist_rtt) or 1
        max_cwnd = max(self.hist_cwnd) or 1

        t = self.hist_t
        self.ln_tput.set_data(t, self.hist_tput)
        self.ln_rtt.set_data( t, [v / max_rtt  * max_tput for v in self.hist_rtt])
        self.ln_cwnd.set_data(t, [v / max_cwnd * max_tput for v in self.hist_cwnd])
        self.ax.relim(); self.ax.autoscale_view()

        # Update metric cards
        self.m_tput.set_text(f'{tput:.1f}')
        self.m_rtt.set_text( f'{rtt:.0f}')
        self.m_loss.set_text(f'{loss:.1f}')
        self.m_util.set_text(f'{util:.0f}')
        self.m_cwnd.set_text(f'{cwnd:.0f}')

        if self.state.t >= 60:
            self.running = False
            self.btn_run.label.set_text('Run')
            self._log('Simulation complete (60s)')

        # Log LSTM training events
        if self.algo == 'Custom' and self.custom_fn is not None:
            ctrl = self.custom_fn
            if hasattr(ctrl, 'train_losses') and ctrl.train_losses:
                n = len(ctrl.train_losses)
                if n % 10 == 0 and n > 0:
                    last_r = ctrl.train_losses[-1]
                    self._log(f'LSTM update #{n} — reward={last_r:.4f} '
                              f'tick={ctrl.tick_n}')

        self.fig.canvas.draw_idle()

# ─────────────────────────────────────────────
# Pure-numpy LSTM (no PyTorch / TensorFlow)
# ─────────────────────────────────────────────
#
# Architecture
# ─────────────
# Input  x_t      : 5 features per tick
#                   [rtt_ratio, loss_rate, bw_norm, cwnd_norm, queued_norm]
# LSTM hidden     : H = 16 units, sequence length SEQ = 16 ticks
# Output          : scalar action in (-1, +1)
#                   mapped to a multiplicative cwnd change factor
#
# Online training
# ───────────────
# After every TRAIN_EVERY ticks the LSTM is updated via BPTT on the
# last SEQ steps using a simple reward signal:
#
#     reward = throughput_gain  -  RTT_penalty  -  loss_penalty
#
# Gradients are clipped to ±GRAD_CLIP to keep training stable.
# ─────────────────────────────────────────────
class LSTMCell:
    """Single LSTM cell — all math in numpy."""
    def __init__(self, input_size, hidden_size, rng):
        H, I = hidden_size, input_size
        k = 1.0 / math.sqrt(H)
        def W(r, c): return rng.uniform(-k, k, (r, c))
        # Weight matrices [forget | input | gate | output]
        self.Wf = W(H, I); self.Uf = W(H, H); self.bf = np.zeros(H)
        self.Wi = W(H, I); self.Ui = W(H, H); self.bi = np.zeros(H)
        self.Wg = W(H, I); self.Ug = W(H, H); self.bg = np.zeros(H)
        self.Wo = W(H, I); self.Uo = W(H, H); self.bo = np.zeros(H)
        # Output head: hidden → scalar action
        self.Wy = W(1, H); self.by = np.zeros(1)
        self.H = H

    def forward(self, x, h, c):
        """One step. Returns (h_new, c_new, cache)."""
        f = _sigmoid(self.Wf @ x + self.Uf @ h + self.bf)
        i = _sigmoid(self.Wi @ x + self.Ui @ h + self.bi)
        g = np.tanh(  self.Wg @ x + self.Ug @ h + self.bg)
        o = _sigmoid(self.Wo @ x + self.Uo @ h + self.bo)
        c_new = f * c + i * g
        h_new = o * np.tanh(c_new)
        y = float(np.tanh((self.Wy @ h_new).item() + self.by[0]))
        return h_new, c_new, (x, h, c, f, i, g, o, c_new, h_new, y)

    def output(self, h):
        return float(np.tanh((self.Wy @ h).item() + self.by[0]))

    def backward(self, caches, d_actions, lr=3e-4, grad_clip=0.5):
        """
        BPTT over a sequence of cached steps.
        d_actions : gradient of loss w.r.t. each output action (list, len=SEQ)
        """
        H = self.H
        # Accumulate gradients
        dWf=np.zeros_like(self.Wf); dUf=np.zeros_like(self.Uf); dbf=np.zeros_like(self.bf)
        dWi=np.zeros_like(self.Wi); dUi=np.zeros_like(self.Ui); dbi=np.zeros_like(self.bi)
        dWg=np.zeros_like(self.Wg); dUg=np.zeros_like(self.Ug); dbg=np.zeros_like(self.bg)
        dWo=np.zeros_like(self.Wo); dUo=np.zeros_like(self.Uo); dbo=np.zeros_like(self.bo)
        dWy=np.zeros_like(self.Wy); dby=np.zeros_like(self.by)

        dh_next = np.zeros(H)
        dc_next = np.zeros(H)

        for t in reversed(range(len(caches))):
            x, h, c, f, i_g, g, o, c_new, h_new, y = caches[t]

            # Output layer gradient
            dy = d_actions[t] * (1 - y**2)   # tanh derivative
            dWy += np.outer(dy * np.ones(1), h_new)
            dby += dy * np.ones(1)
            dh = (self.Wy.T * dy).flatten() + dh_next

            # LSTM gates
            tanh_c = np.tanh(c_new)
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c**2) + dc_next
            df = dc * c
            di = dc * g
            dg = dc * i_g
            dc_prev = dc * f

            # Gate pre-activation gradients
            df_pre = df * f * (1 - f)
            di_pre = di * i_g * (1 - i_g)
            dg_pre = dg * (1 - g**2)
            do_pre = do * o * (1 - o)

            for dpre, dW, dU, db, W_, U_ in [
                (df_pre, dWf, dUf, dbf, self.Wf, self.Uf),
                (di_pre, dWi, dUi, dbi, self.Wi, self.Ui),
                (dg_pre, dWg, dUg, dbg, self.Wg, self.Ug),
                (do_pre, dWo, dUo, dbo, self.Wo, self.Uo),
            ]:
                dW += np.outer(dpre, x)
                dU += np.outer(dpre, h)
                db += dpre

            dh_next = (self.Uf.T @ df_pre + self.Ui.T @ di_pre +
                       self.Ug.T @ dg_pre + self.Uo.T @ do_pre)
            dc_next = dc_prev

        # Gradient clipping + simple SGD update
        def apply(param, grad):
            grad = np.clip(grad, -grad_clip, grad_clip)
            param -= lr * grad
        apply(self.Wf, dWf); apply(self.Uf, dUf); apply(self.bf, dbf)
        apply(self.Wi, dWi); apply(self.Ui, dUi); apply(self.bi, dbi)
        apply(self.Wg, dWg); apply(self.Ug, dUg); apply(self.bg, dbg)
        apply(self.Wo, dWo); apply(self.Uo, dUo); apply(self.bo, dbo)
        apply(self.Wy, dWy); apply(self.by, dby)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


# ─────────────────────────────────────────────
# LSTM Congestion Controller
# ─────────────────────────────────────────────
class LSTMCongestionController:
    """
    Online-learning LSTM congestion controller.

    The LSTM observes a sliding window of network state features and
    outputs a multiplicative adjustment to cwnd each tick. After every
    TRAIN_EVERY ticks it performs one BPTT update using a reward that
    rewards throughput while penalising excess RTT and loss.
    """
    INPUT_SIZE   = 5         # features per tick
    HIDDEN_SIZE  = 16
    SEQ_LEN      = 16        # BPTT window length
    TRAIN_EVERY  = 8         # ticks between gradient updates
    MAX_CWND     = 2000.0
    MIN_CWND     = 4.0
    # Reward shaping weights
    W_TPUT       = 1.0
    W_RTT        = -1.5
    W_LOSS       = -3.0

    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.cell = LSTMCell(self.INPUT_SIZE, self.HIDDEN_SIZE, rng)
        self.h = np.zeros(self.HIDDEN_SIZE)
        self.c = np.zeros(self.HIDDEN_SIZE)
        self.caches = []        # rolling BPTT cache
        self.rewards= []        # per-step reward
        self.tick_n = 0
        self.prev_tput = 0.0
        self.prev_rtt = None
        self.train_losses = []  # for diagnostics

    def reset(self):
        self.h = np.zeros(self.HIDDEN_SIZE)
        self.c = np.zeros(self.HIDDEN_SIZE)
        self.caches = []
        self.rewards= []
        self.tick_n = 0
        self.prev_tput = 0.0
        self.prev_rtt = None

    def _featurise(self, state):
        """Normalise raw state into a bounded feature vector."""
        rtt_ratio = (state.rtt_latest / state.rtt_min) - 1.0 if state.rtt_min else 0.0
        loss_rate = state.lost_rate                               # already 0-1
        bw_norm   = min(state.bw_est / 1e8, 1.0)                  # norm to ~1 Gbps
        cwnd_norm = min(state.cwnd / self.MAX_CWND, 1.0)
        # Proxy for queue occupancy
        bdp_est = state.bw_est * (state.rtt_min or 40) / 1000.0 / 8.0 / 1500.0
        queued_norm = min(max(state.cwnd - bdp_est, 0) / max(bdp_est, 1), 1.0)
        return np.array([rtt_ratio, loss_rate, bw_norm, cwnd_norm, queued_norm],
                        dtype=np.float64)

    def _reward(self, state):
        """Scalar reward for the current tick."""
        tput_norm = state.bw_est / 1e8                            # normalised throughput
        rtt_ratio = (state.rtt_latest / state.rtt_min) - 1.0 if state.rtt_min else 0.0
        r = (self.W_TPUT * tput_norm
             + self.W_RTT  * rtt_ratio
             + self.W_LOSS * state.lost_rate)
        return float(r)

    def __call__(self, state):
        """Called each tick. Returns {'cwnd': new_cwnd}."""
        x = self._featurise(state)
        h_new, c_new, cache = self.cell.forward(x, self.h, self.c)
        action = cache[-1]                                        # tanh output in (-1, +1)
        self.h, self.c = h_new, c_new
        self.caches.append(cache)
        self.rewards.append(self._reward(state))
        self.tick_n += 1

        # Keep rolling window
        if len(self.caches) > self.SEQ_LEN:
            self.caches.pop(0)
            self.rewards.pop(0)

        # BPTT update every TRAIN_EVERY ticks (once we have a full window)
        if self.tick_n % self.TRAIN_EVERY == 0 and len(self.caches) == self.SEQ_LEN:
            self._train()

        # Map action → cwnd multiplier: action ∈ (-1,+1) → factor ∈ (0.7, 1.3)
        factor = 1.0 + action * 0.30
        cwnd = float(np.clip(state.cwnd * factor, self.MIN_CWND, self.MAX_CWND))
        return {'cwnd': cwnd}

    def _train(self):
        """One BPTT pass over the current window."""
        rewards = np.array(self.rewards, dtype=np.float64)
        if rewards.std() > 1e-6:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        d_actions = list(-rewards)                                # negative because we maximise
        self.cell.backward(self.caches, d_actions, lr=3e-4, grad_clip=0.5)
        self.train_losses.append(float(-rewards.mean()))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    lstm_ctrl = LSTMCongestionController(seed=42)

    # Wrap so SimGUI can reset the LSTM state on Reset
    _orig_reset = SimState.reset
    def _patched_reset(self):
        _orig_reset(self)
        lstm_ctrl.reset()
    SimState.reset = _patched_reset

    gui = SimGUI()
    gui.custom_fn = lstm_ctrl
