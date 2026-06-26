"""
Adaptive Filter Designer — LMS & RLS Algorithms
=================================================
Interactive adaptive filter tool showing filter coefficients updating in real
time as the algorithm converges on a signal. Covers the two most important
adaptive filtering algorithms:

  LMS  — Least Mean Squares   (simple, robust, widely used)
  NLMS — Normalized LMS       (step-size auto-scaled by input power)
  RLS  — Recursive Least Squares (fast convergence, higher complexity)

Use-cases demonstrated:
  1. System identification  — unknown system modelling
  2. Noise cancellation     — remove correlated noise from a signal
  3. Echo cancellation      — remove a delayed echo
  4. Channel equalisation   — undo a known channel distortion

Controls:
  Algorithm, filter order, step size (LMS/NLMS), forgetting factor (RLS),
  scenario, signal type, SNR, animation speed

Plots (live):
  • Error signal (time)      • Coefficient evolution (animated)
  • Learning curve (MSE)     • Frequency response of adapted filter
  • Desired vs output        • Coefficient bar chart (current snapshot)

Requirements:
  pip install scipy matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy import signal


# ─── Colour palette (consistent with the rest of the toolkit) ────────────────

BG_DARK  = '#0d0d1a'
BG_PANEL = '#1a1a2e'
BG_CTRL  = '#0d0d1a'
GRID_C   = '#2a2a3e'
TEXT_PRI = '#B4B2A9'
TEXT_SEC = '#888780'
SPINE_C  = '#2a2a3e'

C_DESIRED  = '#378ADD'   # blue
C_OUTPUT   = '#F5A623'   # amber
C_ERROR    = '#E24B4A'   # red
C_MSE      = '#1D9E75'   # green
C_COEFF    = '#C77DFF'   # violet
C_FREQRESP = '#4CAF82'   # teal
C_REF      = '#D85A30'   # orange


# ─── Constants ────────────────────────────────────────────────────────────────

ALGO_NAMES     = ['LMS', 'NLMS', 'RLS']
SCENARIO_NAMES = ['System ID', 'Noise Cancel', 'Echo Cancel', 'Equalisation']
SIG_NAMES      = ['White noise', 'Sine 440 Hz', 'Chirp', 'Speech-like']

N_SAMPLES   = 2000    # total samples per run
ANIM_CHUNK  = 20      # samples processed per animation frame


# ─── Signal generators ────────────────────────────────────────────────────────

def gen_signal(kind, N, fs=8000.0):
    t = np.arange(N) / fs
    if kind == 'White noise':
        return np.random.randn(N)
    elif kind == 'Sine 440 Hz':
        return np.sin(2 * np.pi * 440 * t)
    elif kind == 'Chirp':
        return signal.chirp(t, f0=50, f1=fs / 2 - 50, t1=t[-1], method='linear')
    elif kind == 'Speech-like':
        # Sum of harmonics with varying envelope
        env = np.abs(np.sin(2 * np.pi * 3 * t))
        return env * (0.6 * np.sin(2 * np.pi * 120 * t) +
                      0.3 * np.sin(2 * np.pi * 240 * t) +
                      0.1 * np.sin(2 * np.pi * 480 * t)) + 0.05 * np.random.randn(N)
    return np.random.randn(N)


def make_unknown_system(order=8, kind='lowpass'):
    """Generate a random FIR system to identify."""
    rng = np.random.default_rng(42)
    h = rng.standard_normal(order)
    # Shape it to look like a lowpass
    w = signal.windows.hann(order)
    h = h * w
    h /= np.sum(np.abs(h)) + 1e-12
    return h


# ─── Scenario builders ────────────────────────────────────────────────────────

def build_scenario(name, sig_kind, N, fs, order, snr_db):
    """
    Return (x, d, true_w) — reference input x, desired signal d,
    and the true weights (if known) for comparison.

    x : reference / input to the adaptive filter
    d : desired signal the filter tries to match
    """
    rng   = np.random.default_rng(0)
    x_raw = gen_signal(sig_kind, N, fs)
    noise_var = 10 ** (-snr_db / 10)

    if name == 'System ID':
        # Unknown system h_true applied to x; filter learns h_true
        h_true = make_unknown_system(order)
        # Zero-pad h_true to match order if needed
        h_pad  = np.zeros(order)
        h_pad[:len(h_true)] = h_true
        d = np.convolve(x_raw, h_true, mode='full')[:N]
        d += np.sqrt(noise_var) * rng.standard_normal(N)
        return x_raw, d, h_pad

    elif name == 'Noise Cancel':
        # d = clean signal + noise; x = reference noise (correlated)
        clean   = np.sin(2 * np.pi * 440 * np.arange(N) / fs)
        noise   = 0.5 * x_raw                       # primary noise
        d       = clean + noise                      # noisy observation
        # Reference: same noise, slightly delayed/filtered
        h_path  = np.array([0.8, 0.3, -0.1, 0.05])
        h_pad   = np.zeros(order)
        h_pad[:len(h_path)] = h_path
        x_ref   = np.convolve(x_raw, h_path, mode='full')[:N]
        return x_ref, d, h_pad

    elif name == 'Echo Cancel':
        # d = speech + echo; x = far-end speech
        speech  = gen_signal('Speech-like', N, fs)
        delay   = min(order // 2, 20)
        echo_h  = np.zeros(order)
        echo_h[delay] = 0.6
        if delay + 5 < order:
            echo_h[delay + 5] = 0.2
        x_far   = speech.copy()
        echo    = np.convolve(x_far, echo_h, mode='full')[:N]
        d       = speech + echo + np.sqrt(noise_var) * rng.standard_normal(N)
        return x_far, d, echo_h

    elif name == 'Equalisation':
        # Channel distorts x; filter learns inverse channel
        channel = np.array([1.0, 0.5, -0.25, 0.1])
        ch_pad  = np.zeros(order)
        ch_pad[:len(channel)] = channel
        distorted = np.convolve(x_raw, channel, mode='full')[:N]
        # Desired = original delayed by order//2
        delay = order // 2
        d_sig = np.zeros(N)
        d_sig[delay:] = x_raw[:N - delay]
        return distorted, d_sig, ch_pad

    return x_raw, x_raw, np.zeros(order)


# ─── Adaptive algorithms ──────────────────────────────────────────────────────

def run_lms(x, d, order, mu):
    """
    LMS — Least Mean Squares.
    Returns (output, error, weights_history) where weights_history[n] = w at step n.
    """
    N  = len(x)
    w  = np.zeros(order)
    y  = np.zeros(N)
    e  = np.zeros(N)
    wh = np.zeros((N, order))   # coefficient history

    buf = np.zeros(order)       # circular input buffer

    for n in range(N):
        # Shift buffer
        buf = np.roll(buf, 1)
        buf[0] = x[n]
        # Filter output
        y[n] = np.dot(w, buf)
        # Error
        e[n] = d[n] - y[n]
        # Weight update
        w = w + mu * e[n] * buf
        wh[n] = w.copy()

    return y, e, wh


def run_nlms(x, d, order, mu, eps=1e-6):
    """
    NLMS — Normalized LMS. Step size scaled by input power.
    """
    N  = len(x)
    w  = np.zeros(order)
    y  = np.zeros(N)
    e  = np.zeros(N)
    wh = np.zeros((N, order))

    buf = np.zeros(order)

    for n in range(N):
        buf = np.roll(buf, 1)
        buf[0] = x[n]
        y[n] = np.dot(w, buf)
        e[n] = d[n] - y[n]
        # Normalise step size by input power
        power = np.dot(buf, buf) + eps
        w = w + (mu / power) * e[n] * buf
        wh[n] = w.copy()

    return y, e, wh


def run_rls(x, d, order, lam, delta=1.0):
    """
    RLS — Recursive Least Squares.
    lam : forgetting factor (0 < lam <= 1, typically 0.95–0.999)
    delta : initial P matrix scaling (P = delta * I)
    """
    N   = len(x)
    w   = np.zeros(order)
    P   = delta * np.eye(order)   # inverse correlation matrix
    y   = np.zeros(N)
    e   = np.zeros(N)
    wh  = np.zeros((N, order))

    buf = np.zeros(order)

    for n in range(N):
        buf = np.roll(buf, 1)
        buf[0] = x[n]

        # A priori error
        y[n] = np.dot(w, buf)
        e[n] = d[n] - y[n]

        # Kalman gain
        Pu   = P @ buf
        denom= lam + buf @ Pu
        k    = Pu / (denom + 1e-15)

        # Weight update
        w    = w + k * e[n]

        # P update
        P    = (P - np.outer(k, buf @ P)) / lam

        wh[n] = w.copy()

    return y, e, wh


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_hz(f):
    return f'{f/1000:.1f} kHz' if f >= 1000 else f'{f:.0f} Hz'

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.grid(True, color=GRID_C, linewidth=0.5)
    if title:  ax.set_title(title,  color=TEXT_PRI, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=8)

def moving_average(x, w=50):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


# ─── Main GUI ─────────────────────────────────────────────────────────────────

class AdaptiveFilterDesigner:
    """
    Interactive adaptive filter GUI.
    Runs the algorithm in full, then replays coefficient evolution as animation.
    """

    def __init__(self):
        # Parameters
        self.algo      = 'LMS'
        self.scenario  = 'System ID'
        self.sig_kind  = 'White noise'
        self.order     = 16
        self.mu        = 0.01     # LMS/NLMS step size
        self.lam       = 0.98     # RLS forgetting factor
        self.snr_db    = 20.0
        self.fs        = 8000.0
        self.anim_speed= 3        # frames skip factor

        # Results (computed before animation)
        self.x  = None
        self.d  = None
        self.y  = None
        self.e  = None
        self.wh = None            # (N, order) weight history
        self.tw = None            # true weights

        # Animation state
        self._anim    = None
        self._frame   = 0
        self._running = False

        self._build_gui()
        self._run_algorithm()
        plt.show()

    # ── GUI build ─────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(16, 9.5), facecolor=BG_PANEL)
        self.fig.canvas.manager.set_window_title(
            'Adaptive Filter Designer — LMS / NLMS / RLS')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3.4], wspace=0.04,
            left=0.01, right=0.99, top=0.97, bottom=0.04)

        # ── Left controls ─────────────────────────────────────────────────
        cgs = gridspec.GridSpecFromSubplotSpec(
            14, 1, subplot_spec=outer[0], hspace=0.55)

        ax_algo  = self.fig.add_subplot(cgs[0:3])
        ax_scen  = self.fig.add_subplot(cgs[3:6])
        ax_sig   = self.fig.add_subplot(cgs[6:9])
        ax_ord   = self.fig.add_subplot(cgs[9])
        ax_mu    = self.fig.add_subplot(cgs[10])
        ax_lam   = self.fig.add_subplot(cgs[11])
        ax_snr   = self.fig.add_subplot(cgs[12])
        ax_btns  = self.fig.add_subplot(cgs[13])

        for ax in [ax_algo, ax_scen, ax_sig, ax_ord,
                   ax_mu, ax_lam, ax_snr, ax_btns]:
            ax.set_facecolor(BG_CTRL)

        # Algorithm radio
        self.radio_algo = RadioButtons(
            ax_algo, ALGO_NAMES, active=0, activecolor=C_MSE)
        ax_algo.set_title('Algorithm', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_algo.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(9)
        self.radio_algo.on_clicked(self._on_algo)

        # Scenario radio
        self.radio_scen = RadioButtons(
            ax_scen, SCENARIO_NAMES, active=0, activecolor=C_DESIRED)
        ax_scen.set_title('Scenario', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_scen.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(9)
        self.radio_scen.on_clicked(self._on_scenario)

        # Signal radio
        self.radio_sig = RadioButtons(
            ax_sig, SIG_NAMES, active=0, activecolor=C_REF)
        ax_sig.set_title('Input signal', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_sig.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(9)
        self.radio_sig.on_clicked(self._on_sig)

        # Sliders
        sc = BG_CTRL
        self.sl_ord  = Slider(ax_ord,  'Filter\norder', 2, 64,
                              valinit=self.order, valstep=2,
                              color=C_COEFF, initcolor='none', facecolor=sc)
        self.sl_mu   = Slider(ax_mu,   'Step size μ\n(LMS/NLMS)',
                              0.0001, 0.5, valinit=self.mu,
                              color=C_MSE, initcolor='none', facecolor=sc)
        self.sl_lam  = Slider(ax_lam,  'Forgetting λ\n(RLS)',
                              0.90, 1.0, valinit=self.lam,
                              color=C_OUTPUT, initcolor='none', facecolor=sc)
        self.sl_snr  = Slider(ax_snr,  'SNR (dB)',
                              0, 40, valinit=self.snr_db, valstep=1,
                              color=C_REF, initcolor='none', facecolor=sc)

        for sl in [self.sl_ord, self.sl_mu, self.sl_lam, self.sl_snr]:
            sl.label.set_color(TEXT_PRI);  sl.label.set_fontsize(8)
            sl.valtext.set_color('#ffffff'); sl.valtext.set_fontsize(8)
            sl.on_changed(self._on_param)

        # Buttons (Play / Pause / Reset)
        ax_btns.axis('off')
        self.btn_run = plt.matplotlib.widgets.Button(
            self.fig.add_axes([0.015, 0.05, 0.09, 0.035]),
            'Run', color=BG_CTRL, hovercolor=BG_PANEL)
        self.btn_run.label.set_color('#4CAF82'); self.btn_run.label.set_fontsize(9)
        self.btn_run.on_clicked(self._on_run)

        self.btn_pause = plt.matplotlib.widgets.Button(
            self.fig.add_axes([0.115, 0.05, 0.09, 0.035]),
            'Pause', color=BG_CTRL, hovercolor=BG_PANEL)
        self.btn_pause.label.set_color(C_OUTPUT); self.btn_pause.label.set_fontsize(9)
        self.btn_pause.on_clicked(self._on_pause)

        self.btn_reset_anim = plt.matplotlib.widgets.Button(
            self.fig.add_axes([0.215, 0.05, 0.09, 0.035]),
            'Replay', color=BG_CTRL, hovercolor=BG_PANEL)
        self.btn_reset_anim.label.set_color(C_COEFF); self.btn_reset_anim.label.set_fontsize(9)
        self.btn_reset_anim.on_clicked(self._on_replay)

        # ── Right: 3×2 plot grid ──────────────────────────────────────────
        pgs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer[1], hspace=0.45, wspace=0.32)

        self.ax_sig_plot = self.fig.add_subplot(pgs[0, 0])
        self.ax_err      = self.fig.add_subplot(pgs[0, 1])
        self.ax_mse      = self.fig.add_subplot(pgs[1, 0])
        self.ax_coeffs   = self.fig.add_subplot(pgs[1, 1])
        self.ax_coeff_ev = self.fig.add_subplot(pgs[2, 0])
        self.ax_freq     = self.fig.add_subplot(pgs[2, 1])

        for ax in [self.ax_sig_plot, self.ax_err, self.ax_mse,
                   self.ax_coeffs, self.ax_coeff_ev, self.ax_freq]:
            _style_ax(ax)

        # Metrics bar
        self.met_ax = self.fig.add_axes([0.34, 0.005, 0.65, 0.028])
        self.met_ax.axis('off')
        self.met_txt = self.met_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            color=TEXT_PRI, fontsize=8, transform=self.met_ax.transAxes)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_algo(self, label):
        self.algo = label
        # Update slider visual cues
        lms_active = label in ('LMS', 'NLMS')
        self.sl_mu.label.set_color(TEXT_PRI if lms_active else TEXT_SEC)
        self.sl_mu.valtext.set_color('#ffffff' if lms_active else TEXT_SEC)
        self.sl_lam.label.set_color(TEXT_PRI if not lms_active else TEXT_SEC)
        self.sl_lam.valtext.set_color('#ffffff' if not lms_active else TEXT_SEC)
        self._run_algorithm()

    def _on_scenario(self, label):
        self.scenario = label
        self._run_algorithm()

    def _on_sig(self, label):
        self.sig_kind = label
        self._run_algorithm()

    def _on_param(self, _):
        self.order   = int(self.sl_ord.val)
        self.mu      = float(self.sl_mu.val)
        self.lam     = float(self.sl_lam.val)
        self.snr_db  = float(self.sl_snr.val)
        self._run_algorithm()

    def _on_run(self, _):
        if not self._running:
            self._start_animation()

    def _on_pause(self, _):
        if self._anim is not None:
            if self._running:
                self._anim.pause()
                self._running = False
            else:
                self._anim.resume()
                self._running = True

    def _on_replay(self, _):
        self._stop_animation()
        self._frame = 0
        self._start_animation()

    # ── Algorithm runner ──────────────────────────────────────────────────

    def _run_algorithm(self):
        """Run the full algorithm upfront, then animate the result."""
        self._stop_animation()

        self.order   = int(self.sl_ord.val)
        self.mu      = float(self.sl_mu.val)
        self.lam     = float(self.sl_lam.val)
        self.snr_db  = float(self.sl_snr.val)

        # Build scenario
        x, d, tw = build_scenario(
            self.scenario, self.sig_kind,
            N_SAMPLES, self.fs, self.order, self.snr_db)

        self.x  = x
        self.d  = d
        self.tw = tw

        # Run algorithm
        if self.algo == 'LMS':
            self.y, self.e, self.wh = run_lms(x, d, self.order, self.mu)
        elif self.algo == 'NLMS':
            self.y, self.e, self.wh = run_nlms(x, d, self.order, self.mu)
        elif self.algo == 'RLS':
            self.y, self.e, self.wh = run_rls(x, d, self.order, self.lam)

        # Draw static plots immediately
        self._draw_static()
        self._frame = 0
        self._start_animation()

    # ── Static plots (drawn once after each run) ──────────────────────────

    def _draw_static(self):
        self._plot_signal()
        self._plot_error()
        self._plot_mse()
        self._update_metrics(len(self.e) - 1)
        self.fig.canvas.draw_idle()

    def _plot_signal(self):
        ax = self.ax_sig_plot
        ax.cla()
        _style_ax(ax, 'Desired vs Filter output',
                  'Sample (n)', 'Amplitude')
        n = np.arange(len(self.d))
        ax.plot(n, self.d, color=C_DESIRED, linewidth=0.8,
                label='Desired d[n]', alpha=0.85)
        ax.plot(n, self.y, color=C_OUTPUT,  linewidth=0.8,
                label='Output y[n]', alpha=0.85)
        ax.legend(fontsize=7, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C, loc='upper right')
        ax.set_xlim(0, len(self.d))

    def _plot_error(self):
        ax = self.ax_err
        ax.cla()
        _style_ax(ax, 'Error signal e[n] = d[n] − y[n]',
                  'Sample (n)', 'Error')
        n = np.arange(len(self.e))
        ax.plot(n, self.e, color=C_ERROR, linewidth=0.7, alpha=0.8)
        ax.fill_between(n, self.e, 0, alpha=0.12, color=C_ERROR)
        ax.axhline(0, color=GRID_C, linewidth=0.6)
        ax.set_xlim(0, len(self.e))

    def _plot_mse(self):
        ax = self.ax_mse
        ax.cla()
        _style_ax(ax, 'Learning curve (MSE)',
                  'Sample (n)', 'MSE (dB)')

        e2  = self.e ** 2
        win = max(10, len(e2) // 40)
        mse_smooth = moving_average(e2, win)
        n_s = np.arange(len(mse_smooth)) + win // 2

        mse_db = 10 * np.log10(mse_smooth + 1e-15)
        ax.plot(n_s, mse_db, color=C_MSE, linewidth=1.2)
        ax.fill_between(n_s, mse_db, mse_db.min() - 5,
                        alpha=0.10, color=C_MSE)
        ax.set_xlim(0, len(self.e))

        # Mark convergence point (when MSE drops within 3 dB of final value)
        final_mse = np.mean(mse_db[-100:])
        conv_mask = mse_db < final_mse + 3
        if conv_mask.any():
            conv_n = n_s[np.argmax(conv_mask)]
            ax.axvline(conv_n, color=C_OUTPUT, linewidth=1,
                       linestyle='--', alpha=0.7)
            ax.text(conv_n, mse_db.max() * 0.95,
                    f'≈conv\nn={conv_n}',
                    color=C_OUTPUT, fontsize=6, ha='left')
        ax.text(0.98, 0.92,
                f'Final MSE: {final_mse:.1f} dB',
                transform=ax.transAxes, ha='right', va='top',
                color=C_MSE, fontsize=7)

    # ── Animation ─────────────────────────────────────────────────────────

    def _start_animation(self):
        interval_ms = max(20, 80 - self.anim_speed * 10)
        self._running = True
        self._anim = animation.FuncAnimation(
            self.fig,
            self._anim_frame,
            frames=range(0, N_SAMPLES, ANIM_CHUNK),
            interval=interval_ms,
            repeat=False,
            blit=False)

    def _stop_animation(self):
        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass
            self._anim = None
        self._running = False

    def _anim_frame(self, frame_idx):
        """Called for each animation frame — updates coefficient plots."""
        n = min(frame_idx + ANIM_CHUNK - 1, N_SAMPLES - 1)
        self._frame = n
        self._plot_coeffs_bar(n)
        self._plot_coeff_evolution(n)
        self._plot_freq_response(n)
        self._update_metrics(n)

    # ── Animated plots ────────────────────────────────────────────────────

    def _plot_coeffs_bar(self, n):
        """Current coefficient snapshot as bar chart."""
        ax = self.ax_coeffs
        ax.cla()
        _style_ax(ax, f'Filter coefficients  (n = {n})',
                  'Tap index', 'Weight value')

        w = self.wh[n]
        idx = np.arange(len(w))

        ax.bar(idx, w, color=C_COEFF, alpha=0.8, width=0.7)

        # Overlay true weights if known
        if self.tw is not None and np.any(self.tw != 0):
            ax.bar(idx, self.tw, color=C_DESIRED, alpha=0.35,
                   width=0.7, label='True weights')
            ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                      edgecolor=SPINE_C)

        ax.axhline(0, color=GRID_C, linewidth=0.6)
        ax.set_xlim(-0.5, len(w) - 0.5)

        # Weight norm
        ax.text(0.98, 0.95,
                f'‖w‖ = {np.linalg.norm(w):.4f}',
                transform=ax.transAxes, ha='right', va='top',
                color=C_COEFF, fontsize=7)

    def _plot_coeff_evolution(self, n):
        """Each coefficient's value over time up to current sample."""
        ax = self.ax_coeff_ev
        ax.cla()
        _style_ax(ax, 'Coefficient evolution over time',
                  'Sample (n)', 'Weight value')

        # Show up to 8 coefficients to avoid clutter
        n_show = min(self.order, 8)
        cmap   = plt.cm.plasma
        colours= [cmap(i / n_show) for i in range(n_show)]

        for k in range(n_show):
            ax.plot(np.arange(n + 1), self.wh[:n + 1, k],
                    color=colours[k], linewidth=0.9,
                    label=f'w[{k}]', alpha=0.85)

        ax.set_xlim(0, N_SAMPLES)
        ax.axhline(0, color=GRID_C, linewidth=0.5)

        ax.legend(fontsize=5, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C, loc='upper right',
                  ncol=2, handlelength=1)

        # Progress bar (dim shade for "future")
        ax.axvspan(n, N_SAMPLES, alpha=0.06, color=TEXT_SEC)

    def _plot_freq_response(self, n):
        """Frequency response of the adapted filter at current step."""
        ax = self.ax_freq
        ax.cla()
        _style_ax(ax, 'Frequency response of adapted filter',
                  'Frequency (Hz)', 'Magnitude (dB)')

        w_curr = self.wh[n]
        if np.all(w_curr == 0):
            return

        freqs, H = signal.freqz(w_curr, worN=1024, fs=self.fs)
        mag_db   = 20 * np.log10(np.abs(H) + 1e-15)

        ax.semilogx(freqs[1:], mag_db[1:], color=C_FREQRESP, linewidth=1.3)
        ax.fill_between(freqs[1:], mag_db[1:], mag_db.min() - 5,
                        alpha=0.08, color=C_FREQRESP)

        # True system frequency response (if available)
        if self.tw is not None and np.any(self.tw != 0):
            _, H_true = signal.freqz(self.tw, worN=1024, fs=self.fs)
            mag_true  = 20 * np.log10(np.abs(H_true) + 1e-15)
            ax.semilogx(freqs[1:], mag_true[1:],
                        color=C_DESIRED, linewidth=1.0,
                        linestyle='--', alpha=0.7, label='True system')
            ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                      edgecolor=SPINE_C)

        ax.set_xlim(freqs[1], self.fs / 2)
        ax.set_ylim(max(mag_db.min() - 5, -80), mag_db.max() + 5)
        ax.axhline(-3, color='#5F5E5A', linewidth=0.7, linestyle=':')

        # Progress label
        pct = 100 * n / N_SAMPLES
        ax.text(0.02, 0.06, f'{pct:.0f}% converged',
                transform=ax.transAxes, color=C_FREQRESP,
                fontsize=7, va='bottom')

    # ── Metrics bar ───────────────────────────────────────────────────────

    def _update_metrics(self, n):
        w_now = self.wh[n]
        e_now = self.e[:n + 1]

        # Instantaneous error power
        inst_mse = np.mean(e_now[-50:] ** 2) if len(e_now) >= 50 else np.mean(e_now ** 2)
        inst_db  = 10 * np.log10(inst_mse + 1e-15)

        # Weight misalignment vs true (if known)
        mis_str = ''
        if self.tw is not None and np.any(self.tw != 0):
            mis = np.linalg.norm(self.tw - w_now) / (np.linalg.norm(self.tw) + 1e-12)
            mis_str = f'   Weight misalignment: {mis:.4f}'

        # Convergence %
        final_mse = np.mean(self.e[-100:] ** 2)
        init_mse  = np.mean(self.e[:50]  ** 2) + 1e-12
        conv_pct  = 100 * (1 - np.clip(inst_mse / init_mse, 0, 1))

        algo_params = (f'μ={self.mu:.4f}' if self.algo in ('LMS', 'NLMS')
                       else f'λ={self.lam:.4f}')

        txt = (
            f'  {self.algo}  {self.scenario}   '
            f'Order: {self.order}   {algo_params}   SNR: {self.snr_db:.0f} dB   |   '
            f'n={n}/{N_SAMPLES}   MSE: {inst_db:.1f} dB   '
            f'Conv: {conv_pct:.1f}%'
            f'{mis_str}'
        )
        self.met_txt.set_text(txt)
        self.fig.canvas.draw_idle()


# ─── Non-interactive API ──────────────────────────────────────────────────────

def run_and_print(algo='LMS', scenario='System ID', sig_kind='White noise',
                  order=16, mu=0.01, lam=0.98, snr_db=20, fs=8000.0):
    """
    Run an adaptive filter and print convergence summary.

    Examples
    --------
    >>> run_and_print('LMS',  'System ID',     order=32, mu=0.005)
    >>> run_and_print('NLMS', 'Noise Cancel',  order=16, mu=0.5)
    >>> run_and_print('RLS',  'Echo Cancel',   order=32, lam=0.97)
    >>> run_and_print('RLS',  'Equalisation',  order=16, lam=0.99)
    """
    x, d, tw = build_scenario(scenario, sig_kind, N_SAMPLES, fs, order, snr_db)

    if algo == 'LMS':
        y, e, wh = run_lms(x, d, order, mu)
        params   = f'μ = {mu}'
    elif algo == 'NLMS':
        y, e, wh = run_nlms(x, d, order, mu)
        params   = f'μ = {mu}'
    elif algo == 'RLS':
        y, e, wh = run_rls(x, d, order, lam)
        params   = f'λ = {lam}'
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    init_mse  = 10 * np.log10(np.mean(e[:50]   ** 2) + 1e-15)
    final_mse = 10 * np.log10(np.mean(e[-100:] ** 2) + 1e-15)
    reduction = init_mse - final_mse

    mis = (np.linalg.norm(tw - wh[-1]) / (np.linalg.norm(tw) + 1e-12)
           if tw is not None and np.any(tw != 0) else float('nan'))

    print('=' * 58)
    print(f' Algorithm  : {algo}   ({params})')
    print(f' Scenario   : {scenario}')
    print(f' Signal     : {sig_kind}')
    print(f' Order      : {order}   SNR: {snr_db} dB')
    print('-' * 58)
    print(f' Init  MSE  : {init_mse:.2f} dB')
    print(f' Final MSE  : {final_mse:.2f} dB')
    print(f' Reduction  : {reduction:.2f} dB')
    print(f' Misalignment: {mis:.4f}')
    print(f' Final w    : {np.round(wh[-1], 5).tolist()}')
    print('=' * 58)
    return y, e, wh


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    AdaptiveFilterDesigner()

    # ── Non-interactive examples (uncomment to use) ──────────────────────
    # run_and_print('LMS',  'System ID',    order=32, mu=0.005)
    # run_and_print('NLMS', 'Noise Cancel', order=16, mu=0.5)
    # run_and_print('RLS',  'Echo Cancel',  order=32, lam=0.97)
    # run_and_print('RLS',  'Equalisation', order=16, lam=0.99)
