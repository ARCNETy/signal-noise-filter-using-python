"""
Highpass Filter Designer
========================
Interactive highpass filter design and analysis tool — similar to MATLAB's
fvtool / fdatool, but in Python using scipy.signal and matplotlib.

Supports:
  - Filter types  : Butterworth, Chebyshev I, Chebyshev II, Elliptic, Bessel
  - Parameters    : order, cutoff frequency, sample rate, ripple, stopband attenuation
  - Plots         : Magnitude (dB), Phase, Step Response, Pole-Zero

Requirements:
  pip install scipy matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy import signal


# ─── Filter design ────────────────────────────────────────────────────────────

def design_filter(ftype, order, cutoff_hz, fs_hz, rp_db=1.0, rs_db=40.0):
    """
    Design a digital highpass filter using bilinear transform.

    Parameters
    ----------
    ftype    : str   – 'butter' | 'cheby1' | 'cheby2' | 'ellip' | 'bessel'
    order    : int   – filter order (1–10)
    cutoff_hz: float – -3 dB cutoff frequency in Hz
    fs_hz    : float – sample rate in Hz
    rp_db    : float – passband ripple in dB  (Chebyshev I / Elliptic)
    rs_db    : float – stopband attenuation dB (Chebyshev II / Elliptic)

    Returns
    -------
    b, a : array_like – numerator / denominator coefficients (IIR)
    """
    nyq = fs_hz / 2.0
    wn  = cutoff_hz / nyq          # normalised cutoff [0, 1]
    wn  = np.clip(wn, 1e-4, 0.9999)

    if ftype == 'butter':
        b, a = signal.butter(order, wn, btype='high', analog=False)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, rp_db, wn, btype='high', analog=False)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, rs_db, wn, btype='high', analog=False)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, rp_db, rs_db, wn, btype='high', analog=False)
    elif ftype == 'bessel':
        b, a = signal.bessel(order, wn, btype='high', analog=False, norm='phase')
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    return b, a


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def frequency_response(b, a, fs_hz, n_pts=2048):
    """Return (freqs_hz, H_complex)."""
    w, H = signal.freqz(b, a, worN=n_pts, fs=fs_hz)
    return w, H


def group_delay(b, a, fs_hz, n_pts=2048):
    """Return (freqs_hz, gd_samples)."""
    w, gd = signal.group_delay((b, a), w=n_pts, fs=fs_hz)
    return w, gd


def step_response_fn(b, a, fs_hz):
    """Return (t_sec, y) for a unit-step input."""
    n_samples = min(4000, max(300, int(fs_hz * 0.05)))
    x = np.ones(n_samples)
    t = np.arange(n_samples) / fs_hz
    y = signal.lfilter(b, a, x)
    return t, y


def pole_zero(b, a):
    """Return (zeros, poles, gain)."""
    return signal.tf2zpk(b, a)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_hz(f):
    return f'{f/1000:.1f} kHz' if f >= 1000 else f'{f:.0f} Hz'

def _style_ax(ax):
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#888780', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3e')
    ax.grid(True, color='#2a2a3e', linewidth=0.5)


# ─── Main GUI class ───────────────────────────────────────────────────────────

class HighpassFilterDesigner:
    """
    Interactive matplotlib GUI for highpass filter design.
    """

    FILTER_TYPES  = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    FILTER_LABELS = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptic', 'Bessel']

    # Accent colours
    C_MAG    = '#E24B4A'   # red  — passband is high-freq → warm colour
    C_PHASE  = '#1D9E75'   # green
    C_STEP   = '#D85A30'   # orange
    C_CUTOFF = '#BA7517'   # amber
    C_ZERO   = '#378ADD'   # blue
    C_POLE   = '#E24B4A'   # red

    def __init__(self):
        self.ftype  = 'butter'
        self.order  = 4
        self.cutoff = 1000.0   # Hz  — everything BELOW this is attenuated
        self.fs     = 44100.0  # Hz
        self.rp     = 1.0      # dB  passband ripple
        self.rs     = 40.0     # dB  stopband attenuation

        self._build_gui()
        self._update(None)
        plt.show()

    # ── GUI layout ──────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('Highpass Filter Designer')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3], wspace=0.05,
            left=0.02, right=0.98, top=0.96, bottom=0.04)

        # ── Left controls ────────────────────────────────────────────────────
        ctrl_gs = gridspec.GridSpecFromSubplotSpec(
            9, 1, subplot_spec=outer[0], hspace=0.6)

        ax_type  = self.fig.add_subplot(ctrl_gs[0:3])
        ax_ord   = self.fig.add_subplot(ctrl_gs[3])
        ax_fc    = self.fig.add_subplot(ctrl_gs[4])
        ax_fs    = self.fig.add_subplot(ctrl_gs[5])
        ax_rp    = self.fig.add_subplot(ctrl_gs[6])
        ax_rs    = self.fig.add_subplot(ctrl_gs[7])
        ax_reset = self.fig.add_subplot(ctrl_gs[8])

        for ax in [ax_type, ax_ord, ax_fc, ax_fs, ax_rp, ax_rs, ax_reset]:
            ax.set_facecolor('#0d0d1a')

        # Radio — filter type
        self.radio = RadioButtons(ax_type, self.FILTER_LABELS,
                                  active=0, activecolor=self.C_MAG)
        ax_type.set_title('Filter type', color='#B4B2A9', fontsize=9, pad=2)
        for lbl in self.radio.labels:
            lbl.set_color('#B4B2A9')
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_type)

        # Sliders
        sc = '#0d0d1a'
        self.sl_ord = Slider(ax_ord, 'Order',         1, 10,      valinit=self.order,  valstep=1,   color=self.C_MAG,    initcolor='none', facecolor=sc)
        self.sl_fc  = Slider(ax_fc,  'Cutoff\n(Hz)', 10, self.fs/2-1, valinit=self.cutoff,           color=self.C_PHASE,  initcolor='none', facecolor=sc)
        self.sl_fs  = Slider(ax_fs,  'Sample\nrate (Hz)', 8000, 192000, valinit=self.fs, valstep=100, color='#7F77DD',   initcolor='none', facecolor=sc)
        self.sl_rp  = Slider(ax_rp,  'Ripple\n(dB)', 0.1, 6.0,  valinit=self.rp,                    color=self.C_STEP,   initcolor='none', facecolor=sc)
        self.sl_rs  = Slider(ax_rs,  'Stop\natten (dB)', 10, 80, valinit=self.rs, valstep=1,          color=self.C_CUTOFF, initcolor='none', facecolor=sc)

        for sl in [self.sl_ord, self.sl_fc, self.sl_fs, self.sl_rp, self.sl_rs]:
            sl.label.set_color('#B4B2A9');  sl.label.set_fontsize(8)
            sl.valtext.set_color('#ffffff'); sl.valtext.set_fontsize(8)
            sl.on_changed(self._update)

        # Reset button
        self.btn_reset = Button(ax_reset, 'Reset defaults',
                                color='#0d0d1a', hovercolor='#1a1a2e')
        self.btn_reset.label.set_color('#B4B2A9')
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self._reset)

        # ── Right plots (2 × 2) ──────────────────────────────────────────────
        plot_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[1], hspace=0.4, wspace=0.35)

        self.ax_mag  = self.fig.add_subplot(plot_gs[0, 0])
        self.ax_ph   = self.fig.add_subplot(plot_gs[0, 1])
        self.ax_step = self.fig.add_subplot(plot_gs[1, 0])
        self.ax_pz   = self.fig.add_subplot(plot_gs[1, 1])

        for ax in [self.ax_mag, self.ax_ph, self.ax_step, self.ax_pz]:
            _style_ax(ax)

        # Metrics bar
        self.metrics_ax = self.fig.add_axes([0.35, 0.005, 0.63, 0.03])
        self.metrics_ax.axis('off')
        self.metrics_ax.set_facecolor('#0d0d1a')
        self.metrics_txt = self.metrics_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            color='#B4B2A9', fontsize=8, transform=self.metrics_ax.transAxes)

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_type(self, label):
        self.ftype = self.FILTER_TYPES[self.FILTER_LABELS.index(label)]
        self._update(None)

    def _reset(self, _):
        for sl in [self.sl_ord, self.sl_fc, self.sl_fs, self.sl_rp, self.sl_rs]:
            sl.reset()

    def _update(self, _):
        self.order  = int(self.sl_ord.val)
        self.cutoff = float(self.sl_fc.val)
        self.fs     = float(self.sl_fs.val)
        self.rp     = float(self.sl_rp.val)
        self.rs     = float(self.sl_rs.val)

        nyq = self.fs / 2.0
        if self.cutoff >= nyq:
            self.cutoff = nyq * 0.99
            self.sl_fc.set_val(self.cutoff)
        self.sl_fc.valmax = nyq - 1
        self.sl_fc.ax.set_xlim(10, nyq - 1)

        try:
            b, a = design_filter(
                self.ftype, self.order, self.cutoff, self.fs, self.rp, self.rs)
            self._plot_all(b, a)
        except Exception as e:
            print(f"[Filter design error] {e}")

    # ── Plot helpers ─────────────────────────────────────────────────────────

    def _plot_all(self, b, a):
        self._plot_magnitude(b, a)
        self._plot_phase(b, a)
        self._plot_step(b, a)
        self._plot_pz(b, a)
        self._update_metrics(b, a)
        self.fig.canvas.draw_idle()

    def _plot_magnitude(self, b, a):
        ax = self.ax_mag
        ax.cla(); _style_ax(ax)

        w, H   = frequency_response(b, a, self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)

        ax.semilogx(w, mag_db, color=self.C_MAG, linewidth=1.5)
        ax.axvline(self.cutoff, color=self.C_CUTOFF, linewidth=1, linestyle='--', alpha=0.8)
        ax.axhline(-3, color='#5F5E5A', linewidth=0.8, linestyle=':')

        # Shade the passband (above cutoff)
        ax.fill_between(w, mag_db, -80,
                        where=(w >= self.cutoff),
                        alpha=0.10, color=self.C_MAG)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(-80, 5)
        ax.set_title('Magnitude response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Magnitude (dB)', color='#888780', fontsize=8)

        # Annotate cutoff
        ax.text(self.cutoff * 1.05, -75,
                f'fc={_fmt_hz(self.cutoff)}', color=self.C_CUTOFF, fontsize=7)

        # Annotate passband / stopband regions
        nyq = self.fs / 2
        ax.text(self.cutoff * 1.5, 2, 'PASSBAND',
                color=self.C_MAG, fontsize=7, alpha=0.6,
                ha='left' if self.cutoff * 1.5 < nyq * 0.85 else 'right')
        if self.cutoff > w[2] * 3:
            ax.text(self.cutoff * 0.5, 2, 'STOPBAND',
                    color='#5F5E5A', fontsize=7, alpha=0.6, ha='right')

    def _plot_phase(self, b, a):
        ax = self.ax_ph
        ax.cla(); _style_ax(ax)

        w, H = frequency_response(b, a, self.fs)
        phase_deg = np.unwrap(np.angle(H)) * 180 / np.pi

        ax.semilogx(w, phase_deg, color=self.C_PHASE, linewidth=1.5)
        ax.axvline(self.cutoff, color=self.C_CUTOFF, linewidth=1,
                   linestyle='--', alpha=0.8)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_title('Phase response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Phase (°)', color='#888780', fontsize=8)

    def _plot_step(self, b, a):
        ax = self.ax_step
        ax.cla(); _style_ax(ax)

        t, y = step_response_fn(b, a, self.fs)
        t_ms = t * 1000

        ax.plot(t_ms, y, color=self.C_STEP, linewidth=1.5)
        ax.axhline(0, color='#5F5E5A', linewidth=0.8, linestyle=':')

        # HPF passes the transient and blocks DC — annotate that
        ax.text(0.97, 0.92, 'DC → 0  (blocked)',
                transform=ax.transAxes, ha='right', va='top',
                color='#5F5E5A', fontsize=7)

        ax.set_title('Step response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Time (ms)', color='#888780', fontsize=8)
        ax.set_ylabel('Amplitude', color='#888780', fontsize=8)

    def _plot_pz(self, b, a):
        ax = self.ax_pz
        ax.cla(); _style_ax(ax)

        z, p, _ = pole_zero(b, a)

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color='#444441', linewidth=0.8)
        ax.axhline(0, color='#444441', linewidth=0.5)
        ax.axvline(0, color='#444441', linewidth=0.5)

        ax.scatter(z.real, z.imag, marker='o', s=60,
                   facecolors='none', edgecolors=self.C_ZERO,
                   linewidths=1.5, label='Zeros', zorder=3)
        ax.scatter(p.real, p.imag, marker='x', s=60,
                   color=self.C_POLE, linewidths=1.5,
                   label='Poles', zorder=3)

        lim = 1.4
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title('Pole-zero plot', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Real', color='#888780', fontsize=8)
        ax.set_ylabel('Imaginary', color='#888780', fontsize=8)
        ax.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e', loc='upper right')

        # HPF zeros cluster at z = +1 (DC), annotate
        n_zeros_at_dc = np.sum(np.abs(z - 1.0) < 0.05)
        if n_zeros_at_dc > 0:
            ax.annotate(f'{n_zeros_at_dc} zero(s)\nat z=+1 (DC null)',
                        xy=(1, 0), xytext=(0.3, -0.9),
                        color='#B4B2A9', fontsize=6,
                        arrowprops=dict(arrowstyle='->', color='#5F5E5A', lw=0.8))

    def _update_metrics(self, b, a):
        w, H   = frequency_response(b, a, self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)

        # Attenuation at fc/2  (one octave below cutoff — should be deep in stopband)
        f_half = max(self.cutoff / 2.0, w[1])
        idx_h  = np.argmin(np.abs(w - f_half))
        att_below = -mag_db[idx_h]

        # Gain at Nyquist (top of passband)
        idx_nyq = -1
        gain_nyq = mag_db[idx_nyq]

        # Phase at cutoff
        idx_fc = np.argmin(np.abs(w - self.cutoff))
        ph_fc  = np.angle(H[idx_fc], deg=True)

        # Theoretical roll-off
        rolloff = 20 * self.order

        txt = (
            f'  Cutoff: {_fmt_hz(self.cutoff)}   |   '
            f'Atten @ fc/2: {att_below:.1f} dB   |   '
            f'Gain @ Nyquist: {gain_nyq:.2f} dB   |   '
            f'Phase @ fc: {ph_fc:.1f}°   |   '
            f'Roll-off: {rolloff} dB/oct'
        )
        self.metrics_txt.set_text(txt)


# ─── Non-interactive API ──────────────────────────────────────────────────────

def design_and_print(ftype='butter', order=4, cutoff_hz=1000,
                     fs_hz=44100, rp_db=1.0, rs_db=40.0):
    """
    Design a highpass filter and print key specs + coefficients.

    Examples
    --------
    >>> b, a = design_and_print('butter', order=4, cutoff_hz=1000, fs_hz=44100)
    >>> b, a = design_and_print('cheby1', order=6, cutoff_hz=500,  fs_hz=48000, rp_db=0.5)
    >>> b, a = design_and_print('ellip',  order=5, cutoff_hz=2000, fs_hz=44100, rp_db=1, rs_db=60)
    """
    b, a = design_filter(ftype, order, cutoff_hz, fs_hz, rp_db, rs_db)

    w, H   = frequency_response(b, a, fs_hz)
    mag_db = 20 * np.log10(np.abs(H) + 1e-20)

    idx_fc   = np.argmin(np.abs(w - cutoff_hz))
    f_half   = max(cutoff_hz / 2.0, w[1])
    idx_half = np.argmin(np.abs(w - f_half))

    print("=" * 55)
    print(f" Filter    : {ftype.upper()} highpass  order {order}")
    print(f" Cutoff    : {_fmt_hz(cutoff_hz)}")
    print(f" Sample    : {_fmt_hz(fs_hz)}")
    if ftype in ('cheby1', 'ellip'):
        print(f" Ripple    : {rp_db} dB")
    if ftype in ('cheby2', 'ellip'):
        print(f" Stop att. : {rs_db} dB")
    print("-" * 55)
    print(f" Mag @ fc  : {mag_db[idx_fc]:.2f} dB")
    print(f" Att @ fc/2: {-mag_db[idx_half]:.2f} dB")
    print(f" Roll-off  : {20*order} dB/oct  (theoretical)")
    print("=" * 55)
    print(f" b = {np.round(b, 6).tolist()}")
    print(f" a = {np.round(a, 6).tolist()}")
    print("=" * 55)
    return b, a


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Launch the interactive GUI
    HighpassFilterDesigner()

    # ── Or use the non-interactive API ──────────────────────────────────────
    # Uncomment to design a filter and get coefficients without the GUI:

    # b, a = design_and_print('butter', order=4, cutoff_hz=1000,  fs_hz=44100)
    # b, a = design_and_print('cheby1', order=6, cutoff_hz=500,   fs_hz=48000, rp_db=0.5)
    # b, a = design_and_print('cheby2', order=8, cutoff_hz=2000,  fs_hz=96000, rs_db=60)
    # b, a = design_and_print('ellip',  order=5, cutoff_hz=3000,  fs_hz=44100, rp_db=1, rs_db=50)
    # b, a = design_and_print('bessel', order=4, cutoff_hz=1000,  fs_hz=44100)
