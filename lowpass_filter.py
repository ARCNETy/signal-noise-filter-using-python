"""
Lowpass Filter
=======================
Interactive lowpass filter design and analysis tool — similar to MATLAB's
fvtool / fdatool, but in Python using scipy.signal and matplotlib.

Supports:
  - Filter types  : Butterworth, Chebyshev I, Chebyshev II, Elliptic, Bessel
  - Parameters    : order, cutoff frequency, sample rate, ripple, stopband attenuation
  - Plots         : Magnitude (dB), Phase, Group Delay, Step Response, Pole-Zero

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
    Design a digital lowpass filter using bilinear transform.

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
        b, a = signal.butter(order, wn, btype='low', analog=False)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, rp_db, wn, btype='low', analog=False)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, rs_db, wn, btype='low', analog=False)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, rp_db, rs_db, wn, btype='low', analog=False)
    elif ftype == 'bessel':
        b, a = signal.bessel(order, wn, btype='low', analog=False, norm='phase')
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


def step_response_fn(b, a, fs_hz, n_periods=10):
    """Return (t_sec, y) for a unit-step input."""
    # Enough samples to capture transient
    fc_approx = fs_hz * 0.1          # rough period estimate
    n_samples  = max(200, int(n_periods * fs_hz / max(fc_approx, 1)))
    n_samples  = min(n_samples, 4000)
    x = np.ones(n_samples)
    t = np.arange(n_samples) / fs_hz
    _, y = signal.lfilter(b, a, x, zi=None), None
    y    = signal.lfilter(b, a, x)
    return t, y


def pole_zero(b, a):
    """Return (zeros, poles, gain)."""
    return signal.tf2zpk(b, a)


# ─── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    'mag'  : '#378ADD',
    'phase': '#1D9E75',
    'gd'   : '#D85A30',
    'step' : '#7F77DD',
    'pole' : '#E24B4A',
    'zero' : '#378ADD',
    'cutoff': '#BA7517',
    'passband': 'rgba(55,138,221,0.08)',
}

def _fmt_hz(f):
    return f'{f/1000:.1f} kHz' if f >= 1000 else f'{f:.0f} Hz'


class FilterDesigner:
    """
    Interactive matplotlib GUI for lowpass filter design.
    """

    FILTER_TYPES = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    FILTER_LABELS = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptic', 'Bessel']

    def __init__(self):
        # ── Default parameters ──────────────────────────────────────────────
        self.ftype    = 'butter'
        self.order    = 4
        self.cutoff   = 1000.0    # Hz
        self.fs       = 44100.0   # Hz
        self.rp       = 1.0       # dB  (Cheby1 / Elliptic passband ripple)
        self.rs       = 40.0      # dB  (Cheby2 / Elliptic stopband atten.)

        self._build_gui()
        self._update(None)
        plt.show()

    # ── GUI layout ──────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('Lowpass Filter Designer')

        # Outer grid: left controls | right plots
        outer = gridspec.GridSpec(1, 2, width_ratios=[1, 3], wspace=0.05,
                                  left=0.02, right=0.98, top=0.96, bottom=0.04)

        # ── Left panel ───────────────────────────────────────────────────────
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

        # Radio buttons — filter type
        self.radio = RadioButtons(
            ax_type, self.FILTER_LABELS,
            active=0,
            activecolor='#378ADD',
        )
        ax_type.set_title('Filter type', color='#B4B2A9', fontsize=9, pad=2)
        for lbl in self.radio.labels:
            lbl.set_color('#B4B2A9')
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_type)

        # Sliders
        sc = '#0d0d1a'
        self.sl_ord = Slider(ax_ord,  'Order',   1, 10,  valinit=self.order,  valstep=1,  color='#378ADD', initcolor='none', facecolor=sc)
        self.sl_fc  = Slider(ax_fc,   'Cutoff\n(Hz)', 10, self.fs/2-1, valinit=self.cutoff, color='#1D9E75', initcolor='none', facecolor=sc)
        self.sl_fs  = Slider(ax_fs,   'Sample\nrate (Hz)', 8000, 192000, valinit=self.fs,  valstep=100, color='#7F77DD', initcolor='none', facecolor=sc)
        self.sl_rp  = Slider(ax_rp,   'Ripple\n(dB)',  0.1, 6.0, valinit=self.rp,  color='#D85A30', initcolor='none', facecolor=sc)
        self.sl_rs  = Slider(ax_rs,   'Stop\natten (dB)', 10, 80, valinit=self.rs, valstep=1, color='#BA7517', initcolor='none', facecolor=sc)

        for sl in [self.sl_ord, self.sl_fc, self.sl_fs, self.sl_rp, self.sl_rs]:
            sl.label.set_color('#B4B2A9')
            sl.label.set_fontsize(8)
            sl.valtext.set_color('#ffffff')
            sl.valtext.set_fontsize(8)
            sl.on_changed(self._update)

        # Reset button
        self.btn_reset = Button(ax_reset, 'Reset defaults',
                                color='#0d0d1a', hovercolor='#1a1a2e')
        self.btn_reset.label.set_color('#B4B2A9')
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self._reset)

        # ── Right panel — 4 plots ────────────────────────────────────────────
        plot_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[1], hspace=0.4, wspace=0.35)

        self.ax_mag  = self.fig.add_subplot(plot_gs[0, 0])
        self.ax_ph   = self.fig.add_subplot(plot_gs[0, 1])
        self.ax_step = self.fig.add_subplot(plot_gs[1, 0])
        self.ax_pz   = self.fig.add_subplot(plot_gs[1, 1])

        bg = '#0d0d1a'
        grid_c = '#2a2a3e'
        for ax in [self.ax_mag, self.ax_ph, self.ax_step, self.ax_pz]:
            ax.set_facecolor(bg)
            ax.tick_params(colors='#888780', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#2a2a3e')
            ax.grid(True, color=grid_c, linewidth=0.5)

        self.ax_mag.set_title('Magnitude response', color='#B4B2A9', fontsize=9)
        self.ax_ph.set_title('Phase response', color='#B4B2A9', fontsize=9)
        self.ax_step.set_title('Step response', color='#B4B2A9', fontsize=9)
        self.ax_pz.set_title('Pole-zero plot', color='#B4B2A9', fontsize=9)

        self.ax_mag.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        self.ax_mag.set_ylabel('Magnitude (dB)', color='#888780', fontsize=8)
        self.ax_ph.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        self.ax_ph.set_ylabel('Phase (°)', color='#888780', fontsize=8)
        self.ax_step.set_xlabel('Time (ms)', color='#888780', fontsize=8)
        self.ax_step.set_ylabel('Amplitude', color='#888780', fontsize=8)
        self.ax_pz.set_xlabel('Real', color='#888780', fontsize=8)
        self.ax_pz.set_ylabel('Imaginary', color='#888780', fontsize=8)

        # Metrics text
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
        self.sl_ord.reset()
        self.sl_fc.reset()
        self.sl_fs.reset()
        self.sl_rp.reset()
        self.sl_rs.reset()

    def _update(self, _):
        self.order  = int(self.sl_ord.val)
        self.cutoff = float(self.sl_fc.val)
        self.fs     = float(self.sl_fs.val)
        self.rp     = float(self.sl_rp.val)
        self.rs     = float(self.sl_rs.val)

        # Keep cutoff below Nyquist
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

    # ── Plotting helpers ─────────────────────────────────────────────────────

    def _plot_all(self, b, a):
        self._plot_magnitude(b, a)
        self._plot_phase(b, a)
        self._plot_step(b, a)
        self._plot_pz(b, a)
        self._update_metrics(b, a)
        self.fig.canvas.draw_idle()

    def _plot_magnitude(self, b, a):
        ax = self.ax_mag
        ax.cla()
        ax.set_facecolor('#0d0d1a')
        ax.grid(True, color='#2a2a3e', linewidth=0.5)

        w, H = frequency_response(b, a, self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)

        ax.semilogx(w, mag_db, color='#378ADD', linewidth=1.5)
        ax.axvline(self.cutoff, color='#BA7517', linewidth=1, linestyle='--', alpha=0.8)
        ax.axhline(-3, color='#5F5E5A', linewidth=0.8, linestyle=':')
        ax.fill_between(w, mag_db, -120, alpha=0.08, color='#378ADD')

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(-80, 5)
        ax.set_title('Magnitude response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Magnitude (dB)', color='#888780', fontsize=8)
        ax.tick_params(colors='#888780', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a3e')

        ax.text(self.cutoff * 1.05, -75, f'fc={_fmt_hz(self.cutoff)}',
                color='#BA7517', fontsize=7)

    def _plot_phase(self, b, a):
        ax = self.ax_ph
        ax.cla()
        ax.set_facecolor('#0d0d1a')
        ax.grid(True, color='#2a2a3e', linewidth=0.5)

        w, H = frequency_response(b, a, self.fs)
        phase_deg = np.angle(H, deg=True)
        phase_deg = np.unwrap(np.deg2rad(phase_deg)) * 180 / np.pi

        ax.semilogx(w, phase_deg, color='#1D9E75', linewidth=1.5)
        ax.axvline(self.cutoff, color='#BA7517', linewidth=1, linestyle='--', alpha=0.8)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_title('Phase response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Phase (°)', color='#888780', fontsize=8)
        ax.tick_params(colors='#888780', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a3e')

    def _plot_step(self, b, a):
        ax = self.ax_step
        ax.cla()
        ax.set_facecolor('#0d0d1a')
        ax.grid(True, color='#2a2a3e', linewidth=0.5)

        t, y = step_response_fn(b, a, self.fs)
        t_ms = t * 1000

        ax.plot(t_ms, y, color='#7F77DD', linewidth=1.5)
        ax.axhline(1.0, color='#5F5E5A', linewidth=0.8, linestyle=':')

        ax.set_title('Step response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Time (ms)', color='#888780', fontsize=8)
        ax.set_ylabel('Amplitude', color='#888780', fontsize=8)
        ax.tick_params(colors='#888780', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a3e')

    def _plot_pz(self, b, a):
        ax = self.ax_pz
        ax.cla()
        ax.set_facecolor('#0d0d1a')
        ax.grid(True, color='#2a2a3e', linewidth=0.5)

        z, p, _ = pole_zero(b, a)

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color='#444441', linewidth=0.8)
        ax.axhline(0, color='#444441', linewidth=0.5)
        ax.axvline(0, color='#444441', linewidth=0.5)

        ax.scatter(z.real, z.imag, marker='o', s=60,
                   facecolors='none', edgecolors='#378ADD', linewidths=1.5, label='Zeros')
        ax.scatter(p.real, p.imag, marker='x', s=60,
                   color='#E24B4A', linewidths=1.5, label='Poles')

        lim = 1.4
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title('Pole-zero plot', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Real', color='#888780', fontsize=8)
        ax.set_ylabel('Imaginary', color='#888780', fontsize=8)
        ax.tick_params(colors='#888780', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a3e')
        ax.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e', loc='upper right')

    def _update_metrics(self, b, a):
        w, H = frequency_response(b, a, self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)

        # Attenuation at 2× cutoff
        idx = np.argmin(np.abs(w - min(self.cutoff * 2, self.fs / 2 - 1)))
        att = -mag_db[idx]

        # Roll-off (theoretical)
        rolloff = 20 * self.order

        # Phase at cutoff
        idx_fc = np.argmin(np.abs(w - self.cutoff))
        phase_at_fc = np.angle(H[idx_fc], deg=True)

        # Group delay at DC
        _, gd = group_delay(b, a, self.fs)
        gd_dc = gd[0] / self.fs * 1000  # ms

        txt = (
            f'  Cutoff: {_fmt_hz(self.cutoff)}   |   '
            f'Atten @ 2×fc: {att:.1f} dB   |   '
            f'Roll-off: {rolloff} dB/oct   |   '
            f'Phase @ fc: {phase_at_fc:.1f}°   |   '
            f'Group delay (DC): {gd_dc:.2f} ms'
        )
        self.metrics_txt.set_text(txt)


# ─── Standalone / scripted usage ──────────────────────────────────────────────

def design_and_print(ftype='butter', order=4, cutoff_hz=1000,
                     fs_hz=44100, rp_db=1.0, rs_db=40.0):
    """
    Non-interactive helper: design a filter and print key specs.

    Example
    -------
    >>> b, a = design_and_print('cheby1', order=6, cutoff_hz=2000, fs_hz=48000, rp_db=0.5)
    """
    b, a = design_filter(ftype, order, cutoff_hz, fs_hz, rp_db, rs_db)

    w, H = frequency_response(b, a, fs_hz)
    mag_db = 20 * np.log10(np.abs(H) + 1e-20)

    idx_fc  = np.argmin(np.abs(w - cutoff_hz))
    idx_2fc = np.argmin(np.abs(w - min(cutoff_hz * 2, fs_hz / 2 - 1)))

    print("=" * 55)
    print(f" Filter   : {ftype.upper()} order {order}")
    print(f" Cutoff   : {_fmt_hz(cutoff_hz)}")
    print(f" Sample   : {_fmt_hz(fs_hz)}")
    if ftype in ('cheby1', 'ellip'):
        print(f" Ripple   : {rp_db} dB")
    if ftype in ('cheby2', 'ellip'):
        print(f" Stop att.: {rs_db} dB")
    print("-" * 55)
    print(f" Mag @ fc : {mag_db[idx_fc]:.2f} dB")
    print(f" Att @ 2fc: {-mag_db[idx_2fc]:.2f} dB")
    print(f" Roll-off : {20*order} dB/oct  (theoretical)")
    print("=" * 55)
    print(f" b = {np.round(b, 6).tolist()}")
    print(f" a = {np.round(a, 6).tolist()}")
    print("=" * 55)
    return b, a


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Launch the interactive GUI
    FilterDesigner()

    # ── Or use the non-interactive API directly ──────────────────────────────
    # Uncomment any of the lines below to design a filter without the GUI:

    # b, a = design_and_print('butter',  order=4, cutoff_hz=1000,  fs_hz=44100)
    # b, a = design_and_print('cheby1',  order=6, cutoff_hz=2000,  fs_hz=48000, rp_db=0.5)
    # b, a = design_and_print('cheby2',  order=8, cutoff_hz=5000,  fs_hz=96000, rs_db=60)
    # b, a = design_and_print('ellip',   order=5, cutoff_hz=3000,  fs_hz=44100, rp_db=1, rs_db=50)
    # b, a = design_and_print('bessel',  order=4, cutoff_hz=1000,  fs_hz=44100)
