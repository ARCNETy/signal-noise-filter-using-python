"""
Bandpass Filter Designer
========================
Interactive bandpass filter design and analysis tool — similar to MATLAB's
fvtool / fdatool, but in Python using scipy.signal and matplotlib.

Supports:
  - Filter types  : Butterworth, Chebyshev I, Chebyshev II, Elliptic, Bessel
  - Parameters    : order, low/high cutoff frequencies, sample rate,
                    passband ripple, stopband attenuation
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

def design_filter(ftype, order, f_low, f_high, fs_hz, rp_db=1.0, rs_db=40.0):
    """
    Design a digital bandpass filter using bilinear transform.

    Parameters
    ----------
    ftype   : str   – 'butter' | 'cheby1' | 'cheby2' | 'ellip' | 'bessel'
    order   : int   – per-band order (total filter order = 2 × order)
    f_low   : float – lower -3 dB cutoff frequency in Hz
    f_high  : float – upper -3 dB cutoff frequency in Hz
    fs_hz   : float – sample rate in Hz
    rp_db   : float – passband ripple in dB  (Chebyshev I / Elliptic)
    rs_db   : float – stopband attenuation dB (Chebyshev II / Elliptic)

    Returns
    -------
    b, a : array_like – numerator / denominator coefficients (IIR)
    """
    nyq  = fs_hz / 2.0
    wl   = np.clip(f_low  / nyq, 1e-4, 0.9998)
    wh   = np.clip(f_high / nyq, wl + 1e-4, 0.9999)
    Wn   = [wl, wh]

    if ftype == 'butter':
        b, a = signal.butter(order, Wn, btype='band', analog=False)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, rp_db, Wn, btype='band', analog=False)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, rs_db, Wn, btype='band', analog=False)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, rp_db, rs_db, Wn, btype='band', analog=False)
    elif ftype == 'bessel':
        b, a = signal.bessel(order, Wn, btype='band', analog=False, norm='phase')
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    return b, a


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def frequency_response(b, a, fs_hz, n_pts=2048):
    w, H = signal.freqz(b, a, worN=n_pts, fs=fs_hz)
    return w, H


def group_delay(b, a, fs_hz, n_pts=2048):
    w, gd = signal.group_delay((b, a), w=n_pts, fs=fs_hz)
    return w, gd


def step_response_fn(b, a, fs_hz):
    n_samples = min(4000, max(300, int(fs_hz * 0.05)))
    x = np.ones(n_samples)
    t = np.arange(n_samples) / fs_hz
    y = signal.lfilter(b, a, x)
    return t, y


def pole_zero(b, a):
    return signal.tf2zpk(b, a)


# ─── Shared style helper ──────────────────────────────────────────────────────

def _fmt_hz(f):
    return f'{f/1000:.2f} kHz' if f >= 1000 else f'{f:.0f} Hz'

def _style_ax(ax):
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#888780', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3e')
    ax.grid(True, color='#2a2a3e', linewidth=0.5)


# ─── Main GUI class ───────────────────────────────────────────────────────────

class BandpassFilterDesigner:
    """
    Interactive matplotlib GUI for bandpass filter design.
    """

    FILTER_TYPES  = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    FILTER_LABELS = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptic', 'Bessel']

    C_MAG    = '#F5A623'   # amber  — bandpass
    C_PHASE  = '#1D9E75'   # green
    C_STEP   = '#7F77DD'   # violet
    C_FL     = '#378ADD'   # blue   — lower cutoff marker
    C_FH     = '#E24B4A'   # red    — upper cutoff marker
    C_ZERO   = '#378ADD'
    C_POLE   = '#E24B4A'

    def __init__(self):
        self.ftype  = 'butter'
        self.order  = 4          # per-band order; total = 2× this
        self.f_low  = 500.0      # Hz
        self.f_high = 3000.0     # Hz
        self.fs     = 44100.0    # Hz
        self.rp     = 1.0        # dB
        self.rs     = 40.0       # dB

        self._build_gui()
        self._update(None)
        plt.show()

    # ── GUI layout ──────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(14, 9.5), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('Bandpass Filter Designer')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3], wspace=0.05,
            left=0.02, right=0.98, top=0.96, bottom=0.04)

        # ── Left controls (10 rows to fit extra slider) ───────────────────────
        ctrl_gs = gridspec.GridSpecFromSubplotSpec(
            10, 1, subplot_spec=outer[0], hspace=0.55)

        ax_type  = self.fig.add_subplot(ctrl_gs[0:3])
        ax_ord   = self.fig.add_subplot(ctrl_gs[3])
        ax_fl    = self.fig.add_subplot(ctrl_gs[4])
        ax_fh    = self.fig.add_subplot(ctrl_gs[5])
        ax_fs    = self.fig.add_subplot(ctrl_gs[6])
        ax_rp    = self.fig.add_subplot(ctrl_gs[7])
        ax_rs    = self.fig.add_subplot(ctrl_gs[8])
        ax_reset = self.fig.add_subplot(ctrl_gs[9])

        for ax in [ax_type, ax_ord, ax_fl, ax_fh, ax_fs, ax_rp, ax_rs, ax_reset]:
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
        self.sl_ord = Slider(ax_ord, 'Order\n(per band)', 1, 8,
                             valinit=self.order, valstep=1,
                             color=self.C_MAG, initcolor='none', facecolor=sc)
        self.sl_fl  = Slider(ax_fl,  'Low\ncutoff (Hz)', 10, self.fs/2 - 2,
                             valinit=self.f_low,
                             color=self.C_FL, initcolor='none', facecolor=sc)
        self.sl_fh  = Slider(ax_fh,  'High\ncutoff (Hz)', 11, self.fs/2 - 1,
                             valinit=self.f_high,
                             color=self.C_FH, initcolor='none', facecolor=sc)
        self.sl_fs  = Slider(ax_fs,  'Sample\nrate (Hz)', 8000, 192000,
                             valinit=self.fs, valstep=100,
                             color='#7F77DD', initcolor='none', facecolor=sc)
        self.sl_rp  = Slider(ax_rp,  'Ripple\n(dB)', 0.1, 6.0,
                             valinit=self.rp,
                             color=self.C_STEP, initcolor='none', facecolor=sc)
        self.sl_rs  = Slider(ax_rs,  'Stop\natten (dB)', 10, 80,
                             valinit=self.rs, valstep=1,
                             color='#BA7517', initcolor='none', facecolor=sc)

        for sl in [self.sl_ord, self.sl_fl, self.sl_fh,
                   self.sl_fs, self.sl_rp, self.sl_rs]:
            sl.label.set_color('#B4B2A9');  sl.label.set_fontsize(8)
            sl.valtext.set_color('#ffffff'); sl.valtext.set_fontsize(8)
            sl.on_changed(self._update)

        # Reset button
        self.btn_reset = Button(ax_reset, 'Reset defaults',
                                color='#0d0d1a', hovercolor='#1a1a2e')
        self.btn_reset.label.set_color('#B4B2A9')
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self._reset)

        # ── Right: 2 × 2 plots ────────────────────────────────────────────────
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
        self.metrics_txt = self.metrics_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            color='#B4B2A9', fontsize=8, transform=self.metrics_ax.transAxes)

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_type(self, label):
        self.ftype = self.FILTER_TYPES[self.FILTER_LABELS.index(label)]
        self._update(None)

    def _reset(self, _):
        for sl in [self.sl_ord, self.sl_fl, self.sl_fh,
                   self.sl_fs, self.sl_rp, self.sl_rs]:
            sl.reset()

    def _update(self, _):
        self.order  = int(self.sl_ord.val)
        self.f_low  = float(self.sl_fl.val)
        self.f_high = float(self.sl_fh.val)
        self.fs     = float(self.sl_fs.val)
        self.rp     = float(self.sl_rp.val)
        self.rs     = float(self.sl_rs.val)

        nyq = self.fs / 2.0

        # Enforce fl < fh < nyq with a minimum gap
        min_gap = max(10.0, nyq * 0.01)
        if self.f_high >= nyq:
            self.f_high = nyq * 0.99
            self.sl_fh.set_val(self.f_high)
        if self.f_low >= self.f_high - min_gap:
            self.f_low = max(10.0, self.f_high - min_gap)
            self.sl_fl.set_val(self.f_low)

        # Keep slider ranges sane
        self.sl_fl.valmax = nyq - 2
        self.sl_fh.valmax = nyq - 1
        self.sl_fl.ax.set_xlim(10, nyq - 2)
        self.sl_fh.ax.set_xlim(11, nyq - 1)

        try:
            b, a = design_filter(
                self.ftype, self.order,
                self.f_low, self.f_high,
                self.fs, self.rp, self.rs)
            self._plot_all(b, a)
        except Exception as e:
            print(f"[Filter design error] {e}")

    # ── Plots ────────────────────────────────────────────────────────────────

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

        # Cutoff markers
        ax.axvline(self.f_low,  color=self.C_FL, linewidth=1,
                   linestyle='--', alpha=0.85, label=f'fl={_fmt_hz(self.f_low)}')
        ax.axvline(self.f_high, color=self.C_FH, linewidth=1,
                   linestyle='--', alpha=0.85, label=f'fh={_fmt_hz(self.f_high)}')
        ax.axhline(-3, color='#5F5E5A', linewidth=0.8, linestyle=':')

        # Shade passband
        mask = (w >= self.f_low) & (w <= self.f_high)
        ax.fill_between(w, mag_db, -80, where=mask,
                        alpha=0.15, color=self.C_MAG)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(-80, 5)
        ax.set_title('Magnitude response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Magnitude (dB)', color='#888780', fontsize=8)

        # Centre frequency annotation
        f_c = np.sqrt(self.f_low * self.f_high)
        ax.text(f_c, 2, f'fc={_fmt_hz(f_c)}',
                color=self.C_MAG, fontsize=7, ha='center')

        # Bandwidth annotation
        bw = self.f_high - self.f_low
        ax.text(f_c, -8, f'BW={_fmt_hz(bw)}',
                color='#B4B2A9', fontsize=7, ha='center', alpha=0.7)

        ax.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e', loc='lower right')

    def _plot_phase(self, b, a):
        ax = self.ax_ph
        ax.cla(); _style_ax(ax)

        w, H = frequency_response(b, a, self.fs)
        phase_deg = np.unwrap(np.angle(H)) * 180 / np.pi

        ax.semilogx(w, phase_deg, color=self.C_PHASE, linewidth=1.5)
        ax.axvline(self.f_low,  color=self.C_FL, linewidth=1,
                   linestyle='--', alpha=0.7)
        ax.axvline(self.f_high, color=self.C_FH, linewidth=1,
                   linestyle='--', alpha=0.7)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_title('Phase response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Phase (°)', color='#888780', fontsize=8)

    def _plot_step(self, b, a):
        ax = self.ax_step
        ax.cla(); _style_ax(ax)

        t, y   = step_response_fn(b, a, self.fs)
        t_ms   = t * 1000

        ax.plot(t_ms, y, color=self.C_STEP, linewidth=1.5)
        ax.axhline(0, color='#5F5E5A', linewidth=0.8, linestyle=':')

        ax.text(0.97, 0.92, 'DC blocked — transient only',
                transform=ax.transAxes, ha='right', va='top',
                color='#5F5E5A', fontsize=7)

        ax.set_title('Step response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Time (ms)', color='#888780', fontsize=8)
        ax.set_ylabel('Amplitude', color='#888780', fontsize=8)

    def _plot_pz(self, b, a):
        ax = self.ax_pz
        ax.cla(); _style_ax(ax)

        z, p, _ = pole_zero(b, a)

        # Unit circle + axes
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color='#444441', linewidth=0.8)
        ax.axhline(0, color='#444441', linewidth=0.5)
        ax.axvline(0, color='#444441', linewidth=0.5)

        ax.scatter(z.real, z.imag, marker='o', s=55,
                   facecolors='none', edgecolors=self.C_ZERO,
                   linewidths=1.5, label='Zeros', zorder=3)
        ax.scatter(p.real, p.imag, marker='x', s=55,
                   color=self.C_POLE, linewidths=1.5,
                   label='Poles', zorder=3)

        # Total order annotation
        n_total = 2 * self.order
        ax.text(-1.3, 1.25,
                f'Total order: {n_total}  ({self.order}×2)',
                color='#B4B2A9', fontsize=7)

        lim = 1.4
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title('Pole-zero plot', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Real', color='#888780', fontsize=8)
        ax.set_ylabel('Imaginary', color='#888780', fontsize=8)
        ax.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e', loc='lower right')

    # ── Metrics bar ──────────────────────────────────────────────────────────

    def _update_metrics(self, b, a):
        w, H   = frequency_response(b, a, self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)

        # Centre frequency (geometric mean)
        f_c = np.sqrt(self.f_low * self.f_high)

        # Peak gain (inside passband)
        mask = (w >= self.f_low) & (w <= self.f_high)
        peak_db = np.max(mag_db[mask]) if mask.any() else float('nan')

        # Gain at lower & upper cutoffs
        idx_fl = np.argmin(np.abs(w - self.f_low))
        idx_fh = np.argmin(np.abs(w - self.f_high))
        gain_fl = mag_db[idx_fl]
        gain_fh = mag_db[idx_fh]

        # Bandwidth
        bw = self.f_high - self.f_low

        # Q factor
        q = f_c / bw if bw > 0 else float('nan')

        # Roll-off (theoretical, per side)
        rolloff = 20 * self.order

        txt = (
            f'  fl={_fmt_hz(self.f_low)} ({gain_fl:.1f} dB)   '
            f'fh={_fmt_hz(self.f_high)} ({gain_fh:.1f} dB)   |   '
            f'fc={_fmt_hz(f_c)}   BW={_fmt_hz(bw)}   '
            f'Q={q:.2f}   |   '
            f'Peak={peak_db:.1f} dB   Roll-off={rolloff} dB/oct/side'
        )
        self.metrics_txt.set_text(txt)


# ─── Non-interactive API ──────────────────────────────────────────────────────

def design_and_print(ftype='butter', order=4, f_low=500, f_high=3000,
                     fs_hz=44100, rp_db=1.0, rs_db=40.0):
    """
    Design a bandpass filter and print key specs + coefficients.

    Examples
    --------
    >>> b, a = design_and_print('butter', order=4, f_low=300, f_high=3400, fs_hz=8000)
    >>> b, a = design_and_print('ellip',  order=3, f_low=1000, f_high=4000, fs_hz=44100, rp_db=0.5, rs_db=60)
    """
    b, a = design_filter(ftype, order, f_low, f_high, fs_hz, rp_db, rs_db)

    w, H   = frequency_response(b, a, fs_hz)
    mag_db = 20 * np.log10(np.abs(H) + 1e-20)

    f_c    = np.sqrt(f_low * f_high)
    bw     = f_high - f_low
    q      = f_c / bw

    idx_fc = np.argmin(np.abs(w - f_c))
    idx_fl = np.argmin(np.abs(w - f_low))
    idx_fh = np.argmin(np.abs(w - f_high))

    print("=" * 60)
    print(f" Filter    : {ftype.upper()} bandpass  order {order}×2 = {2*order}")
    print(f" Low  cut  : {_fmt_hz(f_low)}")
    print(f" High cut  : {_fmt_hz(f_high)}")
    print(f" Sample    : {_fmt_hz(fs_hz)}")
    if ftype in ('cheby1', 'ellip'):
        print(f" Ripple    : {rp_db} dB")
    if ftype in ('cheby2', 'ellip'):
        print(f" Stop att. : {rs_db} dB")
    print("-" * 60)
    print(f" Centre fc : {_fmt_hz(f_c)}")
    print(f" Bandwidth : {_fmt_hz(bw)}")
    print(f" Q factor  : {q:.3f}")
    print(f" Mag @ fc  : {mag_db[idx_fc]:.2f} dB")
    print(f" Mag @ fl  : {mag_db[idx_fl]:.2f} dB")
    print(f" Mag @ fh  : {mag_db[idx_fh]:.2f} dB")
    print(f" Roll-off  : {20*order} dB/oct per side (theoretical)")
    print("=" * 60)
    print(f" b = {np.round(b, 6).tolist()}")
    print(f" a = {np.round(a, 6).tolist()}")
    print("=" * 60)
    return b, a


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Launch the interactive GUI
    BandpassFilterDesigner()

    # ── Non-interactive examples (uncomment to use) ──────────────────────────
    # b, a = design_and_print('butter', order=4, f_low=300,  f_high=3400, fs_hz=8000)
    # b, a = design_and_print('cheby1', order=3, f_low=500,  f_high=2000, fs_hz=44100, rp_db=0.5)
    # b, a = design_and_print('cheby2', order=4, f_low=1000, f_high=5000, fs_hz=96000, rs_db=60)
    # b, a = design_and_print('ellip',  order=3, f_low=1000, f_high=4000, fs_hz=44100, rp_db=1, rs_db=50)
    # b, a = design_and_print('bessel', order=4, f_low=500,  f_high=3000, fs_hz=44100)
