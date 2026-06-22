"""
Bandstop (Notch) Filter Designer
=================================
Interactive bandstop filter design and analysis tool — similar to MATLAB's
fvtool / fdatool, but in Python using scipy.signal and matplotlib.

A bandstop (also called a band-reject or notch) filter passes all frequencies
EXCEPT those in the rejection band [f_low, f_high].

Supports:
  - Filter types  : Butterworth, Chebyshev I, Chebyshev II, Elliptic, Bessel
  - Parameters    : order, low/high reject-band edges, sample rate,
                    passband ripple, stopband attenuation
  - Plots         : Magnitude (dB), Phase, Step Response, Pole-Zero
  - Metrics       : notch depth, centre frequency, bandwidth, Q factor

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
    Design a digital bandstop filter using bilinear transform.

    Parameters
    ----------
    ftype   : str   – 'butter' | 'cheby1' | 'cheby2' | 'ellip' | 'bessel'
    order   : int   – per-band order (total filter order = 2 × order)
    f_low   : float – lower rejection-band edge in Hz
    f_high  : float – upper rejection-band edge in Hz
    fs_hz   : float – sample rate in Hz
    rp_db   : float – passband ripple in dB  (Chebyshev I / Elliptic)
    rs_db   : float – stopband attenuation dB (Chebyshev II / Elliptic)

    Returns
    -------
    b, a : array_like – numerator / denominator coefficients (IIR)
    """
    nyq = fs_hz / 2.0
    wl  = np.clip(f_low  / nyq, 1e-4, 0.9998)
    wh  = np.clip(f_high / nyq, wl + 1e-4, 0.9999)
    Wn  = [wl, wh]

    if ftype == 'butter':
        b, a = signal.butter(order, Wn, btype='bandstop', analog=False)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, rp_db, Wn, btype='bandstop', analog=False)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, rs_db, Wn, btype='bandstop', analog=False)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, rp_db, rs_db, Wn, btype='bandstop', analog=False)
    elif ftype == 'bessel':
        b, a = signal.bessel(order, Wn, btype='bandstop', analog=False, norm='phase')
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    return b, a


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def frequency_response(b, a, fs_hz, n_pts=4096):
    """Higher n_pts for sharper notch resolution."""
    w, H = signal.freqz(b, a, worN=n_pts, fs=fs_hz)
    return w, H


def step_response_fn(b, a, fs_hz):
    n_samples = min(4000, max(300, int(fs_hz * 0.05)))
    x = np.ones(n_samples)
    t = np.arange(n_samples) / fs_hz
    y = signal.lfilter(b, a, x)
    return t, y


def pole_zero(b, a):
    return signal.tf2zpk(b, a)


# ─── Style helpers ────────────────────────────────────────────────────────────

def _fmt_hz(f):
    return f'{f/1000:.2f} kHz' if f >= 1000 else f'{f:.1f} Hz'

def _style_ax(ax):
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#888780', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a3e')
    ax.grid(True, color='#2a2a3e', linewidth=0.5)


# ─── Main GUI class ───────────────────────────────────────────────────────────

class BandstopFilterDesigner:
    """
    Interactive matplotlib GUI for bandstop / notch filter design.
    """

    FILTER_TYPES  = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    FILTER_LABELS = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Elliptic', 'Bessel']

    # Colour palette
    C_MAG    = '#C77DFF'   # violet — bandstop / notch
    C_PHASE  = '#1D9E75'   # green
    C_STEP   = '#F5A623'   # amber
    C_FL     = '#378ADD'   # blue   — lower edge
    C_FH     = '#E24B4A'   # red    — upper edge
    C_NOTCH  = '#C77DFF'   # violet — notch centre
    C_ZERO   = '#378ADD'
    C_POLE   = '#E24B4A'

    def __init__(self):
        self.ftype  = 'butter'
        self.order  = 4
        self.f_low  = 950.0    # Hz  — notch lower edge (e.g. 50 Hz hum rejection)
        self.f_high = 1050.0   # Hz  — notch upper edge
        self.fs     = 44100.0  # Hz
        self.rp     = 1.0      # dB
        self.rs     = 40.0     # dB

        self._build_gui()
        self._update(None)
        plt.show()

    # ── GUI layout ──────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(14, 9.5), facecolor='#1a1a2e')
        self.fig.canvas.manager.set_window_title('Bandstop / Notch Filter Designer')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3], wspace=0.05,
            left=0.02, right=0.98, top=0.96, bottom=0.04)

        # ── Left controls ─────────────────────────────────────────────────────
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

        for ax in [ax_type, ax_ord, ax_fl, ax_fh,
                   ax_fs, ax_rp, ax_rs, ax_reset]:
            ax.set_facecolor('#0d0d1a')

        # Radio
        self.radio = RadioButtons(ax_type, self.FILTER_LABELS,
                                  active=0, activecolor=self.C_MAG)
        ax_type.set_title('Filter type', color='#B4B2A9', fontsize=9, pad=2)
        for lbl in self.radio.labels:
            lbl.set_color('#B4B2A9'); lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_type)

        # Sliders
        sc = '#0d0d1a'
        self.sl_ord = Slider(ax_ord, 'Order\n(per band)', 1, 8,
                             valinit=self.order, valstep=1,
                             color=self.C_MAG, initcolor='none', facecolor=sc)
        self.sl_fl  = Slider(ax_fl,  'Reject\nlow (Hz)', 10, self.fs / 2 - 2,
                             valinit=self.f_low,
                             color=self.C_FL, initcolor='none', facecolor=sc)
        self.sl_fh  = Slider(ax_fh,  'Reject\nhigh (Hz)', 11, self.fs / 2 - 1,
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

        nyq     = self.fs / 2.0
        min_gap = max(5.0, nyq * 0.005)

        if self.f_high >= nyq:
            self.f_high = nyq * 0.99
            self.sl_fh.set_val(self.f_high)
        if self.f_low >= self.f_high - min_gap:
            self.f_low = max(10.0, self.f_high - min_gap)
            self.sl_fl.set_val(self.f_low)

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

        # Rejection band edges
        ax.axvline(self.f_low,  color=self.C_FL, linewidth=1,
                   linestyle='--', alpha=0.85, label=f'fl={_fmt_hz(self.f_low)}')
        ax.axvline(self.f_high, color=self.C_FH, linewidth=1,
                   linestyle='--', alpha=0.85, label=f'fh={_fmt_hz(self.f_high)}')
        ax.axhline(-3, color='#5F5E5A', linewidth=0.8, linestyle=':')

        # Shade rejection band (in stopband colour)
        mask = (w >= self.f_low) & (w <= self.f_high)
        ax.fill_between(w, mag_db, -100, where=mask,
                        alpha=0.20, color='#E24B4A')

        # Shade passbands (below fl and above fh)
        mask_lo = w <= self.f_low
        mask_hi = w >= self.f_high
        ax.fill_between(w, mag_db, -100,
                        where=mask_lo, alpha=0.08, color=self.C_MAG)
        ax.fill_between(w, mag_db, -100,
                        where=mask_hi, alpha=0.08, color=self.C_MAG)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(-100, 5)
        ax.set_title('Magnitude response', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Frequency (Hz)', color='#888780', fontsize=8)
        ax.set_ylabel('Magnitude (dB)', color='#888780', fontsize=8)

        # Notch centre annotation
        f_c = np.sqrt(self.f_low * self.f_high)
        ax.axvline(f_c, color=self.C_NOTCH, linewidth=0.8,
                   linestyle=':', alpha=0.6)
        ax.text(f_c, -92, f'fc={_fmt_hz(f_c)}',
                color=self.C_NOTCH, fontsize=7, ha='center')

        # Region labels
        nyq = self.fs / 2
        ax.text(self.f_low * 0.4, 2, 'PASS', color=self.C_MAG,
                fontsize=7, alpha=0.6, ha='center')
        ax.text(f_c, 2, 'REJECT', color='#E24B4A',
                fontsize=7, alpha=0.8, ha='center')
        ax.text(min(self.f_high * 2, nyq * 0.9), 2, 'PASS',
                color=self.C_MAG, fontsize=7, alpha=0.6, ha='center')

        ax.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e', loc='lower center')

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

        # Phase jump at notch centre
        f_c = np.sqrt(self.f_low * self.f_high)
        ax.axvline(f_c, color=self.C_NOTCH, linewidth=0.8,
                   linestyle=':', alpha=0.5)

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
        ax.axhline(1.0, color='#5F5E5A', linewidth=0.8, linestyle=':',
                   label='DC level (passed)')
        ax.legend(fontsize=7, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e')

        # Bandstop passes DC → step settles to 1
        ax.text(0.97, 0.10, 'DC passed → settles to 1',
                transform=ax.transAxes, ha='right', va='bottom',
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

        # Bandstop zeros sit ON the unit circle at the notch frequency
        zeros_on_circle = z[np.abs(np.abs(z) - 1.0) < 0.05]
        if len(zeros_on_circle):
            ax.scatter(zeros_on_circle.real, zeros_on_circle.imag,
                       marker='o', s=90,
                       facecolors='none', edgecolors=self.C_NOTCH,
                       linewidths=2.5, zorder=4,
                       label='Notch zeros\n(on unit circle)')

        n_total = 2 * self.order
        ax.text(-1.35, 1.25, f'Total order: {n_total}  ({self.order}×2)',
                color='#B4B2A9', fontsize=7)

        lim = 1.45
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title('Pole-zero plot', color='#B4B2A9', fontsize=9)
        ax.set_xlabel('Real', color='#888780', fontsize=8)
        ax.set_ylabel('Imaginary', color='#888780', fontsize=8)
        ax.legend(fontsize=6, facecolor='#0d0d1a', labelcolor='#B4B2A9',
                  edgecolor='#2a2a3e', loc='lower right')

    # ── Metrics bar ──────────────────────────────────────────────────────────

    def _update_metrics(self, b, a):
        w, H   = frequency_response(b, a, self.fs)
        mag_db = 20 * np.log10(np.abs(H) + 1e-20)

        # Notch centre
        f_c = np.sqrt(self.f_low * self.f_high)
        idx_fc = np.argmin(np.abs(w - f_c))
        notch_depth = mag_db[idx_fc]   # should be very negative

        # Passband ripple below fl
        mask_lo = (w > w[1]) & (w < self.f_low * 0.9)
        pb_lo   = np.max(mag_db[mask_lo]) if mask_lo.any() else 0.0

        # Passband ripple above fh
        mask_hi = (w > self.f_high * 1.1) & (w < self.fs / 2 * 0.99)
        pb_hi   = np.max(mag_db[mask_hi]) if mask_hi.any() else 0.0

        # Bandwidth of rejection band & Q
        bw = self.f_high - self.f_low
        q  = f_c / bw if bw > 0 else float('nan')

        # Roll-off (theoretical per side)
        rolloff = 20 * self.order

        txt = (
            f'  Reject: {_fmt_hz(self.f_low)} – {_fmt_hz(self.f_high)}   |   '
            f'Centre fc={_fmt_hz(f_c)}   BW={_fmt_hz(bw)}   Q={q:.2f}   |   '
            f'Notch depth={notch_depth:.1f} dB   '
            f'PB ripple low={pb_lo:.2f} dB  high={pb_hi:.2f} dB   |   '
            f'Roll-off={rolloff} dB/oct/side'
        )
        self.metrics_txt.set_text(txt)


# ─── Non-interactive API ──────────────────────────────────────────────────────

def design_and_print(ftype='butter', order=4, f_low=950, f_high=1050,
                     fs_hz=44100, rp_db=1.0, rs_db=40.0):
    """
    Design a bandstop filter and print key specs + coefficients.

    Examples
    --------
    >>> # 50 Hz mains hum rejection (European power line)
    >>> b, a = design_and_print('butter', order=4, f_low=45, f_high=55, fs_hz=8000)

    >>> # 60 Hz mains hum rejection (US/Indian power line)
    >>> b, a = design_and_print('ellip', order=3, f_low=55, f_high=65, fs_hz=44100, rp_db=0.5, rs_db=60)

    >>> # Narrow 1 kHz notch
    >>> b, a = design_and_print('cheby2', order=4, f_low=950, f_high=1050, fs_hz=44100, rs_db=80)
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

    print("=" * 62)
    print(f" Filter     : {ftype.upper()} bandstop  order {order}×2 = {2*order}")
    print(f" Reject low : {_fmt_hz(f_low)}")
    print(f" Reject high: {_fmt_hz(f_high)}")
    print(f" Sample     : {_fmt_hz(fs_hz)}")
    if ftype in ('cheby1', 'ellip'):
        print(f" Ripple     : {rp_db} dB")
    if ftype in ('cheby2', 'ellip'):
        print(f" Stop att.  : {rs_db} dB")
    print("-" * 62)
    print(f" Notch fc   : {_fmt_hz(f_c)}")
    print(f" Bandwidth  : {_fmt_hz(bw)}")
    print(f" Q factor   : {q:.3f}")
    print(f" Notch depth: {mag_db[idx_fc]:.2f} dB")
    print(f" Mag @ fl   : {mag_db[idx_fl]:.2f} dB")
    print(f" Mag @ fh   : {mag_db[idx_fh]:.2f} dB")
    print(f" Roll-off   : {20*order} dB/oct per side (theoretical)")
    print("=" * 62)
    print(f" b = {np.round(b, 6).tolist()}")
    print(f" a = {np.round(a, 6).tolist()}")
    print("=" * 62)
    return b, a


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Launch interactive GUI
    BandstopFilterDesigner()

    # ── Non-interactive examples (uncomment to use) ──────────────────────────
    # 50 Hz mains hum (Europe / India)
    # b, a = design_and_print('butter', order=4, f_low=45,  f_high=55,   fs_hz=8000)

    # 60 Hz mains hum (US)
    # b, a = design_and_print('ellip',  order=3, f_low=55,  f_high=65,   fs_hz=44100, rp_db=0.5, rs_db=60)

    # Narrow 1 kHz notch
    # b, a = design_and_print('cheby2', order=4, f_low=950, f_high=1050, fs_hz=44100, rs_db=80)

    # Wide rejection band
    # b, a = design_and_print('butter', order=6, f_low=200, f_high=2000, fs_hz=44100)
