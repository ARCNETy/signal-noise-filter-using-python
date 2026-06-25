"""
FIR Filter Designer — Windowed Sinc Method, when  your current tools are all IIR
===========================================
Interactive FIR filter design and analysis tool using the windowed sinc
method. Complements the IIR lowpass / highpass / bandpass / bandstop tools
with linear-phase, zero-phase FIR filters.

Filter types  : Lowpass, Highpass, Bandpass, Bandstop
Window types  : Rectangular, Hann, Hamming, Blackman, Blackman-Harris,
                Kaiser, Flat-top, Bartlett
Displays      : Magnitude (dB + linear), Phase, Impulse response,
                Pole-zero map, Window shape, Group delay
Metrics       : Transition width, stopband attenuation, passband ripple,
                group delay (constant for linear-phase FIR), filter order

Requirements:
  pip install scipy matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy import signal
from scipy.fft import fft, fftfreq


# ─── Colour palette (consistent with the IIR tools) ─────────────────────────

BG_DARK  = '#0d0d1a'
BG_PANEL = '#1a1a2e'
BG_CTRL  = '#0d0d1a'
GRID_C   = '#2a2a3e'
TEXT_PRI = '#B4B2A9'
TEXT_SEC = '#888780'
SPINE_C  = '#2a2a3e'

C_MAG_DB  = '#4CAF82'   # teal-green  — FIR uses green to differ from IIR reds
C_MAG_LIN = '#F5A623'   # amber
C_PHASE   = '#1D9E75'   # green
C_IMPULSE = '#378ADD'   # blue
C_WINDOW  = '#C77DFF'   # violet
C_GRPDLY  = '#D85A30'   # orange
C_CUTOFF  = '#BA7517'   # amber cutoff marker
C_ZERO    = '#378ADD'
C_POLE    = '#E24B4A'


# ─── Constants ────────────────────────────────────────────────────────────────

WINDOW_NAMES = [
    'Rectangular', 'Hann', 'Hamming', 'Blackman',
    'Blackman-Harris', 'Kaiser', 'Flat-top', 'Bartlett'
]
WINDOW_KEYS = [
    'boxcar', 'hann', 'hamming', 'blackman',
    'blackmanharris', 'kaiser', 'flattop', 'bartlett'
]

FTYPE_NAMES = ['Lowpass', 'Highpass', 'Bandpass', 'Bandstop']


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_hz(f):
    if f >= 1000: return f'{f/1000:.2f} kHz'
    return f'{f:.1f} Hz'

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.grid(True, color=GRID_C, linewidth=0.5)
    if title:  ax.set_title(title,  color=TEXT_PRI, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=8)

def make_window(key, N, beta=8.6):
    if key == 'kaiser':
        return signal.windows.kaiser(N, beta)
    return signal.get_window(key, N)


# ─── Kaiser order / beta estimation ──────────────────────────────────────────

def kaiser_params(trans_width_hz, fs, atten_db):
    """
    Estimate Kaiser window beta and minimum filter order for a given
    transition width and stopband attenuation (Harris 1978 approximation).
    """
    A = abs(atten_db)
    if A > 50:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21)
    else:
        beta = 0.0
    delta_w = trans_width_hz / (fs / 2)          # normalised [0,1]
    N = int(np.ceil((A - 8) / (2.285 * np.pi * delta_w)))
    N = max(N, 3)
    if N % 2 == 0: N += 1                         # keep odd for Type I
    return beta, N


# ─── FIR design ──────────────────────────────────────────────────────────────

def design_fir(ftype, order, f1, f2, fs, win_key, kaiser_beta=8.6):
    """
    Design a linear-phase FIR filter via windowed sinc (firwin).

    Parameters
    ----------
    ftype       : str   – 'lowpass'|'highpass'|'bandpass'|'bandstop'
    order       : int   – filter order (number of taps = order + 1)
    f1          : float – cutoff / lower cutoff (Hz)
    f2          : float – upper cutoff (Hz, bandpass/bandstop only)
    fs          : float – sample rate (Hz)
    win_key     : str   – window key from WINDOW_KEYS
    kaiser_beta : float – Kaiser window β parameter

    Returns
    -------
    h : ndarray – FIR coefficients (taps)
    """
    nyq = fs / 2.0
    f1c = np.clip(f1 / nyq, 1e-4, 0.9999)
    f2c = np.clip(f2 / nyq, f1c + 1e-4, 0.9999)

    if win_key == 'kaiser':
        window = ('kaiser', kaiser_beta)
    else:
        window = win_key

    # firwin needs odd number of taps for highpass / bandstop
    N = order + 1
    if ftype in ('highpass', 'bandstop') and N % 2 == 0:
        N += 1

    if ftype == 'lowpass':
        h = signal.firwin(N, f1c, window=window, pass_zero=True)
    elif ftype == 'highpass':
        h = signal.firwin(N, f1c, window=window, pass_zero=False)
    elif ftype == 'bandpass':
        h = signal.firwin(N, [f1c, f2c], window=window, pass_zero=False)
    elif ftype == 'bandstop':
        h = signal.firwin(N, [f1c, f2c], window=window, pass_zero=True)
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    return h


# ─── Analysis ─────────────────────────────────────────────────────────────────

def freq_response(h, fs, n=4096):
    w, H = signal.freqz(h, worN=n, fs=fs)
    mag_db  = 20 * np.log10(np.abs(H) + 1e-15)
    mag_lin = np.abs(H)
    phase   = np.unwrap(np.angle(H)) * 180 / np.pi
    return w, H, mag_db, mag_lin, phase

def grp_delay(h, fs, n=4096):
    w, gd = signal.group_delay((h, [1.0]), w=n, fs=fs)
    return w, gd

def pole_zero_fir(h):
    z = np.roots(h)
    # FIR has all poles at origin
    p = np.zeros(len(h) - 1, dtype=complex)
    return z, p


# ─── Metrics helpers ──────────────────────────────────────────────────────────

def passband_ripple(freqs, mag_db, f_lo, f_hi):
    """Max deviation from 0 dB in passband [f_lo, f_hi]."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any(): return float('nan')
    pb = mag_db[mask]
    return np.max(pb) - np.min(pb)

def stopband_atten(freqs, mag_db, f_lo, f_hi):
    """Min attenuation in stopband."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any(): return float('nan')
    return -np.max(mag_db[mask])


# ─── Main GUI ─────────────────────────────────────────────────────────────────

class FIRDesigner:
    """Interactive FIR filter designer — windowed sinc method."""

    def __init__(self):
        # Parameters
        self.ftype     = 'lowpass'
        self.order     = 64       # number of taps = order + 1
        self.f1        = 1000.0   # lower / only cutoff (Hz)
        self.f2        = 3000.0   # upper cutoff for BP/BS (Hz)
        self.fs        = 44100.0
        self.win_key   = 'hamming'
        self.kaiser_b  = 8.6
        self.show_db   = True
        self.show_lin  = False
        self.show_gd   = True
        self.auto_ord  = False    # auto-order from Kaiser estimate

        self._build_gui()
        self._update(None)
        plt.show()

    # ── GUI build ──────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(15, 9.5), facecolor=BG_PANEL)
        self.fig.canvas.manager.set_window_title('FIR Filter Designer — Windowed Sinc')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3.2], wspace=0.04,
            left=0.01, right=0.99, top=0.97, bottom=0.04)

        # ── Left controls ───────────────────────────────────────────────────
        cgs = gridspec.GridSpecFromSubplotSpec(
            13, 1, subplot_spec=outer[0], hspace=0.55)

        ax_ftype  = self.fig.add_subplot(cgs[0:3])
        ax_wtype  = self.fig.add_subplot(cgs[3:7])
        ax_ord    = self.fig.add_subplot(cgs[7])
        ax_f1     = self.fig.add_subplot(cgs[8])
        ax_f2     = self.fig.add_subplot(cgs[9])
        ax_fs     = self.fig.add_subplot(cgs[10])
        ax_kaiser = self.fig.add_subplot(cgs[11])
        ax_reset  = self.fig.add_subplot(cgs[12])

        for ax in [ax_ftype, ax_wtype, ax_ord, ax_f1, ax_f2,
                   ax_fs, ax_kaiser, ax_reset]:
            ax.set_facecolor(BG_CTRL)

        # Filter type radio
        self.radio_ftype = RadioButtons(
            ax_ftype, FTYPE_NAMES, active=0, activecolor=C_MAG_DB)
        ax_ftype.set_title('Filter type', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_ftype.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(9)
        self.radio_ftype.on_clicked(self._on_ftype)

        # Window type radio
        self.radio_win = RadioButtons(
            ax_wtype, WINDOW_NAMES, active=2, activecolor=C_WINDOW)
        ax_wtype.set_title('Window function', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_win.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(8)
        self.radio_win.on_clicked(self._on_window)

        # Sliders
        sc = BG_CTRL
        self.sl_ord = Slider(ax_ord, 'Order\n(taps−1)', 4, 512,
                             valinit=self.order, valstep=2,
                             color=C_MAG_DB, initcolor='none', facecolor=sc)
        self.sl_f1  = Slider(ax_f1,  'Cutoff f1\n(Hz)', 10, self.fs / 2 - 1,
                             valinit=self.f1,
                             color=C_CUTOFF, initcolor='none', facecolor=sc)
        self.sl_f2  = Slider(ax_f2,  'Cutoff f2\n(Hz, BP/BS)', 11, self.fs / 2 - 1,
                             valinit=self.f2,
                             color=C_POLE, initcolor='none', facecolor=sc)
        self.sl_fs  = Slider(ax_fs,  'Sample\nrate (Hz)', 8000, 192000,
                             valinit=self.fs, valstep=100,
                             color='#7F77DD', initcolor='none', facecolor=sc)
        self.sl_kb  = Slider(ax_kaiser, 'Kaiser β\n(attenuation)', 1, 20,
                             valinit=self.kaiser_b,
                             color=C_WINDOW, initcolor='none', facecolor=sc)

        for sl in [self.sl_ord, self.sl_f1, self.sl_f2, self.sl_fs, self.sl_kb]:
            sl.label.set_color(TEXT_PRI);  sl.label.set_fontsize(8)
            sl.valtext.set_color('#ffffff'); sl.valtext.set_fontsize(8)
            sl.on_changed(self._update)

        # Reset button
        self.btn_reset = Button(ax_reset, 'Reset defaults',
                                color=BG_CTRL, hovercolor=BG_PANEL)
        self.btn_reset.label.set_color(TEXT_PRI)
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self._reset)

        # ── Right: 3×2 plot grid ────────────────────────────────────────────
        pgs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer[1], hspace=0.48, wspace=0.32)

        self.ax_mag   = self.fig.add_subplot(pgs[0, 0])
        self.ax_magL  = self.fig.add_subplot(pgs[0, 1])
        self.ax_phase = self.fig.add_subplot(pgs[1, 0])
        self.ax_gd    = self.fig.add_subplot(pgs[1, 1])
        self.ax_imp   = self.fig.add_subplot(pgs[2, 0])
        self.ax_win   = self.fig.add_subplot(pgs[2, 1])

        for ax in [self.ax_mag, self.ax_magL, self.ax_phase,
                   self.ax_gd, self.ax_imp, self.ax_win]:
            _style_ax(ax)

        # Metrics bar
        self.met_ax = self.fig.add_axes([0.35, 0.005, 0.64, 0.028])
        self.met_ax.axis('off')
        self.met_txt = self.met_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            color=TEXT_PRI, fontsize=8, transform=self.met_ax.transAxes)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_ftype(self, label):
        self.ftype = label.lower()
        self._toggle_f2_label()
        self._update(None)

    def _on_window(self, label):
        self.win_key = WINDOW_KEYS[WINDOW_NAMES.index(label)]
        active = self.win_key == 'kaiser'
        self.sl_kb.label.set_color(TEXT_PRI if active else TEXT_SEC)
        self.sl_kb.valtext.set_color('#ffffff' if active else TEXT_SEC)
        self._update(None)

    def _toggle_f2_label(self):
        needs_f2 = self.ftype in ('bandpass', 'bandstop')
        c = TEXT_PRI if needs_f2 else TEXT_SEC
        self.sl_f2.label.set_color(c)
        self.sl_f2.valtext.set_color('#ffffff' if needs_f2 else TEXT_SEC)

    def _reset(self, _):
        for sl in [self.sl_ord, self.sl_f1, self.sl_f2, self.sl_fs, self.sl_kb]:
            sl.reset()

    def _update(self, _):
        self.order    = int(self.sl_ord.val)
        self.f1       = float(self.sl_f1.val)
        self.f2       = float(self.sl_f2.val)
        self.fs       = float(self.sl_fs.val)
        self.kaiser_b = float(self.sl_kb.val)

        nyq = self.fs / 2.0
        min_gap = max(10.0, nyq * 0.01)

        # Clamp f1, f2
        self.f1 = np.clip(self.f1, 10, nyq - min_gap - 1)
        if self.f2 <= self.f1 + min_gap:
            self.f2 = self.f1 + min_gap
        self.f2 = np.clip(self.f2, self.f1 + min_gap, nyq - 1)

        self.sl_f1.valmax = nyq - min_gap - 1
        self.sl_f2.valmax = nyq - 1
        self.sl_f1.ax.set_xlim(10, nyq - min_gap - 1)
        self.sl_f2.ax.set_xlim(11, nyq - 1)

        try:
            h = design_fir(
                self.ftype, self.order,
                self.f1, self.f2,
                self.fs, self.win_key, self.kaiser_b)
            self._plot_all(h)
        except Exception as e:
            print(f'[FIR design error] {e}')

    # ── Plots ──────────────────────────────────────────────────────────────

    def _plot_all(self, h):
        self._plot_mag_db(h)
        self._plot_mag_lin(h)
        self._plot_phase(h)
        self._plot_group_delay(h)
        self._plot_impulse(h)
        self._plot_window()
        self._update_metrics(h)
        self.fig.canvas.draw_idle()

    # — Magnitude (dB) ──────────────────────────────────────────────────────

    def _plot_mag_db(self, h):
        ax = self.ax_mag
        ax.cla(); _style_ax(ax, 'Magnitude response (dB)',
                            'Frequency (Hz)', 'Magnitude (dB)')

        w, H, mag_db, mag_lin, phase = freq_response(h, self.fs)

        ax.semilogx(w, mag_db, color=C_MAG_DB, linewidth=1.4)
        ax.fill_between(w, mag_db, -160, alpha=0.09, color=C_MAG_DB)

        # Cutoff markers
        self._add_cutoff_lines(ax, mag_db.min() - 5, 5)
        ax.axhline(-3,   color='#5F5E5A', linewidth=0.8, linestyle=':', label='−3 dB')
        ax.axhline(-6,   color='#3a3a3a', linewidth=0.6, linestyle=':')
        ax.axhline(-60,  color='#3a3a3a', linewidth=0.6, linestyle=':')

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(max(mag_db.min() - 10, -160), 5)
        ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C, loc='lower left')

        # Annotate min stopband attenuation
        sb_min = mag_db.min()
        ax.text(0.98, 0.04, f'Min: {sb_min:.1f} dB',
                transform=ax.transAxes, ha='right', va='bottom',
                color=C_MAG_DB, fontsize=7)

    # — Magnitude (linear) ──────────────────────────────────────────────────

    def _plot_mag_lin(self, h):
        ax = self.ax_magL
        ax.cla(); _style_ax(ax, 'Magnitude response (linear)',
                            'Frequency (Hz)', 'Magnitude')

        w, H, mag_db, mag_lin, phase = freq_response(h, self.fs)

        ax.semilogx(w, mag_lin, color=C_MAG_LIN, linewidth=1.4)
        ax.fill_between(w, mag_lin, 0, alpha=0.10, color=C_MAG_LIN)
        self._add_cutoff_lines(ax, 0, 1.1)
        ax.axhline(1.0 / np.sqrt(2), color='#5F5E5A', linewidth=0.8,
                   linestyle=':', label='1/√2 (−3 dB)')

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C, loc='lower right')

    # — Phase ────────────────────────────────────────────────────────────────

    def _plot_phase(self, h):
        ax = self.ax_phase
        ax.cla(); _style_ax(ax, 'Phase response (linear-phase FIR)',
                            'Frequency (Hz)', 'Phase (°)')

        w, H, mag_db, mag_lin, phase = freq_response(h, self.fs)

        ax.semilogx(w, phase, color=C_PHASE, linewidth=1.2)
        self._add_cutoff_lines(ax, phase.min() - 10, phase.max() + 10)

        ax.set_xlim(w[1], self.fs / 2)
        ax.text(0.02, 0.04,
                'Linear phase → constant group delay',
                transform=ax.transAxes, color=TEXT_SEC,
                fontsize=7, va='bottom')

    # — Group delay ──────────────────────────────────────────────────────────

    def _plot_group_delay(self, h):
        ax = self.ax_gd
        ax.cla(); _style_ax(ax, 'Group delay',
                            'Frequency (Hz)', 'Group delay (samples)')

        w, gd = grp_delay(h, self.fs)

        ax.semilogx(w, gd, color=C_GRPDLY, linewidth=1.2)

        # Theoretical constant group delay for linear-phase FIR = (N-1)/2
        N   = len(h)
        gd0 = (N - 1) / 2.0
        ax.axhline(gd0, color='#5F5E5A', linewidth=0.9,
                   linestyle='--', label=f'(N−1)/2 = {gd0:.1f}')

        # Latency in ms
        lat_ms = gd0 / self.fs * 1000
        ax.text(0.98, 0.92, f'Latency: {lat_ms:.2f} ms',
                transform=ax.transAxes, ha='right', va='top',
                color=C_GRPDLY, fontsize=7)

        ax.set_xlim(w[1], self.fs / 2)
        ax.set_ylim(0, gd0 * 1.5)
        ax.legend(fontsize=7, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C)

    # — Impulse response ─────────────────────────────────────────────────────

    def _plot_impulse(self, h):
        ax = self.ax_imp
        ax.cla(); _style_ax(ax, 'Impulse response (FIR coefficients)',
                            'Tap index (n)', 'h[n]')

        n = np.arange(len(h))
        ax.stem(n, h, linefmt=f'{C_IMPULSE}', markerfmt=f'o',
                basefmt='none', use_line_collection=True)

        # Set stem colours manually
        markerline, stemlines, _ = ax.stem(
            n, h, linefmt='-', markerfmt='o', basefmt='none',
            use_line_collection=True)
        markerline.set_color(C_IMPULSE)
        markerline.set_markersize(3)
        stemlines.set_color(C_IMPULSE)
        stemlines.set_linewidth(0.8)
        stemlines.set_alpha(0.7)

        # Highlight centre tap (linear-phase symmetry axis)
        mid = (len(h) - 1) // 2
        ax.axvline(mid, color=C_CUTOFF, linewidth=1,
                   linestyle='--', alpha=0.7, label=f'Centre tap (n={mid})')

        ax.set_xlim(-1, len(h))
        ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C)

        # Symmetry annotation
        ax.text(0.98, 0.92,
                f'{len(h)} taps  (order {len(h)-1})',
                transform=ax.transAxes, ha='right', va='top',
                color=TEXT_SEC, fontsize=7)

    # — Window shape ─────────────────────────────────────────────────────────

    def _plot_window(self):
        ax = self.ax_win
        ax.cla(); _style_ax(ax, f'Window shape — {WINDOW_NAMES[WINDOW_KEYS.index(self.win_key)]}',
                            'Sample index', 'Amplitude')

        N   = self.order + 1
        w   = make_window(self.win_key, N, self.kaiser_b)
        n   = np.arange(N)

        ax.fill_between(n, w, 0, alpha=0.25, color=C_WINDOW)
        ax.plot(n, w, color=C_WINDOW, linewidth=1.5)
        ax.set_xlim(0, N - 1)
        ax.set_ylim(-0.05, 1.1)

        # Overlay frequency response of the window itself
        ax2 = ax.twinx()
        ax2.set_facecolor('none')
        ax2.tick_params(colors=TEXT_SEC, labelsize=6)
        W    = np.abs(fft(w, n=4096))
        W_db = 20 * np.log10(W / W.max() + 1e-15)
        freqs_w = np.arange(len(W)) / len(W)
        # Show first half, up to first null
        half = len(W) // 2
        ax2.plot(freqs_w[:half] * (N - 1),
                 W_db[:half], color='#E24B4A',
                 linewidth=0.8, alpha=0.6, linestyle='--')
        ax2.set_ylim(-120, 10)
        ax2.set_ylabel('Window spectrum (dB)', color=TEXT_SEC, fontsize=7)
        ax2.tick_params(colors=TEXT_SEC, labelsize=6)

        # Key stats
        sl = -np.min(W_db[1:])          # approx sidelobe level
        ax.text(0.02, 0.06,
                f'Sidelobe: −{sl:.0f} dB',
                transform=ax.transAxes, color=C_WINDOW,
                fontsize=7, va='bottom')

    # ── Cutoff marker helper ───────────────────────────────────────────────

    def _add_cutoff_lines(self, ax, ymin, ymax):
        """Draw vertical cutoff lines appropriate for current filter type."""
        style = dict(linewidth=1, linestyle='--', alpha=0.8)
        if self.ftype in ('lowpass', 'highpass'):
            ax.axvline(self.f1, color=C_CUTOFF,
                       label=f'f1={_fmt_hz(self.f1)}', **style)
        else:
            ax.axvline(self.f1, color='#378ADD',
                       label=f'f1={_fmt_hz(self.f1)}', **style)
            ax.axvline(self.f2, color=C_POLE,
                       label=f'f2={_fmt_hz(self.f2)}', **style)

    # ── Metrics bar ───────────────────────────────────────────────────────

    def _update_metrics(self, h):
        w, H, mag_db, mag_lin, phase = freq_response(h, self.fs)
        N   = len(h)
        gd0 = (N - 1) / 2.0
        lat = gd0 / self.fs * 1000    # ms

        # Stopband attenuation & passband ripple depend on filter type
        nyq = self.fs / 2
        if self.ftype == 'lowpass':
            pb_rip = passband_ripple(w, mag_db, 0, self.f1 * 0.8)
            sb_att = stopband_atten(w, mag_db, self.f1 * 1.2, nyq)
        elif self.ftype == 'highpass':
            pb_rip = passband_ripple(w, mag_db, self.f1 * 1.2, nyq)
            sb_att = stopband_atten(w, mag_db, 0, self.f1 * 0.8)
        elif self.ftype == 'bandpass':
            pb_rip = passband_ripple(w, mag_db,
                                     self.f1 * 1.1, self.f2 * 0.9)
            sb_att = min(
                stopband_atten(w, mag_db, 0, self.f1 * 0.9),
                stopband_atten(w, mag_db, self.f2 * 1.1, nyq))
        else:  # bandstop
            pb_rip = min(
                passband_ripple(w, mag_db, 0, self.f1 * 0.9),
                passband_ripple(w, mag_db, self.f2 * 1.1, nyq))
            sb_att = stopband_atten(w, mag_db,
                                    self.f1 * 1.1, self.f2 * 0.9)

        win_lbl = WINDOW_NAMES[WINDOW_KEYS.index(self.win_key)]
        kb_str  = f'  β={self.kaiser_b:.1f}' if self.win_key == 'kaiser' else ''

        txt = (
            f'  {self.ftype.capitalize()}   '
            f'Window: {win_lbl}{kb_str}   '
            f'Order: {N-1}  ({N} taps)   |   '
            f'Passband ripple: {pb_rip:.2f} dB   '
            f'Stopband atten: {sb_att:.1f} dB   |   '
            f'Group delay: {gd0:.1f} samples = {lat:.2f} ms'
        )
        self.met_txt.set_text(txt)


# ─── Non-interactive API ──────────────────────────────────────────────────────

def design_and_print(ftype='lowpass', order=64, f1=1000.0, f2=3000.0,
                     fs=44100.0, window='hamming', kaiser_beta=8.6):
    """
    Design an FIR filter and print key specs + first/last 5 coefficients.

    Examples
    --------
    >>> h = design_and_print('lowpass',  order=64,  f1=1000, fs=44100, window='hamming')
    >>> h = design_and_print('highpass', order=128, f1=500,  fs=44100, window='blackman')
    >>> h = design_and_print('bandpass', order=128, f1=300,  f2=3400,  fs=8000, window='kaiser', kaiser_beta=8.6)
    >>> h = design_and_print('bandstop', order=256, f1=45,   f2=55,    fs=8000, window='blackmanharris')
    """
    h = design_fir(ftype, order, f1, f2, fs, window, kaiser_beta)

    w, H, mag_db, mag_lin, phase = freq_response(h, fs)
    gd0 = (len(h) - 1) / 2.0
    lat = gd0 / fs * 1000

    print('=' * 60)
    print(f' FIR {ftype.upper()}   order {len(h)-1}  ({len(h)} taps)')
    print(f' Window : {window}' + (f'  β={kaiser_beta}' if window=='kaiser' else ''))
    print(f' f1     : {_fmt_hz(f1)}')
    if ftype in ('bandpass', 'bandstop'):
        print(f' f2     : {_fmt_hz(f2)}')
    print(f' Sample : {_fmt_hz(fs)}')
    print('-' * 60)
    print(f' Taps         : {len(h)}')
    print(f' Group delay  : {gd0:.1f} samples = {lat:.3f} ms')
    print(f' Min stopband : {mag_db.min():.2f} dB')
    print(f' h[0:5]       : {np.round(h[:5],  8).tolist()}')
    print(f' h[-5:]       : {np.round(h[-5:], 8).tolist()}')
    print('=' * 60)
    return h


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    FIRDesigner()

    # ── Non-interactive examples (uncomment to use) ──────────────────────────
    # h = design_and_print('lowpass',  order=64,  f1=1000, fs=44100, window='hamming')
    # h = design_and_print('highpass', order=128, f1=500,  fs=44100, window='blackman')
    # h = design_and_print('bandpass', order=128, f1=300,  f2=3400,  fs=8000,  window='kaiser', kaiser_beta=8.6)
    # h = design_and_print('bandstop', order=256, f1=45,   f2=55,    fs=44100, window='blackmanharris')
