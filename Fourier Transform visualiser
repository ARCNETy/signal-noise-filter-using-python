"""
Fourier Transform Visualiser
=============================
Interactive FFT/DFT analysis tool for WAV and CSV signals.
Upload a file, control the window, see the spectrum live.

Features:
  - File support  : WAV (mono & stereo), CSV (any numeric column)
  - Windows       : Rectangular, Hann, Hamming, Blackman, Kaiser, Flat-top
  - Displays      : Time-domain waveform, Magnitude spectrum (linear + dB),
                    Phase spectrum, Spectrogram (STFT)
  - Controls      : Window type, FFT size, overlap, zoom, channel select
  - Extras        : Peak frequency readout, harmonic markers, export spectrum CSV

Requirements:
scipy matplotlib numpy soundfile
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, RadioButtons, Button, CheckButtons
from scipy import signal
from scipy.fft import fft, fftfreq
import tkinter as tk
from tkinter import filedialog

# Optional: soundfile for WAV reading (fallback to scipy.io.wavfile)
try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False
    from scipy.io import wavfile


# ─── Style constants ──────────────────────────────────────────────────────────

BG_DARK   = '#0d0d1a'
BG_PANEL  = '#1a1a2e'
BG_CTRL   = '#0d0d1a'
GRID_C    = '#2a2a3e'
TEXT_PRI  = '#B4B2A9'
TEXT_SEC  = '#888780'
SPINE_C   = '#2a2a3e'

C_WAVEFORM  = '#378ADD'   # blue
C_MAG_LIN   = '#F5A623'   # amber
C_MAG_DB    = '#E24B4A'   # red
C_PHASE     = '#1D9E75'   # green
C_PEAK      = '#C77DFF'   # violet
C_HARM      = '#FF6B6B'   # coral

WINDOW_NAMES  = ['Rectangular', 'Hann', 'Hamming', 'Blackman', 'Kaiser', 'Flat-top']
WINDOW_KEYS   = ['boxcar',      'hann', 'hamming', 'blackman', 'kaiser', 'flattop']
FFT_SIZES     = [256, 512, 1024, 2048, 4096, 8192]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_hz(f):
    if f >= 1e6: return f'{f/1e6:.2f} MHz'
    if f >= 1e3: return f'{f/1e3:.2f} kHz'
    return f'{f:.1f} Hz'

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(SPINE_C)
    ax.grid(True, color=GRID_C, linewidth=0.5)
    if title:  ax.set_title(title,  color=TEXT_PRI, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=8)

def make_window(name, N, beta=8.0):
    if name == 'kaiser':
        return signal.windows.kaiser(N, beta)
    return signal.get_window(name, N)

def compute_fft(x, fs, win_key, nfft, beta=8.0):
    """Return (freqs, magnitude_linear, magnitude_dB, phase_deg)."""
    N   = min(nfft, len(x))
    seg = x[:N].copy()
    w   = make_window(win_key, N, beta)

    # Coherent gain correction
    cg  = np.sum(w)
    seg = seg * w

    X       = fft(seg, n=nfft)
    freqs   = fftfreq(nfft, d=1.0 / fs)

    half    = nfft // 2
    freqs   = freqs[:half]
    X       = X[:half]

    mag_lin = (2 / cg) * np.abs(X)          # two-sided → one-sided
    mag_lin[0] /= 2                          # DC bin: no doubling
    mag_db  = 20 * np.log10(mag_lin + 1e-12)
    phase   = np.angle(X, deg=True)

    return freqs, mag_lin, mag_db, phase

def compute_spectrogram(x, fs, win_key, nfft, overlap_pct, beta=8.0):
    hop    = int(nfft * (1 - overlap_pct / 100))
    hop    = max(hop, 1)
    win    = make_window(win_key, nfft, beta)
    f, t, S = signal.spectrogram(x, fs=fs, window=win,
                                  nperseg=nfft, noverlap=nfft - hop,
                                  scaling='spectrum')
    S_db   = 10 * np.log10(S + 1e-12)
    return f, t, S_db


# ─── File loading ─────────────────────────────────────────────────────────────

def load_wav(path):
    """Return (samples_float, fs, n_channels)."""
    if HAS_SF:
        data, fs = sf.read(path, always_2d=True)   # shape (N, ch)
    else:
        fs, data = wavfile.read(path)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        data = data.astype(np.float64)
        # Normalise int types
        if data.dtype.kind == 'i':
            data /= np.iinfo(data.dtype).max
    return data.astype(np.float64), int(fs), data.shape[1]


def load_csv(path):
    """Return (samples_float, assumed_fs=1000, n_columns)."""
    raw = np.genfromtxt(path, delimiter=',', skip_header=1)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    # Drop non-finite rows
    mask = np.all(np.isfinite(raw), axis=1)
    raw  = raw[mask]
    # If first column looks like a time axis, derive fs from it
    fs = 1000.0
    start_col = 0
    if raw.shape[1] > 1:
        t_col = raw[:, 0]
        diffs = np.diff(t_col)
        if np.all(diffs > 0) and np.std(diffs) / np.mean(diffs) < 0.05:
            fs = 1.0 / np.mean(diffs)
            start_col = 1
    data = raw[:, start_col:]
    return data.astype(np.float64), fs, data.shape[1]


# ─── Main GUI ─────────────────────────────────────────────────────────────────

class FFTVisualiser:

    def __init__(self, filepath=None):
        # Signal state
        self.raw_data   = None   # (N, ch) float64
        self.fs         = 1000.0
        self.n_ch       = 1
        self.ch_idx     = 0
        self.filepath   = None

        # FFT parameters
        self.win_key    = 'hann'
        self.nfft       = 2048
        self.overlap    = 75     # %
        self.kaiser_b   = 8.0
        self.show_db    = True
        self.show_phase = False
        self.show_spect = True
        self.show_harm  = False

        # View zoom
        self.f_min = 0.0
        self.f_max = None        # set after load

        self._build_gui()

        if filepath:
            self._load_file(filepath)
        else:
            self._load_demo()

        plt.show()

    # ── GUI build ─────────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(15, 9), facecolor=BG_PANEL)
        self.fig.canvas.manager.set_window_title('Fourier Transform Visualiser')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3.2], wspace=0.04,
            left=0.01, right=0.99, top=0.97, bottom=0.04)

        # ── Left controls ─────────────────────────────────────────────────────
        cgs = gridspec.GridSpecFromSubplotSpec(
            12, 1, subplot_spec=outer[0], hspace=0.55)

        ax_open   = self.fig.add_subplot(cgs[0])
        ax_info   = self.fig.add_subplot(cgs[1])
        ax_win    = self.fig.add_subplot(cgs[2:5])
        ax_nfft   = self.fig.add_subplot(cgs[5])
        ax_ovlp   = self.fig.add_subplot(cgs[6])
        ax_kaiser = self.fig.add_subplot(cgs[7])
        ax_fmin   = self.fig.add_subplot(cgs[8])
        ax_fmax   = self.fig.add_subplot(cgs[9])
        ax_opts   = self.fig.add_subplot(cgs[10])
        ax_export = self.fig.add_subplot(cgs[11])

        for ax in [ax_open, ax_info, ax_win, ax_nfft, ax_ovlp,
                   ax_kaiser, ax_fmin, ax_fmax, ax_opts, ax_export]:
            ax.set_facecolor(BG_CTRL)

        # Open file button
        self.btn_open = Button(ax_open, 'Open WAV / CSV',
                               color=BG_CTRL, hovercolor='#1a1a2e')
        self.btn_open.label.set_color(TEXT_PRI)
        self.btn_open.label.set_fontsize(9)
        self.btn_open.on_clicked(self._on_open)

        # Info text
        ax_info.axis('off')
        self.txt_info = ax_info.text(
            0.5, 0.5, 'No file loaded — using demo signal',
            ha='center', va='center', color=TEXT_SEC,
            fontsize=7, transform=ax_info.transAxes, wrap=True)

        # Window type radio
        self.radio_win = RadioButtons(ax_win, WINDOW_NAMES,
                                      active=1, activecolor=C_MAG_LIN)
        ax_win.set_title('Window function', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_win.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(8)
        self.radio_win.on_clicked(self._on_window)

        # FFT size slider  (index into FFT_SIZES list)
        sc = BG_CTRL
        self.sl_nfft = Slider(ax_nfft, 'FFT size\n(index)',
                              0, len(FFT_SIZES) - 1,
                              valinit=FFT_SIZES.index(self.nfft), valstep=1,
                              color=C_MAG_LIN, initcolor='none', facecolor=sc)
        self.sl_nfft.label.set_color(TEXT_PRI); self.sl_nfft.label.set_fontsize(8)
        self.sl_nfft.valtext.set_color('#ffffff'); self.sl_nfft.valtext.set_fontsize(8)
        self.sl_nfft.on_changed(self._update_nfft_label)

        self.sl_ovlp = Slider(ax_ovlp, 'Overlap\n(%)',
                              0, 95, valinit=self.overlap, valstep=5,
                              color=C_PHASE, initcolor='none', facecolor=sc)
        self.sl_ovlp.label.set_color(TEXT_PRI); self.sl_ovlp.label.set_fontsize(8)
        self.sl_ovlp.valtext.set_color('#ffffff'); self.sl_ovlp.valtext.set_fontsize(8)
        self.sl_ovlp.on_changed(self._update)

        self.sl_kaiser = Slider(ax_kaiser, 'Kaiser β',
                                1, 20, valinit=self.kaiser_b,
                                color=C_PEAK, initcolor='none', facecolor=sc)
        self.sl_kaiser.label.set_color(TEXT_SEC); self.sl_kaiser.label.set_fontsize(8)
        self.sl_kaiser.valtext.set_color(TEXT_SEC); self.sl_kaiser.valtext.set_fontsize(8)
        self.sl_kaiser.on_changed(self._update)

        self.sl_fmin = Slider(ax_fmin, 'Zoom\nf min (Hz)',
                              0, 20000, valinit=0, valstep=10,
                              color=C_WAVEFORM, initcolor='none', facecolor=sc)
        self.sl_fmin.label.set_color(TEXT_PRI); self.sl_fmin.label.set_fontsize(8)
        self.sl_fmin.valtext.set_color('#ffffff'); self.sl_fmin.valtext.set_fontsize(8)
        self.sl_fmin.on_changed(self._update)

        self.sl_fmax = Slider(ax_fmax, 'Zoom\nf max (Hz)',
                              1, 96000, valinit=22050, valstep=10,
                              color=C_WAVEFORM, initcolor='none', facecolor=sc)
        self.sl_fmax.label.set_color(TEXT_PRI); self.sl_fmax.label.set_fontsize(8)
        self.sl_fmax.valtext.set_color('#ffffff'); self.sl_fmax.valtext.set_fontsize(8)
        self.sl_fmax.on_changed(self._update)

        # Toggle checkboxes
        self.chk = CheckButtons(ax_opts,
                                ['Magnitude (dB)', 'Phase', 'Spectrogram', 'Harmonics'],
                                [True, False, True, False])
        ax_opts.set_title('Display options', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.chk.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(8)
        self.chk.on_clicked(self._on_checks)

        # Export button
        self.btn_exp = Button(ax_export, 'Export spectrum CSV',
                              color=BG_CTRL, hovercolor='#1a1a2e')
        self.btn_exp.label.set_color(TEXT_PRI)
        self.btn_exp.label.set_fontsize(9)
        self.btn_exp.on_clicked(self._export_csv)

        # ── Right: 2×2 plot grid ──────────────────────────────────────────────
        pgs = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[1], hspace=0.42, wspace=0.32)

        self.ax_time  = self.fig.add_subplot(pgs[0, 0])
        self.ax_mag   = self.fig.add_subplot(pgs[0, 1])
        self.ax_phase = self.fig.add_subplot(pgs[1, 0])
        self.ax_spect = self.fig.add_subplot(pgs[1, 1])

        for ax in [self.ax_time, self.ax_mag, self.ax_phase, self.ax_spect]:
            _style_ax(ax)

        # Status / metrics bar
        self.met_ax = self.fig.add_axes([0.34, 0.005, 0.65, 0.028])
        self.met_ax.axis('off')
        self.met_txt = self.met_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            color=TEXT_PRI, fontsize=8, transform=self.met_ax.transAxes)

    # ── File handling ─────────────────────────────────────────────────────────

    def _on_open(self, _):
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title='Open audio or CSV file',
            filetypes=[('WAV files', '*.wav'),
                       ('CSV files', '*.csv'),
                       ('All files', '*.*')])
        root.destroy()
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.wav':
                data, fs, n_ch = load_wav(path)
            elif ext == '.csv':
                data, fs, n_ch = load_csv(path)
            else:
                print(f"Unsupported file type: {ext}")
                return

            self.raw_data  = data
            self.fs        = fs
            self.n_ch      = n_ch
            self.ch_idx    = 0
            self.filepath  = path
            self.f_max     = fs / 2

            # Reset zoom sliders
            self.sl_fmin.set_val(0)
            self.sl_fmax.set_val(self.f_max)
            self.sl_fmax.valmax = self.f_max
            self.sl_fmax.ax.set_xlim(1, self.f_max)

            name = os.path.basename(path)
            dur  = len(data) / fs
            info = (f'{name}\n'
                    f'{_fmt_hz(fs)} · {n_ch}ch · {dur:.2f}s · {len(data)} samples')
            self.txt_info.set_text(info)
            self._update(None)

        except Exception as e:
            self.txt_info.set_text(f'Error loading file:\n{e}')
            print(f"Load error: {e}")

    def _load_demo(self):
        """Multi-tone demo signal: 100 + 500 + 1200 + 3000 Hz."""
        fs  = 44100
        t   = np.linspace(0, 1.0, fs, endpoint=False)
        sig = (0.6 * np.sin(2 * np.pi * 100  * t) +
               0.4 * np.sin(2 * np.pi * 500  * t) +
               0.3 * np.sin(2 * np.pi * 1200 * t) +
               0.15* np.sin(2 * np.pi * 3000 * t) +
               0.05* np.random.randn(len(t)))
        self.raw_data  = sig[:, np.newaxis].astype(np.float64)
        self.fs        = float(fs)
        self.n_ch      = 1
        self.ch_idx    = 0
        self.filepath  = None
        self.f_max     = fs / 2

        self.sl_fmax.set_val(self.f_max)
        self.sl_fmax.valmax = self.f_max
        self.sl_fmax.ax.set_xlim(1, self.f_max)

        self.txt_info.set_text(
            'DEMO: 100 + 500 + 1200 + 3000 Hz + noise\n(open a file to replace)')
        self._update(None)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_window(self, label):
        idx = WINDOW_NAMES.index(label)
        self.win_key = WINDOW_KEYS[idx]
        # Enable/disable Kaiser slider visual cue
        active = self.win_key == 'kaiser'
        self.sl_kaiser.label.set_color(TEXT_PRI if active else TEXT_SEC)
        self.sl_kaiser.valtext.set_color('#ffffff' if active else TEXT_SEC)
        self._update(None)

    def _update_nfft_label(self, val):
        idx = int(round(val))
        self.nfft = FFT_SIZES[idx]
        self.sl_nfft.valtext.set_text(str(self.nfft))
        self._update(None)

    def _on_checks(self, label):
        states = self.chk.get_status()
        labels = ['Magnitude (dB)', 'Phase', 'Spectrogram', 'Harmonics']
        d      = dict(zip(labels, states))
        self.show_db    = d['Magnitude (dB)']
        self.show_phase = d['Phase']
        self.show_spect = d['Spectrogram']
        self.show_harm  = d['Harmonics']
        self._update(None)

    def _update(self, _):
        if self.raw_data is None:
            return

        self.overlap   = int(self.sl_ovlp.val)
        self.kaiser_b  = float(self.sl_kaiser.val)
        self.f_min     = float(self.sl_fmin.val)
        self.f_max_view= float(self.sl_fmax.val)
        if self.f_min >= self.f_max_view:
            self.f_min = 0.0

        # Get mono signal for selected channel
        x = self.raw_data[:, min(self.ch_idx, self.raw_data.shape[1] - 1)]

        freqs, mag_lin, mag_db, phase = compute_fft(
            x, self.fs, self.win_key, self.nfft, self.kaiser_b)

        self._plot_time(x)
        self._plot_mag(freqs, mag_lin, mag_db)
        self._plot_phase(freqs, phase)
        self._plot_spectrogram(x)
        self._update_metrics(freqs, mag_lin, mag_db)
        self.fig.canvas.draw_idle()

    # ── Individual plots ──────────────────────────────────────────────────────

    def _plot_time(self, x):
        ax = self.ax_time
        ax.cla(); _style_ax(ax, 'Time domain', 'Time (s)', 'Amplitude')

        N  = len(x)
        t  = np.arange(N) / self.fs

        # Downsample for display if very long
        max_pts = 8000
        if N > max_pts:
            step = N // max_pts
            t = t[::step]; x_plot = x[::step]
        else:
            x_plot = x

        ax.plot(t, x_plot, color=C_WAVEFORM, linewidth=0.7)
        ax.set_xlim(t[0], t[-1])

        # RMS line
        rms = np.sqrt(np.mean(x ** 2))
        ax.axhline( rms, color='#BA7517', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.axhline(-rms, color='#BA7517', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.text(t[-1], rms * 1.05, f'RMS={rms:.3f}',
                color='#BA7517', fontsize=6, ha='right')

    def _plot_mag(self, freqs, mag_lin, mag_db):
        ax = self.ax_mag
        ax.cla()

        y     = mag_db if self.show_db else mag_lin
        color = C_MAG_DB if self.show_db else C_MAG_LIN
        ylabel= 'Magnitude (dB)' if self.show_db else 'Magnitude (linear)'

        _style_ax(ax, 'Magnitude spectrum', 'Frequency (Hz)', ylabel)

        # Mask to zoom range
        mask   = (freqs >= self.f_min) & (freqs <= self.f_max_view)
        fz, yz = freqs[mask], y[mask]

        ax.plot(fz, yz, color=color, linewidth=0.9)
        ax.fill_between(fz, yz, yz.min() - 5, alpha=0.12, color=color)

        # Peak markers
        if len(yz) > 0:
            top_n  = min(5, len(yz))
            peaks, props = signal.find_peaks(yz, height=np.percentile(yz, 70),
                                              distance=max(1, len(yz) // 40))
            if len(peaks) > 0:
                # Sort by height, take top N
                heights = yz[peaks]
                order   = np.argsort(heights)[::-1][:top_n]
                for i, pk in enumerate(peaks[order]):
                    fx = fz[pk]
                    fy = yz[pk]
                    ax.plot(fx, fy, 'v', color=C_PEAK, markersize=5, zorder=5)
                    ax.annotate(
                        _fmt_hz(fx),
                        xy=(fx, fy), xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', fontsize=6, color=C_PEAK,
                        arrowprops=dict(arrowstyle='-', color=C_PEAK,
                                        lw=0.5) if i < 3 else None)

            # Harmonic markers from strongest peak
            if self.show_harm and len(peaks) > 0:
                f0 = fz[peaks[np.argmax(yz[peaks])]]
                for n in range(2, 8):
                    fh = f0 * n
                    if fh > self.f_max_view: break
                    ax.axvline(fh, color=C_HARM, linewidth=0.7,
                               linestyle=':', alpha=0.7)
                    ax.text(fh, yz.max() * 0.95,
                            f'×{n}', color=C_HARM, fontsize=6, ha='center')

        ax.set_xlim(self.f_min, self.f_max_view)

    def _plot_phase(self, freqs, phase):
        ax = self.ax_phase
        ax.cla()

        if self.show_phase:
            _style_ax(ax, 'Phase spectrum', 'Frequency (Hz)', 'Phase (°)')
            mask = (freqs >= self.f_min) & (freqs <= self.f_max_view)
            ax.plot(freqs[mask], phase[mask], color=C_PHASE,
                    linewidth=0.8, alpha=0.9)
            ax.set_xlim(self.f_min, self.f_max_view)
            ax.set_ylim(-185, 185)
            ax.axhline(0, color=GRID_C, linewidth=0.6)
        else:
            _style_ax(ax, 'Phase (disabled)', '', '')
            ax.text(0.5, 0.5, 'Enable "Phase" in options',
                    ha='center', va='center', color=TEXT_SEC,
                    fontsize=9, transform=ax.transAxes)

    def _plot_spectrogram(self, x):
        ax = self.ax_spect
        ax.cla()
        _style_ax(ax, 'Spectrogram (STFT)', 'Time (s)', 'Frequency (Hz)')

        if not self.show_spect:
            ax.text(0.5, 0.5, 'Enable "Spectrogram" in options',
                    ha='center', va='center', color=TEXT_SEC,
                    fontsize=9, transform=ax.transAxes)
            return

        try:
            f, t, S_db = compute_spectrogram(
                x, self.fs, self.win_key, self.nfft, self.overlap, self.kaiser_b)

            vmin = np.percentile(S_db, 20)
            vmax = np.percentile(S_db, 99.5)

            im = ax.pcolormesh(t, f, S_db,
                               shading='auto', cmap='magma',
                               vmin=vmin, vmax=vmax)

            # Frequency zoom
            ax.set_ylim(self.f_min, self.f_max_view)

            cb = self.fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
            cb.ax.tick_params(colors=TEXT_SEC, labelsize=6)
            cb.set_label('dB', color=TEXT_SEC, fontsize=7)

        except Exception as e:
            ax.text(0.5, 0.5, f'Spectrogram error:\n{e}',
                    ha='center', va='center', color=C_MAG_DB,
                    fontsize=8, transform=ax.transAxes)

    def _update_metrics(self, freqs, mag_lin, mag_db):
        x   = self.raw_data[:, self.ch_idx]
        rms = np.sqrt(np.mean(x ** 2))
        peak_amp = np.max(np.abs(x))

        # Peak frequency
        mask = (freqs >= self.f_min) & (freqs <= self.f_max_view)
        if mask.any():
            sub_f   = freqs[mask]
            sub_db  = mag_db[mask]
            pk_idx  = np.argmax(sub_db)
            pk_freq = sub_f[pk_idx]
            pk_db   = sub_db[pk_idx]
        else:
            pk_freq = pk_db = 0.0

        # THD estimate (if harmonics visible)
        thd_str = ''
        if self.show_harm and pk_freq > 0:
            fund_mag = mag_lin[np.argmin(np.abs(freqs - pk_freq))]
            harm_power = 0.0
            for n in range(2, 6):
                fh = pk_freq * n
                if fh >= self.fs / 2: break
                ih = np.argmin(np.abs(freqs - fh))
                harm_power += mag_lin[ih] ** 2
            thd = 100 * np.sqrt(harm_power) / (fund_mag + 1e-12)
            thd_str = f'   THD≈{thd:.1f}%'

        n_pts  = len(x)
        dur    = n_pts / self.fs
        win_lbl = WINDOW_NAMES[WINDOW_KEYS.index(self.win_key)]

        txt = (
            f'  {_fmt_hz(self.fs)} · {n_pts} pts · {dur:.3f}s   |   '
            f'Window: {win_lbl}   FFT: {self.nfft}   Overlap: {self.overlap}%   |   '
            f'Peak: {_fmt_hz(pk_freq)} ({pk_db:.1f} dB)   '
            f'RMS: {rms:.4f}   Peak amp: {peak_amp:.4f}'
            f'{thd_str}'
        )
        self.met_txt.set_text(txt)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self, _):
        if self.raw_data is None:
            return
        x = self.raw_data[:, self.ch_idx]
        freqs, mag_lin, mag_db, phase = compute_fft(
            x, self.fs, self.win_key, self.nfft, self.kaiser_b)

        root = tk.Tk(); root.withdraw()
        out_path = filedialog.asksaveasfilename(
            title='Save spectrum as CSV',
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')])
        root.destroy()

        if out_path:
            header = 'frequency_hz,magnitude_linear,magnitude_db,phase_deg'
            data   = np.column_stack([freqs, mag_lin, mag_db, phase])
            np.savetxt(out_path, data, delimiter=',',
                       header=header, comments='')
            self.txt_info.set_text(
                self.txt_info.get_text() + f'\nExported: {os.path.basename(out_path)}')
            self.fig.canvas.draw_idle()
            print(f"Spectrum exported to: {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    FFTVisualiser(filepath)
