"""
pSTAM Design Toolbox
====================
A desktop interface for sizing Pneumatic Silicone Tube Artificial Muscles (pSTAM).

Workflow: bound the four design variables (L, D, T, P), review the ANN
surrogate models (M1 contraction, M2 inflation time, M3 deflation time),
define an objective and constraints over those models, then run particle-swarm
optimisation and export the recommended design.

Modules used:
    config.get_default_config()     configuration schema
    pstam.optimizer.PAMOptimizer    particle-swarm optimiser
    pstam.objective.CostFunction    objective and detailed evaluation
    pstam.ann.PAMModels             surrogate forward inference

Required packages: numpy, pandas, pyswarms, PyQt5. matplotlib is optional and
enables the on-screen convergence plot and the PDF report; the optimiser and
the CSV / JSON / TXT exports do not depend on it.
"""

import os
import sys
import json
import math
import glob

import numpy as np
import pandas as pd

# Plots are rendered to images via the Agg backend; matplotlib is optional.
import io
HAVE_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    HAVE_MPL = False

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QTimer, QRectF, QPointF, QSize
from PyQt5.QtGui import (
    QFont, QFontDatabase, QColor, QPainter, QPen, QBrush, QPixmap, QMovie
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QStackedWidget,
    QLineEdit, QComboBox, QCheckBox, QRadioButton, QButtonGroup, QSlider,
    QTextBrowser, QFileDialog, QMessageBox, QSizePolicy,
    QGraphicsDropShadowEffect
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_default_config

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ASSET_DIR   = os.path.join(BASE_DIR, "assets")

# ════════════════════════════════════════════════════════════════════
#  PALETTE — "Atelier" (light theme)
# ════════════════════════════════════════════════════════════════════
CANVAS    = "#EEF1F3"   # cool paper background
CANVAS2   = "#E7ECEF"   # rails / deeper ground
SURFACE   = "#FFFFFF"   # cards
SURFACE2  = "#F6F8FA"   # nested / inset panels
LINE      = "#D7DEE4"   # borders
LINE_SOFT = "#E7ECF0"   # subtle dividers

INK       = "#16242E"   # primary text
BODY      = "#34434F"   # secondary text
MUTED     = "#4E5C68"   # tertiary text
FAINT     = "#6C7A85"   # quaternary text

PRIMARY   = "#0E7C86"   # petrol teal — accent
PRIMARY_D = "#0A626B"   # darker (hover/pressed)
PRIMARY_SOFT = "#DDEFF1"  # tint (selected backgrounds)
ACCENT    = "#C9772C"   # amber — used sparingly

# semantic / math-token colours (legible on white)
VAR    = "#1F7A4D"   # design variables (green)
FIX    = "#B0466E"   # fixed inputs (rose)
MODEL  = "#0E6E78"   # model calls (teal)
OP     = "#B5701C"   # operators / mode (amber-brown)
SWEEP  = "#B5701C"   # sweep
AGG    = "#6A52B0"   # aggregation (violet)
CON    = "#C0413B"   # constraints (brick)
GOOD   = "#2E9E6B"   # pass
BAD    = "#C0413B"   # violation

MODELS  = ["M1", "M2", "M3"]
AGGS    = ["single", "mean", "min", "max"]
DIRS    = ["≤", "≥"]
UNIT    = {"M1": "mm", "M2": "ms", "M3": "ms"}

MODEL_PHYS = {
    "M1": ("Quasi-static contraction", "mm"),
    "M2": ("Inflation time",           "ms"),
    "M3": ("Deflation time",           "ms"),
}
MODEL_USES_STROKE = {"M1": False, "M2": True, "M3": True}
MODEL_SWEEPABLES  = {"M1": ["F"], "M2": ["F", "stroke"], "M3": ["F", "stroke"]}

VAR_RANGE = {"L": (150, 350, "mm"), "D": (14, 26, "mm"),
             "T": (3, 6, "mm"),     "P": (20, 200, "kPa")}


# ════════════════════════════════════════════════════════════════════
#  Runtime diagnostics — real validation RMSE from weight files
# ════════════════════════════════════════════════════════════════════
def load_model_diagnostics():
    diag = {}
    for m in MODELS:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{m}_weights.csv"))
            tr = float(pd.to_numeric(df["train_mse_error"], errors="coerce").dropna().iloc[0])
            te = float(pd.to_numeric(df["test_mse_error"],  errors="coerce").dropna().iloc[0])
            hn = int(float(pd.to_numeric(df["hidden_layers_neurons"],
                                         errors="coerce").dropna().iloc[0]))
            act = str(df["layer_activation_function"].dropna().iloc[0])
            diag[m] = {"rmse_train": math.sqrt(tr), "rmse_test": math.sqrt(te),
                       "hidden": hn, "act": act, "ok": True}
        except Exception:
            diag[m] = {"ok": False, "rmse_train": None, "rmse_test": None,
                       "hidden": None, "act": None}
    return diag


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════
def pick_font(candidates, default="DejaVu Sans"):
    fams = set(QFontDatabase().families())
    for c in candidates:
        if c in fams:
            return c
    return default


def shadow(widget, *_a, **_k):
    # Depth is carried by the card borders; retained for call-site symmetry.
    return widget


def fig_to_pixmap(fig, width_px=None):
    """Render a matplotlib Figure to a QPixmap via Agg (no live Qt canvas)."""
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    pm = QPixmap()
    pm.loadFromData(buf.getvalue())
    if width_px and pm.width() > int(width_px):
        pm = pm.scaledToWidth(int(width_px), Qt.SmoothTransformation)
    return pm


def hrule(color=LINE_SOFT, h=1):
    f = QFrame(); f.setFixedHeight(h)
    f.setStyleSheet(f"background:{color}; border:none;")
    f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return f


def eyebrow(text, color=PRIMARY):
    l = QLabel(text.upper())
    l.setStyleSheet(f"color:{color}; letter-spacing:2px; font-size:11px;"
                    f"font-weight:800;")
    return l


def h1(text):
    l = QLabel(text); l.setObjectName("h1")
    return l


def caption(text, color=BODY):
    l = QLabel(text); l.setWordWrap(True)
    l.setStyleSheet(f"color:{color}; font-size:12.5px;")
    return l


def section(title, color=INK, rule=True):
    w = QWidget()
    lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)
    lb = QLabel(title)
    lb.setStyleSheet(f"color:{color}; font-size:13px; font-weight:800;"
                     f"letter-spacing:0.2px;")
    lay.addWidget(lb)
    if rule:
        lay.addWidget(hrule(), 1)
    return w


def card():
    c = QFrame(); c.setObjectName("card")
    return c


class Emblem(QWidget):
    """Vector pSTAM mark — nested contraction arcs."""
    def __init__(self, d=32, parent=None):
        super().__init__(parent); self._d = d; self.setFixedSize(d, d)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(2, 2, -2, -2)
        cx, cy = r.center().x(), r.center().y()
        pen = QPen(QColor(PRIMARY), 2.4); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        for sp in (0.0, 0.17, 0.34):
            rad = (r.width() / 2) * (1.0 - sp)
            from PyQt5.QtCore import QRectF
            p.drawArc(QRectF(cx - rad, cy - rad, rad * 2, rad * 2),
                      int(38 * 16), int(284 * 16))
        p.setPen(Qt.NoPen); p.setBrush(QColor(ACCENT))
        from PyQt5.QtCore import QPointF
        p.drawEllipse(QPointF(cx, cy), 2.4, 2.4)
        p.end()


class RangeBar(QWidget):
    """
    Training extent (lo..hi) with the selected [min,max] band highlighted.
    The selected min/max values are printed at the band's ends; the training
    endpoints are printed faintly at the extremes — so the bar reads as data.
    """
    def __init__(self, lo, hi, unit="", decimals=0, parent=None):
        super().__init__(parent)
        self._lo, self._hi = float(lo), float(hi)
        self._a, self._b = float(lo), float(hi)
        self._unit, self._dec = unit, decimals
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_band(self, a, b):
        self._a, self._b = float(a), float(b)
        self.update()

    def _fmt(self, v):
        s = f"{float(v):.{self._dec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"

    def paintEvent(self, _):
        from PyQt5.QtGui import QFont as _QFont
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        pad = 6
        track_y = H * 0.60
        x_lo, x_hi = pad, W - pad
        span = max(self._hi - self._lo, 1e-9)
        a, b = min(self._a, self._b), max(self._a, self._b)
        xa = x_lo + (x_hi - x_lo) * (a - self._lo) / span
        xb = x_lo + (x_hi - x_lo) * (b - self._lo) / span

        # full training track
        p.setPen(QPen(QColor(LINE), 4, cap=Qt.RoundCap))
        p.drawLine(int(x_lo), int(track_y), int(x_hi), int(track_y))
        # active band
        p.setPen(QPen(QColor(PRIMARY), 4, cap=Qt.RoundCap))
        p.drawLine(int(xa), int(track_y), int(max(xb, xa + 1)), int(track_y))
        # handles
        p.setPen(Qt.NoPen)
        for x in (xa, xb):
            p.setBrush(QColor("#FFFFFF"))
            p.drawEllipse(QPointF(x, track_y), 5.0, 5.0)
            p.setBrush(QColor(PRIMARY))
            p.drawEllipse(QPointF(x, track_y), 2.6, 2.6)

        small = _QFont(self.font()); small.setPointSizeF(8.0)
        big = _QFont(self.font()); big.setPointSizeF(8.6); big.setBold(True)

        # training endpoints (faint, below track at the extremes)
        p.setFont(small); p.setPen(QColor(MUTED))
        u2 = (" " + self._unit) if self._unit else ""
        lo_txt, hi_txt = self._fmt(self._lo) + u2, self._fmt(self._hi) + u2
        p.drawText(QRectF(x_lo - 2, track_y + 6, 110, 14),
                   Qt.AlignLeft | Qt.AlignVCenter, lo_txt)
        p.drawText(QRectF(x_hi - 110 + 2, track_y + 6, 110, 14),
                   Qt.AlignRight | Qt.AlignVCenter, hi_txt)

        # selected min / max (teal bold, above the band ends, clamped to widget)
        p.setFont(big); p.setPen(QColor(PRIMARY_D))
        u = (" " + self._unit) if self._unit else ""
        min_txt, max_txt = self._fmt(a) + u, self._fmt(b) + u
        fm = p.fontMetrics()
        wmin = fm.horizontalAdvance(min_txt); wmax = fm.horizontalAdvance(max_txt)
        xmin = max(0, min(xa - wmin / 2, W - wmin))
        xmax = max(0, min(xb - wmax / 2, W - wmax))
        # avoid overlap when band is narrow
        if xmax < xmin + wmin + 6:
            xmin = max(0, xa - wmin - 4)
            xmax = min(W - wmax, xb + 4)
        p.drawText(QRectF(xmin, track_y - 20, wmin + 4, 15),
                   Qt.AlignLeft | Qt.AlignVCenter, min_txt)
        p.drawText(QRectF(xmax, track_y - 20, wmax + 4, 15),
                   Qt.AlignLeft | Qt.AlignVCenter, max_txt)
        p.end()


class SliderField(QWidget):
    """Label · slider · numeric entry · unit, two-way bound."""
    def __init__(self, label, lo, hi, val, step=1.0, unit="",
                 decimals=2, on_change=None, label_w=128):
        super().__init__()
        self._lo, self._hi, self._step = float(lo), float(hi), float(step)
        self._decimals = decimals; self._on_change = on_change
        self._mult = 1.0 / step
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(10)
        if label:
            lb = QLabel(label); lb.setMinimumWidth(label_w)
            lb.setStyleSheet(f"color:{BODY}; font-size:12px;")
            lay.addWidget(lb)
        self._sl = QSlider(Qt.Horizontal)
        self._sl.setMinimum(int(round(lo * self._mult)))
        self._sl.setMaximum(int(round(hi * self._mult)))
        self._sl.setValue(int(round(val * self._mult)))
        self._sl.setMinimumWidth(140)
        self._sl.setFixedHeight(22)
        self._sl.valueChanged.connect(self._from_slider)
        lay.addWidget(self._sl, 1)
        self._ed = QLineEdit(self._fmt(val)); self._ed.setFixedWidth(72)
        self._ed.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._ed.editingFinished.connect(self._from_edit)
        lay.addWidget(self._ed)
        if unit:
            u = QLabel(unit); u.setFixedWidth(32)
            u.setStyleSheet(f"color:{FAINT}; font-size:11px;")
            lay.addWidget(u)

    def _fmt(self, v):
        s = f"{float(v):.{self._decimals}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"

    def _from_slider(self, _):
        self._ed.setText(self._fmt(self._sl.value() / self._mult))
        if self._on_change:
            self._on_change()

    def _from_edit(self):
        try:
            v = max(self._lo, min(self._hi, float(self._ed.text())))
        except ValueError:
            v = self._sl.value() / self._mult
        self._ed.setText(self._fmt(v))
        self._sl.blockSignals(True)
        self._sl.setValue(int(round(v * self._mult)))
        self._sl.blockSignals(False)
        if self._on_change:
            self._on_change()

    def value(self):
        try:
            return max(self._lo, min(self._hi, float(self._ed.text())))
        except ValueError:
            return self._sl.value() / self._mult

    def set_value(self, v):
        v = max(self._lo, min(self._hi, float(v)))
        self._ed.setText(self._fmt(v))
        self._sl.blockSignals(True)
        self._sl.setValue(int(round(v * self._mult)))
        self._sl.blockSignals(False)


class Placeholder(QFrame):
    """Show an asset from assets/ if present, else a labelled drop-zone.

    ``filename`` may be a single name or a list of candidate names (the first
    that exists is used). With ``fit_width=True`` the asset is scaled to the
    widget's width with its aspect ratio preserved (so a portrait clip grows
    taller rather than being squashed) and it re-scales as the window resizes.
    ``max_h`` optionally caps that height.
    """
    def __init__(self, filename, caption_text, min_h=160, fit_width=False, max_h=None):
        super().__init__()
        self.setObjectName("ph"); self.setMinimumHeight(min_h)
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 10, 10, 10)
        self._movie = None; self._pixmap = None; self._img_lbl = None
        self._native = None; self._fit_width = fit_width; self._max_h = max_h
        self._pad = 20

        candidates = filename if isinstance(filename, (list, tuple)) else [filename]
        chosen = None; path = None
        for cand in candidates:
            p = os.path.join(ASSET_DIR, cand)
            if os.path.exists(p):
                chosen, path = cand, p; break
        loaded = False
        if path:
            if chosen.lower().endswith(".gif"):
                self._movie = QMovie(path)
                self._movie.jumpToFrame(0)
                self._native = self._movie.currentImage().size()
                lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter)
                lbl.setMovie(self._movie); self._movie.start()
                self._img_lbl = lbl; lay.addWidget(lbl, 1); loaded = True
            else:
                pm = QPixmap(path)
                if not pm.isNull():
                    self._pixmap = pm; self._native = pm.size()
                    lbl = QLabel(); lbl.setAlignment(Qt.AlignCenter)
                    self._img_lbl = lbl; lay.addWidget(lbl, 1); loaded = True
        if not loaded:
            lay.addStretch(1)
            icon = QLabel("⊞"); icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet(f"color:{PRIMARY}; font-size:24px;")
            lay.addWidget(icon)
            name = QLabel(f"assets/{candidates[0]}"); name.setAlignment(Qt.AlignCenter)
            name.setStyleSheet(f"color:{BODY}; font-family:monospace; font-size:11px;")
            lay.addWidget(name)
            cap = QLabel(caption_text); cap.setAlignment(Qt.AlignCenter)
            cap.setWordWrap(True)
            cap.setStyleSheet(f"color:{FAINT}; font-size:11px;")
            lay.addWidget(cap)
            lay.addStretch(1)
        if loaded:
            self.setObjectName("media")
            # the media label must be free to shrink so it never forces the
            # card (and the row of cards) wider than the window.
            self._img_lbl.setMinimumSize(0, 0)
            self._img_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            self.setMinimumWidth(0)
            self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        if self._movie is not None:
            self._movie.setCacheMode(QMovie.CacheNone)
            self._movie.setPaused(True)   # resumed by showEvent when visible
        if loaded and self._native and self._native.width() > 0:
            QTimer.singleShot(0, self._apply_fit)

    def showEvent(self, e):
        super().showEvent(e)
        m = self._movie
        if m is not None:
            try:
                m.setPaused(False)
            except RuntimeError:
                pass

    def hideEvent(self, e):
        super().hideEvent(e)
        m = self._movie
        if m is not None:
            try:
                m.setPaused(True)
            except RuntimeError:
                pass

    def _apply_fit(self):
        if not (self._native and self._native.width() > 0):
            return
        w = max(10, self.width() - self._pad)
        h = int(round(w * self._native.height() / self._native.width()))
        if self._max_h and h > self._max_h:
            h = self._max_h
            w = int(round(h * self._native.width() / self._native.height()))
        if self._movie is not None:
            self._movie.setScaledSize(QSize(w, h))
        elif self._pixmap is not None and self._img_lbl is not None:
            self._img_lbl.setPixmap(self._pixmap.scaled(
                QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.minimumHeight() != h + self._pad:
            self.setMinimumHeight(h + self._pad)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._native is not None:
            self._apply_fit()


def gallery_sources():
    """Absolute paths of gallery images in assets/, sorted.

    Uses any ``gallery*.<img>`` files if present; otherwise falls back to the
    showcase collages so the carousel is never empty.
    """
    exts = ("png", "jpg", "jpeg", "gif", "webp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(ASSET_DIR, f"gallery*.{e}"))
        files += glob.glob(os.path.join(ASSET_DIR, f"gallery*.{e.upper()}"))
    files = sorted(set(files))
    if not files:
        for nm in ("hero_collage.png", "pstam_collage.png"):
            p = os.path.join(ASSET_DIR, nm)
            if os.path.exists(p):
                files.append(p)
    return files


class Carousel(QFrame):
    """A simple image gallery: left / right navigation, dot indicators and a
    timed auto-advance that loops. Purely presentational."""
    def __init__(self, sources, min_h=232, interval_ms=3000):
        super().__init__()
        self.setObjectName("media"); self.setMinimumHeight(min_h)
        self._sources = list(sources); self._idx = 0
        self._cur_path = None; self._movie = None; self._pixmap = None; self._native = None

        root = QVBoxLayout(self); root.setContentsMargins(8, 8, 8, 6); root.setSpacing(6)
        row = QHBoxLayout(); row.setSpacing(6)
        self._prev = QPushButton("‹"); self._prev.setObjectName("navarrow")
        self._prev.setCursor(Qt.PointingHandCursor); self._prev.setFixedSize(34, 34)
        self._prev.clicked.connect(lambda: self._go(-1, True))
        self._img = QLabel(); self._img.setAlignment(Qt.AlignCenter)
        self._img.setMinimumHeight(min_h - 46)
        self._img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._next = QPushButton("›"); self._next.setObjectName("navarrow")
        self._next.setCursor(Qt.PointingHandCursor); self._next.setFixedSize(34, 34)
        self._next.clicked.connect(lambda: self._go(1, True))
        row.addWidget(self._prev, 0, Qt.AlignVCenter)
        row.addWidget(self._img, 1)
        row.addWidget(self._next, 0, Qt.AlignVCenter)
        root.addLayout(row, 1)

        dotrow = QHBoxLayout(); dotrow.setSpacing(6); dotrow.addStretch(1)
        self._dots = []
        for _ in self._sources:
            d = QLabel("●"); d.setCursor(Qt.PointingHandCursor)
            self._dots.append(d); dotrow.addWidget(d)
        dotrow.addStretch(1); root.addLayout(dotrow)
        for i, d in enumerate(self._dots):
            d.mousePressEvent = lambda _e, ix=i: self._jump(ix)

        if len(self._sources) <= 1:
            self._prev.hide(); self._next.hide()
            for d in self._dots:
                d.hide()

        self._timer = QTimer(self); self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(lambda: self._go(1, False))
        if len(self._sources) > 1:
            self._timer.start()

        if self._sources:
            QTimer.singleShot(0, self._render)
        else:
            self._img.setText("Add gallery images to assets/ (gallery_1.png, …)")
            self._img.setStyleSheet(f"color:{FAINT}; font-size:12px;")

    def _jump(self, ix):
        if 0 <= ix < len(self._sources) and ix != self._idx:
            self._idx = ix; self._render()
            if self._timer.isActive():
                self._timer.start()

    def _go(self, delta, manual):
        if len(self._sources) <= 1:
            return
        self._idx = (self._idx + delta) % len(self._sources)
        self._render()
        if manual and self._timer.isActive():
            self._timer.start()

    def _render(self):
        if not self._sources:
            return
        path = self._sources[self._idx]
        aw = max(60, self._img.width()); ah = max(60, self._img.height())
        if path.lower().endswith(".gif"):
            if self._movie is None or self._cur_path != path:
                if self._movie:
                    self._movie.stop()
                self._pixmap = None
                self._movie = QMovie(path); self._movie.setCacheMode(QMovie.CacheNone)
                self._movie.jumpToFrame(0)
                self._native = self._movie.currentImage().size()
                self._img.setMovie(self._movie); self._movie.start()
                self._movie.setPaused(not self.isVisible())
            if self._native and self._native.width() > 0:
                self._movie.setScaledSize(self._native.scaled(aw, ah, Qt.KeepAspectRatio))
        else:
            if self._pixmap is None or self._cur_path != path:
                self._movie = None
                self._pixmap = QPixmap(path); self._native = self._pixmap.size()
            if self._pixmap and not self._pixmap.isNull():
                self._img.setPixmap(self._pixmap.scaled(
                    aw, ah, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._cur_path = path
        for i, d in enumerate(self._dots):
            on = (i == self._idx)
            d.setStyleSheet(f"color:{PRIMARY if on else LINE}; font-size:{13 if on else 11}px;")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    def showEvent(self, e):
        super().showEvent(e)
        m = self._movie
        if m is not None:
            try:
                m.setPaused(False)
            except RuntimeError:
                pass
        if len(self._sources) > 1 and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        m = self._movie
        if m is not None:
            try:
                m.setPaused(True)
            except RuntimeError:
                pass
        self._timer.stop()


# ════════════════════════════════════════════════════════════════════
#  Cost-function preview (HTML, never wraps)
# ════════════════════════════════════════════════════════════════════
def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _span(txt, color, bold=False):
    return (f'<span style="color:{color}; font-weight:{"700" if bold else "500"};">'
            f'{_esc(txt)}</span>')


def _model_call(model, fF, fS, sweep_var):
    inputs = {
        "M1": [("L", "var"), ("D", "var"), ("T", "var"), ("F", "F"), ("P", "var")],
        "M2": [("L", "var"), ("D", "var"), ("T", "var"), ("P", "var"),
               ("F", "F"), ("s", "stroke")],
        "M3": [("L", "var"), ("D", "var"), ("T", "var"), ("P", "var"),
               ("F", "F"), ("s", "stroke")],
    }[model]
    out = [_span(f"{model}(", MODEL, True)]; parts = []
    for sym, kind in inputs:
        if kind == "var":
            parts.append(_span(sym, VAR, True))
        elif kind == "F":
            parts.append(_span("F", SWEEP, True) if sweep_var == "F"
                         else _span(f"{fF:.4g} N", FIX))
        elif kind == "stroke":
            parts.append(_span("s%", SWEEP, True) if sweep_var == "stroke"
                         else _span(f"{fS:.4g} %", FIX))
    out.append(_span(", ", OP).join(parts)); out.append(_span(")", MODEL, True))
    return "".join(out)


def cost_preview_html(mode, terms, cons):
    rows = [_span("Objective", MUTED) + "&nbsp;&nbsp;" + _span(mode.upper(), OP, True)]
    active = [t for t in terms if float(t.get("weight", 1.0)) != 0.0]
    multi = len(active) > 1
    for i, t in enumerate(active):
        model = t["model"]; agg = t["aggregate"]; wt = float(t.get("weight", 1.0))
        fF = float(t["fixed"].get("F", 75)); fS = float(t["fixed"].get("stroke", 55))
        sw = t.get("sweep"); sv = sw["variable"] if sw else None
        line = ("&nbsp;&nbsp;&nbsp;&nbsp;" if i == 0
                else "&nbsp;&nbsp;" + _span("+", OP, True) + "&nbsp;")
        if multi:
            line += _span(f"{wt:.4g}", OP, True) + _span(" × ", OP)
        if sw and agg != "single":
            sym = "F" if sv == "F" else "s"; un = "N" if sv == "F" else "%"
            line += _span(agg, AGG, True)
            line += _span(f"[{sym}={sw['from']:.4g}→{sw['to']:.4g}, "
                          f"Δ{sw['step']:.4g}{un}] ", SWEEP)
        line += _model_call(model, fF, fS, sv); rows.append(line)

    actc = [c for c in cons if c.get("enabled", True)]
    if actc:
        rows.append(""); rows.append(_span("Subject to", MUTED))
        for i, c in enumerate(actc):
            model = c["model"]; agg = c["aggregate"]
            thr = float(c["threshold"]); dirn = c["direction"]; pen = float(c["penalty_weight"])
            fF = float(c["fixed"].get("F", 75)); fS = float(c["fixed"].get("stroke", 55))
            sw = c.get("sweep"); sv = sw["variable"] if sw else None
            line = "&nbsp;&nbsp;&nbsp;&nbsp;" + _span(f"[C{i+1}] ", CON, True)
            if sw and agg != "single":
                sym = "F" if sv == "F" else "s"; un = "N" if sv == "F" else "%"
                line += _span(agg, AGG, True)
                line += _span(f"[{sym}={sw['from']:.4g}→{sw['to']:.4g}, "
                              f"Δ{sw['step']:.4g}{un}] ", SWEEP)
            line += _model_call(model, fF, fS, sv)
            line += _span(f" {dirn} ", OP, True)
            line += _span(f"{thr:.4g} {UNIT[model]}", FIX)
            line += _span(f"  · penalty {pen:,.0f}", FAINT)
            rows.append(line)

    rows.append("")
    tail = _span("PSO minimises ", MUTED)
    tail += _span("−objective" if mode == "maximize" else "objective", OP, True)
    if actc:
        tail += _span("  +  Σ penalty · max(0, violation)", FAINT)
    rows.append(tail)

    body = "<br>".join(r if r else "&nbsp;" for r in rows)
    return (f'<div style="font-family:\'JetBrains Mono\',\'Cascadia Code\',Consolas,'
            f'monospace; font-size:13px; line-height:1.75; white-space:nowrap; '
            f'color:{INK};">{body}</div>')


def report_html(result):
    pos = result["best_position"]; rep = result["report"]; mode = rep["mode"]
    L, D, T, P = pos[0], pos[1], pos[2], pos[3]
    sense = "maximised" if mode == "maximize" else "minimised"

    def heading(text, color):
        return (f'<div style="color:{color}; letter-spacing:1.5px; font-size:11px;'
                f'font-weight:800; margin:16px 0 6px 0;">{text}</div>')

    def chip(sym, val, unit):
        return (f'<td align="center" style="padding:2px 20px;">'
                f'<div style="color:{MUTED}; font-size:11px; letter-spacing:0.5px;'
                f'font-weight:700;">{sym}</div>'
                f'<div style="color:{INK}; font-size:22px; font-weight:800;">'
                f'{val:.2f}<span style="color:{MUTED}; font-size:12px;"> {unit}</span></div></td>')

    th = (f'padding:6px 8px; color:{MUTED}; font-size:10px; letter-spacing:0.5px;'
          f'font-weight:700; border-bottom:1.5px solid {LINE};')
    td = f'padding:6px 8px; font-size:12px; border-bottom:1px solid {LINE_SOFT};'

    h = [f'<div style="font-family:Inter,\'Segoe UI\',sans-serif; color:{INK};">']

    h.append('<div style="text-align:center;">')
    h.append(f'<div style="color:{PRIMARY}; letter-spacing:2px; font-size:11px;'
             f'font-weight:800; margin-bottom:6px;">OPTIMAL DESIGN</div>')
    h.append('<table align="center" style="margin:0 auto;"><tr>')
    h.append(chip("L · length", L, "mm")); h.append(chip("D · diameter", D, "mm"))
    h.append(chip("T · wall", T, "mm")); h.append(chip("P · pressure", P, "kPa"))
    h.append('</tr></table></div>')

    h.append(heading("OBJECTIVE TERMS", MODEL))
    h.append('<table width="100%" style="border-collapse:collapse;">')
    h.append(f'<tr><th align="left" style="{th}">TERM</th>'
             f'<th align="left" style="{th}">MODEL</th>'
             f'<th align="left" style="{th}">CHARACTERISTIC</th>'
             f'<th align="center" style="{th}">AGGREGATE</th>'
             f'<th align="right" style="{th}">PREDICTION</th>'
             f'<th align="right" style="{th}">WEIGHT</th>'
             f'<th align="right" style="{th}">CONTRIBUTION</th></tr>')
    for t in rep["objective_terms"]:
        m = t["model"]; phys, un = MODEL_PHYS[m]
        h.append(
            f'<tr><td style="{td} color:{MUTED};">T{t["term_index"]}</td>'
            f'<td style="{td} color:{MODEL}; font-weight:800;">{m}</td>'
            f'<td style="{td} color:{INK};">{phys}</td>'
            f'<td align="center" style="{td} color:{BODY};">{t["aggregate"]}</td>'
            f'<td align="right" style="{td} color:{INK}; font-weight:700;">'
            f'{t["prediction"]:.3f} {un}</td>'
            f'<td align="right" style="{td} color:{BODY};">× {t["weight"]:.3g}</td>'
            f'<td align="right" style="{td} color:{PRIMARY}; font-weight:800;">'
            f'{t["weighted"]:.3f}</td></tr>')
    h.append('</table>')
    h.append(f'<div style="text-align:right; margin:8px 8px 0 0; color:{BODY}; font-size:13px;">'
             f'Objective value ({sense}): '
             f'<span style="color:{PRIMARY}; font-weight:800; font-size:16px;">'
             f'{rep["total_objective"]:.4f}</span></div>')

    cons = rep["constraints"]
    if cons:
        h.append(heading("CONSTRAINTS", CON))
        h.append('<table width="100%" style="border-collapse:collapse;">')
        h.append(f'<tr><th align="left" style="{th}">ID</th>'
                 f'<th align="left" style="{th}">MODEL</th>'
                 f'<th align="left" style="{th}">CHARACTERISTIC</th>'
                 f'<th align="center" style="{th}">LIMIT</th>'
                 f'<th align="right" style="{th}">PREDICTION</th>'
                 f'<th align="right" style="{th}">STATUS</th></tr>')
        for c in cons:
            m = c["model"]; phys, un = MODEL_PHYS[m]
            ok = not c["violated"]; col = GOOD if ok else BAD
            h.append(
                f'<tr><td style="{td} color:{MUTED};">C{c["constraint_index"]}</td>'
                f'<td style="{td} color:{MODEL}; font-weight:800;">{m}</td>'
                f'<td style="{td} color:{INK};">{phys}</td>'
                f'<td align="center" style="{td} color:{BODY};">'
                f'{c["aggregate"]} {c["direction"]} {c["threshold"]:.4g} {un}</td>'
                f'<td align="right" style="{td} color:{INK}; font-weight:700;">'
                f'{c["prediction"]:.3f} {un}</td>'
                f'<td align="right" style="{td}">'
                f'<span style="color:{col}; font-weight:800; font-size:11px;">'
                f'{"PASS" if ok else "VIOLATED"}</span></td></tr>')
        h.append('</table>')

    if result.get("csv_path"):
        h.append(f'<div style="margin-top:14px; color:{MUTED}; font-size:11px;">'
                 f'Saved to {os.path.basename(result["csv_path"])}</div>')
    h.append('</div>')
    return "".join(h)


# ════════════════════════════════════════════════════════════════════
#  Term & Constraint editor blocks  (aligned grid layout)
# ════════════════════════════════════════════════════════════════════
class _BlockBase(QFrame):
    LABW = 62

    def _combo(self, values, current, w=64):
        cb = QComboBox(); cb.addItems([str(v) for v in values])
        if str(current) in [str(v) for v in values]:
            cb.setCurrentText(str(current))
        cb.setFixedWidth(w); return cb

    def _edit(self, value, w=58):
        e = QLineEdit(str(value)); e.setFixedWidth(w)
        e.setAlignment(Qt.AlignRight | Qt.AlignVCenter); return e

    def _key(self, t):
        l = QLabel(t); l.setMinimumWidth(self.LABW)
        l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        l.setStyleSheet(f"color:{MUTED}; font-size:11px;"); return l

    def _tag(self, t, color=FAINT):
        l = QLabel(t); l.setStyleSheet(f"color:{color}; font-size:11px;"); return l


class TermBlock(_BlockBase):
    def __init__(self, idx, data, on_change, on_remove=None):
        super().__init__(); self.setObjectName("block")
        self._on_change = on_change; self._on_remove = on_remove; self.idx = idx
        self._build(data); self._sync_dynamic()

    def set_index(self, i):
        self.idx = i; self._title.setText(f"TERM {i}")

    def _build(self, d):
        v = QVBoxLayout(self); v.setContentsMargins(14, 9, 14, 10); v.setSpacing(6)
        head = QHBoxLayout()
        self._title = QLabel(f"TERM {self.idx}")
        self._title.setStyleSheet(f"color:{PRIMARY}; font-weight:800;"
                                  f"letter-spacing:1.5px; font-size:11px;")
        head.addWidget(self._title); head.addSpacing(10)
        self._out = QLabel(); self._out.setMinimumWidth(0)
        self._out.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._out.setStyleSheet(
            f"color:{MODEL}; font-size:11px; font-weight:700;")
        head.addWidget(self._out, 1)
        self._rm = QPushButton("✕"); self._rm.setObjectName("rmbtn")
        self._rm.setToolTip("Remove this term"); self._rm.setCursor(Qt.PointingHandCursor)
        self._rm.setFixedSize(22, 22)
        self._rm.clicked.connect(lambda: self._on_remove and self._on_remove(self))
        head.addWidget(self._rm)
        v.addLayout(head)
        v.addWidget(hrule())

        g = QGridLayout(); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        g.setColumnStretch(4, 1)

        g.addWidget(self._key("Model"), 0, 0)
        self._model = self._combo(MODELS, d.get("model", "M1"), 58)
        self._model.currentTextChanged.connect(
            lambda *_: (self._sync_dynamic(), self._emit()))
        g.addWidget(self._model, 0, 1)
        g.addWidget(self._key("Aggregate"), 0, 2)
        self._agg = self._combo(AGGS, d.get("aggregate", "single"), 80)
        self._agg.currentTextChanged.connect(lambda *_: self._emit())
        g.addWidget(self._agg, 0, 3)

        g.addWidget(self._key("Weight"), 1, 0)
        self._wt = self._edit(d.get("weight", 1.0), 58)
        self._wt.editingFinished.connect(self._emit)
        g.addWidget(self._wt, 1, 1)


        fx = d.get("fixed", {})
        self._fixed_key = self._key("Fixed")
        g.addWidget(self._fixed_key, 2, 0)
        fixrow = QWidget(); fr = QHBoxLayout(fixrow)
        fr.setContentsMargins(0, 0, 0, 0); fr.setSpacing(6)
        self._F_lab = self._tag("F"); fr.addWidget(self._F_lab)
        self._F = self._edit(fx.get("F", 75.0), 54); self._F.editingFinished.connect(self._emit)
        fr.addWidget(self._F)
        self._F_unit = self._tag("N"); fr.addWidget(self._F_unit)
        self._fix_gap = QWidget(); self._fix_gap.setFixedWidth(14); fr.addWidget(self._fix_gap)
        self._S_lab = self._tag("stroke"); fr.addWidget(self._S_lab)
        self._S = self._edit(fx.get("stroke", 55.0), 54); self._S.editingFinished.connect(self._emit)
        fr.addWidget(self._S); self._S_pct = self._tag("%"); fr.addWidget(self._S_pct)
        fr.addStretch(1)
        g.addWidget(fixrow, 2, 1, 1, 4)
        v.addLayout(g)

        self._sweep_on = QCheckBox("Sweep one input over a range")
        self._sweep_on.setChecked(d.get("sweep") is not None)
        self._sweep_on.toggled.connect(lambda *_: (self._sync_dynamic(), self._emit()))
        v.addWidget(self._sweep_on)

        sw = d.get("sweep") or {}
        self._sweep_row = QWidget()
        sg = QGridLayout(self._sweep_row); sg.setContentsMargins(self.LABW + 8, 0, 0, 0)
        sg.setHorizontalSpacing(8); sg.setVerticalSpacing(6); sg.setColumnStretch(7, 1)
        sg.addWidget(self._tag("Variable"), 0, 0)
        self._sw_var = self._combo(["F", "stroke"], sw.get("variable", "F"), 80)
        self._sw_var.currentTextChanged.connect(
            lambda *_: (self._sync_dynamic(), self._emit()))
        sg.addWidget(self._sw_var, 0, 1)
        sg.addWidget(self._tag("From"), 0, 2)
        self._sw_from = self._edit(sw.get("from", 20), 46); sg.addWidget(self._sw_from, 0, 3)
        sg.addWidget(self._tag("To"), 0, 4)
        self._sw_to = self._edit(sw.get("to", 76), 46); sg.addWidget(self._sw_to, 0, 5)
        sg.addWidget(self._tag("Step"), 0, 6)
        self._sw_step = self._edit(sw.get("step", 10), 46); sg.addWidget(self._sw_step, 0, 7)
        for e in (self._sw_from, self._sw_to, self._sw_step):
            e.editingFinished.connect(self._emit)
        v.addWidget(self._sweep_row)

    def _sync_dynamic(self):
        model = self._model.currentText()
        phys, un = MODEL_PHYS[model]
        self._out.setText(f"→ {phys} ({un})")
        uses_stroke = MODEL_USES_STROKE[model]
        sweeping = self._sweep_on.isChecked()
        allowed = MODEL_SWEEPABLES[model]; cur = self._sw_var.currentText()
        self._sw_var.blockSignals(True); self._sw_var.clear()
        self._sw_var.addItems(allowed)
        self._sw_var.setCurrentText(cur if cur in allowed else allowed[0])
        self._sw_var.blockSignals(False)
        sw_var = self._sw_var.currentText()
        show_stroke = uses_stroke and not (sweeping and sw_var == "stroke")
        for w in (self._S_lab, self._S, self._S_pct):
            w.setVisible(show_stroke)
        show_F = not (sweeping and sw_var == "F")
        for w in (self._F_lab, self._F, self._F_unit):
            w.setVisible(show_F)
        self._fix_gap.setVisible(show_F and show_stroke)
        self._fixed_key.setVisible(show_F or show_stroke)
        self._sweep_row.setVisible(sweeping)

    def _emit(self):
        if self._on_change:
            self._on_change()

    def get_dict(self):
        d = {"model": self._model.currentText(),
             "fixed": {"F": float(self._F.text() or 75),
                       "stroke": float(self._S.text() or 55)},
             "aggregate": self._agg.currentText(),
             "weight": float(self._wt.text() or 1.0)}
        if self._sweep_on.isChecked():
            d["sweep"] = {"variable": self._sw_var.currentText(),
                          "from": float(self._sw_from.text() or 20),
                          "to": float(self._sw_to.text() or 90),
                          "step": float(self._sw_step.text() or 10)}
        else:
            d["sweep"] = None
        return d


class ConstraintBlock(_BlockBase):
    def __init__(self, idx, data, on_change, on_remove=None):
        super().__init__(); self.setObjectName("block")
        self._on_change = on_change; self._on_remove = on_remove; self.idx = idx
        self._build(data); self._sync_dynamic()

    def set_index(self, i):
        self.idx = i; self._title.setText(f"CONSTRAINT {i}")

    def _build(self, d):
        v = QVBoxLayout(self); v.setContentsMargins(14, 9, 14, 10); v.setSpacing(6)
        head = QHBoxLayout()
        self._title = QLabel(f"CONSTRAINT {self.idx}")
        self._title.setStyleSheet(f"color:{CON}; font-weight:800;"
                                  f"letter-spacing:1.5px; font-size:11px;")
        head.addWidget(self._title); head.addSpacing(10)
        self._out = QLabel(); self._out.setMinimumWidth(0)
        self._out.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._out.setStyleSheet(
            f"color:{MODEL}; font-size:11px; font-weight:700;")
        head.addWidget(self._out, 1)
        self._enabled = QCheckBox("enabled")
        self._enabled.setChecked(d.get("enabled", True))
        self._enabled.toggled.connect(lambda *_: self._emit())
        head.addWidget(self._enabled); head.addSpacing(6)
        self._rm = QPushButton("✕"); self._rm.setObjectName("rmbtn")
        self._rm.setToolTip("Remove this constraint"); self._rm.setCursor(Qt.PointingHandCursor)
        self._rm.setFixedSize(22, 22)
        self._rm.clicked.connect(lambda: self._on_remove and self._on_remove(self))
        head.addWidget(self._rm)
        v.addLayout(head)
        v.addWidget(hrule())

        g = QGridLayout(); g.setHorizontalSpacing(8); g.setVerticalSpacing(6)
        g.setColumnStretch(4, 1)
        g.addWidget(self._key("Model"), 0, 0)
        self._model = self._combo(MODELS, d.get("model", "M2"), 58)
        self._model.currentTextChanged.connect(
            lambda *_: (self._sync_dynamic(), self._emit()))
        g.addWidget(self._model, 0, 1)
        g.addWidget(self._key("Aggregate"), 0, 2)
        self._agg = self._combo(AGGS, d.get("aggregate", "max"), 80)
        self._agg.currentTextChanged.connect(lambda *_: self._emit())
        g.addWidget(self._agg, 0, 3)

        g.addWidget(self._key("Limit"), 1, 0)
        limrow = QWidget(); lr = QHBoxLayout(limrow)
        lr.setContentsMargins(0, 0, 0, 0); lr.setSpacing(6)
        self._dir = self._combo(DIRS, d.get("direction", "≤"), 50)
        self._dir.currentTextChanged.connect(lambda *_: self._emit())
        lr.addWidget(self._dir)
        self._thr = self._edit(d.get("threshold", 500.0), 72)
        self._thr.editingFinished.connect(self._emit); lr.addWidget(self._thr)
        lr.addSpacing(14); lr.addWidget(self._tag("penalty"))
        self._pen = self._edit(d.get("penalty_weight", 100_000), 92)
        self._pen.editingFinished.connect(self._emit); lr.addWidget(self._pen)
        lr.addStretch(1)
        g.addWidget(limrow, 1, 1, 1, 4)

        fx = d.get("fixed", {})
        self._fixed_key = self._key("Fixed")
        g.addWidget(self._fixed_key, 2, 0)
        fixrow = QWidget(); fr = QHBoxLayout(fixrow)
        fr.setContentsMargins(0, 0, 0, 0); fr.setSpacing(6)
        self._F_lab = self._tag("F"); fr.addWidget(self._F_lab)
        self._F = self._edit(fx.get("F", 75.0), 54); self._F.editingFinished.connect(self._emit)
        fr.addWidget(self._F)
        self._F_unit = self._tag("N"); fr.addWidget(self._F_unit)
        self._fix_gap = QWidget(); self._fix_gap.setFixedWidth(14); fr.addWidget(self._fix_gap)
        self._S_lab = self._tag("stroke"); fr.addWidget(self._S_lab)
        self._S = self._edit(fx.get("stroke", 55.0), 54); self._S.editingFinished.connect(self._emit)
        fr.addWidget(self._S); self._S_pct = self._tag("%"); fr.addWidget(self._S_pct)
        fr.addStretch(1)
        g.addWidget(fixrow, 2, 1, 1, 4)
        v.addLayout(g)

        self._sweep_on = QCheckBox("Sweep one input over a range")
        self._sweep_on.setChecked(d.get("sweep") is not None)
        self._sweep_on.toggled.connect(lambda *_: (self._sync_dynamic(), self._emit()))
        v.addWidget(self._sweep_on)

        sw = d.get("sweep") or {}
        self._sweep_row = QWidget()
        sg = QGridLayout(self._sweep_row); sg.setContentsMargins(self.LABW + 8, 0, 0, 0)
        sg.setHorizontalSpacing(8); sg.setVerticalSpacing(6); sg.setColumnStretch(7, 1)
        sg.addWidget(self._tag("Variable"), 0, 0)
        self._sw_var = self._combo(["F", "stroke"], sw.get("variable", "stroke"), 80)
        self._sw_var.currentTextChanged.connect(
            lambda *_: (self._sync_dynamic(), self._emit()))
        sg.addWidget(self._sw_var, 0, 1)
        sg.addWidget(self._tag("From"), 0, 2)
        self._sw_from = self._edit(sw.get("from", 20), 46); sg.addWidget(self._sw_from, 0, 3)
        sg.addWidget(self._tag("To"), 0, 4)
        self._sw_to = self._edit(sw.get("to", 76), 46); sg.addWidget(self._sw_to, 0, 5)
        sg.addWidget(self._tag("Step"), 0, 6)
        self._sw_step = self._edit(sw.get("step", 10), 46); sg.addWidget(self._sw_step, 0, 7)
        for e in (self._sw_from, self._sw_to, self._sw_step):
            e.editingFinished.connect(self._emit)
        v.addWidget(self._sweep_row)

    def _sync_dynamic(self):
        model = self._model.currentText()
        phys, un = MODEL_PHYS[model]
        self._out.setText(f"→ {phys} ({un})")
        uses_stroke = MODEL_USES_STROKE[model]
        sweeping = self._sweep_on.isChecked()
        allowed = MODEL_SWEEPABLES[model]; cur = self._sw_var.currentText()
        self._sw_var.blockSignals(True); self._sw_var.clear()
        self._sw_var.addItems(allowed)
        self._sw_var.setCurrentText(cur if cur in allowed else allowed[0])
        self._sw_var.blockSignals(False)
        sw_var = self._sw_var.currentText()
        show_stroke = uses_stroke and not (sweeping and sw_var == "stroke")
        for w in (self._S_lab, self._S, self._S_pct):
            w.setVisible(show_stroke)
        show_F = not (sweeping and sw_var == "F")
        for w in (self._F_lab, self._F, self._F_unit):
            w.setVisible(show_F)
        self._fix_gap.setVisible(show_F and show_stroke)
        self._fixed_key.setVisible(show_F or show_stroke)
        self._sweep_row.setVisible(sweeping)

    def _emit(self):
        if self._on_change:
            self._on_change()

    def get_dict(self):
        d = {"enabled": self._enabled.isChecked(),
             "model": self._model.currentText(),
             "fixed": {"F": float(self._F.text() or 75),
                       "stroke": float(self._S.text() or 55)},
             "aggregate": self._agg.currentText(),
             "threshold": float(self._thr.text() or 500),
             "direction": self._dir.currentText(),
             "penalty_weight": float(self._pen.text() or 100000)}
        if self._sweep_on.isChecked():
            d["sweep"] = {"variable": self._sw_var.currentText(),
                          "from": float(self._sw_from.text() or 20),
                          "to": float(self._sw_to.text() or 90),
                          "step": float(self._sw_step.text() or 10)}
        else:
            d["sweep"] = None
        return d


# ════════════════════════════════════════════════════════════════════
#  Optimisation worker
# ════════════════════════════════════════════════════════════════════
class OptWorker(QObject):
    progress = pyqtSignal(int, int)
    done     = pyqtSignal(object)
    failed   = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg

    @pyqtSlot()
    def run(self):
        try:
            from pstam.optimizer import PAMOptimizer
            opt = PAMOptimizer(cfg=self.cfg, data_dir=DATA_DIR, results_dir=RESULTS_DIR,
                               progress_callback=lambda i, t: self.progress.emit(i, t))
            self.done.emit(opt.run())
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════
#  Main window
# ════════════════════════════════════════════════════════════════════
class Toolbox(QMainWindow):
    NAV = ["Home", "Design Space", "Surrogate Models",
           "Objective & Cost", "Optimise & Report"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pSTAM Design Toolbox")
        # Size to the available screen so the window never opens larger than the
        # display; a lower minimum lets it shrink on small laptops, with the
        # per-page scrollbars covering any overflow.
        self.setMinimumSize(940, 560)
        scr = QApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            w0 = max(940, min(1360, av.width() - 40))
            h0 = max(560, min(880, av.height() - 60))
            self.resize(w0, h0)
            self.move(av.x() + (av.width() - w0) // 2,
                      av.y() + (av.height() - h0) // 2)
        else:
            self.resize(1200, 780)
        self._cfg = get_default_config()
        self._diag = load_model_diagnostics()
        self._t_blocks, self._c_blocks = [], []
        self._last_result = None; self._thread = None

        self.DISPLAY = pick_font(["Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans"])
        self.MONO = pick_font(["JetBrains Mono", "Cascadia Code", "Consolas",
                               "DejaVu Sans Mono"], "DejaVu Sans Mono")
        QApplication.instance().setFont(QFont(self.DISPLAY, 10))

        self._build_shell(); self._apply_qss(); self._refresh_preview()
        if int(str(np.__version__).split(".")[0]) >= 2:
            QTimer.singleShot(400, lambda: self._toast(
                "NumPy 2 detected — if a run fails, install the pinned stack "
                "in requirements.txt (numpy 1.23.5).", ACCENT))

    def _build_shell(self):
        root = QWidget(); self.setCentralWidget(root)
        cv = QVBoxLayout(root); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(0)

        bar = QWidget(); bar.setObjectName("topbar"); bar.setFixedHeight(58)
        bh = QHBoxLayout(bar); bh.setContentsMargins(22, 0, 20, 0); bh.setSpacing(14)
        bt = QVBoxLayout(); bt.setSpacing(0); bt.setContentsMargins(0, 0, 0, 0)
        nm = QLabel("pSTAM"); nm.setStyleSheet(
            f"color:{INK}; font-size:16px; font-weight:800; letter-spacing:0.5px;")
        sub = QLabel("DESIGN TOOLBOX"); sub.setStyleSheet(
            f"color:{PRIMARY}; font-size:8px; letter-spacing:2.5px; font-weight:800;")
        bt.addWidget(nm); bt.addWidget(sub)
        bh.addLayout(bt)
        sep = QFrame(); sep.setFixedSize(1, 28); sep.setStyleSheet(f"background:{LINE};")
        bh.addSpacing(6); bh.addWidget(sep); bh.addSpacing(6)

        self._nav_btns = []
        for i, label in enumerate(self.NAV):
            b = QPushButton(label.replace('&', '&&'))
            b.setCheckable(True); b.setObjectName("nav")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _, ix=i: self._goto(ix))
            self._nav_btns.append(b); bh.addWidget(b)
        bh.addStretch(1)
        loadb = QPushButton("Load config"); loadb.setObjectName("ghost")
        loadb.setToolTip("Load a saved configuration (.json)")
        loadb.setCursor(Qt.PointingHandCursor); loadb.clicked.connect(self._load_config)
        bh.addWidget(loadb)
        saveb = QPushButton("Save config"); saveb.setObjectName("ghost")
        saveb.setToolTip("Save the current configuration (.json)")
        saveb.setCursor(Qt.PointingHandCursor); saveb.clicked.connect(self._save_config_only)
        bh.addWidget(saveb)
        cv.addWidget(bar)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._page_home())
        self._stack.addWidget(self._page_design())
        self._stack.addWidget(self._page_models())
        self._stack.addWidget(self._page_objcon())
        self._stack.addWidget(self._page_optimise())
        cv.addWidget(self._stack, 1)

        self._nav_btns[0].setChecked(True); self._goto(0)

    def _scroll_page(self):
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setObjectName("scroll")
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget(); inner.setObjectName("pagebg")
        lay = QVBoxLayout(inner); lay.setContentsMargins(28, 20, 28, 24); lay.setSpacing(13)
        sa.setWidget(inner)
        return sa, lay

    def _goto(self, ix):
        self._stack.setCurrentIndex(ix)
        for j, b in enumerate(self._nav_btns):
            b.setChecked(j == ix)
        if ix == 3:
            self._refresh_preview()

    # ════════════════════════════════════════════════════════════════
    #  Home
    # ════════════════════════════════════════════════════════════════
    def _page_home(self):
        sa, lay = self._scroll_page()

        # masthead: a generous logo area beside the headline
        mast = card(); mr = QHBoxLayout(mast)
        mr.setContentsMargins(24, 22, 26, 22); mr.setSpacing(26)
        logo = QFrame(); logo.setObjectName("logoslot")
        logo.setFixedWidth(260); logo.setMinimumHeight(168)
        logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        lgl = QVBoxLayout(logo); lgl.setContentsMargins(10, 10, 10, 10)
        lgl.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(ASSET_DIR, "logo.png")
        if os.path.exists(logo_path) and not QPixmap(logo_path).isNull():
            li = QLabel(); li.setAlignment(Qt.AlignCenter)
            li.setPixmap(QPixmap(logo_path).scaled(238, 200, Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))
            lgl.addWidget(li)
        else:
            lt = QLabel("Your logo"); lt.setAlignment(Qt.AlignCenter)
            lt.setStyleSheet(f"color:{MUTED}; font-size:13px; font-weight:700;")
            lp = QLabel("assets/logo.png"); lp.setAlignment(Qt.AlignCenter)
            lp.setStyleSheet(f"color:{FAINT}; font-family:monospace; font-size:10.5px;")
            lgl.addWidget(lt); lgl.addWidget(lp)
        mr.addWidget(logo)

        txt = QVBoxLayout(); txt.setSpacing(8)
        txt.addWidget(eyebrow("soft pneumatic actuator design · human-centred robotics"))
        t = QLabel("Design the muscle, not just the model.")
        t.setStyleSheet(f"color:{INK}; font-size:26px; font-weight:800; letter-spacing:-0.4px;")
        t.setWordWrap(True); txt.addWidget(t)
        d = QLabel("A surrogate-driven toolbox for sizing Pneumatic Silicone Tube Artificial "
                   "Muscles (pSTAM). Three ANN models map geometry and actuation to "
                   "contraction and dynamic response; particle-swarm optimisation searches "
                   "the four-variable design space against your objective and limits.")
        d.setWordWrap(True); d.setStyleSheet(f"color:{BODY}; font-size:12.5px;")
        txt.addWidget(d)
        row = QHBoxLayout(); row.setSpacing(10)
        start = QPushButton("Start a new design  →"); start.setObjectName("cta")
        start.setCursor(Qt.PointingHandCursor); start.clicked.connect(lambda: self._goto(1))
        row.addWidget(start)
        loadc = QPushButton("Load a saved configuration"); loadc.setObjectName("ghost")
        loadc.setCursor(Qt.PointingHandCursor); loadc.clicked.connect(self._load_config)
        row.addWidget(loadc); row.addStretch(1)
        txt.addLayout(row)
        mr.addLayout(txt, 1)
        shadow(mast); lay.addWidget(mast)

        # compact stat strip
        grid = QGridLayout(); grid.setSpacing(14)
        facts = [("4", "design variables", "L · D · T · P"),
                 ("3", "ANN surrogate models", "contraction · inflation · deflation"),
                 ("PSO", "global-best optimiser", "swarm over the bounded space")]
        for i, (big, small, sub) in enumerate(facts):
            c = card(); cy = QHBoxLayout(c); cy.setContentsMargins(18, 12, 18, 12); cy.setSpacing(12)
            b = QLabel(big); b.setStyleSheet(f"color:{PRIMARY}; font-size:26px; font-weight:800;")
            cy.addWidget(b)
            tw = QVBoxLayout(); tw.setSpacing(0)
            s = QLabel(small); s.setStyleSheet(f"color:{INK}; font-size:13px; font-weight:700;")
            ss = QLabel(sub); ss.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            tw.addWidget(s); tw.addWidget(ss); cy.addLayout(tw); cy.addStretch(1)
            shadow(c); grid.addWidget(c, 0, i)
        gw = QWidget(); gw.setLayout(grid); lay.addWidget(gw)

        # lower: collage (left) + about/credits/refs (right)
        low = QHBoxLayout(); low.setSpacing(14)
        coll = card(); cl = QVBoxLayout(coll); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(8)
        cl.addWidget(section("Gallery", color=INK))
        cl.addWidget(Carousel(gallery_sources(), min_h=232, interval_ms=3000))
        shadow(coll); low.addWidget(coll, 1)

        ac = card(); al = QVBoxLayout(ac); al.setContentsMargins(20, 16, 20, 16); al.setSpacing(8)
        al.addWidget(section("About this toolbox", color=INK))
        about = QLabel("The toolbox replaces trial-and-error sizing with a closed loop: measured "
                       "actuator behaviour trains the ANN surrogates, and PSO inverts them to "
                       "recommend a buildable design that meets stroke, force and timing "
                       "targets. Use it as a working datasheet — state what the muscle must do, "
                       "and read back the geometry that delivers it.")
        about.setWordWrap(True); about.setStyleSheet(f"color:{BODY}; font-size:12px;")
        al.addWidget(about)

        au = QFrame(); au.setObjectName("authors")
        aul = QVBoxLayout(au); aul.setContentsMargins(14, 11, 14, 12); aul.setSpacing(5)
        ah = QLabel("AUTHORS"); ah.setStyleSheet(
            f"color:{PRIMARY_D}; font-size:11.5px; font-weight:800; letter-spacing:1.5px;"
            f"background:transparent; border:none;")
        aul.addWidget(ah)
        for name, role in [("Debadrata Sarkar (contact: deba00126@gmail.com)", "Lead Developer, Design & Experiments"),
                           ("Aman Arora", "Design and Experiments, Principal Investigator"),
                           ("Arunabha Majumder", "Experiments"),
                           ("Soumen Sen", "Co-Principal Investigator"),
                           ("Shibendu Shekhar Roy", "Principal Investigator")]:
            r = QLabel(f'<span style="color:{INK}; font-weight:800;">{name}</span>'
                       f'<span style="color:{MUTED};">  —  {role}</span>')
            r.setTextFormat(Qt.RichText)
            r.setStyleSheet("background:transparent; border:none; font-size:12.5px;")
            aul.addWidget(r)
        al.addWidget(au)

        rf = QFrame(); rf.setObjectName("refs")
        rfl = QVBoxLayout(rf); rfl.setContentsMargins(14, 11, 14, 12); rfl.setSpacing(5)
        rh = QLabel("MANUSCRIPTS &amp; PATENTS"); rh.setTextFormat(Qt.RichText)
        rh.setStyleSheet("color:#A65E1E; font-size:11.5px; font-weight:800;"
                         "letter-spacing:1.5px; background:transparent; border:none;")
        rfl.addWidget(rh)
        for tag, txt in [("[1]", "Hybridized neural network inspired behavioural modelling of pneumatic artificial muscles for assistive robotic applications, Arora A., Sarkar D. et al. (2023). Industrial Robot, Vol. 50 No. 1 pp. 56–69, doi: https://doi.org/10.1108/IR-03-2022-0060"),
                         ("[2]", "Low-pressure pneumatic muscles: development, phenomenological modeling, and evaluation in assistive applications through sEMG analysis, Arora A., Sarkar D., Kumar A. et al. Journal of Mechanical Science and  Technology, Vol. 36, 4719–4733 (2022), doi: https://doi.org/10.1007/s12206-022-0832-0"),
                         ("[3]", "Custom made silicone bladder based low pressure actuated pneumatic artificial muscle, Indian Patent Application: IN 202211021971 (Filed: 12/04/2022; Status: Pending)"),
                         ("[4]", "A method and apparatus for characterizing pneumatic artificial muscle actuators along with measurement of its air consumption under varying operating conditions, Indian Patent Application: IN 202211071703 (Filed: 13/12/2022; Status: Pending)")]:
            r = QLabel(f'<span style="color:{ACCENT}; font-weight:800;">{tag}</span>  '
                       f'<span style="color:{BODY};">{txt}</span>')
            r.setTextFormat(Qt.RichText); r.setWordWrap(True)
            r.setStyleSheet("background:transparent; border:none; font-size:12px;")
            rfl.addWidget(r)
        al.addWidget(rf); al.addStretch(1)
        shadow(ac); low.addWidget(ac, 1)
        lw = QWidget(); lw.setLayout(low); lay.addWidget(lw)
        lay.addStretch(1)
        return sa

    # ════════════════════════════════════════════════════════════════
    #  Design Space
    # ════════════════════════════════════════════════════════════════
    def _page_design(self):
        sa, lay = self._scroll_page()
        lay.addWidget(caption("The optimiser searches only within these limits. Keep each "
                              "range inside the ANN training extent shown on the bar — "
                              "predictions outside it are unreliable."))

        body = QHBoxLayout(); body.setSpacing(16)
        left = card(); lv = QVBoxLayout(left); lv.setContentsMargins(22, 18, 22, 18); lv.setSpacing(13)
        ds = self._cfg["design_space"]; self._ds = {}
        meta = [("L", "Muscle length", 1.0, 0), ("D", "Outer diameter", 0.5, 1),
                ("T", "Wall thickness", 0.1, 2), ("P", "Actuation pressure", 5.0, 0)]
        for key, name, step, dec in meta:
            lo, hi, unit = VAR_RANGE[key]
            lv.addWidget(section(f"{key} — {name}  ({unit})", color=INK))
            mn = SliderField("Minimum", lo, hi, ds[f"{key}_min"], step, unit, dec,
                             on_change=lambda k=key: self._ds_changed(k), label_w=72)
            mx = SliderField("Maximum", lo, hi, ds[f"{key}_max"], step, unit, dec,
                             on_change=lambda k=key: self._ds_changed(k), label_w=72)
            lv.addWidget(mn); lv.addWidget(mx)
            bar = RangeBar(lo, hi, unit, dec); bar.set_band(ds[f"{key}_min"], ds[f"{key}_max"])
            lv.addWidget(bar)
            cap = QLabel(f"Teal band = selected range · light ends = training limits "
                         f"{lo}–{hi} {unit}")
            cap.setStyleSheet(f"color:{MUTED}; font-size:11px;"); lv.addWidget(cap)
            self._ds[key] = (mn, mx, bar)
        shadow(left); body.addWidget(left, 1)

        right = QVBoxLayout(); right.setSpacing(16)
        ev = card(); evl = QVBoxLayout(ev); evl.setContentsMargins(20, 16, 20, 16); evl.setSpacing(9)
        evl.addWidget(section("Design parameters — exploded view", color=INK))
        evl.addWidget(Placeholder("exploded_view.png",
                      "Exploded render labelling L (length, mm), D (outer diameter, mm), "
                      "T (wall thickness, mm) and the P actuation port (kPa).", min_h=220))
        shadow(ev); right.addWidget(ev)
        col = card(); cvl = QVBoxLayout(col); cvl.setContentsMargins(20, 16, 20, 16); cvl.setSpacing(9)
        cvl.addWidget(section("Experimental specimens", color=INK))
        cvl.addWidget(Placeholder("pstam_collage.png",
                      "Collage of the fabricated pSTAM units used across the M1–M3 "
                      "experiments, annotated with their L / D / T values.", min_h=180))
        shadow(col); right.addWidget(col)
        rw = QWidget(); rw.setLayout(right); body.addWidget(rw, 1)
        bw = QWidget(); bw.setLayout(body); lay.addWidget(bw)
        lay.addStretch(1)
        return sa

    def _ds_changed(self, key):
        mn, mx, bar = self._ds[key]; bar.set_band(mn.value(), mx.value())

    # ════════════════════════════════════════════════════════════════
    #  Surrogate Models
    # ════════════════════════════════════════════════════════════════
    def _page_models(self):
        sa, lay = self._scroll_page()
        lay.addWidget(caption("These are data-driven surrogates, not physics models: each is "
                              "an artificial neural network fitted to bench measurements and "
                              "validated on a held-out split. The RMSE shown is the held-out "
                              "error in the model's own output unit."))
        info = [
            ("M1", "Static contraction", "f(L, D, T, F, P) → δ", "mm",
             [("L", "Muscle length", "150–350 mm"), ("D", "Outer diameter", "14–26 mm"),
              ("T", "Wall thickness", "3–6 mm"), ("F", "Axial load", "10–150 N"),
              ("P", "Actuation pressure", "20–200 kPa"), ("δ", "Contraction (output)", "0.1–89.6 mm")],
             "Quasi-static shortening under a constant axial load at the supply "
             "pressure. This is the primary objective driver.",
             ["M1_experiment.gif", "contraction.gif"]),
            ("M2", "Inflation time", "f(L, D, T, P, F, s%) → t_inf", "ms",
             [("L", "Muscle length", "150–350 mm"), ("D", "Outer diameter", "14–26 mm"),
              ("T", "Wall thickness", "3–6 mm"), ("P", "Actuation pressure", "20–200 kPa"),
              ("F", "Axial load", "10–150 N"), ("s%", "Stroke fraction", "20–90 %"),
              ("t_inf", "Inflation time (output)", "25–1652 ms")],
             "Time to reach the target stroke fraction during pressurisation. "
             "Lower means a faster response.",
             ["M2_experiment.gif", "inflation.gif"]),
            ("M3", "Deflation time", "f(L, D, T, P, F, s%) → t_def", "ms",
             [("L", "Muscle length", "150–350 mm"), ("D", "Outer diameter", "14–26 mm"),
              ("T", "Wall thickness", "3–6 mm"), ("P", "Actuation pressure", "20–200 kPa"),
              ("F", "Axial load", "10–150 N"), ("s%", "Stroke fraction", "20–90 %"),
              ("t_def", "Deflation time (output)", "8.6–2853 ms")],
             "Time to release from the stroke fraction during venting. With M2 this "
             "describes the full actuation cycle.",
             ["M3_experiment.gif", "deflation.gif"]),
        ]
        max_inputs = max(len([p for p in m[4] if "output" not in p[1]]) for m in info)

        cols = QHBoxLayout(); cols.setSpacing(14)
        for mid, title, sig, un, params, desc, gif in info:
            c = card(); cl = QVBoxLayout(c); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
            band = QFrame(); band.setObjectName("modelhead")
            bl = QVBoxLayout(band); bl.setContentsMargins(18, 13, 18, 12); bl.setSpacing(1)
            bl.setAlignment(Qt.AlignHCenter)
            tt = QLabel(mid); tt.setAlignment(Qt.AlignHCenter)
            tt.setStyleSheet("color:#FFFFFF; font-size:21px; font-weight:800;")
            bl.addWidget(tt)
            st = QLabel(title); st.setAlignment(Qt.AlignHCenter)
            st.setStyleSheet("color:#EAF6F7; font-size:13px; font-weight:700;")
            bl.addWidget(st); cl.addWidget(band)

            inner = QVBoxLayout(); inner.setContentsMargins(16, 13, 16, 14); inner.setSpacing(11)

            sigwrap = QFrame(); sigwrap.setObjectName("inset")
            sw = QVBoxLayout(sigwrap); sw.setContentsMargins(10, 8, 10, 8); sw.setSpacing(2)
            sw.setAlignment(Qt.AlignHCenter)
            seb = QLabel("SURROGATE FUNCTION"); seb.setAlignment(Qt.AlignHCenter)
            seb.setStyleSheet(f"color:{MUTED}; font-size:10px; letter-spacing:1.5px;"
                              f"font-weight:700;")
            sw.addWidget(seb)
            sigl = QLabel(f'<span style="white-space:nowrap;">{_esc(sig)}</span>')
            sigl.setTextFormat(Qt.RichText); sigl.setAlignment(Qt.AlignHCenter)
            sigl.setStyleSheet(f"color:{MODEL}; font-family:monospace; font-size:14px;"
                               f"font-weight:700;")
            sw.addWidget(sigl)
            inner.addWidget(sigwrap)

            inner.addWidget(Placeholder(gif, f"{title} — experiment clip.",
                                        min_h=118, fit_width=True, max_h=360))

            dg = self._diag.get(mid, {})
            rmse = QFrame(); rmse.setObjectName("inset")
            rl = QHBoxLayout(rmse); rl.setContentsMargins(12, 9, 12, 9); rl.setSpacing(22)
            if dg.get("ok"):
                rl.addStretch(1)
                rl.addWidget(self._stat("RMSE · train", f"{dg['rmse_train']:.2f} {un}", GOOD))
                rl.addWidget(self._stat("RMSE · test", f"{dg['rmse_test']:.2f} {un}", BAD))
                rl.addStretch(1)
            else:
                rl.addWidget(QLabel("RMSE unavailable"))
            inner.addWidget(rmse)

            inputs = [p for p in params if "output" not in p[1]]
            outputs = [p for p in params if "output" in p[1]]
            tbl = QGridLayout(); tbl.setHorizontalSpacing(10); tbl.setVerticalSpacing(5)
            tbl.setColumnStretch(1, 1)
            hsym = QLabel("SYMBOL"); hsym.setStyleSheet(
                f"color:{MUTED}; font-size:9.5px; letter-spacing:0.8px; font-weight:700;")
            hpar = QLabel("PARAMETER"); hpar.setStyleSheet(
                f"color:{MUTED}; font-size:9.5px; letter-spacing:0.8px; font-weight:700;")
            hrng = QLabel("TRAINING RANGE"); hrng.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            hrng.setStyleSheet(
                f"color:{MUTED}; font-size:9.5px; letter-spacing:0.8px; font-weight:700;")
            tbl.addWidget(hsym, 0, 0); tbl.addWidget(hpar, 0, 1); tbl.addWidget(hrng, 0, 2)
            r = 1
            for sym, nm, rng in inputs:
                a = QLabel(sym); a.setStyleSheet(
                    f"color:{INK}; font-family:monospace; font-size:12px; font-weight:700;")
                b = QLabel(nm); b.setStyleSheet(f"color:{BODY}; font-size:12px;")
                cc = QLabel(rng); cc.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cc.setStyleSheet(f"color:{BODY}; font-family:monospace; font-size:11px;")
                tbl.addWidget(a, r, 0); tbl.addWidget(b, r, 1); tbl.addWidget(cc, r, 2)
                r += 1
            while r <= max_inputs:                      # pad so all cards align
                sp = QLabel(" "); sp.setStyleSheet("font-size:12px;")
                tbl.addWidget(sp, r, 0); r += 1
            div = QFrame(); div.setFixedHeight(1); div.setStyleSheet(f"background:{LINE};")
            tbl.addWidget(div, r, 0, 1, 3); r += 1
            otag = QLabel("OUTPUT"); otag.setStyleSheet(
                f"color:{GOOD}; font-size:9.5px; letter-spacing:0.8px; font-weight:800;")
            tbl.addWidget(otag, r, 0, 1, 3); r += 1
            for sym, nm, rng in outputs:
                a = QLabel(sym); a.setStyleSheet(
                    f"color:{VAR}; font-family:monospace; font-size:13px; font-weight:800;")
                b = QLabel(nm.replace(" (output)", "")); b.setStyleSheet(
                    f"color:{INK}; font-size:12px; font-weight:600;")
                cc = QLabel(rng); cc.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cc.setStyleSheet(f"color:{BODY}; font-family:monospace; font-size:11px;")
                tbl.addWidget(a, r, 0); tbl.addWidget(b, r, 1); tbl.addWidget(cc, r, 2)
                r += 1
            tw = QWidget(); tw.setLayout(tbl); inner.addWidget(tw)

            dl = QLabel(desc); dl.setWordWrap(True)
            dl.setStyleSheet(f"color:{BODY}; font-size:11.5px;")
            inner.addWidget(dl)
            inner.addStretch(1)
            cl.addLayout(inner)
            shadow(c); cols.addWidget(c, 1)
        cw = QWidget(); cw.setLayout(cols); lay.addWidget(cw)
        lay.addStretch(1)
        return sa

    def _stat(self, label, value, color):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(1)
        v.setAlignment(Qt.AlignHCenter)
        l = QLabel(label.upper()); l.setAlignment(Qt.AlignHCenter)
        l.setStyleSheet(f"color:{MUTED}; font-size:10px; letter-spacing:1px; font-weight:700;")
        val = QLabel(value); val.setAlignment(Qt.AlignHCenter)
        val.setStyleSheet(f"color:{color}; font-size:16px; font-weight:800;")
        v.addWidget(l); v.addWidget(val)
        return w

    # ════════════════════════════════════════════════════════════════
    #  Objective & Constraints
    # ════════════════════════════════════════════════════════════════
    def _page_objcon(self):
        page = QWidget()
        outer = QVBoxLayout(page); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)

        # ---- scrollable area: header + the two editor columns ----
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setObjectName("scroll")
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inner = QWidget(); inner.setObjectName("pagebg")
        lay = QVBoxLayout(inner); lay.setContentsMargins(28, 16, 28, 8); lay.setSpacing(10)
        lay.addWidget(caption("The objective is a weighted sum of model terms. Each "
                              "constraint adds a penalty when its limit is crossed. "
                              "Weight 0 mutes a term; ✕ removes a row."))

        twocol = QHBoxLayout(); twocol.setSpacing(16)

        # objective column
        oc = card(); ov = QVBoxLayout(oc); ov.setContentsMargins(20, 16, 20, 16); ov.setSpacing(10)
        ov.addWidget(section("Objective function", color=INK))
        mode_row = QHBoxLayout(); mode_row.setSpacing(10)
        ml = QLabel("Direction"); ml.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        ml.setMinimumWidth(70); ml.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        mode_row.addWidget(ml)
        self._mode_grp = QButtonGroup(self)
        self._rb_max = QRadioButton("Maximise"); self._rb_min = QRadioButton("Minimise")
        self._mode_grp.addButton(self._rb_max); self._mode_grp.addButton(self._rb_min)
        (self._rb_max if self._cfg["objective"]["mode"] == "maximize"
         else self._rb_min).setChecked(True)
        self._rb_max.toggled.connect(self._refresh_preview)
        mode_row.addWidget(self._rb_max); mode_row.addWidget(self._rb_min); mode_row.addStretch(1)
        ov.addLayout(mode_row)
        objcap = QLabel("Objective = Σ  weight<sub>i</sub> × term<sub>i</sub>")
        objcap.setTextFormat(Qt.RichText)
        objcap.setStyleSheet(f"color:{MUTED}; font-size:11.5px;")
        ov.addWidget(objcap)
        self._term_box = QVBoxLayout(); self._term_box.setSpacing(10)
        ov.addLayout(self._term_box)
        for t in self._cfg["objective"]["terms"]:
            self._add_term(t)
        addt = QPushButton("+ Add term"); addt.setObjectName("ghost")
        addt.setCursor(Qt.PointingHandCursor); addt.clicked.connect(lambda: self._add_term())
        ar = QHBoxLayout(); ar.addWidget(addt); ar.addStretch(1); ov.addLayout(ar)
        ov.addStretch(1)
        shadow(oc); twocol.addWidget(oc, 1)

        # constraints column
        cc = card(); cv = QVBoxLayout(cc); cv.setContentsMargins(20, 16, 20, 16); cv.setSpacing(10)
        cv.addWidget(section("Constraints", color=INK))
        cv.addWidget(caption("Penalty = weight × max(0, violation). Use sweep + max/min "
                             "for worst-case limits.", MUTED))
        self._con_box = QVBoxLayout(); self._con_box.setSpacing(10)
        cv.addLayout(self._con_box)
        for c in self._cfg["constraints"]:
            self._add_con(c)
        addc = QPushButton("+ Add constraint"); addc.setObjectName("ghost")
        addc.setCursor(Qt.PointingHandCursor); addc.clicked.connect(lambda: self._add_con())
        ar2 = QHBoxLayout(); ar2.addWidget(addc); ar2.addStretch(1); cv.addLayout(ar2)
        cv.addStretch(1)
        shadow(cc); twocol.addWidget(cc, 1)

        twW = QWidget(); twW.setLayout(twocol); lay.addWidget(twW)
        lay.addStretch(1)
        sa.setWidget(inner)
        outer.addWidget(sa, 1)

        # pinned footer: cost readout (left) beside model reference (right)
        footer = QWidget(); footer.setObjectName("pagebg")
        fl = QVBoxLayout(footer); fl.setContentsMargins(28, 6, 28, 12); fl.setSpacing(8)
        fl.addWidget(hrule(LINE))
        frow = QHBoxLayout(); frow.setSpacing(14)

        prev = card(); pv = QVBoxLayout(prev); pv.setContentsMargins(18, 11, 18, 11); pv.setSpacing(7)
        pv.addWidget(section("Cost function — live readout", color=INK))
        self._preview = QTextBrowser(); self._preview.setObjectName("preview")
        self._preview.setLineWrapMode(QTextBrowser.NoWrap)
        self._preview.setMinimumHeight(112); self._preview.setMaximumHeight(150)
        pv.addWidget(self._preview)
        leg = QHBoxLayout(); leg.setSpacing(12)
        for col, lab in [(VAR, "design variable"), (FIX, "fixed input"), (MODEL, "model call"),
                         (SWEEP, "sweep"), (AGG, "aggregation"), (CON, "constraint")]:
            dot = QLabel("●"); dot.setStyleSheet(f"color:{col}; font-size:11px;")
            tx = QLabel(lab); tx.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            leg.addWidget(dot); leg.addWidget(tx)
        leg.addStretch(1); pv.addLayout(leg)
        shadow(prev); frow.addWidget(prev, 3)

        ref = card(); rv = QVBoxLayout(ref); rv.setContentsMargins(18, 11, 18, 11); rv.setSpacing(6)
        rv.addWidget(section("Surrogate models", color=INK))
        ref_meta = {"M1": ("Static contraction", "f(L, D, T, F, P) → δ [mm]"),
                    "M2": ("Inflation time", "f(L, D, T, P, F, s%) → t_inf [ms]"),
                    "M3": ("Deflation time", "f(L, D, T, P, F, s%) → t_def [ms]")}
        for m in MODELS:
            phys, sig = ref_meta[m]
            mc = QFrame(); mc.setObjectName("inset")
            mlc = QHBoxLayout(mc); mlc.setContentsMargins(11, 6, 11, 6); mlc.setSpacing(10)
            left = QVBoxLayout(); left.setSpacing(0)
            t = QLabel(f"{m} · {phys}")
            t.setStyleSheet(f"color:{MODEL}; font-size:12px; font-weight:800;")
            s = QLabel(f'<span style="white-space:nowrap;">{_esc(sig)}</span>')
            s.setTextFormat(Qt.RichText)
            s.setStyleSheet(f"color:{BODY}; font-family:monospace; font-size:10.5px;")
            left.addWidget(t); left.addWidget(s); mlc.addLayout(left); mlc.addStretch(1)
            dg = self._diag.get(m, {})
            if dg.get("ok"):
                rr = QVBoxLayout(); rr.setSpacing(0); rr.setAlignment(Qt.AlignRight)
                rl = QLabel("RMSE train · test")
                rl.setStyleSheet(f"color:{MUTED}; font-size:9px; letter-spacing:0.5px;")
                rl.setAlignment(Qt.AlignRight)
                rv2 = QLabel(f"{dg['rmse_train']:.1f} · {dg['rmse_test']:.1f} {UNIT[m]}")
                rv2.setStyleSheet(f"color:{BAD}; font-size:11px; font-weight:700;")
                rv2.setAlignment(Qt.AlignRight)
                rr.addWidget(rl); rr.addWidget(rv2); mlc.addLayout(rr)
            rv.addWidget(mc)
        rv.addStretch(1)
        shadow(ref); frow.addWidget(ref, 2)

        fl.addLayout(frow)
        outer.addWidget(footer)
        return page

    def _add_term(self, data=None):
        if len(self._t_blocks) >= 8:
            self._toast("At most 8 objective terms.", ACCENT); return
        if data is None:
            data = {"model": "M1", "fixed": {"F": 75, "stroke": 55}, "sweep": None,
                    "aggregate": "single", "weight": 1.0}
        b = TermBlock(len(self._t_blocks) + 1, data,
                      self._refresh_preview, self._remove_term_block)
        self._t_blocks.append(b); self._term_box.addWidget(b); self._refresh_preview()

    def _remove_term_block(self, b):
        if len(self._t_blocks) <= 1:
            self._toast("At least one objective term is required.", ACCENT); return
        self._t_blocks.remove(b); b.setParent(None); b.deleteLater()
        for i, blk in enumerate(self._t_blocks):
            blk.set_index(i + 1)
        self._refresh_preview()

    def _add_con(self, data=None):
        if len(self._c_blocks) >= 8:
            self._toast("At most 8 constraints.", ACCENT); return
        if data is None:
            data = {"enabled": True, "model": "M2", "fixed": {"F": 75, "stroke": 55},
                    "sweep": None, "aggregate": "single", "threshold": 500,
                    "direction": "≤", "penalty_weight": 100_000}
        b = ConstraintBlock(len(self._c_blocks) + 1, data,
                            self._refresh_preview, self._remove_con_block)
        self._c_blocks.append(b); self._con_box.addWidget(b); self._refresh_preview()

    def _remove_con_block(self, b):
        self._c_blocks.remove(b); b.setParent(None); b.deleteLater()
        for i, blk in enumerate(self._c_blocks):
            blk.set_index(i + 1)
        self._refresh_preview()

    def _refresh_preview(self):
        if not hasattr(self, "_preview"):
            return
        try:
            mode = "maximize" if self._rb_max.isChecked() else "minimize"
            terms = [b.get_dict() for b in self._t_blocks]
            cons = [b.get_dict() for b in self._c_blocks]
            self._preview.setHtml(cost_preview_html(mode, terms, cons))
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════
    #  Optimise & Report
    # ════════════════════════════════════════════════════════════════
    def _page_optimise(self):
        sa, lay = self._scroll_page()
        top = QHBoxLayout(); top.setSpacing(16)
        pso = card(); pv = QVBoxLayout(pso); pv.setContentsMargins(22, 18, 22, 18); pv.setSpacing(9)
        pv.addWidget(section("Particle-swarm settings", color=INK))
        p = self._cfg["pso"]

        def pso_row(field, hint):
            pv.addWidget(field)
            h = QLabel(hint); h.setWordWrap(True)
            h.setStyleSheet(f"color:{MUTED}; font-size:11px; margin-left:2px;")
            pv.addWidget(h)

        self._pso_n = SliderField("Swarm size", 5, 200, p["n_particles"], 1, "", 0)
        pso_row(self._pso_n, "More particles cover more of the space each iteration, "
                "but each iteration costs more.")
        self._pso_it = SliderField("Iterations", 10, 500, p["n_iterations"], 1, "", 0)
        pso_row(self._pso_it, "More iterations give the swarm longer to converge; "
                "runtime grows in proportion.")
        self._pso_c1 = SliderField("c1 — cognitive", 0.01, 2.0, p["c1"], 0.01, "", 2)
        pso_row(self._pso_c1, "Higher pulls each particle toward its own best — "
                "more exploration, slower agreement.")
        self._pso_c2 = SliderField("c2 — social", 0.01, 2.0, p["c2"], 0.01, "", 2)
        pso_row(self._pso_c2, "Higher pulls the swarm toward the global best — "
                "faster convergence, more risk of local optima.")
        self._pso_w = SliderField("w — inertia", 0.1, 1.5, p["w"], 0.01, "", 2)
        pso_row(self._pso_w, "Higher keeps momentum and explores; lower settles sooner. "
                "A good start is 0.7–0.9.")
        shadow(pso); top.addWidget(pso, 1)

        rc = card(); rv = QVBoxLayout(rc); rv.setContentsMargins(22, 20, 22, 20); rv.setSpacing(13)
        rv.addWidget(section("Run", color=INK))
        brow = QHBoxLayout(); brow.setSpacing(10)
        self._run_btn = QPushButton("Run optimisation"); self._run_btn.setObjectName("cta")
        self._run_btn.setCursor(Qt.PointingHandCursor); self._run_btn.clicked.connect(self._start)
        brow.addWidget(self._run_btn)
        self._stop_btn = QPushButton("Stop"); self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False); self._stop_btn.clicked.connect(self._stop)
        brow.addWidget(self._stop_btn); brow.addStretch(1)
        rv.addLayout(brow)
        self._prog_lbl = QLabel("Idle"); self._prog_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        rv.addWidget(self._prog_lbl)
        self._progbar = QFrame(); self._progbar.setObjectName("progtrack"); self._progbar.setFixedHeight(8)
        self._progfill = QFrame(self._progbar); self._progfill.setObjectName("progfill")
        self._progfill.setGeometry(0, 0, 0, 8)
        rv.addWidget(self._progbar)

        self._plot_lbl = QLabel(); self._plot_lbl.setObjectName("plot")
        self._plot_lbl.setAlignment(Qt.AlignCenter); self._plot_lbl.setWordWrap(True)
        self._plot_lbl.setMinimumHeight(230)
        if HAVE_MPL:
            self._plot_lbl.setText("Run an optimisation to see the convergence curve.")
        else:
            self._plot_lbl.setText("Convergence plot needs matplotlib (optional). The "
                                   "optimiser, the report and CSV / JSON / TXT export all "
                                   "work without it.")
        rv.addWidget(self._plot_lbl)
        shadow(rc); top.addWidget(rc, 1)
        tw = QWidget(); tw.setLayout(top); lay.addWidget(tw)

        rep = card(); rl = QVBoxLayout(rep); rl.setContentsMargins(22, 20, 22, 20); rl.setSpacing(11)
        rl.addWidget(section("Optimal-design report", color=INK))
        self._report = QTextBrowser(); self._report.setObjectName("report")
        self._report.setMinimumHeight(260)
        self._report.setHtml(f'<div style="color:{FAINT}; font-family:Inter,sans-serif;'
                             f'font-size:12px;">Run an optimisation to populate the report.</div>')
        rl.addWidget(self._report)
        shadow(rep); lay.addWidget(rep)

        exp = card(); el = QVBoxLayout(exp); el.setContentsMargins(22, 20, 22, 20); el.setSpacing(11)
        el.addWidget(section("Export results", color=INK))
        el.addWidget(caption("Writes to results/: a full CSV (design, evaluated objective "
                             "terms, constraints, convergence history), a round-trippable "
                             "config JSON, a plain-text report"
                             + (", and a rendered PDF." if HAVE_MPL else
                                ". (PDF needs matplotlib.)")))
        erow = QHBoxLayout(); erow.setSpacing(10)
        fl = QLabel("File name"); fl.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        fl.setMinimumWidth(88); fl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        erow.addWidget(fl)
        self._fname = QLineEdit("pstam_design"); self._fname.setMinimumWidth(240)
        erow.addWidget(self._fname)
        self._save_btn = QPushButton("Save results"); self._save_btn.setObjectName("cta")
        self._save_btn.setEnabled(False); self._save_btn.setCursor(Qt.PointingHandCursor)
        self._save_btn.clicked.connect(self._save_results)
        erow.addWidget(self._save_btn); erow.addStretch(1)
        el.addLayout(erow)
        self._save_note = QLabel(""); self._save_note.setWordWrap(True)
        self._save_note.setStyleSheet(f"color:{GOOD}; font-size:11px;")
        el.addWidget(self._save_note)
        shadow(exp); lay.addWidget(exp)
        lay.addStretch(1)
        return sa

    def _set_progress(self, frac):
        w = int(self._progbar.width() * max(0.0, min(1.0, frac)))
        self._progfill.setGeometry(0, 0, w, 8)

    # ════════════════════════════════════════════════════════════════
    #  Config collect / apply / load / save
    # ════════════════════════════════════════════════════════════════
    def _collect(self):
        cfg = get_default_config()
        for key, (mn, mx, _bar) in self._ds.items():
            cfg["design_space"][f"{key}_min"] = mn.value()
            cfg["design_space"][f"{key}_max"] = mx.value()
        cfg["objective"]["mode"] = "maximize" if self._rb_max.isChecked() else "minimize"
        cfg["objective"]["terms"] = [b.get_dict() for b in self._t_blocks]
        cfg["constraints"] = [b.get_dict() for b in self._c_blocks]
        cfg["pso"]["n_particles"] = int(self._pso_n.value())
        cfg["pso"]["n_iterations"] = int(self._pso_it.value())
        cfg["pso"]["c1"] = self._pso_c1.value()
        cfg["pso"]["c2"] = self._pso_c2.value()
        cfg["pso"]["w"] = self._pso_w.value()
        return cfg

    def _apply_cfg(self, cfg):
        ds = cfg.get("design_space", {})
        for key, (mn, mx, bar) in self._ds.items():
            if f"{key}_min" in ds:
                mn.set_value(ds[f"{key}_min"]); mx.set_value(ds[f"{key}_max"])
                bar.set_band(ds[f"{key}_min"], ds[f"{key}_max"])
        mode = cfg.get("objective", {}).get("mode", "maximize")
        (self._rb_max if mode == "maximize" else self._rb_min).setChecked(True)
        for b in self._t_blocks:
            b.setParent(None); b.deleteLater()
        self._t_blocks = []
        for t in cfg.get("objective", {}).get("terms", []) or [{}]:
            self._add_term(t)
        for b in self._c_blocks:
            b.setParent(None); b.deleteLater()
        self._c_blocks = []
        for c in cfg.get("constraints", []):
            self._add_con(c)
        pso = cfg.get("pso", {})
        if "n_particles" in pso:
            self._pso_n.set_value(pso["n_particles"]); self._pso_it.set_value(pso["n_iterations"])
            self._pso_c1.set_value(pso["c1"]); self._pso_c2.set_value(pso["c2"])
            self._pso_w.set_value(pso["w"])
        self._refresh_preview()

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load configuration", RESULTS_DIR,
            "Configuration (*.json);;All files (*.*)")
        if not path:
            return
        try:
            try:
                with open(path, encoding="utf-8-sig") as fh:
                    cfg = json.load(fh)
            except UnicodeDecodeError:
                with open(path, encoding="cp1252") as fh:
                    cfg = json.load(fh)
            self._apply_cfg(cfg)
            self._toast(f"Loaded {os.path.basename(path)}", GOOD)
        except Exception as e:
            self._toast(f"Could not load: {e}", BAD)

    def _save_config_only(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", os.path.join(RESULTS_DIR, "pstam_config.json"),
            "Configuration (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._collect(), fh, indent=2, ensure_ascii=False)
            self._toast(f"Saved {os.path.basename(path)}", GOOD)
        except Exception as e:
            self._toast(f"Could not save: {e}", BAD)

    # ════════════════════════════════════════════════════════════════
    #  Run
    # ════════════════════════════════════════════════════════════════
    def _start(self):
        if self._thread is not None:
            return
        try:
            cfg = self._collect()
        except Exception as e:
            self._toast(f"Configuration error: {e}", BAD); return
        self._cancelled = False
        self._run_btn.setEnabled(False); self._stop_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._prog_lbl.setText("Running…"); self._set_progress(0)
        if HAVE_MPL:
            self._plot_lbl.setPixmap(QPixmap())
            self._plot_lbl.setText("Running…")

        thread = QThread(self)                  # owned by the window
        worker = OptWorker(cfg)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        worker.done.connect(self._on_done, Qt.QueuedConnection)
        worker.failed.connect(self._on_failed, Qt.QueuedConnection)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished, Qt.QueuedConnection)
        self._thread, self._worker = thread, worker
        thread.start()

    @pyqtSlot(int, int)
    def _on_progress(self, it, total):
        self._set_progress(it / total)
        self._prog_lbl.setText(f"Iteration {it} / {total}")

    @pyqtSlot()
    def _on_thread_finished(self):
        # Runs on the GUI thread after the worker thread has fully stopped,
        # so the thread object is never released while it is still running.
        thread = self._thread
        self._thread = None; self._worker = None
        if thread is not None:
            thread.wait(); thread.deleteLater()
        self._run_btn.setEnabled(True); self._stop_btn.setEnabled(False)

    @pyqtSlot(object)
    def _on_done(self, result):
        self._stop_btn.setEnabled(False)
        if getattr(self, "_cancelled", False):
            self._prog_lbl.setText("Stopped"); return
        self._set_progress(1.0); self._prog_lbl.setText("Complete")
        self._last_result = result; self._save_btn.setEnabled(True)
        self._report.setHtml(report_html(result)); self._plot_convergence(result)

    @pyqtSlot(str)
    def _on_failed(self, msg):
        self._stop_btn.setEnabled(False)
        self._prog_lbl.setText("Error")
        QMessageBox.critical(self, "Optimisation error", msg[-1500:])

    def closeEvent(self, event):
        thread = self._thread
        if thread is not None and thread.isRunning():
            answer = QMessageBox.question(
                self, "Optimisation running",
                "An optimisation is still running. Close the toolbox once it "
                "finishes? The window closes as soon as the current run ends.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore(); return
            self.hide()
            QApplication.processEvents()
            thread.quit(); thread.wait()
        event.accept()

    def _stop(self):
        self._cancelled = True; self._stop_btn.setEnabled(False)
        self._prog_lbl.setText("Stopping after current run…")

    def _plot_convergence(self, result):
        if not HAVE_MPL:
            return
        try:
            hist = list(result["cost_history"]); mode = result["report"]["mode"]
            vals = [-v for v in hist] if mode == "maximize" else hist
            xs = list(range(1, len(vals) + 1))
            fig = Figure(figsize=(5.6, 2.9), facecolor="white")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#FBFDFE")
            ymin, ymax = min(vals), max(vals)
            pad = max((ymax - ymin) * 0.12, abs(ymax) * 0.03, 0.5)
            ax.plot(xs, vals, color=PRIMARY, linewidth=2.0)
            if len(vals) > 1:
                ax.fill_between(xs, vals, ymin - pad, color=PRIMARY, alpha=0.08)
                ax.set_xlim(1, len(vals))
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlabel("Iteration", color=BODY, fontsize=10)
            ax.set_ylabel("Objective value " + ("(maximised)" if mode == "maximize"
                                                else "(minimised)"), color=BODY, fontsize=10)
            for sp in ax.spines.values():
                sp.set_color(LINE)
            ax.tick_params(colors=BODY, labelsize=9)
            ax.grid(True, color=LINE_SOFT, linewidth=0.7)
            fig.tight_layout(pad=1.2)
            w = max(360, min(self._plot_lbl.width() or 520, 720))
            pm = fig_to_pixmap(fig, w)
            self._plot_lbl.setText("")
            self._plot_lbl.setMinimumHeight(pm.height() + 6)
            self._plot_lbl.setPixmap(pm)
        except Exception as e:
            self._plot_lbl.setText(f"Plot unavailable: {e}")

    # ════════════════════════════════════════════════════════════════
    #  Save results — CSV + JSON + TXT (+ PDF if matplotlib present)
    # ════════════════════════════════════════════════════════════════
    def _save_results(self):
        if not self._last_result:
            return
        base = "".join(ch for ch in self._fname.text().strip()
                       if ch.isalnum() or ch in ("_", "-")) or "pstam_design"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stem = os.path.join(RESULTS_DIR, base)
        written = []
        try:
            self._write_csv(stem + ".csv"); written.append(base + ".csv")
            with open(stem + "_config.json", "w", encoding="utf-8") as fh:
                json.dump(self._collect(), fh, indent=2, ensure_ascii=False)
            written.append(base + "_config.json")
            txt = self._report_text()
            with open(stem + "_report.txt", "w", encoding="utf-8") as fh:
                fh.write(txt)
            written.append(base + "_report.txt")
            if HAVE_MPL:
                try:
                    self._write_pdf(stem + "_report.pdf", txt)
                    written.append(base + "_report.pdf")
                except Exception:
                    pass
            self._save_note.setStyleSheet(f"color:{GOOD}; font-size:11px;")
            self._save_note.setText("Saved " + " · ".join(written) + "  →  results/")
        except Exception as e:
            self._save_note.setStyleSheet(f"color:{BAD}; font-size:11px;")
            self._save_note.setText(f"Save failed: {e}")

    def _write_csv(self, path):
        r = self._last_result; rep = r["report"]; pos = r["best_position"]
        hist = pd.DataFrame(list(r["cost_history"]), columns=["solver_internal_value"])
        hist.index = hist.index + 1; hist.index.name = "iteration"
        sense = "maximised" if rep["mode"] == "maximize" else "minimised"
        design = pd.DataFrame([{"L_mm": pos[0], "D_mm": pos[1], "T_mm": pos[2], "P_kPa": pos[3],
                                "objective_sense": sense,
                                "objective_value": rep["total_objective"]}])
        terms = pd.DataFrame([{"term": t["term_index"], "model": t["model"],
                               "meaning": MODEL_PHYS[t["model"]][0], "aggregate": t["aggregate"],
                               "prediction": t["prediction"], "unit": MODEL_PHYS[t["model"]][1],
                               "weight": t["weight"], "weighted": t["weighted"]}
                              for t in rep["objective_terms"]])
        cons = pd.DataFrame([{"constraint": c["constraint_index"], "model": c["model"],
                              "meaning": MODEL_PHYS[c["model"]][0], "aggregate": c["aggregate"],
                              "direction": c["direction"], "threshold": c["threshold"],
                              "prediction": c["prediction"], "unit": MODEL_PHYS[c["model"]][1],
                              "violated": c["violated"]} for c in rep["constraints"]])
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            fh.write("# OPTIMAL DESIGN\n"); design.to_csv(fh, index=False)
            fh.write("\n# OBJECTIVE TERMS (evaluated)\n"); terms.to_csv(fh, index=False)
            fh.write("\n# CONSTRAINTS (evaluated)\n"); cons.to_csv(fh, index=False)
            fh.write("\n# CONVERGENCE HISTORY\n"); hist.to_csv(fh)

    def _report_text(self):
        r = self._last_result; rep = r["report"]; pos = r["best_position"]
        sense = "maximised" if rep["mode"] == "maximize" else "minimised"
        L = ["=" * 62, "  pSTAM DESIGN OPTIMISATION — REPORT", "=" * 62, "",
             "OPTIMAL DESIGN",
             f"  L  muscle length        {pos[0]:9.3f} mm",
             f"  D  outer diameter       {pos[1]:9.3f} mm",
             f"  T  wall thickness       {pos[2]:9.3f} mm",
             f"  P  actuation pressure   {pos[3]:9.3f} kPa", "",
             f"OBJECTIVE TERMS  (objective is {sense})"]
        for t in rep["objective_terms"]:
            phys, un = MODEL_PHYS[t["model"]]
            L.append(f"  T{t['term_index']}  {t['model']} · {phys:<24} "
                     f"{t['aggregate']:>6}  {t['prediction']:10.3f} {un:<3} "
                     f"× {t['weight']:<5g} = {t['weighted']:10.3f}")
        L += [f"  {'-'*54}",
              f"  Objective value ({sense}): {rep['total_objective']:.4f}", "",
              "CONSTRAINTS"]
        for c in rep["constraints"]:
            phys, un = MODEL_PHYS[c["model"]]
            status = "VIOLATED" if c["violated"] else "PASS"
            L.append(f"  C{c['constraint_index']}  {c['model']} · {phys:<24} "
                     f"{c['aggregate']:>6}  {c['direction']} {c['threshold']:.4g} {un:<3} | "
                     f"predicted {c['prediction']:10.3f} {un:<3} [{status}]")
        L += ["", "MODEL VALIDATION (held-out RMSE)"]
        for m in MODELS:
            dg = self._diag.get(m, {})
            if dg.get("ok"):
                L.append(f"  {m} · {MODEL_PHYS[m][0]:<24} "
                         f"RMSE(test) {dg['rmse_test']:8.3f} {MODEL_PHYS[m][1]}")
        L += ["", "=" * 62]
        return "\n".join(L)

    def _write_pdf(self, path, text):
        with PdfPages(path) as pdf:
            fig = Figure(figsize=(8.27, 11.69)); fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111); ax.axis("off")
            ax.text(0.06, 0.97, "pSTAM Design Optimisation — Report", fontsize=15,
                    fontweight="bold", va="top", color=PRIMARY_D)
            ax.text(0.06, 0.93, text, fontsize=7.5, family="monospace", va="top", color="#13242C")
            pdf.savefig(fig)
            r = self._last_result; hist = list(r["cost_history"]); mode = r["report"]["mode"]
            vals = [-v for v in hist] if mode == "maximize" else hist
            f2 = Figure(figsize=(8.27, 5.5)); f2.patch.set_facecolor("white")
            a2 = f2.add_subplot(111); a2.plot(vals, color=PRIMARY, linewidth=2)
            a2.set_title("PSO convergence"); a2.set_xlabel("Iteration"); a2.set_ylabel("Objective")
            a2.grid(True, alpha=0.3); pdf.savefig(f2)

    def _toast(self, msg, color=PRIMARY):
        self.statusBar().setStyleSheet(
            f"color:{color}; background:{SURFACE}; font-size:11px; padding-left:8px;")
        self.statusBar().showMessage(msg, 4500)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "_progbar"):
            QTimer.singleShot(0, lambda: self._set_progress(
                self._progfill.width() / max(1, self._progbar.width())))

    # ════════════════════════════════════════════════════════════════
    #  QSS — light "Atelier" theme
    # ════════════════════════════════════════════════════════════════
    def _apply_qss(self):
        self.setStyleSheet(f"""
        QMainWindow {{ background:{CANVAS}; }}
        QWidget {{ color:{INK}; font-family:'{self.DISPLAY}','Segoe UI',sans-serif; }}
        QLabel, QCheckBox, QRadioButton {{ background:transparent; }}
        QScrollArea {{ background:transparent; }}
        #topbar {{ background:{SURFACE}; border-bottom:1px solid {LINE}; }}
        #pagebg, #scroll {{ background:{CANVAS}; border:none; }}
        QLabel#h1 {{ color:{INK}; font-size:22px; font-weight:800; letter-spacing:-0.3px; }}

        QPushButton#nav {{ padding:7px 11px; border:none; border-radius:8px;
            color:{BODY}; font-size:13px; font-weight:600; background:transparent; }}
        QPushButton#nav:hover {{ background:{SURFACE2}; color:{INK}; }}
        QPushButton#nav:checked {{ color:#FFFFFF; background:{PRIMARY};
            font-weight:700; }}

        #card {{ background:{SURFACE}; border:1px solid {LINE}; border-radius:16px; }}
        #block, #inset {{ background:{SURFACE2}; border:1px solid {LINE}; border-radius:12px; }}
        #media {{ background:{SURFACE}; border:1px solid {LINE_SOFT}; border-radius:12px; }}
        #authors {{ background:{PRIMARY_SOFT}; border:1px solid {PRIMARY}; border-radius:11px; }}
        #refs {{ background:#FBF0E3; border:1px solid {ACCENT}; border-radius:11px; }}
        #modelhead {{ background:{PRIMARY};
            border-top-left-radius:16px; border-top-right-radius:16px; }}
        #ph {{ border:1.4px dashed {LINE}; border-radius:13px; background:{SURFACE2}; }}
        #logoslot {{ border:1px solid {LINE_SOFT}; border-radius:13px; background:{SURFACE2}; }}
        QPushButton#navarrow {{ background:{SURFACE}; color:{PRIMARY};
            border:1px solid {LINE}; border-radius:17px; font-size:17px;
            font-weight:800; padding-bottom:3px; }}
        QPushButton#navarrow:hover {{ background:{PRIMARY}; color:#FFFFFF;
            border-color:{PRIMARY}; }}

        QPushButton#cta {{ background:{PRIMARY}; color:#FFFFFF; border:none;
            border-radius:10px; padding:11px 20px; font-weight:800; font-size:13px; }}
        QPushButton#cta:hover {{ background:{PRIMARY_D}; }}
        QPushButton#cta:disabled {{ background:#C2CDD3; color:#FFFFFF; }}
        QPushButton#ghost {{ background:{SURFACE}; color:{PRIMARY_D};
            border:1.4px solid {PRIMARY}; border-radius:9px; padding:8px 15px;
            font-weight:700; font-size:12px; }}
        QPushButton#ghost:hover {{ background:{PRIMARY_SOFT}; }}
        QPushButton#ghostMuted {{ background:{SURFACE}; color:{MUTED};
            border:1.4px solid {LINE}; border-radius:9px; padding:8px 15px;
            font-weight:600; font-size:12px; }}
        QPushButton#ghostMuted:hover {{ border-color:{MUTED}; color:{INK}; }}
        QPushButton#danger {{ background:{SURFACE}; color:{BAD};
            border:1.4px solid {BAD}; border-radius:9px; padding:8px 16px;
            font-weight:700; font-size:12px; }}
        QPushButton#danger:disabled {{ color:{FAINT}; border-color:{LINE}; }}
        QPushButton#rmbtn {{ background:transparent; color:{FAINT};
            border:1px solid {LINE}; border-radius:11px; font-size:12px;
            font-weight:700; padding:0; }}
        QPushButton#rmbtn:hover {{ background:#FCEDEC; color:{BAD}; border-color:{BAD}; }}

        QLineEdit {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE};
            border-radius:8px; padding:6px 9px; selection-background-color:{PRIMARY};
            selection-color:#FFFFFF; }}
        QLineEdit:focus {{ border:1.5px solid {PRIMARY}; }}
        QComboBox {{ background:#FFFFFF; color:{INK}; border:1px solid {LINE};
            border-radius:8px; padding:5px 9px; }}
        QComboBox:focus, QComboBox:hover {{ border:1.5px solid {PRIMARY}; }}
        QComboBox QAbstractItemView {{ background:#FFFFFF; color:{INK};
            selection-background-color:{PRIMARY_SOFT}; selection-color:{INK};
            border:1px solid {LINE}; outline:none; }}
        QComboBox::drop-down {{ border:none; width:18px; }}

        QCheckBox, QRadioButton {{ color:{BODY}; font-size:12px; spacing:7px; }}
        QCheckBox::indicator, QRadioButton::indicator {{ width:15px; height:15px; }}
        QCheckBox::indicator {{ border:1.4px solid {LINE}; border-radius:4px; background:#FFFFFF; }}
        QCheckBox::indicator:checked {{ background:{PRIMARY}; border-color:{PRIMARY}; }}
        QRadioButton::indicator {{ border:1.4px solid {LINE}; border-radius:8px; background:#FFFFFF; }}
        QRadioButton::indicator:checked {{ background:{PRIMARY}; border-color:{PRIMARY}; }}

        QSlider::groove:horizontal {{ height:6px; background:{LINE}; border-radius:3px; }}
        QSlider::sub-page:horizontal {{ background:{PRIMARY}; border-radius:3px; }}
        QSlider::add-page:horizontal {{ background:{LINE}; border-radius:3px; }}
        QSlider::handle:horizontal {{ width:18px; height:18px; margin:-7px 0;
            border-radius:9px; background:#FFFFFF; border:3px solid {PRIMARY}; }}
        QSlider::handle:horizontal:hover {{ border:3px solid {PRIMARY_D}; }}
        QSlider::handle:horizontal:pressed {{ background:{PRIMARY_SOFT}; }}

        #preview, #report {{ background:#FFFFFF; border:1px solid {LINE};
            border-radius:12px; padding:12px; }}
        #plot {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px;
            color:{FAINT}; font-size:11px; padding:8px; }}
        #progtrack {{ background:{LINE}; border-radius:4px; }}
        #progfill {{ background:{PRIMARY}; border-radius:4px; }}

        QScrollBar:vertical {{ background:transparent; width:11px; margin:2px; }}
        QScrollBar::handle:vertical {{ background:{LINE}; border-radius:5px; min-height:30px; }}
        QScrollBar::handle:vertical:hover {{ background:{PRIMARY}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}
        QScrollBar:horizontal {{ background:transparent; height:11px; margin:2px; }}
        QScrollBar::handle:horizontal {{ background:{LINE}; border-radius:5px; min-width:30px; }}
        QScrollBar::handle:horizontal:hover {{ background:{PRIMARY}; }}
        QStatusBar {{ background:{SURFACE}; color:{MUTED}; border-top:1px solid {LINE}; }}
        QToolTip {{ background:{INK}; color:#FFFFFF; border:none; padding:5px; }}
        """)



# Backward-compatible alias for scripts that import the previous class name.
Studio = Toolbox

def _install_error_reporting():
    """Report unexpected errors instead of letting Qt abort the process."""
    import faulthandler, traceback
    try:
        faulthandler.enable()               # Python stack on hard crashes
    except Exception:
        pass

    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        sys.__stderr__.write(text)
        app = QApplication.instance()
        if app is not None and QThread.currentThread() == app.thread():
            QMessageBox.critical(None, "Unexpected error", text[-2000:])
    sys.excepthook = hook


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ASSET_DIR, exist_ok=True)
    _install_error_reporting()
    app = QApplication(sys.argv)
    window = Toolbox()       # keep this reference for the life of the app
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
