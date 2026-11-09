"""Preview cursors from the vector set in the cursors/ folder.

The set is mocu-xcursor (CC0, https://github.com/sevmeyer/mocu-xcursor) plus a
rotate cursor drawn in its style. Files keep mocu's placeholder colours and
hotspot marker; here the white theme's colours are applied and the SVG is
rendered at the screen's scale, so the cursors stay crisp at any scaling.
Without the folder (or a file), the matching system cursor is used.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QCursor, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

GRID = 24          # mocu's design grid
SIZE = 32          # logical cursor size in pixels, as the system default
THEME = {          # mocu "White": placeholder -> colour
    '#fafbfc': '#fcfcfc',  # fill
    '#1a1b1c': '#222222',  # outline
    '#0a0b0c': '#000000',  # shadow
}

# name -> (file, system cursor used when the file is missing)
FILES = {
    'arrow': ('default', Qt.CursorShape.ArrowCursor),
    'cross': ('crosshair', Qt.CursorShape.CrossCursor),
    'open_hand': ('grab', Qt.CursorShape.OpenHandCursor),
    'closed_hand': ('grabbing', Qt.CursorShape.ClosedHandCursor),
    'size_hor': ('ew-resize', Qt.CursorShape.SizeHorCursor),
    'size_ver': ('ns-resize', Qt.CursorShape.SizeVerCursor),
    'size_fdiag': ('nwse-resize', Qt.CursorShape.SizeFDiagCursor),
    'size_bdiag': ('nesw-resize', Qt.CursorShape.SizeBDiagCursor),
}

_HOT = re.compile(r'<circle[^>]*id="hot"[^>]*/>')
_cache: dict[tuple, QCursor] = {}


def folder() -> Path:
    base = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
    return base / 'cursors'


def _load(name: str) -> tuple[str, float, float] | None:
    """Themed SVG text and hotspot (grid units), or None if unavailable."""
    try:
        text = (folder() / f'{name}.svg').read_text(encoding='utf-8')
    except OSError:
        return None
    marker = _HOT.search(text)
    if marker:
        cx = float(re.search(r'cx="([\d.]+)"', marker.group()).group(1))
        cy = float(re.search(r'cy="([\d.]+)"', marker.group()).group(1))
        text = text.replace(marker.group(), '')
    else:
        cx = cy = GRID / 2
    for placeholder, colour in THEME.items():
        text = text.replace(placeholder, colour)
    return text, cx, cy


def _scale() -> float:
    screen = QApplication.primaryScreen() if QApplication.instance() else None
    return max(1.0, screen.devicePixelRatio()) if screen else 1.0


def _cursor(text: str, cx: float, cy: float, scale: float) -> QCursor:
    pixmap = QPixmap(round(SIZE * scale), round(SIZE * scale))
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(text.encode('utf-8'))).render(painter, QRectF(0, 0, SIZE, SIZE))
    painter.end()
    ratio = SIZE / GRID
    return QCursor(pixmap, round(cx * ratio), round(cy * ratio))


def cursor(name: str) -> QCursor:
    scale = _scale()
    key = (name, scale)
    if key not in _cache:
        file, fallback = FILES[name]
        loaded = _load(file)
        _cache[key] = _cursor(*loaded, scale) if loaded else QCursor(fallback)
    return _cache[key]


def rotate_cursor(direction: float) -> QCursor:
    """Curved arrow bulging toward ``direction`` (screen degrees, 0 = right,
    90 = down): away from the crop box, wrapping a corner outside it."""
    scale = _scale()
    key = ('rotate', round(direction, 3), scale)
    if key not in _cache:
        loaded = _load('rotate')
        if loaded is None:
            _cache[key] = QCursor(Qt.CursorShape.SizeAllCursor)
        else:
            text, cx, cy = loaded
            # The file bulges upward (-90°); turn the shape, not its shadow.
            text = text.replace(
                '<path id="c"',
                f'<path id="c" transform="rotate({direction + 90:.3f} {cx} {cy})"', 1)
            _cache[key] = _cursor(text, cx, cy, scale)
    return _cache[key]
