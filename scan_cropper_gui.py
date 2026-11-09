#!/usr/bin/env python3
"""PySide6 GUI for the top-aligned batch scan cropper."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import statistics
import threading
import time
import traceback
import weakref
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from PIL import Image
from PySide6.QtCore import (
    QEvent,
    QByteArray,
    QObject,
    QPoint,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QSignalBlocker,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QCursor,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from scan_cropper import (
    IMAGE_SUFFIXES,
    ChannelShiftMeasurement,
    Detection,
    centered_crop_box,
    choose_common_size,
    detect_page,
    detect_rgb_channel_shifts,
    inner_side_for,
    outer_anchored_crop_box,
    render_transformed,
    rotated_full_box,
    save_crop,
    transformed_crop,
    transformed_crop_with_channel_shifts,
)


from scan_cropper_auto import (
    solve_page,
    reference_angle,
    PageAnalysis,
    CropPlan,
    Cancelled,
    analyze_batch,
    analyze_page,
    clipped_side,
    after_single_page,
    solve_batch,
    natural_key,
    rotated,
    paper_reference,
    inner_side as auto_inner_side,
    solve_bound_batch,
)

import scan_cropper_cursors as cursors
from scan_cropper_perspective import (
    recommend as recommend_perspective,
    state_matrix,
    validate_state,
    manual_state,
    project,
    rotation_matrix,
    warp_analysis,
    combined_matrix,
    transformed_bounds,
    homography,
)


def perspective_cache_key(path, size, angle, center, matrix=None, channel_shifts=None):
    key = (path, size, round(angle, 4), center)
    if matrix is not None:
        key = key + (tuple(np.asarray(matrix).ravel()),)
    if channel_shifts:
        key = key + (
            "rgb",
            tuple(
                tuple(channel_shifts.get(c, (0.0, 0.0)))
                for c in ("red", "green", "blue")
            ),
        )
    return key


INNER_MODES = [
    ("指定宽度：内侧在左", "bound-left"),
    ("指定宽度：内侧在右", "bound-right"),
    ("指定宽度：交替，第一张内侧在左", "bound-alternate-left"),
    ("指定宽度：交替，第一张内侧在右", "bound-alternate-right"),
    ("固定：内侧在左", "left"),
    ("固定：内侧在右", "right"),
    ("交替：第一张内侧在左", "alternate-left"),
    ("交替：第一张内侧在右", "alternate-right"),
    ("启发式自动判断", "auto"),
    ("不区分内外（左右都按外侧）", "none"),
]

PREVIEW_BACKGROUNDS = [
    ("绿色", "#4F7774"),
    ("蓝色", "#607A96"),
    ("黄色", "#A8895F"),
    ("紫色", "#806D88"),
    ("红色", "#986B62"),
    ("黑色", "#000000"),
    ("白色", "#FFFFFF"),
    ("灰色", "#808080"),
]

BATCH_WORKERS = 8
COLOR_EXPORT_WORKERS = 3
RGB_DETECT_WORKERS = 3
DETECTION_BRIGHT_FRACTION = 0.16
ROTATION_CACHE_MARGIN = 256
COLOR_PROFILE_FILENAME = "scan_cropper_color_profiles.json"
BATCH_CONFIG_VERSION = 11
SPIN_INT_MIN = -(2**31)
SPIN_INT_MAX = 2**31 - 1


class CompactAngleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:
        return f"{value:.4f}".rstrip("0").rstrip(".")


# Offsets depend on the scanner and its settings, so nothing is assumed:
# a fresh install has a zero profile and colour correction switched off.
DEFAULT_COLOR_PROFILE_NAME = "无偏移"
DEFAULT_COLOR_PROFILES = {
    DEFAULT_COLOR_PROFILE_NAME: {
        "red_x": 0.0,
        "red_y": 0.0,
        "green_x": 0.0,
        "green_y": 0.0,
        "blue_x": 0.0,
        "blue_y": 0.0,
    },
}
PROFILE_CHANNELS = ("red", "green", "blue")


def normalized_color_profile(values: object) -> dict[str, float] | None:
    if not isinstance(values, dict):
        return None
    result: dict[str, float] = {}
    for channel in PROFILE_CHANNELS:
        for axis in ("x", "y"):
            key = f"{channel}_{axis}"
            try:
                value = float(values.get(key, 0.0))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            result[key] = value
    return result


def color_profile_path() -> Path:
    base = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    return base / COLOR_PROFILE_FILENAME


def application_settings_path() -> Path:
    return color_profile_path().with_name("scan_cropper_settings.json")


def load_color_profiles() -> tuple[dict[str, dict[str, float]], str, bool]:
    profiles = {name: values.copy() for name, values in DEFAULT_COLOR_PROFILES.items()}
    selected = DEFAULT_COLOR_PROFILE_NAME
    enabled = False
    path = color_profile_path()
    if not path.is_file():
        return profiles, selected, enabled
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("颜色 Profile 配置的顶层必须是对象")
        raw_profiles = raw.get("profiles", {})
        if not isinstance(raw_profiles, dict):
            raise ValueError("颜色 Profile 配置中的 profiles 必须是对象")
        loaded: dict[str, dict[str, float]] = {}
        for name, values in raw_profiles.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(values, dict)
            ):
                continue
            profile = normalized_color_profile(values)
            if profile is not None:
                loaded[name.strip()] = profile
        if loaded:
            profiles = loaded
        requested = raw.get("selected_profile")
        if isinstance(requested, str) and requested in profiles:
            selected = requested
        else:
            selected = next(iter(profiles))
        enabled = bool(raw.get("enabled", True))
    except (OSError, ValueError, TypeError):
        # A damaged optional settings file must not prevent cropping.
        pass
    return profiles, selected, enabled


def write_color_profiles(
    profiles: dict[str, dict[str, float]],
    selected: str,
    enabled: bool,
) -> None:
    path = color_profile_path()
    payload = {
        "version": 2,
        "selected_profile": selected,
        "enabled": enabled,
        "profiles": profiles,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ColorProfileDialog(QDialog):
    def __init__(
        self,
        profiles: dict[str, dict[str, float]],
        selected: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑颜色偏移 Profiles")
        self.profiles = {name: values.copy() for name, values in profiles.items()}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(list(self.profiles))
        self.profile_combo.setCurrentText(selected)
        form.addRow("Profile", self.profile_combo)

        self.shift_spins: dict[str, QDoubleSpinBox] = {}
        channel_labels = {"red": "红", "green": "绿", "blue": "蓝"}
        for channel in PROFILE_CHANNELS:
            row = QHBoxLayout()
            for axis, direction in (("x", "X"), ("y", "Y")):
                row.addWidget(QLabel(direction))
                spin = QDoubleSpinBox()
                spin.setRange(-sys.float_info.max, sys.float_info.max)
                spin.setDecimals(3)
                spin.setSingleStep(0.05)
                spin.setSuffix(" px")
                spin.setKeyboardTracking(False)
                self.shift_spins[f"{channel}_{axis}"] = spin
                row.addWidget(spin)
            form.addRow(f"{channel_labels[channel]}通道", row)
        layout.addLayout(form)

        note = QLabel(
            "偏移以未旋转的原图坐标为准：X 右为正，Y 下为正。"
            "绿通道通常作为检测基准，可保持 0。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        profile_buttons = QHBoxLayout()
        save_button = QPushButton("保存当前")
        save_as_button = QPushButton("另存为…")
        delete_button = QPushButton("删除")
        profile_buttons.addWidget(save_button)
        profile_buttons.addWidget(save_as_button)
        profile_buttons.addWidget(delete_button)
        layout.addLayout(profile_buttons)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.profile_combo.currentTextChanged.connect(self.load_profile)
        save_button.clicked.connect(self.save_current)
        save_as_button.clicked.connect(self.save_as)
        delete_button.clicked.connect(self.delete_current)
        self.load_profile(self.profile_combo.currentText())

    def spin_values(self) -> dict[str, float]:
        return {key: spin.value() for key, spin in self.shift_spins.items()}

    def load_profile(self, name: str) -> None:
        values = self.profiles.get(name)
        if values is None:
            return
        for key, spin in self.shift_spins.items():
            with QSignalBlocker(spin):
                spin.setValue(values.get(key, 0.0))

    def save_current(self) -> None:
        name = self.profile_combo.currentText()
        if name:
            self.profiles[name] = self.spin_values()

    def save_as(self) -> None:
        name, accepted = QInputDialog.getText(
            self,
            "另存颜色 Profile",
            "Profile 名称：",
        )
        name = name.strip()
        if not accepted or not name:
            return
        if name in self.profiles:
            answer = QMessageBox.question(
                self,
                "覆盖 Profile？",
                f"颜色 Profile“{name}”已经存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.profiles[name] = self.spin_values()
        with QSignalBlocker(self.profile_combo):
            self.profile_combo.clear()
            self.profile_combo.addItems(list(self.profiles))
            self.profile_combo.setCurrentText(name)

    def delete_current(self) -> None:
        if len(self.profiles) <= 1:
            QMessageBox.information(
                self,
                "不能删除",
                "至少需要保留一个颜色 Profile。",
            )
            return
        name = self.profile_combo.currentText()
        answer = QMessageBox.question(
            self,
            "删除颜色 Profile？",
            f"确定删除“{name}”吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        del self.profiles[name]
        with QSignalBlocker(self.profile_combo):
            self.profile_combo.clear()
            self.profile_combo.addItems(list(self.profiles))
            self.profile_combo.setCurrentIndex(0)
        self.load_profile(self.profile_combo.currentText())

    def accept(self) -> None:
        self.save_current()
        super().accept()

    @property
    def selected_profile(self) -> str:
        return self.profile_combo.currentText()


def repair_width_outliers(detections: list[Detection]) -> list[Detection]:
    """Infer a missed page edge from the stable page width of the batch."""
    if len(detections) < 3:
        return detections

    initial_width = statistics.median(item.width for item in detections)
    ordinary = [
        item
        for item in detections
        if initial_width * 0.85 <= item.width <= initial_width * 1.15
    ]
    if len(ordinary) < 2:
        return detections

    common_width = statistics.median(item.width for item in ordinary)
    common_center = statistics.median(item.center_x for item in ordinary)
    repaired: list[Detection] = []
    for item in detections:
        if common_width * 0.85 <= item.width <= common_width * 1.15:
            repaired.append(item)
            continue

        right_from_left = item.left + common_width
        left_from_right = item.right - common_width
        left_anchor_overflow = max(0.0, right_from_left - item.source_width)
        right_anchor_overflow = max(0.0, -left_from_right)
        left_anchor_score = left_anchor_overflow * 10 + abs(
            (item.left + right_from_left) / 2 - common_center
        )
        right_anchor_score = right_anchor_overflow * 10 + abs(
            (left_from_right + item.right) / 2 - common_center
        )
        if left_anchor_score <= right_anchor_score:
            repaired.append(replace(item, right=right_from_left))
        else:
            repaired.append(replace(item, left=left_from_right))
    return repaired


def inferred_detection(path: Path, references: list[Detection]) -> Detection:
    """Create an editable initial page box when content-based detection fails."""
    with Image.open(path) as source:
        source_width, source_height = source.size

    if references:
        left_ratio = statistics.median(
            item.left / item.source_width for item in references
        )
        right_ratio = statistics.median(
            item.right / item.source_width for item in references
        )
        bottom_ratio = statistics.median(
            item.bottom / item.source_height for item in references
        )
        threshold = statistics.median(item.threshold for item in references)
    else:
        # Scanner-bed fallback for an all-dark batch.  The result is only an
        # editable starting point; the GUI marks it for manual verification.
        left_ratio = 0.075
        right_ratio = 0.925
        bottom_ratio = 0.87
        threshold = 0.0

    return Detection(
        path=path,
        source_width=source_width,
        source_height=source_height,
        left=left_ratio * source_width,
        right=right_ratio * source_width,
        bottom=bottom_ratio * source_height,
        threshold=threshold,
        left_roughness=0.0,
        right_roughness=0.0,
    )


class FolderLineEdit(QLineEdit):
    """A path field that accepts a folder (or a file, meaning its folder)
    dragged in from the file manager."""

    folder_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText("选择或拖入文件夹")

    @staticmethod
    def dropped_folder(mime) -> Path | None:
        for url in mime.urls() if mime.hasUrls() else []:
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_dir():
                    return path
                if path.is_file():
                    return path.parent
        return None

    # Only folders (or files, meaning their folder) are accepted; dragged
    # text or web links are ignored rather than pasted into the path.
    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.dropped_folder(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.dropped_folder(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        folder = self.dropped_folder(event.mimeData())
        if folder is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.folder_dropped.emit(str(folder))


ROTATE_CURSOR_STEP = 15


class PreviewView(QGraphicsView):
    crop_dragged = Signal(int, int)
    level_line_picked = Signal(QPointF, QPointF)
    edge_dragged = Signal(str, int, int)
    corner_dragged = Signal(int, float, float)
    corner_started = Signal(int)
    corner_moved = Signal(int, float, float)
    # Ctrl+drag outside the crop box: degrees turned so far (clockwise).
    rotate_started = Signal()
    rotate_moved = Signal(float)
    rotate_finished = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.preview_levels: list[tuple[float, QPixmap]] = []
        self.preview_source_ref: weakref.ReferenceType[Image.Image] | None = None
        self.preview_source_size = (0, 0)
        self.active_preview_level = -1
        self.preview_offset = QPointF()
        self.provisional_rotation: tuple[float, object] = (0.0, None)
        self.is_draft = False
        # Two-point leveling: Ctrl+click two points of a line meant to be
        # horizontal or vertical.
        self.level_mode = False
        self.level_points: list[QPointF] = []
        self.level_item = QGraphicsPathItem()
        level_pen = QPen(QColor("#FF4FA3"), 2)
        level_pen.setCosmetic(True)
        self.level_item.setPen(level_pen)
        self.level_item.setZValue(40)
        self.level_item.setVisible(False)
        # Ctrl+drag outside the crop box turns the page about this scene point.
        self.rotate_enabled = False
        self.rotate_pivot: QPointF | None = None
        self.rotate_drag: tuple[QPointF, float, float] | None = None
        self.panning = False
        self.pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation
        )
        self.scene().addItem(self.pixmap_item)
        self.clip_item = QGraphicsPathItem()
        self.clip_item.setPen(QPen(Qt.PenStyle.NoPen))
        self.scene().addItem(self.clip_item)
        self.pixmap_item.setParentItem(self.clip_item)
        self.boundary_item = QGraphicsRectItem()
        boundary_pen = QPen(QColor("#FFD166"), 2)
        boundary_pen.setCosmetic(True)
        self.boundary_item.setPen(boundary_pen)
        self.boundary_item.setZValue(10)
        self.boundary_item.setVisible(False)
        self.scene().addItem(self.boundary_item)
        self.crop_outline_item = QGraphicsPolygonItem()
        crop_outline_pen = QPen(QColor("#FFD166"), 2)
        crop_outline_pen.setCosmetic(True)
        self.crop_outline_item.setPen(crop_outline_pen)
        self.crop_outline_item.setZValue(20)
        self.crop_outline_item.setVisible(False)
        self.scene().addItem(self.crop_outline_item)
        self.detected_edge_items = []
        for _ in range(4):
            edge_item = QGraphicsPathItem()
            edge_pen = QPen(QColor("#37DBFF"), 1.5)
            edge_pen.setCosmetic(True)
            edge_item.setPen(edge_pen)
            edge_item.setZValue(15)
            edge_item.setVisible(False)
            self.scene().addItem(edge_item)
            self.detected_edge_items.append(edge_item)
        self.full_preview_mode = False
        self.crop_drag_start: QPointF | None = None
        self.crop_drag_offset = (0, 0)
        self.edit_polygon = QPolygonF()
        self.edit_enabled = False
        self.handle_drag = None
        self.edit_gesture = 0
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.message_item = QGraphicsSimpleTextItem("裁切预览")
        self.message_item.setBrush(QColor("#FFF4D6"))
        self.scene().addItem(self.message_item)
        self.scene().addItem(self.level_item)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Setting the drag mode installs the system hand; use the built-in set.
        self.viewport().setCursor(cursors.cursor("arrow"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def set_background_color(self, color: str) -> None:
        self.setBackgroundBrush(QColor(color))

    def set_detected_edges(self, curves):
        for i, item in enumerate(self.detected_edge_items):
            item.setVisible(i < len(curves))
            if i < len(curves):
                curve = QPainterPath()
                for j, (x, y) in enumerate(curves[i]):
                    if j == 0:
                        curve.moveTo(float(x), float(y))
                    else:
                        curve.lineTo(float(x), float(y))
                item.setPath(curve)

    def set_preview(
        self,
        image,
        reset_view: bool = False,
        offset: QPointF | None = None,
        logical_size=None,
        rotation: float = 0.0,
        rotation_center: tuple[float, float] | None = None,
    ) -> None:
        """Show a PIL image, a worker-built PreviewRaster or a PreviewDraft.

        ``rotation`` (degrees, clockwise, about ``rotation_center`` in scene
        coordinates) is a provisional display rotation used only until the
        full-quality render for the requested angle is ready.
        """
        self.set_image_clip()
        self.message_item.setVisible(False)
        self.set_detected_edges([])
        current_source = (
            self.preview_source_ref() if self.preview_source_ref is not None else None
        )
        if current_source is not image:
            if isinstance(image, PreviewRaster):
                # QImages were converted to a pixmap-native format by the
                # worker, so this is a cheap handover on the UI thread.
                self.preview_levels = [
                    (scale, QPixmap.fromImage(level)) for scale, level in image.levels
                ]
                size = image.size
            elif isinstance(image, PreviewDraft):
                pixmap = QPixmap.fromImage(image.image)
                self.preview_levels = [(pixmap.width() / image.source_size[0], pixmap)]
                size = image.source_size
            else:
                self.preview_levels = self._build_preview_levels(image)
                size = image.size
            self.preview_source_ref = weakref.ref(image)
            self.preview_source_size = logical_size or size
            self.active_preview_level = -1
        self.is_draft = isinstance(image, PreviewDraft)
        self.preview_offset = offset or QPointF()
        self.provisional_rotation = (float(rotation), rotation_center)
        self.pixmap_item.setOffset(QPointF())
        self.pixmap_item.setPos(QPointF())
        if not self._apply_preview_level(0):
            self._apply_item_transform()
        self.boundary_item.setRect(self.pixmap_item.sceneBoundingRect())
        self.boundary_item.setVisible(False)
        self.crop_outline_item.setVisible(False)
        self._update_sampling_mode()
        if reset_view:
            self.fit_preview()
        else:
            self._update_pan_bounds()

    @staticmethod
    def _pil_to_pixmap(image: Image.Image) -> QPixmap:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        raw = rgba.tobytes()
        qimage = QImage(
            raw,
            rgba.width,
            rgba.height,
            rgba.width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(qimage)

    @classmethod
    def _build_preview_levels(cls, image: Image.Image) -> list[tuple[float, QPixmap]]:
        levels = [(1.0, cls._pil_to_pixmap(image))]
        reduced = image
        for scale in (0.5, 0.25, 0.125):
            size = (
                max(1, round(reduced.width * 0.5)),
                max(1, round(reduced.height * 0.5)),
            )
            if size[0] < 128 or size[1] < 128:
                break
            reduced = reduced.resize(size, Image.Resampling.LANCZOS)
            levels.append((scale, cls._pil_to_pixmap(reduced)))
        return levels

    def _apply_preview_level(self, index: int) -> bool:
        if not self.preview_levels or index == self.active_preview_level:
            return False
        _, pixmap = self.preview_levels[index]
        self.pixmap_item.setPixmap(pixmap)
        self.active_preview_level = index
        self._apply_item_transform()
        return True

    def _apply_item_transform(self) -> None:
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return
        source_width, source_height = self.preview_source_size
        # Points map: level pixels -> source scale -> scene offset -> rotation.
        transform = QTransform.fromScale(
            source_width / pixmap.width(), source_height / pixmap.height()
        ) * QTransform.fromTranslate(self.preview_offset.x(), self.preview_offset.y())
        angle, center = self.provisional_rotation
        if abs(angle) > 1e-9 and center is not None:
            rotation = QTransform()
            rotation.translate(center[0], center[1])
            rotation.rotate(angle)
            rotation.translate(-center[0], -center[1])
            transform = transform * rotation
        self.pixmap_item.setTransform(transform)

    def set_provisional_rotation(self, angle: float, center) -> None:
        """Rotate what is shown now; the exact render replaces it later."""
        self.provisional_rotation = (float(angle), center)
        self._apply_item_transform()
        self.boundary_item.setRect(self.pixmap_item.sceneBoundingRect())
        self._update_pan_bounds()
        self.viewport().update()

    @property
    def shows_exact_pixels(self) -> bool:
        return not self.is_draft and abs(self.provisional_rotation[0]) <= 1e-9

    def _select_preview_level(self) -> None:
        if not self.preview_levels:
            return
        view_scale = abs(self.transform().m11()) * self.devicePixelRatioF()
        candidates = [
            (scale, index)
            for index, (scale, _) in enumerate(self.preview_levels)
            if scale >= view_scale
        ]
        index = min(candidates)[1] if candidates else 0
        if index != self.active_preview_level:
            self._apply_preview_level(index)
            self.boundary_item.setRect(self.pixmap_item.sceneBoundingRect())

    def show_message(self, text: str) -> None:
        self.preview_levels = []
        self.preview_source_ref = None
        self.active_preview_level = -1
        self.pixmap_item.setPixmap(QPixmap())
        self.boundary_item.setVisible(False)
        self.crop_outline_item.setVisible(False)
        self.message_item.setText(text)
        self.message_item.setVisible(True)
        self.scene().setSceneRect(self.message_item.boundingRect())

    def set_crop_outline(self, polygon: QPolygonF, visible: bool = True) -> None:
        self.boundary_item.setVisible(False)
        self.crop_outline_item.setPolygon(polygon)
        self.crop_outline_item.setVisible(visible)

    def set_image_clip(self, polygon=None):
        self.clip_item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, polygon is not None
        )
        if polygon is not None:
            path = QPainterPath()
            path.addPolygon(polygon)
            path.closeSubpath()
            self.clip_item.setPath(path)

    def capture_view_state(self) -> tuple[QTransform, QPointF]:
        center = self.mapToScene(self.viewport().rect().center())
        return self.transform(), center

    def restore_view_state(self, state: tuple[QTransform, QPointF]) -> None:
        transform, center = state
        self.setTransform(transform)
        self._update_pan_bounds()
        self.centerOn(center)
        self._update_sampling_mode()

    def restore_aligned_view(
        self,
        transform: QTransform,
        scene_anchor: QPointF,
        viewport_anchor: QPoint,
    ) -> None:
        """Restore zoom while pinning one scene point to an exact screen pixel."""
        self.setTransform(transform)
        self._update_pan_bounds()
        self.centerOn(scene_anchor)
        for _ in range(2):
            actual = self.mapFromScene(scene_anchor)
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() + actual.x() - viewport_anchor.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + actual.y() - viewport_anchor.y()
            )
        self._update_sampling_mode()

    def fit_preview(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(
            self.pixmap_item,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._select_preview_level()
        self._update_pan_bounds()
        self._update_sampling_mode()

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.pixmap_item.pixmap().isNull():
            super().wheelEvent(event)
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current_scale = abs(self.transform().m11())
        next_scale = current_scale * factor
        if 0.01 <= next_scale <= 64:
            self.scale(factor, factor)
            self._select_preview_level()
            self._update_pan_bounds()
            self._update_sampling_mode()
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._update_pan_bounds()

    def _update_pan_bounds(self) -> None:
        """Allow every image corner to be panned as far as the view center."""
        if self.pixmap_item.pixmap().isNull():
            return
        scale_x = max(abs(self.transform().m11()), 1e-9)
        scale_y = max(abs(self.transform().m22()), 1e-9)
        padding_x = self.viewport().width() / scale_x / 2
        padding_y = self.viewport().height() / scale_y / 2
        bounds = self.pixmap_item.sceneBoundingRect()
        self.scene().setSceneRect(
            bounds.adjusted(-padding_x, -padding_y, padding_x, padding_y)
        )

    def _update_sampling_mode(self) -> None:
        scale = abs(self.transform().m11())
        if scale >= 3.0:
            self.pixmap_item.setTransformationMode(
                Qt.TransformationMode.FastTransformation
            )
            self.setRenderHints(QPainter.RenderHint.Antialiasing)
        else:
            self.pixmap_item.setTransformationMode(
                Qt.TransformationMode.SmoothTransformation
            )
            self.setRenderHints(
                QPainter.RenderHint.Antialiasing
                | QPainter.RenderHint.SmoothPixmapTransform
            )
        self.viewport().update()

    def drawForeground(self, painter, rect) -> None:  # type: ignore[no-untyped-def]
        if (
            self.pixmap_item.pixmap().isNull()
            or abs(self.transform().m11()) < 6.0
            or not self.shows_exact_pixels
        ):
            return

        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        visible = visible.intersected(self.pixmap_item.sceneBoundingRect())
        if visible.isEmpty():
            return

        painter.save()
        painter.setWorldMatrixEnabled(False)
        if (
            self.clip_item.flags()
            & QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape
        ):
            painter.setClipPath(self.viewportTransform().map(self.clip_item.path()))
        pen = QPen(QColor(0, 0, 0, 40))
        pen.setCosmetic(True)
        painter.setPen(pen)

        x0 = max(0, int(visible.left()))
        x1 = min(self.pixmap_item.pixmap().width(), int(visible.right()) + 1)
        y0 = max(0, int(visible.top()))
        y1 = min(self.pixmap_item.pixmap().height(), int(visible.bottom()) + 1)
        for x in range(x0, x1 + 1):
            painter.drawLine(
                self.mapFromScene(x, y0),
                self.mapFromScene(x, y1),
            )
        for y in range(y0, y1 + 1):
            painter.drawLine(
                self.mapFromScene(x0, y),
                self.mapFromScene(x1, y),
            )
        painter.restore()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.fit_preview()
        event.accept()

    def edit_hit(self, pos):
        if not self.edit_enabled or len(self.edit_polygon) != 4:
            return None
        points = [self.mapFromScene(p) for p in self.edit_polygon]
        for i, p in enumerate(points):
            if math.hypot(pos.x() - p.x(), pos.y() - p.y()) <= 11:
                return ("corner", i)
        for i, (p, q) in enumerate(zip(points, points[1:] + points[:1])):
            vx, vy = q.x() - p.x(), q.y() - p.y()
            t = max(
                0,
                min(
                    1,
                    ((pos.x() - p.x()) * vx + (pos.y() - p.y()) * vy)
                    / max(1, vx * vx + vy * vy),
                ),
            )
            if math.hypot(pos.x() - p.x() - t * vx, pos.y() - p.y() - t * vy) <= 8:
                return ("edge", ("top", "right", "bottom", "left")[i])
        return None

    def set_level_mode(self, enabled: bool) -> None:
        self.level_mode = enabled
        self.level_points = []
        self.level_item.setPath(QPainterPath())
        self.level_item.setVisible(enabled)
        self.update_edit_cursor()

    def _draw_level_overlay(self, cursor_scene: QPointF | None = None) -> None:
        path = QPainterPath()
        radius = 5 / max(abs(self.transform().m11()), 1e-9)
        for point in self.level_points:
            path.addEllipse(point, radius, radius)
        if self.level_points and cursor_scene is not None:
            path.moveTo(self.level_points[0])
            path.lineTo(cursor_scene)
        self.level_item.setPath(path)

    def update_edit_cursor(self, pos=None, control=None):
        if self.level_mode:
            self.viewport().setCursor(cursors.cursor("cross"))
            return
        pos = pos if pos is not None else self.viewport().mapFromGlobal(QCursor.pos())
        control = (
            bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)
            if control is None
            else control
        )
        hit = self.edit_hit(pos) if control else None
        if self.panning:
            # Plain left-drag pans the view; keep the grab cursor throughout.
            self.viewport().setCursor(cursors.cursor("closed_hand"))
        elif self.rotate_drag is not None:
            self.viewport().setCursor(self.rotate_cursor_at(pos))
        elif self.crop_drag_start is not None:
            self.viewport().setCursor(cursors.cursor("closed_hand"))
        elif hit:
            if hit[0] == "corner":
                shape = (
                    cursors.cursor("size_fdiag")
                    if hit[1] in (0, 2)
                    else cursors.cursor("size_bdiag")
                )
            else:
                shape = (
                    cursors.cursor("size_ver")
                    if hit[1] in ("top", "bottom")
                    else cursors.cursor("size_hor")
                )
            self.viewport().setCursor(shape)
        elif control and self.crop_drag_hit(pos):
            self.viewport().setCursor(cursors.cursor("open_hand"))
        elif control and self.rotate_hit(pos):
            self.viewport().setCursor(self.rotate_cursor_at(pos))
        elif self.handle_drag is None:
            self.viewport().setCursor(cursors.cursor("arrow"))

    def rotate_hit(self, pos):
        """Outside the crop box, away from its edges and corners."""
        return (
            self.rotate_enabled
            and self.edit_enabled
            and self.full_preview_mode
            and self.rotate_pivot is not None
            and not self.pixmap_item.pixmap().isNull()
            and len(self.edit_polygon) == 4
            and self.edit_hit(pos) is None
            and not self.edit_polygon.containsPoint(
                self.mapToScene(pos), Qt.FillRule.OddEvenFill
            )
        )

    def rotate_cursor_at(self, pos) -> QCursor:
        """Curved arrow facing away from the crop box, per 15° step."""
        box = self.edit_polygon.boundingRect() if len(self.edit_polygon) else QRectF()
        center = box.center() if not box.isEmpty() else (self.rotate_pivot or self.mapToScene(pos))
        # View coordinates, so the arrow matches what is seen on screen.
        here, middle = QPointF(pos), QPointF(self.mapFromScene(center))
        angle = math.degrees(math.atan2(here.y() - middle.y(), here.x() - middle.x()))
        step = round(angle / ROTATE_CURSOR_STEP) % (360 // ROTATE_CURSOR_STEP)
        return cursors.rotate_cursor(step * ROTATE_CURSOR_STEP)

    def _pointer_angle(self, pos) -> float:
        point = self.mapToScene(pos)
        return math.degrees(
            math.atan2(point.y() - self.rotate_pivot.y(), point.x() - self.rotate_pivot.x())
        )

    def _rotate_delta(self, pos) -> float:
        # Screen y points down, so a growing atan2 angle is clockwise, which
        # matches the page rotation convention.
        _, start, _ = self.rotate_drag
        return (self._pointer_angle(pos) - start + 180.0) % 360.0 - 180.0

    def crop_drag_hit(self, pos):
        return (
            self.edit_enabled
            and self.full_preview_mode
            and len(self.edit_polygon) == 4
            and self.edit_polygon.containsPoint(
                self.mapToScene(pos), Qt.FillRule.OddEvenFill
            )
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if (
            self.level_mode
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            point = self.mapToScene(event.pos())
            self.level_points.append(point)
            self._draw_level_overlay(point)
            if len(self.level_points) == 2:
                first, second = self.level_points
                self.level_points = []
                self.level_line_picked.emit(first, second)
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            hit = self.edit_hit(event.pos())
            if hit:
                self.edit_gesture += 1
                self.handle_drag = (
                    hit,
                    event.position(),
                    abs(self.transform().m11()),
                    (0, 0),
                )
                if hit[0] == "corner":
                    self.corner_started.emit(hit[1])
                event.accept()
                return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.crop_drag_hit(event.pos())
        ):
            self.crop_drag_start = self.mapToScene(event.pos())
            self.edit_gesture += 1
            self.viewport().setCursor(cursors.cursor("closed_hand"))
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.rotate_hit(event.pos())
        ):
            self.rotate_drag = (self.mapToScene(event.pos()), self._pointer_angle(event.pos()), 0.0)
            self.edit_gesture += 1
            self.viewport().setCursor(self.rotate_cursor_at(event.pos()))
            self.rotate_started.emit()
            event.accept()
            return
        super().mousePressEvent(event)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
            and not self.pixmap_item.pixmap().isNull()
        ):
            self.panning = True
            self.viewport().setCursor(cursors.cursor("closed_hand"))

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.handle_drag is not None:
            hit, start, scale, previous = self.handle_drag
            dx = (event.position().x() - start.x()) / scale
            dy = (event.position().y() - start.y()) / scale
            if hit[0] == "edge":
                now = (round(dx), round(dy))
                self.edge_dragged.emit(
                    hit[1], now[0] - previous[0], now[1] - previous[1]
                )
                self.handle_drag = (hit, start, scale, now)
            else:
                self.corner_moved.emit(hit[1], dx, dy)
            event.accept()
            return
        if self.rotate_drag is not None:
            delta = self._rotate_delta(event.pos())
            press, start, _ = self.rotate_drag
            self.rotate_drag = (press, start, delta)
            self.viewport().setCursor(self.rotate_cursor_at(event.pos()))
            self.rotate_moved.emit(delta)
            event.accept()
            return
        if self.level_mode:
            if self.level_points:
                self._draw_level_overlay(self.mapToScene(event.pos()))
            self.update_edit_cursor()
            super().mouseMoveEvent(event)
            return
        self.update_edit_cursor(
            event.pos(), bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        )
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and self.edit_hit(event.pos()) is not None
        ):
            event.accept()
            return
        if self.crop_drag_start is not None:
            scene_pos = self.mapToScene(event.pos())
            dx = round(scene_pos.x() - self.crop_drag_start.x())
            dy = round(self.crop_drag_start.y() - scene_pos.y())
            if (dx, dy) != self.crop_drag_offset:
                self.crop_dragged.emit(
                    dx - self.crop_drag_offset[0], dy - self.crop_drag_offset[1]
                )
                self.crop_drag_offset = (dx, dy)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.panning and event.button() == Qt.MouseButton.LeftButton:
            self.panning = False
            super().mouseReleaseEvent(event)
            self.update_edit_cursor(
                event.pos(), bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            )
            return
        if self.rotate_drag is not None and event.button() == Qt.MouseButton.LeftButton:
            delta = self._rotate_delta(event.pos())
            self.rotate_drag = None
            self.rotate_finished.emit(delta)
            self.update_edit_cursor(
                event.pos(), bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            )
            event.accept()
            return
        if self.handle_drag is not None and event.button() == Qt.MouseButton.LeftButton:
            hit, start, scale, previous = self.handle_drag
            self.handle_drag = None
            if hit[0] == "corner":
                self.corner_dragged.emit(
                    hit[1],
                    (event.position().x() - start.x()) / scale,
                    (event.position().y() - start.y()) / scale,
                )
            self.update_edit_cursor(event.pos())
            event.accept()
            return
        if (
            self.crop_drag_start is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.crop_drag_start = None
            self.crop_drag_offset = (0, 0)
            self.update_edit_cursor(
                event.pos(),
                bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)


class DetectWorker(QRunnable):
    def __init__(
        self, paths: list[Path], angles: dict[Path, float] | None = None
    ) -> None:
        super().__init__()
        self.paths = paths
        self.angles = angles or {}
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            total = len(self.paths)
            detections: list[Detection | None] = [None] * total
            failures: list[tuple[int, Path, str]] = []
            completed = 0
            self.signals.progress.emit(0, total, "正在并行检测图片")
            # Full-resolution rotation is done before boundary analysis.
            # Limit concurrent scans so large pages do not exhaust memory.
            with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, 2)) as executor:
                futures = {
                    executor.submit(
                        detect_page,
                        path,
                        1275,
                        DETECTION_BRIGHT_FRACTION,
                        self.angles.get(path, 0.1),
                    ): (index, path)
                    for index, path in enumerate(self.paths)
                }
                for future in as_completed(futures):
                    index, path = futures[future]
                    try:
                        detections[index] = future.result()
                    except Exception as exc:
                        failures.append((index, path, str(exc)))
                    completed += 1
                    self.signals.progress.emit(completed, total, f"已检测 {path.name}")
            successful = repair_width_outliers(
                [item for item in detections if item is not None]
            )
            repaired_by_path = {item.path: item for item in successful}
            for index, item in enumerate(detections):
                if item is not None:
                    detections[index] = repaired_by_path[item.path]

            fallback_details: list[tuple[Path, str]] = []
            for index, path, reason in failures:
                try:
                    detections[index] = inferred_detection(path, successful)
                except Exception as exc:
                    raise RuntimeError(f"无法读取图片 {path.name}: {exc}") from exc
                fallback_details.append((path, reason))

            ordered = [item for item in detections if item is not None]
            self.signals.finished.emit((ordered, fallback_details))
        except Exception:
            self.signals.error.emit(traceback.format_exc())


def manual_cover_page(path):
    """Image extent only, not detected paper edges; excluded from all solving."""
    with Image.open(path) as source:
        width, height = source.size
    edges = {
        "left": [[0, height * i / 3] for i in range(4)],
        "right": [[width, height * i / 3] for i in range(4)],
        "top": [[width * i / 3, 0] for i in range(4)],
        "bottom": [[width * i / 3, height] for i in range(4)],
    }
    return PageAnalysis(
        path.name,
        width,
        height,
        edges,
        {side: [0.0] * 4 for side in edges},
        {side: 0.0 for side in edges},
        {side: 0.0 for side in edges},
        {side: 0.0 for side in edges},
        ["排除页：未自动分析，请手动调整"],
    )


def centered_plan(page, size, warning="排除页：未自动定位，请手动调整"):
    x, y = (page.width - size[0]) // 2, (page.height - size[1]) // 2
    return CropPlan(
        (x, y, x + size[0], y + size[1]), 0.0, "none", 0.0, 0.0, 0.0, 0.0,
        list(page.warnings) + [warning],
    )


def place_excluded_page(path, size, cache_dir=None, cancel=None, policy="balanced"):
    """Detect an excluded page and place the batch size on it.

    Typically a cover scanned as half of a spread: the inner part runs off
    the scan, so the crop is anchored on the outer edge plus top and bottom,
    exactly as in fixed-width mode. A page with both edges visible is
    centred on its detected paper. Returns (analysis, plan); falls back to a
    centred box on the image when nothing reliable is found.
    """
    width, height = map(int, size)
    try:
        side = clipped_side(path)
        if side is not None:
            page = analyze_page(path, cache_dir, cancel, width, side)
            plan = solve_bound_batch(
                [page], width, height, (0, 0, 0, 0), policy, mode="bound-" + side
            )[1][0]
            note = "排除页：内侧在扫描范围外，按外侧及上下纸边定位"
        else:
            page = analyze_page(path, cache_dir, cancel)
            plan = solve_page(page, (width, height), "none", (0, 0, 0, 0), policy)
            note = "排除页：按检测到的纸张居中定位"
        plan.warnings = list(dict.fromkeys(plan.warnings + [note]))
        return page, plan
    except Cancelled:
        raise
    except Exception:
        page = manual_cover_page(path)
        return page, centered_plan(page, size, "排除页：未能自动定位，请手动调整")


class AutoCropWorker(QRunnable):
    def __init__(
        self,
        paths,
        pages,
        size,
        mode,
        margins,
        policy,
        cache_dir,
        excluded_paths=(),
        bound_width=None,
        auto_crop=True,
        auto_perspective=False,
        perspectives=None,
    ):
        super().__init__()
        self.paths, self.pages, self.size = paths, pages, size
        self.mode, self.margins, self.policy = (
            mode,
            ((0, 0, 0, 0) if pages is None else margins),
            policy,
        )
        self.cache_dir = cache_dir
        self.excluded_paths = set(excluded_paths)
        self.bound_width = bound_width
        self.requested_margins = tuple(margins)
        self.auto_crop = auto_crop
        self.auto_perspective = auto_perspective
        self.perspectives = deepcopy(perspectives) if perspectives else {}
        self.raw_pages = []
        self.cancel = threading.Event()
        self.signals = WorkerSignals()

    def run(self):
        try:
            body_paths = [p for p in self.paths if p not in self.excluded_paths]
            # Alternation counts analyzed pages only: the chosen inner side is
            # that of the first analyzed page, and excluded pages do not count.
            body_indices = list(range(len(body_paths)))
            must_analyze = (
                self.pages is None
                or self.mode.startswith("bound-")
                or any(
                    "未拆书：内侧由指定宽度确定" in p.warnings
                    for p in (self.pages or [])
                )
            )
            if self.pages is None and not self.auto_crop and not self.auto_perspective:
                pages = [manual_cover_page(p) for p in self.paths]
                body_pages = [
                    p
                    for p, path in zip(pages, self.paths)
                    if path not in self.excluded_paths
                ]
                by_path = dict(zip(self.paths, pages))
            elif must_analyze:
                if body_paths and self.mode.startswith("bound-"):
                    body_pages = analyze_batch(
                        body_paths,
                        self.cache_dir,
                        self.signals.progress.emit,
                        self.cancel,
                        bound_width=self.bound_width,
                        bound_mode=self.mode,
                        page_indices=body_indices,
                    )
                else:
                    body_pages = (
                        analyze_batch(
                            body_paths,
                            self.cache_dir,
                            self.signals.progress.emit,
                            self.cancel,
                        )
                        if body_paths
                        else []
                    )
                by_path = dict(zip(body_paths, body_pages))
                by_path.update(
                    {
                        p: manual_cover_page(p)
                        for p in self.paths
                        if p in self.excluded_paths
                    }
                )
                pages = [by_path[p] for p in self.paths]
            else:
                pages = self.pages
                by_path = dict(zip(self.paths, pages))
                body_pages = [by_path[p] for p in body_paths]
            self.raw_pages = pages
            warped = []
            for index, path in zip(body_indices, body_paths):
                page = by_path[path]
                state = self.perspectives.get(path)
                changed_kind = bool(
                    state
                    and state.get("automatic")
                    and bool(state.get("fixed_width")) != self.mode.startswith("bound-")
                )
                if self.auto_perspective and (self.pages is None or changed_kind):
                    try:
                        state = recommend_perspective(
                            page,
                            auto_inner_side(page, self.mode, index),
                            (
                                self.bound_width
                                if self.mode.startswith("bound-")
                                else None
                            ),
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        state = manual_state(
                            (0, 0, page.width, page.height), 0, page.width, page.height
                        )
                        state["reason"] = "四边无法可靠拟合；未自动开启，可手动调整"
                    if state is not None:
                        self.perspectives[path] = state
                warped.append(warp_analysis(page, state))
            body_pages = warped
            by_path.update(dict(zip(body_paths, body_pages)))
            pages = [by_path[p] for p in self.paths]
            if not self.auto_crop:
                size = self.size or (
                    (
                        self.bound_width
                        if self.mode.startswith("bound-")
                        else min(p.width for p in pages)
                    ),
                    min(p.height for p in pages),
                )
                body_plans = []
                for index, path, page in zip(body_indices, body_paths, body_pages):
                    x, y = (page.width - size[0]) // 2, (page.height - size[1]) // 2
                    side = auto_inner_side(page, self.mode, index)
                    angle = (
                        reference_angle(page, side)[0]
                        if state_matrix(self.perspectives.get(path)) is not None
                        else 0.0
                    )
                    body_plans.append(
                        CropPlan(
                            (x, y, x + size[0], y + size[1]),
                            angle,
                            side,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            page.warnings,
                        )
                    )
            elif body_pages:
                if self.mode.startswith("bound-"):
                    size, body_plans = solve_bound_batch(
                        body_pages,
                        self.bound_width,
                        None if self.size is None else self.size[1],
                        self.margins,
                        self.policy,
                        self.signals.progress.emit,
                        self.cancel,
                        mode=self.mode,
                        page_indices=body_indices,
                    )
                else:
                    size, body_plans = solve_batch(
                        body_pages,
                        self.size,
                        self.mode,
                        self.margins,
                        self.policy,
                        self.signals.progress.emit,
                        self.cancel,
                        page_indices=body_indices,
                    )
            else:
                size = self.size or (
                    (
                        self.bound_width
                        if self.mode.startswith("bound-")
                        else min(p.width for p in pages)
                    ),
                    min(p.height for p in pages),
                )
                body_plans = []
            plans_by_path = dict(zip(body_paths, body_plans))
            for path in self.paths:
                if path in self.excluded_paths:
                    if self.pages is None and self.auto_crop:
                        # Excluded pages do not shape the batch, but still get
                        # the batch size placed on their own detected paper.
                        by_path[path], plans_by_path[path] = place_excluded_page(
                            path, size, self.cache_dir, self.cancel, self.policy
                        )
                    else:
                        plans_by_path[path] = centered_plan(by_path[path], size)
            pages = [by_path[p] for p in self.paths]
            self.raw_pages = [
                by_path[p] if p in self.excluded_paths else raw
                for p, raw in zip(self.paths, self.raw_pages)
            ]
            plans = [plans_by_path[p] for p in self.paths]
            self.signals.finished.emit((self.paths, pages, size, plans))
        except Cancelled:
            self.signals.finished.emit(None)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class ReanalyzePageWorker(QRunnable):
    """Re-detect one (typically rescanned) page against the batch's size.

    Mirrors AutoCropWorker for a single page: the inner side follows the
    page's position in the batch, and the plan is solved at the batch's
    zero-trim paper size so the batch trims apply to it like any other page.
    """

    def __init__(
        self,
        path: Path,
        index: int,
        mode: str,
        paper_size: tuple[int, int],
        auto_crop: bool,
        auto_perspective: bool,
        cache_dir: Path,
        batch_pages=(),
        excluded: bool = False,
    ) -> None:
        super().__init__()
        self.excluded = excluded
        self.batch_pages = list(batch_pages)
        self.path, self.index, self.mode = path, index, mode
        self.paper_size = tuple(int(v) for v in paper_size)
        self.auto_crop, self.auto_perspective = auto_crop, auto_perspective
        self.cache_dir = cache_dir
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            bound = self.mode.startswith("bound-")
            width, height = self.paper_size
            if self.excluded:
                page, plan = place_excluded_page(self.path, self.paper_size, self.cache_dir)
                _, origins = paper_reference([page], plan.inner, (width, height), [plan.angle])
                self.signals.finished.emit((self.path, page, page, plan, None, origins[0]))
                return
            page = analyze_page(
                self.path,
                self.cache_dir,
                None,
                width if bound else None,
                auto_inner_side(None, self.mode, self.index) if bound else "left",
            )
            # Scanner-adapter steps that check against the rest of the batch.
            page = after_single_page(self.path, page, self.batch_pages)
            inner = auto_inner_side(page, self.mode, self.index)
            state = None
            if self.auto_perspective:
                try:
                    state = recommend_perspective(page, inner, width if bound else None)
                except (ValueError, np.linalg.LinAlgError):
                    state = manual_state(
                        (0, 0, page.width, page.height), 0, page.width, page.height
                    )
                    state["reason"] = "四边无法可靠拟合；未自动开启，可手动调整"
            warped = warp_analysis(page, state)
            if not self.auto_crop:
                x, y = (page.width - width) // 2, (page.height - height) // 2
                angle = (
                    reference_angle(page, inner)[0]
                    if state_matrix(state) is not None
                    else 0.0
                )
                plan = CropPlan(
                    (x, y, x + width, y + height), angle, inner, 0.0, 0.0, 0.0, 0.0,
                    page.warnings,
                )
            elif bound:
                plan = solve_bound_batch(
                    [warped], width, height, (0, 0, 0, 0), "balanced",
                    mode=self.mode, page_indices=[self.index],
                )[1][0]
            else:
                plan = solve_batch(
                    [warped], (width, height), self.mode, (0, 0, 0, 0), "balanced",
                    page_indices=[self.index],
                )[1][0]
            # A fixed side reproduces the batch's side for this page index.
            _, origins = paper_reference([warped], plan.inner, (width, height), [plan.angle])
            self.signals.finished.emit((self.path, page, warped, plan, state, origins[0]))
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class RotatedBoundaryWorker(QRunnable):
    def __init__(self, generation: int, path: Path, angle: float) -> None:
        super().__init__()
        self.generation = generation
        self.path = path
        self.angle = angle
        self.signals = PreviewCacheSignals()

    def run(self) -> None:
        try:
            detection = detect_page(
                self.path, 1275, DETECTION_BRIGHT_FRACTION, self.angle
            )
            self.signals.ready.emit((self.generation, self.path, self.angle, detection))
        except Exception as exc:
            self.signals.ready.emit((self.generation, self.path, self.angle, exc))
        finally:
            self.signals.finished.emit(self.generation)


def parse_page_numbers(text: str, count: int) -> set[int]:
    """One-based displayed page numbers; never filename numbers."""
    if not text.strip():
        return set()
    result = set()
    for part in text.replace("，", ",").split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            raise ValueError("页码格式应为 1,2,5-8；留空表示不排除。")
        first = int(match[1])
        last = int(match[2] or first)
        if not 1 <= first <= last <= count:
            raise ValueError(f"页码须在 1–{count} 内，范围起点不能大于终点。")
        result.update(range(first, last + 1))
    return result


def reliable_channel_measurement(m: ChannelShiftMeasurement) -> bool:
    values = (
        m.red_x,
        m.red_y,
        m.blue_x,
        m.blue_y,
        m.red_response,
        m.blue_response,
        m.red_mad,
        m.blue_mad,
        m.neutral_fraction,
    )
    return (
        all(math.isfinite(v) for v in values)
        and m.neutral_fraction >= 0.55
        and min(m.red_response, m.blue_response) >= 0.10
        and max(m.red_mad, m.blue_mad) <= 0.35
    )


def combine_channel_measurements(
    items: list[ChannelShiftMeasurement],
) -> ChannelShiftMeasurement:
    # Each reliable page has equal weight; a highly textured page cannot dominate.
    def median(field):
        return statistics.median(getattr(m, field) for m in items)

    rx, ry, bx, by = [median(k) for k in ("red_x", "red_y", "blue_x", "blue_y")]
    # Include between-page disagreement, not only within-page tile stability.
    red_spread = statistics.median(
        math.hypot(m.red_x - rx, m.red_y - ry) for m in items
    )
    blue_spread = statistics.median(
        math.hypot(m.blue_x - bx, m.blue_y - by) for m in items
    )
    return ChannelShiftMeasurement(
        rx,
        ry,
        bx,
        by,
        sum(m.red_samples for m in items),
        sum(m.blue_samples for m in items),
        median("red_response"),
        median("blue_response"),
        max(median("red_mad"), red_spread),
        max(median("blue_mad"), blue_spread),
        median("neutral_fraction"),
    )


class ChannelShiftDetectWorker(QRunnable):
    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self.paths = paths
        self.signals = WorkerSignals()

    def run(self) -> None:
        started = time.perf_counter()
        accepted, rejected = [], []
        try:
            total = len(self.paths)
            outcomes = {}
            workers = max(1, min(RGB_DETECT_WORKERS, total))
            self.signals.progress.emit(
                0, total, f"正在并行测量 RGB 偏移（最多 {workers} 张）"
            )
            # Each task decodes its own image and owns all measurement buffers.
            # Limit concurrent full-resolution decodes to bound memory usage.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(detect_rgb_channel_shifts, path): i
                    for i, path in enumerate(self.paths)
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    i = futures[future]
                    path = self.paths[i]
                    try:
                        m = future.result()
                        outcomes[i] = (
                            (m, None)
                            if reliable_channel_measurement(m)
                            else (None, "彩色占比过高或测量不稳定")
                        )
                    except Exception as exc:
                        outcomes[i] = (None, str(exc))
                    self.signals.progress.emit(
                        completed,
                        total,
                        f"已测量 RGB 偏移 {completed}/{total}：{path.name}",
                    )
            # Keep summaries in page order despite out-of-order completion.
            for i, path in enumerate(self.paths):
                m, error = outcomes[i]
                if m is not None:
                    accepted.append(m)
                else:
                    rejected.append((path.name, error))
            if not accepted:
                raise ValueError(
                    "范围内没有可用于标定的可靠图片。\n"
                    + "\n".join(f"{n}：{reason}" for n, reason in rejected)
                )
            measurement = combine_channel_measurements(accepted)
            self.signals.finished.emit(
                (
                    self.paths,
                    measurement,
                    time.perf_counter() - started,
                    len(accepted),
                    rejected,
                )
            )
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class ExportWorker(QRunnable):
    def __init__(
        self,
        detections: list[Detection],
        boxes: list[tuple[int, int, int, int]],
        rotation_centers: list[tuple[float, float]],
        rotation_angles: list[float],
        output_dir: Path,
        channel_shifts: dict[str, tuple[float, float]] | None,
        perspectives=None,
    ) -> None:
        super().__init__()
        self.perspectives = perspectives or {}
        self.detections = detections
        self.boxes = boxes
        self.rotation_centers = rotation_centers
        self.rotation_angles = rotation_angles
        self.output_dir = output_dir
        self.channel_shifts = channel_shifts
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            total = len(self.detections)
            completed = 0
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.signals.progress.emit(0, total, "正在并行导出图片")
            max_workers = COLOR_EXPORT_WORKERS if self.channel_shifts else BATCH_WORKERS
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for item, box, center, angle in zip(
                    self.detections,
                    self.boxes,
                    self.rotation_centers,
                    self.rotation_angles,
                ):
                    destination = self.output_dir / item.path.name
                    future = executor.submit(
                        save_crop,
                        item.path,
                        destination,
                        box,
                        angle,
                        center,
                        self.channel_shifts,
                        self.perspectives.get(item.path),
                    )
                    futures[future] = item.path
                for future in as_completed(futures):
                    path = futures[future]
                    future.result()
                    completed += 1
                    self.signals.progress.emit(completed, total, f"已导出 {path.name}")
            self.signals.finished.emit(self.output_dir)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class PreviewCacheSignals(QObject):
    ready = Signal(object)
    draft = Signal(object)
    finished = Signal(int)


class PreviewCacheWorker(QRunnable):
    def __init__(
        self,
        generation: int,
        jobs: list[
            tuple[
                Path,
                tuple[int, int, int, int],
                tuple[float, float],
            ]
        ],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.jobs = jobs
        self.cancel_event = cancel_event
        self.signals = PreviewCacheSignals()

    def run(self) -> None:
        try:
            for path, preview_box, rotation_center in self.jobs:
                if self.cancel_event.is_set():
                    break
                with Image.open(path) as source_file:
                    source = source_file.convert("RGBA")
                left, top, right, bottom = preview_box
                cache_box = (
                    left - ROTATION_CACHE_MARGIN,
                    top - ROTATION_CACHE_MARGIN,
                    right + ROTATION_CACHE_MARGIN,
                    bottom + ROTATION_CACHE_MARGIN,
                )
                for step in (-1, 1, -2, 2):
                    if self.cancel_event.is_set():
                        break
                    angle = step / 10
                    transformed = transformed_crop(
                        source,
                        cache_box,
                        angle,
                        rotation_center,
                    )
                    cache_key = (
                        path,
                        source.size,
                        round(angle, 4),
                        rotation_center,
                    )
                    self.signals.ready.emit(
                        (
                            self.generation,
                            cache_key,
                            cache_box,
                            transformed,
                        )
                    )
        except Exception:
            pass
        finally:
            self.signals.finished.emit(self.generation)


class CurrentRotationWorker(QRunnable):
    def __init__(
        self,
        generation: int,
        path: Path,
        preview_box: tuple[int, int, int, int],
        rotation_angle: float,
        rotation_center: tuple[float, float],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.path = path
        self.preview_box = preview_box
        self.rotation_angle = rotation_angle
        self.rotation_center = rotation_center
        self.cancel_event = cancel_event
        self.signals = PreviewCacheSignals()

    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                return
            with Image.open(self.path) as source_file:
                source = source_file.convert("RGBA")
            left, top, right, bottom = self.preview_box
            cache_box = (
                left - ROTATION_CACHE_MARGIN,
                top - ROTATION_CACHE_MARGIN,
                right + ROTATION_CACHE_MARGIN,
                bottom + ROTATION_CACHE_MARGIN,
            )
            transformed = transformed_crop(
                source,
                cache_box,
                self.rotation_angle,
                self.rotation_center,
            )
            if self.cancel_event.is_set():
                return
            cache_key = (
                self.path,
                source.size,
                round(self.rotation_angle, 4),
                self.rotation_center,
            )
            self.signals.ready.emit(
                (
                    self.generation,
                    cache_key,
                    cache_box,
                    transformed,
                )
            )
        except Exception:
            pass
        finally:
            self.signals.finished.emit(self.generation)


PREVIEW_RENDER_THREADS = max(2, min(12, (os.cpu_count() or 4) - 2))
PREVIEW_RENDER_EXECUTOR = ThreadPoolExecutor(
    max_workers=PREVIEW_RENDER_THREADS, thread_name_prefix="preview-render"
)
DRAFT_LONG_SIDE = 1100


def load_preview_source(path: Path) -> Image.Image:
    """Decode a source exactly as preview rendering and export expect it."""
    with Image.open(path) as source_file:
        # Preserve grayscale until rendering so RGB alignment is skipped just
        # as it is when exporting the original file.
        return (
            source_file.copy()
            if source_file.mode in {"L", "LA"}
            else source_file.convert("RGBA" if "A" in source_file.getbands() else "RGB")
        )


class SourceCache:
    """Small LRU of decoded scans, so re-rendering a page skips PNG decoding.

    Keys include size and mtime: a rescanned file is decoded again.
    """

    def __init__(self, capacity: int = 2) -> None:
        self.capacity = capacity
        self.lock = threading.Lock()
        self.items: OrderedDict[tuple[object, ...], Image.Image] = OrderedDict()

    @staticmethod
    def key(path: Path) -> tuple[object, ...]:
        stat = path.stat()
        return (str(path), stat.st_size, stat.st_mtime_ns)

    def get(self, path: Path, store: bool = True) -> Image.Image:
        key = self.key(path)
        with self.lock:
            image = self.items.get(key)
            if image is not None:
                self.items.move_to_end(key)
                return image
        image = load_preview_source(path)
        if store:
            with self.lock:
                self.items[key] = image
                self.items.move_to_end(key)
                while len(self.items) > self.capacity:
                    self.items.popitem(last=False)
        return image

    def discard(self, path: Path) -> None:
        with self.lock:
            for key in [k for k in self.items if k[0] == str(path)]:
                del self.items[key]


PREVIEW_SOURCES = SourceCache()


def qimage_from_pil(image: Image.Image) -> QImage:
    """Convert to a pixmap-native format off the UI thread (pixels unchanged)."""
    if image.mode == "L":
        pixels = np.ascontiguousarray(np.asarray(image))
        fmt, stride, target = QImage.Format.Format_Grayscale8, image.width, QImage.Format.Format_RGB32
    elif image.mode == "RGB":
        pixels = np.ascontiguousarray(np.asarray(image))
        fmt, stride, target = QImage.Format.Format_RGB888, image.width * 3, QImage.Format.Format_RGB32
    else:
        pixels = np.ascontiguousarray(np.asarray(image.convert("RGBA")))
        fmt, stride, target = (
            QImage.Format.Format_RGBA8888,
            image.width * 4,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
    return QImage(pixels.data, image.width, image.height, stride, fmt).convertToFormat(target)


def build_preview_qimages(image: Image.Image) -> list[tuple[float, QImage]]:
    """The same LANCZOS pyramid the view used to build on the UI thread."""
    levels = [(1.0, qimage_from_pil(image))]
    reduced = image
    for scale in (0.5, 0.25, 0.125):
        size = (max(1, round(reduced.width * 0.5)), max(1, round(reduced.height * 0.5)))
        if size[0] < 128 or size[1] < 128:
            break
        reduced = reduced.resize(size, Image.Resampling.LANCZOS)
        levels.append((scale, qimage_from_pil(reduced)))
    return levels


class SourceInfo:
    """Size of the current scan, known without decoding its pixels."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = tuple(size)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]


class PreviewRaster:
    """A rendered page ready for display: prebuilt levels, no PIL pixels kept."""

    def __init__(self, levels: list[tuple[float, QImage]], size: tuple[int, int]) -> None:
        self.levels = levels
        self.size = size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]


class PreviewDraft:
    """A small unrotated copy shown immediately while a page is rendered."""

    def __init__(self, image: QImage, source_size: tuple[int, int]) -> None:
        self.image = image
        self.source_size = source_size


def make_draft(source: Image.Image) -> PreviewDraft:
    factor = max(1, math.ceil(max(source.size) / DRAFT_LONG_SIDE))
    small = source.reduce(factor) if factor > 1 else source
    return PreviewDraft(qimage_from_pil(small), source.size)


class DraftWorker(QRunnable):
    """Prepare quick drafts for pages ahead, so fast flipping never stalls."""

    def __init__(self, generation: int, path: Path, cancel_event: threading.Event) -> None:
        super().__init__()
        self.generation = generation
        self.path = path
        self.cancel_event = cancel_event
        self.signals = PreviewCacheSignals()

    def run(self) -> None:
        try:
            if self.cancel_event.is_set():
                return
            draft = make_draft(PREVIEW_SOURCES.get(self.path, store=False))
            if not self.cancel_event.is_set():
                self.signals.ready.emit((self.generation, self.path, draft))
        except Exception:
            pass
        finally:
            self.signals.finished.emit(self.generation)


class FullPreviewWorker(QRunnable):
    def __init__(
        self,
        generation: int,
        jobs: list[tuple[Path, float, tuple[float, float]]],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.jobs = jobs
        self.cancel_event = cancel_event
        self.signals = PreviewCacheSignals()

    def run(self) -> None:
        try:
            for job in self.jobs:
                path, rotation_angle, rotation_center = job[:3]
                perspective = job[3] if len(job) > 3 else None
                channel_shifts = job[4] if len(job) > 4 else None
                if self.cancel_event.is_set():
                    break
                source = PREVIEW_SOURCES.get(path)
                if self.cancel_event.is_set():
                    break
                # The draft lets the page appear at once next time it is shown.
                self.signals.draft.emit((self.generation, path, make_draft(source)))
                width, height = source.size
                if (
                    abs(rotation_angle) < 1e-9
                    and perspective is None
                    and not channel_shifts
                ):
                    full_box = (0, 0, width, height)
                    # Own copy: levels are attached below, and the decoded
                    # source is shared through the cache.
                    transformed = source.copy()
                else:
                    full_box = (
                        rotated_full_box(width, height, rotation_angle, rotation_center)
                        if perspective is None
                        else transformed_bounds(
                            width,
                            height,
                            combined_matrix(
                                width,
                                height,
                                rotation_angle,
                                rotation_center,
                                perspective,
                            ),
                        )
                    )
                    transformed = render_transformed(
                        source,
                        full_box,
                        rotation_angle,
                        rotation_center,
                        channel_shifts,
                        perspective,
                        PREVIEW_RENDER_EXECUTOR,
                        self.cancel_event,
                    )
                if self.cancel_event.is_set():
                    break
                # Build display levels here rather than on the UI thread.
                transformed.preview_levels = build_preview_qimages(transformed)
                cache_key = perspective_cache_key(
                    path,
                    source.size,
                    rotation_angle,
                    rotation_center,
                    perspective,
                    channel_shifts,
                )
                self.signals.ready.emit(
                    (self.generation, cache_key, full_box, transformed)
                )
        except Exception:
            pass
        finally:
            self.signals.finished.emit(self.generation)


class CropperWindow(QMainWindow):
    def __init__(self, initial_input: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("扫图裁切")
        self.resize(1280, 900)
        self.thread_pool = QThreadPool.globalInstance()
        self.rotation_thread_pool = QThreadPool(self)
        self.rotation_thread_pool.setMaxThreadCount(1)
        self.full_preview_thread_pool = QThreadPool(self)
        # Two pages render at once: one decodes while the other warps.
        self.full_preview_thread_pool.setMaxThreadCount(2)
        self.draft_thread_pool = QThreadPool(self)
        self.draft_thread_pool.setMaxThreadCount(1)
        # key -> (serial, worker); only the registered worker may fill a key.
        self.full_preview_inflight: dict[tuple[object, ...], tuple[int, QRunnable]] = {}
        self.full_preview_serial = 0
        self.preview_drafts: dict[Path, PreviewDraft] = {}
        self.draft_inflight: dict[Path, threading.Event] = {}
        # (path, rendered angle, render signature, "full" | "draft") on screen.
        self.displayed_preview: tuple[object, ...] | None = None
        self.level_mode_active = False
        self.mouse_rotation_base: float | None = None
        self.page_reanalysis_worker: ReanalyzePageWorker | None = None
        # (auto crop, auto perspective) used by the analysis now on screen.
        self.applied_analysis_flags: tuple[bool, bool] = (True, False)
        self.detections: list[Detection] = []
        self.crop_undo_history = []
        self.crop_redo_history = []
        self.undo_restoring = False
        self.undo_batch_values = None
        self.inferred_paths: set[Path] = set()
        self.page_offsets: dict[Path, tuple[int, int]] = {}
        self.page_rotations: dict[Path, float] = {}
        self.page_size_deltas: dict[Path, tuple[int, int]] = {}
        self.perspectives = {}
        self.raw_smart_pages = {}
        self.corner_live = None
        self.edge_sweep = None
        self.edge_sweep_pending = None
        self.edge_sweep_saved = None
        self.sweep_short_duration = 1.5
        self.sweep_start_delay = 0.5
        self.inspection_scale = 2.0
        self.edge_jump_pending = None
        self.position_jump = None
        self.edge_sweep_timer = QTimer(self)
        self.edge_sweep_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.edge_sweep_timer.setInterval(16)
        self.edge_sweep_timer.timeout.connect(self.advance_edge_sweep)
        self.boundary_generation = 0
        self.boundary_worker: RotatedBoundaryWorker | None = None
        self.import_boundary_generation = 0
        self.import_boundary_worker: DetectWorker | None = None
        self.smart_pages: dict[Path, PageAnalysis] = {}
        self.smart_plans: dict[Path, CropPlan] = {}
        self.smart_size: tuple[int, int] | None = None
        self.smart_paper_size: tuple[int, int] | None = None
        self.smart_origins: dict[Path, tuple[int, int]] = {}
        self.smart_symmetric = False
        self.smart_bound = False
        self.bound_width = 4300
        self.saved_inner_trim = 0
        self.mode_control_bound = False
        self.mode_follows_ui = False
        self.smart_default_offsets: dict[Path, tuple[int, int]] = {}
        self.smart_manual_pages: set[Path] = set()
        self.smart_worker: AutoCropWorker | None = None
        self.smart_is_new = False
        self.smart_applied_settings = None
        self.smart_timer = QTimer(self)
        self.smart_timer.setSingleShot(True)
        self.smart_timer.setInterval(500)
        self.smart_timer.timeout.connect(self.recalculate_smart)
        self.current_index = 0
        self.preview_image: Image.Image | None = None
        self.preview_path: Path | None = None
        self.preview_transform_cache_key: tuple[object, ...] | None = None
        self.preview_transform_cache_box: tuple[int, int, int, int] | None = None
        self.preview_transform_cache_image: Image.Image | None = None
        self.prefetched_rotation_cache: dict[
            tuple[object, ...],
            tuple[tuple[int, int, int, int], Image.Image],
        ] = {}
        self.preview_cache_generation = 0
        self.preview_cache_cancel_event: threading.Event | None = None
        self.preview_cache_worker: PreviewCacheWorker | None = None
        self.current_rotation_generation = 0
        self.current_rotation_cancel_event: threading.Event | None = None
        self.current_rotation_worker: CurrentRotationWorker | None = None
        self.pending_rotation_request: (
            tuple[
                Path,
                tuple[int, int, int, int],
                float,
                tuple[float, float],
            ]
            | None
        ) = None
        self.full_preview_generation = 0
        self.full_preview_cancel_event: threading.Event | None = None
        self.full_preview_worker: FullPreviewWorker | None = None
        self.full_preview_cache: dict[
            tuple[object, ...], tuple[tuple[int, int, int, int], Image.Image]
        ] = {}
        self.full_preview_cache_key: tuple[object, ...] | None = None
        self.full_preview_image: Image.Image | None = None
        self.full_preview_box: tuple[int, int, int, int] | None = None
        self.pending_full_preview_keys: tuple[tuple[object, ...], ...] = ()
        self.pending_full_toggle = False
        self.current_preview_box: tuple[int, int, int, int] | None = None
        self.show_full_preview = True
        self.prefer_full_preview = True
        self.reset_view_on_refresh = True
        self.active_worker: QRunnable | None = None
        self.channel_shift_worker: ChannelShiftDetectWorker | None = None
        self.pending_color_profile_name: str | None = None
        (
            self.color_profiles,
            self.selected_color_profile,
            self.color_correction_enabled,
        ) = load_color_profiles()
        self.preview_cache_timer = QTimer(self)
        self.preview_cache_timer.setSingleShot(True)
        self.preview_cache_timer.setInterval(250)
        self.preview_cache_timer.timeout.connect(self.schedule_preview_cache)
        self.rotation_preview_timer = QTimer(self)
        self.rotation_preview_timer.setSingleShot(True)
        self.rotation_preview_timer.setInterval(60)
        self.rotation_preview_timer.timeout.connect(self.start_current_rotation_worker)
        self.full_preview_timer = QTimer(self)
        self.full_preview_timer.setSingleShot(True)
        self.full_preview_timer.setInterval(300)
        self.full_preview_timer.timeout.connect(self.prepare_current_full_preview)

        self._build_ui()
        self._build_menus()
        self._install_shortcuts()
        QApplication.instance().installEventFilter(self)

        if initial_input is not None and initial_input.is_dir():
            self.input_edit.setText(str(initial_input))
            self.output_edit.setText(str(initial_input / "cropped"))
        self.restore_user_settings(restore_paths=initial_input is None)
        self._set_controls_enabled(False)
        self.settings_save_timer = QTimer(self)
        self.settings_save_timer.setSingleShot(True)
        self.settings_save_timer.setInterval(300)
        self.settings_save_timer.timeout.connect(self.save_user_settings)
        for control, signal in (
            (self.input_edit, "textChanged"),
            (self.output_edit, "textChanged"),
            (self.mode_side_combo, "currentIndexChanged"),
            (self.alternate_check, "toggled"),
            (self.fixed_width_check, "toggled"),
            (self.excluded_pages_edit, "textChanged"),
            (self.color_correction_check, "toggled"),
            (self.color_profile_combo, "currentIndexChanged"),
            (self.background_combo, "currentIndexChanged"),
            (self.auto_check, "toggled"),
            (self.perspective_auto_check, "toggled"),
            (self.show_edges_check, "toggled"),
        ):
            getattr(control, signal).connect(
                lambda *_: self.settings_save_timer.start()
            )
        self.main_splitter.splitterMoved.connect(
            lambda *_: self.settings_save_timer.start()
        )
        self.crop_inner_spin.valueChanged.connect(
            lambda *_: (
                self.settings_save_timer.start()
                if self.fixed_width_check.isChecked()
                else None
            )
        )

    def restore_user_settings(self, restore_paths=True):
        try:
            raw = json.loads(application_settings_path().read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            for key, minimum, maximum in [
                ("sweep_short_duration", 0.1, 60.0),
                ("sweep_start_delay", 0.0, 10.0),
                ("inspection_scale", 0.1, 16.0),
            ]:
                try:
                    value = float(raw.get(key, getattr(self, key)))
                    if math.isfinite(value) and minimum <= value <= maximum:
                        setattr(self, key, value)
                except (ValueError, TypeError, OverflowError):
                    pass
            width = int(raw.get("image_width", 4300))
            if not 16 <= width <= 100000:
                width = 4300
            controls = [
                self.input_edit,
                self.output_edit,
                self.inner_mode,
                self.mode_side_combo,
                self.alternate_check,
                self.fixed_width_check,
                self.crop_inner_spin,
                self.excluded_pages_edit,
                self.color_correction_check,
                self.color_profile_combo,
                self.background_combo,
                self.auto_check,
                self.perspective_auto_check,
                self.show_edges_check,
                self.crop_top_spin,
                self.crop_bottom_spin,
                self.crop_outer_spin,
                self.top_spin,
                self.bottom_spin,
                self.outer_spin,
                self.inner_spin,
            ]
            with ExitStack() as blockers:
                for control in controls:
                    blockers.enter_context(QSignalBlocker(control))
                self.bound_width = width
                side = raw.get("inner_side", "right")
                if side not in ("left", "right"):
                    side = "right"
                mode = (
                    ("bound-" if raw.get("fixed_width", False) else "")
                    + ("alternate-" if raw.get("alternate", True) else "")
                    + side
                )
                self.inner_mode.setCurrentIndex(self.inner_mode.findData(mode))
                self.update_mode_controls()
                # Batch trims are intentionally not application preferences.
                # Ignore trim values written by older settings versions.
                if restore_paths:
                    for key, control in [
                        ("input_dir", self.input_edit),
                        ("output_dir", self.output_edit),
                    ]:
                        if isinstance(raw.get(key), str):
                            control.setText(raw[key])
                self.excluded_pages_edit.setText(
                    str(
                        raw.get(
                            "excluded_pages",
                            "1-2" if raw.get("skip_first_two", False) else "",
                        )
                    )
                )
                self.auto_check.setChecked(bool(raw.get("auto_detect", True)))
                self.show_edges_check.setChecked(bool(raw.get("show_edges", True)))
                self.prefer_full_preview = bool(raw.get("show_full_preview", True))
                self.show_full_preview = self.prefer_full_preview
                self.perspective_auto_check.setChecked(
                    bool(raw.get("auto_perspective", False))
                )
                self.color_correction_check.setChecked(
                    bool(
                        raw.get(
                            "color_enabled", self.color_correction_check.isChecked()
                        )
                    )
                )
                for control, key in [(self.background_combo, "background")]:
                    index = control.findData(raw.get(key))
                    if index >= 0:
                        control.setCurrentIndex(index)
                profile = raw.get("color_profile")
                if (
                    isinstance(profile, str)
                    and self.color_profile_combo.findText(profile) >= 0
                ):
                    self.color_profile_combo.setCurrentText(profile)
            self.update_preview_background()
            geometry = raw.get("window_geometry")
            if isinstance(geometry, str) and len(geometry) < 4096:
                self.restoreGeometry(
                    QByteArray.fromBase64(geometry.encode("ascii", errors="ignore"))
                )
            sizes = raw.get("splitter_sizes")
            if (
                isinstance(sizes, list)
                and len(sizes) == 2
                and all(isinstance(v, int) and 0 < v <= 100000 for v in sizes)
            ):
                self.main_splitter.setSizes(sizes)
        except (OSError, ValueError, TypeError):
            pass  # Optional user settings must never prevent opening the app.

    def save_user_settings(self):
        width = (
            self.crop_inner_spin.value()
            if self.fixed_width_check.isChecked()
            else self.bound_width
        )
        payload = {
            "version": 4,
            "input_dir": self.input_edit.text(),
            "output_dir": self.output_edit.text(),
            "inner_side": self.mode_side_combo.currentData(),
            "alternate": self.alternate_check.isChecked(),
            "fixed_width": self.fixed_width_check.isChecked(),
            "image_width": width,
            "excluded_pages": self.excluded_pages_edit.text(),
            "auto_detect": self.auto_check.isChecked(),
            "auto_perspective": self.perspective_auto_check.isChecked(),
            "color_enabled": self.color_correction_check.isChecked(),
            "color_profile": self.color_profile_combo.currentText(),
            "background": self.background_combo.currentData(),
            "policy": "balanced",
            "show_edges": self.show_edges_check.isChecked(),
            "show_full_preview": self.prefer_full_preview,
            "window_geometry": self.saveGeometry().toBase64().data().decode("ascii"),
            "splitter_sizes": self.main_splitter.sizes(),
            "sweep_short_duration": self.sweep_short_duration,
            "sweep_start_delay": self.sweep_start_delay,
            "inspection_scale": self.inspection_scale,
        }
        try:
            path = application_settings_path()
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            self.status_label.setText(f"设置保存失败：{exc}")

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        self.setCentralWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(350)
        controls.setMaximumWidth(470)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(6)
        self.controls_panel = controls

        paths_group = QGroupBox("文件夹")
        paths_layout = QFormLayout(paths_group)
        self.input_edit = FolderLineEdit()
        self.input_edit.folder_dropped.connect(self.set_input_folder)
        input_button = QPushButton("选择…")
        input_button.clicked.connect(self.choose_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(input_button)
        paths_layout.addRow("扫描原图", input_row)

        self.output_edit = FolderLineEdit()
        self.output_edit.folder_dropped.connect(self.output_edit.setText)
        output_button = QPushButton("选择…")
        output_button.clicked.connect(self.choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(output_button)
        paths_layout.addRow("输出目录", output_row)

        self.inner_mode = QComboBox()
        for label, value in INNER_MODES:
            self.inner_mode.addItem(label, value)
        self.inner_mode.setCurrentIndex(self.inner_mode.findData("alternate-right"))

        self.inner_mode.setParent(self)
        self.inner_mode.hide()
        self.mode_side_combo = QComboBox()
        self.mode_side_combo.addItem("内侧在左", "left")
        self.mode_side_combo.addItem("内侧在右", "right")
        self.mode_side_combo.setCurrentIndex(1)
        self.alternate_check = QCheckBox("交替")
        self.alternate_check.setChecked(True)
        self.alternate_check.setToolTip(
            "按参与分析的页面顺序左右交替：下拉所选方向对应第一张参与分析的页。"
            "排除的页不计入交替，其内侧由扫描图自动判断。"
        )
        self.fixed_width_check = QCheckBox("指定宽度")
        self.fixed_width_check.setToolTip(
            "不检测内侧边缘；按外侧旋转和定位，内侧裁量框改为图像宽度。自动校正仅按上下纸边调整内侧角点高度。可与交替同时使用。"
        )
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mode_side_combo, 1)
        mode_row.addWidget(self.alternate_check)
        mode_row.addWidget(self.fixed_width_check)
        paths_layout.addRow("处理模式", mode_row)
        self.analyze_button = QPushButton("分析文件夹")
        self.analyze_button.clicked.connect(self.analyze_folder)
        paths_layout.addRow(self.analyze_button)
        self.auto_check = QCheckBox("自动裁切")
        self.auto_check.setToolTip("下次分析时生效；可与自动梯形校正独立使用。")
        self.auto_check.setChecked(True)
        self.perspective_auto_check = QCheckBox("自动梯形校正")
        self.perspective_auto_check.setToolTip(
            "下次分析时生效；按可靠纸边推荐梯形或斜切校正。指定宽度时只按外侧及上下纸边调整内侧角点高度，保持输出宽度。"
        )
        auto_row = QHBoxLayout()
        auto_row.addWidget(self.auto_check)
        auto_row.addWidget(self.perspective_auto_check)
        paths_layout.insertRow(paths_layout.rowCount() - 1, auto_row)
        self.excluded_pages_edit = QLineEdit()
        self.excluded_pages_edit.setPlaceholderText("留空全部分析，如 1,2,5-8")
        self.excluded_pages_edit.setToolTip(
            "按文件排序后的页序号，从 1 开始。排除的页不参与整批尺寸统计，也不计入左右交替；"
            "仍按整批尺寸自动定位（内侧超出扫描范围的封面等按外侧及上下纸边定位），可手调并导出。修改后下次分析生效。"
        )
        paths_layout.insertRow(
            paths_layout.rowCount() - 1, "排除分析页码", self.excluded_pages_edit
        )
        self.cancel_auto_button = QPushButton("取消自动分析")
        self.cancel_auto_button.setEnabled(False)
        self.cancel_auto_button.hide()
        self.cancel_auto_button.clicked.connect(self.cancel_smart)
        paths_layout.addRow(self.cancel_auto_button)
        controls_layout.addWidget(paths_group)

        self.smart_group = QGroupBox("整批裁剪微调")
        smart_layout = QFormLayout(self.smart_group)
        self.smart_form = smart_layout
        self.crop_top_spin = self._spinbox(0)
        self.crop_bottom_spin = self._spinbox(0)
        self.crop_outer_spin = self._spinbox(0)
        self.crop_inner_spin = self._spinbox(0)
        for label, spin in zip(
            ("顶部裁掉", "底部裁掉", "外侧裁掉", "内侧裁掉"), self.smart_crop_spins()
        ):
            spin.setToolTip(
                "自动裁剪结果上的追加微调：0 不追加裁剪，正数向内裁，负数向外扩展；内外侧随装订方向映射。"
            )
            smart_layout.addRow(label, spin)
        self.smart_size_label = QLabel("分析后自动计算裁剪框；四边微调从 0 开始")
        self.smart_size_label.setWordWrap(True)
        smart_layout.addRow(self.smart_size_label)
        self.recalculate_action = QAction("按当前设置重新计算未排除页", self)
        self.recalculate_action.triggered.connect(self.recalculate_smart)
        self.show_edges_check = QAction("显示检测纸边（青色）", self)
        self.show_edges_check.setCheckable(True)
        self.show_edges_check.setChecked(True)
        self.show_edges_check.toggled.connect(self.refresh_preview)
        self.smart_group.setToolTip(
            "自动结果上的追加微调；0 不追加，正数向内裁，负数向外扩。内外随装订方向切换。"
        )
        controls_layout.addWidget(self.smart_group)

        crop_group = QGroupBox("手动模式裁量")
        self.legacy_crop_group = crop_group
        crop_layout = QFormLayout(crop_group)
        self.legacy_crop_layout = crop_layout
        self.background_combo = QComboBox()
        for label, color in PREVIEW_BACKGROUNDS:
            self.background_combo.addItem(label, color)

        self.top_spin = self._spinbox(15)
        self.bottom_spin = self._spinbox(0)
        self.outer_spin = self._spinbox(15)
        self.inner_spin = self._spinbox(25)
        crop_layout.addRow("顶部裁掉", self.top_spin)
        crop_layout.addRow("底部裁掉", self.bottom_spin)
        crop_layout.addRow("外侧裁掉", self.outer_spin)
        crop_layout.addRow("内侧裁掉", self.inner_spin)
        self.legacy_crop_note = QLabel(
            "正数向内裁，负数向外扩展；此组仅用于旧版手动模式"
        )
        crop_layout.addRow(self.legacy_crop_note)
        self.color_correction_check = QAction("启用颜色校正", self)
        self.color_correction_check.setCheckable(True)
        self.color_correction_check.setChecked(self.color_correction_enabled)
        self.color_profile_combo = QComboBox()
        for name in self.color_profiles:
            self.color_profile_combo.addItem(name)
        selected_index = self.color_profile_combo.findText(self.selected_color_profile)
        self.color_profile_combo.setCurrentIndex(max(0, selected_index))
        controls_layout.addWidget(crop_group)

        offset_group = QGroupBox("当前页微调")
        offset_layout = QFormLayout(offset_group)
        self.rotation_spin = CompactAngleSpinBox()
        self.rotation_spin.setRange(-sys.float_info.max, sys.float_info.max)
        self.rotation_spin.setDecimals(4)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setValue(0.1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setKeyboardTracking(False)
        self.offset_x_spin = self._spinbox(0)
        self.offset_y_spin = self._spinbox(0)
        self.width_delta_spin = self._spinbox(0)
        self.height_delta_spin = self._spinbox(0)
        self.level_button = QPushButton("参考线")
        self.level_button.setToolTip(
            "按下后在预览中 Ctrl+左键点两点，按两点连线调成水平或竖直并退出。"
            "快捷键 Ctrl+R 进入/退出，Esc 取消。"
        )
        self.level_button.clicked.connect(self.toggle_level_mode)
        rotation_row = QHBoxLayout()
        rotation_row.addWidget(self.rotation_spin, 1)
        rotation_row.addWidget(self.level_button)
        offset_layout.addRow("旋转角度", rotation_row)
        for label, first, second in [
            ("偏移 X / Y", self.offset_x_spin, self.offset_y_spin),
            ("增量 宽 / 高", self.width_delta_spin, self.height_delta_spin),
        ]:
            row = QHBoxLayout()
            row.addWidget(first)
            row.addWidget(second)
            offset_layout.addRow(label, row)
        self.page_output_size_label = QLabel("—")
        self.page_output_size_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        offset_layout.addRow("本页输出", self.page_output_size_label)
        self.page_perspective_check = QCheckBox("本页梯形校正")
        self.page_perspective_check.setToolTip(
            "关闭时保留四角参数；预览只显示四边形框，梯形校正仅在导出时应用。角点偏移单位为原图像素。"
        )
        offset_layout.addRow(self.page_perspective_check)
        self.perspective_panel = QWidget()
        corners = QGridLayout(self.perspective_panel)
        corners.setContentsMargins(0, 0, 0, 0)
        corners.setVerticalSpacing(2)
        self.corner_spins = []
        corners.addWidget(QLabel("角点偏移"), 0, 0)
        corners.addWidget(QLabel("X（原图向右）"), 0, 1)
        corners.addWidget(QLabel("Y（向上）"), 0, 2)
        for i, label in enumerate(("左上", "右上", "右下", "左下")):
            row = []
            corners.addWidget(QLabel(label), i + 1, 0)
            for j in range(2):
                spin = QDoubleSpinBox()
                spin.setRange(-100000, 100000)
                spin.setDecimals(1)
                spin.setSingleStep(1)
                spin.setKeyboardTracking(False)
                corners.addWidget(spin, i + 1, j + 1)
                row.append(spin)
            self.corner_spins.append(row)
        self.perspective_note = QLabel()
        self.perspective_note.setWordWrap(True)
        corners.addWidget(self.perspective_note, 5, 0, 1, 3)
        offset_layout.addRow(self.perspective_panel)
        self.perspective_panel.hide()
        offset_layout.setVerticalSpacing(3)
        self.page_perspective_check.toggled.connect(self.update_page_perspective)
        for pair in self.corner_spins:
            for spin in pair:
                spin.valueChanged.connect(self.update_page_perspective)
        self.offset_x_spin.setToolTip("自动位置上的追加偏移；初始为 0，正数向右。")
        self.offset_y_spin.setToolTip("自动位置上的追加偏移；初始为 0，正数向上。")
        self.reset_pose_button = QPushButton("重置本页微调")
        self.reset_pose_button.clicked.connect(self.reset_smart_pose)
        self.reanalyze_page_button = QPushButton("重新分析本页")
        self.reanalyze_page_button.setToolTip(
            "重扫替换原图后使用：重新检测本页纸边，按整批尺寸重新推荐位置、角度"
            "（及梯形校正），并清除本页微调。其余页面不变。"
        )
        self.reanalyze_page_button.clicked.connect(self.reanalyze_current_page)
        self.page_buttons = QWidget()
        page_buttons_row = QHBoxLayout(self.page_buttons)
        page_buttons_row.setContentsMargins(0, 0, 0, 0)
        page_buttons_row.addWidget(self.reset_pose_button)
        page_buttons_row.addWidget(self.reanalyze_page_button)
        offset_layout.addRow(self.page_buttons)
        controls_layout.addWidget(offset_group)

        self.geometry_label = QLabel("请先分析扫描目录", self)
        self.geometry_label.hide()
        self.geometry_label.setWordWrap(True)
        self.geometry_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        nav_row = QHBoxLayout()
        self.previous_button = QPushButton("◀ 上一页")
        self.page_jump_spin = QSpinBox()
        self.page_jump_spin.setRange(1, 1)
        self.page_jump_spin.setSuffix(" / 0")
        self.page_jump_spin.setKeyboardTracking(False)
        self.page_jump_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_button = QPushButton("下一页 ▶")
        self.previous_button.clicked.connect(self.previous_page)
        self.page_jump_spin.valueChanged.connect(self.jump_to_page)
        self.next_button.clicked.connect(self.next_page)
        nav_row.addWidget(self.previous_button)
        nav_row.addWidget(self.page_jump_spin)
        nav_row.addWidget(self.next_button)
        controls_layout.addLayout(nav_row)

        self.page_label = QLabel("0 / 0")
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.page_label)

        self.export_button = QPushButton("导出全部裁切图")
        self.export_button.setMinimumHeight(30)
        self.export_button.clicked.connect(self.export_all)
        self.export_selected_button = QPushButton("导出指定页码…")
        self.export_selected_button.setMinimumHeight(30)
        self.export_selected_button.clicked.connect(self.export_selected)
        self.export_config_button = QPushButton("导出批次配置…")
        self.export_config_button.setMinimumHeight(30)
        self.export_config_button.setToolTip(
            "保存整批裁量与每页角度、偏移、宽高增量和梯形参数，可通过“批次 → 导入批次配置…”恢复。"
        )
        self.export_config_button.setEnabled(False)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_button)
        export_row.addWidget(self.export_selected_button)
        export_row.addWidget(self.export_config_button)
        controls_layout.addLayout(export_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        for form in (paths_layout, smart_layout, crop_layout, offset_layout):
            form.setVerticalSpacing(4)
        splitter.addWidget(controls)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        self.preview_view = PreviewView()
        self.preview_view.crop_dragged.connect(self.drag_current_crop)
        self.preview_view.level_line_picked.connect(self.apply_level_line)
        self.preview_view.rotate_started.connect(self.start_mouse_rotation)
        self.preview_view.rotate_moved.connect(self.preview_mouse_rotation)
        self.preview_view.rotate_finished.connect(self.finish_mouse_rotation)
        self.preview_view.edge_dragged.connect(self.drag_current_edge)
        self.preview_view.corner_dragged.connect(self.drag_current_corner)
        self.preview_view.corner_started.connect(self.start_corner_drag)
        self.preview_view.corner_moved.connect(self.preview_corner_drag)
        self.preview_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_view.setMinimumSize(QSize(500, 600))
        self.update_preview_background()
        preview_layout.addWidget(self.preview_view, 1)
        preview_help = QLabel(
            "滚轮缩放 · 左键拖拽平移 · Ctrl + 左键拖拽裁切框 · 双击适合窗口　"
            "空格切换完整图像与裁切预览　"
            "3× 像素显示 · 6× 像素网格　"
            "方向键 / WAXD 每次移动裁切框 1px · Q / E、PageUp / PageDown 翻页"
        )
        preview_help.setToolTip(preview_help.text())
        preview_help.setText(
            "Ctrl+拖框 · Ctrl+框外拖动旋转 · Shift+QWE/AD/ZXC 定位 · Shift+S 全图 · 重按回退 · Alt+WAXD 单边／Alt+S 四边 · Q/E 翻页"
        )
        preview_help.setWordWrap(True)
        preview_layout.addWidget(preview_help)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(1, 1)

        for widget in (
            self.top_spin,
            self.bottom_spin,
            self.outer_spin,
            self.inner_spin,
        ):
            widget.valueChanged.connect(self.update_crop_settings)
        self.offset_x_spin.valueChanged.connect(self.update_current_adjustment)
        self.offset_y_spin.valueChanged.connect(self.update_current_adjustment)
        self.rotation_spin.valueChanged.connect(self.update_current_adjustment)
        self.width_delta_spin.valueChanged.connect(self.update_current_adjustment)
        self.height_delta_spin.valueChanged.connect(self.update_current_adjustment)
        self.inner_mode.currentIndexChanged.connect(self.mode_changed)
        self.mode_side_combo.currentIndexChanged.connect(self.selection_mode_changed)
        self.alternate_check.toggled.connect(self.selection_mode_changed)
        self.fixed_width_check.toggled.connect(self.selection_mode_changed)
        self.background_combo.currentIndexChanged.connect(
            self.update_preview_background
        )
        self.color_correction_check.toggled.connect(
            self.update_color_correction_enabled
        )
        self.color_profile_combo.currentIndexChanged.connect(
            self.load_selected_color_profile
        )
        for spin in self.smart_crop_spins():
            spin.valueChanged.connect(self.update_smart_crop)
        self._set_controls_enabled(False)

    def selection_mode_changed(self, *_):
        side = str(self.mode_side_combo.currentData())
        mode = ("alternate-" if self.alternate_check.isChecked() else "") + side
        if self.fixed_width_check.isChecked():
            mode = "bound-" + mode
        self.inner_mode.setCurrentIndex(self.inner_mode.findData(mode))

    def update_mode_controls(self):
        mode = str(self.inner_mode.currentData())
        side = "left" if mode.endswith("left") else "right"
        with QSignalBlocker(self.mode_side_combo), QSignalBlocker(
            self.alternate_check
        ), QSignalBlocker(self.fixed_width_check):
            self.mode_side_combo.setCurrentIndex(self.mode_side_combo.findData(side))
            self.alternate_check.setChecked("alternate-" in mode)
            self.fixed_width_check.setChecked(mode.startswith("bound-"))
        bound = str(self.inner_mode.currentData()).startswith("bound-")
        if self.smart_pages and not self.mode_follows_ui:
            # Mode changes apply at the next 分析文件夹. Until then the batch
            # spins keep the meaning of the analyzed mode (inner trim versus
            # image width), so the current crop is not reinterpreted.
            bound = self.smart_bound
        self.perspective_auto_check.setEnabled(self.smart_worker is None)
        if bound != self.mode_control_bound:
            value = self.crop_inner_spin.value()
            if bound:
                self.saved_inner_trim = value
            else:
                self.bound_width = value
            with QSignalBlocker(self.crop_inner_spin):
                self.crop_inner_spin.setRange(
                    16 if bound else SPIN_INT_MIN, 100000 if bound else SPIN_INT_MAX
                )
                self.crop_inner_spin.setValue(
                    self.bound_width if bound else self.saved_inner_trim
                )
            self.mode_control_bound = bound
        self.smart_form.labelForField(self.crop_inner_spin).setText(
            "图像宽度" if bound else "内侧裁掉"
        )
        self.crop_inner_spin.setToolTip(
            "指定输出图像宽度；按所选内侧方向，以另一边的外侧定位。"
            if bound
            else "自动结果上的追加内侧裁量；0 不追加，正数向内裁，负数向外扩。"
        )

    def applied_inner_mode(self) -> str:
        """The mode the current crop results were computed with."""
        if self.smart_pages and self.smart_applied_settings:
            return str(self.smart_applied_settings[0])
        return str(self.inner_mode.currentData())

    def mode_changed(self, *_):
        self.update_mode_controls()
        if self.smart_pages:
            # Never re-analyze automatically; only 分析文件夹 applies it.
            pending = str(self.inner_mode.currentData()) != self.applied_inner_mode()
            self.status_label.setText(
                "处理模式已更改：点击“分析文件夹”后生效，当前结果保持不变"
                if pending
                else "处理模式与当前结果一致"
            )
            return
        self.clear_crop_undo()
        self.undo_batch_values = self.smart_crop_values()
        if str(self.inner_mode.currentData()).startswith("bound-"):
            self.smart_group.setVisible(True)
            self.smart_group.setEnabled(True)
        else:
            self.update_crop_settings()

    def smart_crop_spins(self):
        return (
            self.crop_top_spin,
            self.crop_bottom_spin,
            self.crop_outer_spin,
            self.crop_inner_spin,
        )

    def smart_crop_values(self):
        return tuple(spin.value() for spin in self.smart_crop_spins())

    def set_smart_crop(self, paper_size, crop):
        self.smart_paper_size = tuple(paper_size)
        for spin, value in zip(self.smart_crop_spins(), crop):
            with QSignalBlocker(spin):
                spin.setValue(value)
        self.sync_smart_size()
        self.undo_batch_values = self.smart_crop_values()

    def smart_base(self, path, side=None):
        top, bottom, outer, inner = self.smart_crop_values()
        if side is None:
            side = self.smart_plans[path].inner
        x, y = self.smart_origins[path]
        if self.smart_bound:
            return (
                x + self.smart_paper_size[0] - inner - outer
                if side == "left"
                else x + outer
            ), y + top
        return x + (inner if side == "left" else outer), y + top

    def smart_preferences_changed(self, *_):
        if self.smart_pages:
            if self.smart_worker is None:
                self.export_button.setEnabled(False)
                self.export_selected_button.setEnabled(False)
                self.export_batch_config_action.setEnabled(False)
                self.smart_timer.start()
                self.status_label.setText("正在按新设置重算；排除页保留手调")

    def update_smart_crop(self, *_):
        values = self.smart_crop_values()
        if self.smart_pages and self.undo_batch_values is not None and values != self.undo_batch_values:
            self.push_crop_undo(('batch', self.undo_batch_values, None))
        self.undo_batch_values = values
        # Follow what the spin currently means, not a pending mode change.
        if self.mode_control_bound:
            self.bound_width = self.crop_inner_spin.value()
        if not self.smart_paper_size or not self.smart_pages:
            return
        self.sync_smart_size()
        valid = min(self.smart_size) >= 16 and max(self.smart_size) <= 100000
        ready = valid and self.smart_worker is None and not self.smart_timer.isActive()
        self.export_button.setEnabled(ready)
        self.export_selected_button.setEnabled(ready)
        self.export_batch_config_action.setEnabled(ready)
        self.refresh_preview()
        self.status_label.setText(
            "已更新裁剪微调" if valid else "裁剪宽高需在 16～100000 像素之间"
        )

    def reset_smart_pose(self):
        if not self.smart_pages or not self.detections or self.smart_worker:
            return
        path = self.detections[self.current_index].path
        self.record_page_undo(path)
        self.page_offsets[path] = (0, 0)
        self.page_rotations[path] = self.smart_plans[path].angle
        self.page_size_deltas[path] = (0, 0)
        if path in self.perspectives:
            state = deepcopy(self.perspectives[path])
            state["offsets"] = [[0.0, 0.0] for _ in range(4)]
            state["enabled"] = bool(state.get("recommended_enabled", False))
            self.apply_perspective_state(state, record_undo=False)
        self.invalidate_full_preview_cache(path)
        self.load_current_preview()

    def reanalyze_current_page(self):
        if (
            not self.smart_pages
            or not self.detections
            or self.smart_worker is not None
            or self.page_reanalysis_worker is not None
            or not self.smart_paper_size
        ):
            return
        item = self.detections[self.current_index]
        base = (
            Path(sys.executable).parent
            if getattr(sys, "frozen", False)
            else Path(__file__).parent
        )
        excluded = item.path in self.smart_manual_pages
        # Alternation counts analyzed pages only.
        body_index = sum(
            1 for d in self.detections[: self.current_index]
            if d.path not in self.smart_manual_pages
        )
        worker = ReanalyzePageWorker(
            item.path,
            body_index,
            self.applied_inner_mode(),
            self.smart_paper_size,
            *self.applied_analysis_flags,
            base / ".scan-cropper-cache",
            [p for path, p in self.raw_smart_pages.items()
             if path != item.path and path not in self.smart_manual_pages],
            excluded,
        )
        worker.signals.finished.connect(self.page_reanalysis_finished)
        worker.signals.error.connect(self.page_reanalysis_error)
        self.page_reanalysis_worker = worker
        self.reanalyze_page_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.status_label.setText(f"正在重新分析 {item.path.name}…")
        self.thread_pool.start(worker)

    def page_reanalysis_done(self):
        self.page_reanalysis_worker = None
        self.analyze_button.setEnabled(self.smart_worker is None)
        self.reanalyze_page_button.setEnabled(
            bool(self.smart_pages) and self.offset_x_spin.isEnabled()
        )

    def page_reanalysis_error(self, details):
        self.page_reanalysis_done()
        self.status_label.setText("重新分析本页失败；保留原有结果")
        QMessageBox.warning(self, "重新分析本页失败", details)

    def page_reanalysis_finished(self, result):
        self.page_reanalysis_done()
        path, raw, warped, plan, state, origin = result
        index = next((i for i, d in enumerate(self.detections) if d.path == path), None)
        if index is None or path not in self.smart_pages:
            return
        # Analysis results changed underneath the history; as after a full
        # analysis, undo starts afresh.
        self.clear_crop_undo()
        self.raw_smart_pages[path] = raw
        self.smart_pages[path] = warped
        self.smart_plans[path] = plan
        if state is not None:
            self.perspectives[path] = state
        else:
            self.perspectives.pop(path, None)
        self.smart_origins[path] = origin
        self.smart_default_offsets[path] = (plan.box[0] - origin[0], origin[1] - plan.box[1])
        self.detections[index] = Detection(
            path,
            raw.width,
            raw.height,
            statistics.median(p[0] for p in raw.edges["left"]),
            statistics.median(p[0] for p in raw.edges["right"]),
            statistics.median(p[1] for p in raw.edges["bottom"]),
            0.0,
            raw.roughness["left"],
            raw.roughness["right"],
        )
        self.inferred_paths.discard(path)
        self.page_offsets[path] = (0, 0)
        self.page_rotations[path] = plan.angle
        self.page_size_deltas[path] = (0, 0)
        # The file itself may have been replaced: drop every cached pixel.
        PREVIEW_SOURCES.discard(path)
        self.preview_drafts.pop(path, None)
        self.invalidate_full_preview_cache(path)
        if self.displayed_preview is not None and self.displayed_preview[0] == path:
            self.displayed_preview = None
        if index == self.current_index:
            self.preview_path = None
            self.load_current_preview()
        self.status_label.setText(
            f"已重新分析 {path.name}：按整批尺寸重新推荐，本页微调已清除"
        )

    def clear_smart(self):
        self.clear_crop_undo()
        self.smart_timer.stop()
        self.smart_pages = {}
        self.smart_plans = {}
        self.smart_size = None
        self.smart_paper_size = None
        self.smart_origins = {}
        self.smart_symmetric = False
        self.smart_bound = False
        self.smart_default_offsets = {}
        self.smart_manual_pages = set()
        self.smart_applied_settings = None

    def start_smart(self, paths, pages=None, size=None, mode=None):
        try:
            excluded = (
                {
                    paths[i - 1]
                    for i in parse_page_numbers(
                        self.excluded_pages_edit.text(), len(paths)
                    )
                }
                if pages is None
                else self.smart_manual_pages
            )
        except ValueError as exc:
            QMessageBox.warning(self, "排除页码无效", str(exc))
            return
        self.smart_timer.stop()
        self.smart_is_new = pages is None
        self._set_controls_enabled(False)
        self.analyze_button.setEnabled(False)
        self.auto_check.setEnabled(False)
        self.perspective_auto_check.setEnabled(False)
        self.excluded_pages_edit.setEnabled(False)
        self.input_edit.setEnabled(False)
        self.import_batch_config_action.setEnabled(False)
        self.export_batch_config_action.setEnabled(False)
        self.detect_color_shift_action.setEnabled(False)
        self.cancel_auto_button.setEnabled(True)
        self.cancel_auto_button.show()
        base = (
            Path(sys.executable).parent
            if getattr(sys, "frozen", False)
            else Path(__file__).parent
        )
        mode = str(self.inner_mode.currentData()) if mode is None else mode
        margins = self.smart_crop_values()
        if mode.startswith("bound-"):
            self.bound_width = self.crop_inner_spin.value()
            margins = margins[:3] + (0,)
        worker = AutoCropWorker(
            paths,
            pages,
            size,
            mode,
            margins,
            "balanced",
            base / ".scan-cropper-cache",
            excluded,
            self.bound_width if mode.startswith("bound-") else None,
            auto_crop=self.auto_check.isChecked(),
            auto_perspective=self.perspective_auto_check.isChecked(),
            perspectives=None if pages is None else self.perspectives,
        )
        worker.signals.progress.connect(self.worker_progress)
        worker.signals.finished.connect(self.smart_finished)
        worker.signals.error.connect(self.smart_error)
        self.smart_worker = worker
        self.inner_mode.setEnabled(False)
        self.mode_side_combo.setEnabled(False)
        self.alternate_check.setEnabled(False)
        self.fixed_width_check.setEnabled(False)
        self.smart_group.setEnabled(False)
        self.thread_pool.start(worker)

    def recalculate_smart(self, *_):
        if not self.smart_pages or self.smart_worker is not None:
            return
        top, bottom, outer, inner = self.smart_crop_values()
        width, height = self.smart_paper_size
        # Recalculate with the analyzed mode; a pending mode change needs
        # 分析文件夹 so that detection and the batch spins match it.
        mode = self.applied_inner_mode()
        if mode.startswith("bound-"):
            size = (inner, height - top - bottom)
        else:
            size = (
                width
                - outer
                - (outer if mode == "none" else inner),
                height - top - bottom,
            )
        if not size or min(size) < 16 or max(size) > 100000:
            self.status_label.setText("四边裁量使尺寸无效，无法重算")
            return
        self.start_smart(
            list(self.smart_pages),
            [
                self.raw_smart_pages.get(p, self.smart_pages[p])
                for p in self.smart_pages
            ],
            size,
            mode,
        )

    def cancel_smart(self):
        if self.smart_worker:
            self.smart_worker.cancel.set()
            self.cancel_auto_button.setEnabled(False)
            self.status_label.setText("正在取消，保留上次完成的方案…")

    def finish_smart_controls(self):
        self.smart_worker = None
        self.mode_follows_ui = False
        self.cancel_auto_button.setEnabled(False)
        self.cancel_auto_button.hide()
        self.analyze_button.setEnabled(True)
        self.auto_check.setEnabled(True)
        self.perspective_auto_check.setEnabled(True)
        self.excluded_pages_edit.setEnabled(True)
        self.input_edit.setEnabled(True)
        self.import_batch_config_action.setEnabled(True)
        self.export_batch_config_action.setEnabled(bool(self.detections))
        self.detect_color_shift_action.setEnabled(bool(self.detections))
        self._set_controls_enabled(bool(self.detections))

    def sync_smart_size(self):
        if self.smart_paper_size:
            top, bottom, outer, inner = self.smart_crop_values()
            pw, ph = self.smart_paper_size
            self.smart_size = (
                (
                    inner
                    if self.smart_bound
                    else pw - outer - (outer if self.smart_symmetric else inner)
                ),
                ph - top - bottom,
            )
            self.smart_size_label.setText(
                f"输出：{self.smart_size[0]} × {self.smart_size[1]} px"
            )

    def restore_applied_smart_settings(self):
        self.sync_smart_size()
        if self.smart_applied_settings:
            mode, margins, policy = self.smart_applied_settings
            with QSignalBlocker(self.inner_mode):
                self.inner_mode.setCurrentIndex(self.inner_mode.findData(mode))
            self.update_mode_controls()
            self.sync_smart_size()

    def smart_finished(self, result):
        was_new = self.smart_is_new
        worker = self.smart_worker
        self.finish_smart_controls()
        if result is None:
            self.restore_applied_smart_settings()
            if self.detections:
                self.input_edit.setText(str(self.detections[0].path.parent))
            self.status_label.setText("已取消；保留上次完成的裁剪方案")
            return
        paths, pages, size, plans = result
        self.clear_crop_undo()
        if worker is not None and was_new:
            self.applied_analysis_flags = (worker.auto_crop, worker.auto_perspective)
        if worker is not None:
            self.perspectives = worker.perspectives
            self.raw_smart_pages = dict(zip(paths, worker.raw_pages))
        elif was_new:
            self.perspectives = {}
            self.raw_smart_pages = dict(zip(paths, pages))
        self.smart_applied_settings = (
            (worker.mode, worker.margins, worker.policy)
            if worker
            else (
                str(self.inner_mode.currentData()),
                self.smart_crop_values(),
                "balanced",
            )
        )
        self.smart_pages = dict(zip(paths, pages))
        mode = worker.mode if worker else str(self.inner_mode.currentData())
        previous_bound = self.smart_bound
        self.smart_symmetric = mode == "none"
        self.smart_bound = mode.startswith("bound-")
        if was_new:
            # Analyze a zero-trim baseline, then apply the current session's
            # adjustments without making them part of the automatic offsets.
            _, origins = paper_reference(
                pages, mode, size, [p.angle for p in plans], [p.inner for p in plans]
            )
            self.smart_origins = dict(zip(paths, origins))
            trims = worker.requested_margins if worker else self.smart_crop_values()
            self.set_smart_crop(
                size,
                (
                    trims[0],
                    trims[1],
                    trims[2],
                    size[0] if self.smart_bound else trims[3],
                ),
            )
        else:
            if previous_bound != self.smart_bound:
                top, bottom, outer, inner = self.smart_crop_values()
                # Keep the original automatic baseline height while switching models.
                self.smart_paper_size = (
                    (
                        size[0]
                        if self.smart_bound
                        else size[0]
                        + outer
                        + (outer if self.smart_symmetric else inner)
                    ),
                    self.smart_paper_size[1],
                )
                _, origins = paper_reference(
                    pages,
                    mode,
                    self.smart_paper_size,
                    [p.angle for p in plans],
                    [p.inner for p in plans],
                )
                for path, origin in zip(paths, origins):
                    if path not in self.smart_manual_pages:
                        self.smart_origins[path] = origin
            self.sync_smart_size()
        if was_new:
            self.smart_manual_pages = set(worker.excluded_paths) if worker else set()
            self.smart_default_offsets = {}
        for path, plan in zip(paths, plans):
            if not was_new and path in self.smart_manual_pages:
                continue
            x, y = (
                self.smart_origins[path]
                if was_new
                else self.smart_base(path, plan.inner)
            )
            self.smart_default_offsets[path] = (plan.box[0] - x, y - plan.box[1])
        if was_new:
            self.smart_plans = dict(zip(paths, plans))
            detections = []
            for path, page in zip(paths, pages):
                left = statistics.median(p[0] for p in page.edges["left"])
                right = statistics.median(p[0] for p in page.edges["right"])
                bottom = statistics.median(p[1] for p in page.edges["bottom"])
                detections.append(
                    Detection(
                        path,
                        page.width,
                        page.height,
                        left,
                        right,
                        bottom,
                        0.0,
                        page.roughness["left"],
                        page.roughness["right"],
                    )
                )
            self.input_edit.setText(str(paths[0].parent))
            self._activate_imported_batch(
                detections,
                {p: (0, 0) for p in paths},
                {p: v.angle for p, v in zip(paths, plans)},
                {p: (0, 0) for p in paths},
                set(),
                0,
            )
        else:
            for path, plan in zip(paths, plans):
                if path in self.smart_manual_pages:
                    continue
                self.smart_plans[path] = plan
                self.page_offsets[path] = (0, 0)
                self.page_rotations[path] = plan.angle
            self.invalidate_full_preview_cache()
            self.load_current_preview()
        self._set_controls_enabled(True)
        self.status_label.setText(
            f"已载入 {len(paths)} 页；{len(self.smart_manual_pages)} 页排除自动分析、保留手调"
        )

    def smart_error(self, details):
        self.finish_smart_controls()
        self.restore_applied_smart_settings()
        if self.detections:
            self.input_edit.setText(str(self.detections[0].path.parent))
        self.status_label.setText("自动推荐失败；保留上次方案")
        QMessageBox.warning(self, "自动推荐失败", details)

    def smart_note(self, path):
        if path not in self.smart_plans:
            return ""
        if path in self.smart_manual_pages:
            return "\n排除页：未自动分析；手动补正后与正文一起导出"
        plan = self.smart_plans[path]
        prefix = "最近一次自动计算的参考值（不随手调裁量更新）"
        warnings = "；".join(plan.warnings) or "未触发几何警告，仍需抽查内容"
        return (
            f"\n{prefix}：边界检测评分 {plan.confidence:.0%}\n"
            f"估计最宽缺边 {plan.max_outside:.0f}px；最深额外裁入 {plan.max_trim:.0f}px\n{warnings}"
        )

    def smart_config(self):
        if not self.smart_size:
            return None
        return {
            "size": self.smart_size,
            "paper_size": self.smart_paper_size,
            "crop": self.smart_crop_values(),
            "symmetric": self.smart_symmetric,
            "bound": self.smart_bound,
            "offset_mode": "delta",
            "policy": "balanced",
            "excluded_pages": self.excluded_pages_edit.text(),
            "pages": {
                p.name: {
                    "analysis": a.to_dict(),
                    "plan": self.smart_plans[p].to_dict(),
                    "manual": p in self.smart_manual_pages,
                    "default_offset": self.smart_default_offsets[p],
                    "origin": self.smart_origins[p],
                }
                for p, a in self.smart_pages.items()
            },
        }

    def restore_smart_config(self, raw, detections):
        self.smart_timer.stop()
        pages, plans, manual = {}, {}, set()
        if raw is None:
            self.clear_smart()
            return
        if not isinstance(raw, dict) or raw.get("policy") not in (
            "balanced",
            "less_fill",
            "preserve",
        ):
            raise ValueError("自动裁剪配置无效")
        size = tuple(int(x) for x in raw["size"])
        if len(size) != 2 or min(size) < 16 or max(size) > 100000:
            raise ValueError("自动裁剪宽高无效")
        for detection in detections:
            record = raw["pages"][detection.path.name]
            analysis = PageAnalysis.from_dict(record["analysis"])
            plan = CropPlan.from_dict(record["plan"])
            if analysis.name != detection.path.name or (
                analysis.width,
                analysis.height,
            ) != (detection.source_width, detection.source_height):
                raise ValueError("自动边界与原图不匹配")
            pages[detection.path] = analysis
            plans[detection.path] = plan
            if record.get("manual", False):
                manual.add(detection.path)
        mode = str(self.inner_mode.currentData())
        symmetric = bool(raw.get("symmetric", mode == "none"))
        bound = bool(raw.get("bound", mode.startswith("bound-")))
        self.update_mode_controls()
        shifts = {}
        if "paper_size" in raw:
            paper_size = tuple(int(v) for v in raw["paper_size"])
            crop = tuple(int(v) for v in raw["crop"])
            if (
                len(paper_size) != 2
                or min(paper_size) < 16
                or max(paper_size) > 100000
                or len(crop) != 4
                or any(v < SPIN_INT_MIN or v > SPIN_INT_MAX for v in crop)
                or (
                    (
                        crop[3]
                        if bound
                        else paper_size[0]
                        - crop[2]
                        - (crop[2] if symmetric else crop[3])
                    ),
                    paper_size[1] - crop[0] - crop[1],
                )
                != size
            ):
                raise ValueError("纸边裁剪配置无效")
            defaults = {
                d.path: tuple(
                    int(v) for v in raw["pages"][d.path.name]["default_offset"]
                )
                for d in detections
            }
            origins = {
                d.path: tuple(int(v) for v in raw["pages"][d.path.name]["origin"])
                for d in detections
            }
        else:
            # Versions 5 and 6 use different offset origins. Preserve both the
            # current pose and the stored default pose when changing coordinates.
            paper_size, _ = paper_reference(
                list(pages.values()), mode, angles=[p.angle for p in plans.values()]
            )
            dw, dh = paper_size[0] - size[0], paper_size[1] - size[1]
            if symmetric and dw % 2:
                paper_size = (paper_size[0] + 1, paper_size[1])
                dw += 1
            outer, top = round(dw / 2), round(dh / 2)
            crop = (top, dh - top, outer, dw - outer)
            _, anchors = paper_reference(
                list(pages.values()),
                mode,
                paper_size,
                [p.angle for p in plans.values()],
            )
            origins = dict(zip(pages, anchors))
            old_crop = raw.get("crop")
            if "canvas" in raw:
                canvas = tuple(int(v) for v in raw["canvas"])
                if (
                    len(canvas) != 2
                    or not isinstance(old_crop, (list, tuple))
                    or len(old_crop) != 4
                    or (
                        canvas[0] - old_crop[2] - old_crop[3],
                        canvas[1] - old_crop[0] - old_crop[1],
                    )
                    != size
                ):
                    raise ValueError("旧版画布裁剪配置无效")
            defaults = {}
            for path, plan in plans.items():
                x = origins[path][0] + (crop[3] if plan.inner == "left" else crop[2])
                y = origins[path][1] + crop[0]
                if "canvas" in raw:
                    old_x, old_y = old_crop[2], old_crop[0]
                    old_default = raw["pages"][path.name]["default_offset"]
                else:
                    old_x, old_y = plan.box[:2]
                    old_default = (0, 0)
                shifts[path] = (old_x - x, y - old_y)
                defaults[path] = (
                    old_default[0] + shifts[path][0],
                    old_default[1] + shifts[path][1],
                )
        if any(
            len(v) != 2 or any(x < SPIN_INT_MIN or x > SPIN_INT_MAX for x in v)
            for v in list(defaults.values()) + list(origins.values())
        ):
            raise ValueError("默认偏移或纸张基准配置无效")
        if raw.get("offset_mode") != "delta":
            shifts = {
                p: (
                    shifts.get(p, (0, 0))[0] - defaults[p][0],
                    shifts.get(p, (0, 0))[1] - defaults[p][1],
                )
                for p in pages
            }
        self.smart_pages, self.smart_plans, self.smart_manual_pages = (
            pages,
            plans,
            manual,
        )
        self.smart_default_offsets, self.smart_origins = defaults, origins
        self.smart_symmetric = symmetric
        self.smart_bound = bound
        self.set_smart_crop(paper_size, crop)
        self.excluded_pages_edit.setText(
            str(
                raw.get(
                    "excluded_pages", "1-2" if raw.get("skip_covers", False) else ""
                )
            )
        )
        self.smart_applied_settings = (
            str(self.inner_mode.currentData()),
            self.smart_crop_values(),
            "balanced",
        )
        return shifts

    def _spinbox(self, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(SPIN_INT_MIN, SPIN_INT_MAX)
        box.setValue(value)
        box.setSuffix(" px")
        box.setKeyboardTracking(False)
        return box

    def add_control_menu(self, menu, title, widget):
        submenu = menu.addMenu(title)
        action = QWidgetAction(submenu)
        action.setDefaultWidget(widget)
        submenu.addAction(action)
        return submenu

    def _build_menus(self) -> None:
        self.batch_menu = self.menuBar().addMenu("批次")
        edit_menu = self.menuBar().addMenu("编辑")
        self.undo_crop_action = edit_menu.addAction("撤销裁剪调整（Ctrl+Z）")
        self.undo_crop_action.setEnabled(False)
        self.undo_crop_action.triggered.connect(self.undo_crop_adjustment)
        self.redo_crop_action = edit_menu.addAction("重做裁剪调整（Ctrl+Shift+Z）")
        self.redo_crop_action.setEnabled(False)
        self.redo_crop_action.triggered.connect(self.redo_crop_adjustment)
        self.export_batch_config_action = self.batch_menu.addAction("导出当前批次配置…")
        self.import_batch_config_action = self.batch_menu.addAction("导入批次配置…")
        self.export_batch_config_action.setEnabled(False)
        self.export_batch_config_action.triggered.connect(self.export_batch_config)
        # The button mirrors the menu action wherever its state is updated.
        self.export_config_button.clicked.connect(self.export_batch_config_action.trigger)
        self.export_batch_config_action.changed.connect(
            lambda: self.export_config_button.setEnabled(
                self.export_batch_config_action.isEnabled()
            )
        )
        self.import_batch_config_action.triggered.connect(self.import_batch_config)

        self.batch_menu.addSeparator()
        self.batch_menu.addAction(self.recalculate_action)
        view_menu = self.menuBar().addMenu("视图")
        self.add_control_menu(view_menu, "背景颜色", self.background_combo)
        view_menu.addAction(self.show_edges_check)
        view_menu.addAction("扫边与定位设置…").triggered.connect(
            self.edit_inspection_settings
        )
        view_menu.addAction("当前页详细信息…").triggered.connect(
            lambda: QMessageBox.information(
                self, "当前页详细信息", self.geometry_label.text()
            )
        )
        page_menu = self.menuBar().addMenu("本页")
        page_menu.addAction("重置本页微调").triggered.connect(self.reset_smart_pose)
        self.color_menu = self.menuBar().addMenu("颜色校正")
        self.color_menu.addAction(self.color_correction_check)
        self.add_control_menu(self.color_menu, "选择 Profile", self.color_profile_combo)
        self.edit_color_profiles_action = self.color_menu.addAction("编辑 Profiles…")
        self.edit_color_profiles_action.triggered.connect(self.edit_color_profiles)
        self.detect_color_shift_action = self.color_menu.addAction(
            "按页码范围共同标定 RGB 偏移…"
        )
        self.detect_color_shift_action.setEnabled(False)
        self.detect_color_shift_action.triggered.connect(
            self.detect_current_color_shift
        )

    def edit_inspection_settings(self):
        self.stop_edge_sweep()
        self.edge_jump_pending = None
        dialog = QDialog(self)
        dialog.setWindowTitle("扫边与定位设置")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        values = []
        for key, label, minimum, maximum, suffix in [
            ("sweep_short_duration", "短边移动时长", 0.1, 60.0, " 秒"),
            ("sweep_start_delay", "起点停留／启动延迟", 0.0, 10.0, " 秒"),
            ("inspection_scale", "扫边和 Shift 定位倍率", 0.1, 16.0, " ×"),
        ]:
            spin = QDoubleSpinBox(dialog)
            spin.setObjectName(key)
            spin.setRange(minimum, maximum)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setSuffix(suffix)
            spin.setValue(getattr(self, key))
            form.addRow(label, spin)
            values.append((key, spin))
        layout.addLayout(form)
        note = QLabel(
            "移动时长不含启动延迟；长边按实际长度比例增加时长。\n倍率按实际屏幕像素计算，2× 表示 1 图片像素占 2 屏幕像素。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for key, spin in values:
                setattr(self, key, spin.value())
            self.settings_save_timer.start()
            self.status_label.setText("已保存扫边与定位设置；下次操作生效")

    def _persist_color_profiles(self) -> bool:
        try:
            write_color_profiles(
                self.color_profiles,
                self.color_profile_combo.currentText(),
                self.color_correction_check.isChecked(),
            )
            return True
        except OSError as exc:
            QMessageBox.warning(
                self,
                "无法保存颜色 Profile",
                f"无法写入：\n{color_profile_path()}\n\n{exc}",
            )
            return False

    def load_selected_color_profile(
        self,
        _index: int | None = None,
        *,
        persist: bool = True,
    ) -> None:
        name = self.color_profile_combo.currentText()
        if name in self.color_profiles and persist:
            self._persist_color_profiles()
        self.refresh_color_preview()

    def update_color_correction_enabled(self, enabled: bool) -> None:
        self.color_profile_combo.setEnabled(enabled)
        self._persist_color_profiles()
        self.refresh_color_preview()

    def refresh_color_preview(self):
        if hasattr(self, "settings_save_timer"):
            self.settings_save_timer.start()
        if self.detections and self.preview_image is not None:
            self.stop_edge_sweep()
            self.invalidate_full_preview_cache()
            self.refresh_preview()

    def edit_color_profiles(self) -> None:
        dialog = ColorProfileDialog(
            self.color_profiles,
            self.color_profile_combo.currentText(),
            self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.color_profiles = dialog.profiles
        selected = dialog.selected_profile
        with QSignalBlocker(self.color_profile_combo):
            self.color_profile_combo.clear()
            self.color_profile_combo.addItems(list(self.color_profiles))
            self.color_profile_combo.setCurrentText(selected)
        if self._persist_color_profiles():
            self.status_label.setText(f"已保存颜色 Profile：{selected}")
        self.refresh_color_preview()

    def current_channel_shifts(
        self,
    ) -> dict[str, tuple[float, float]] | None:
        if not self.color_correction_check.isChecked():
            return None
        values = self.color_profiles.get(self.color_profile_combo.currentText())
        if values is None:
            return None
        shifts = {
            channel: (
                values.get(f"{channel}_x", 0.0),
                values.get(f"{channel}_y", 0.0),
            )
            for channel in PROFILE_CHANNELS
        }
        if all(
            abs(axis) < 1e-9
            for channel_shift in shifts.values()
            for axis in channel_shift
        ):
            return None
        return shifts

    def export_batch_config(self) -> None:
        if self.smart_worker is not None or self.smart_timer.isActive():
            return
        if not self.detections:
            QMessageBox.warning(self, "没有批次", "请先分析扫描目录。")
            return
        input_dir = self.detections[0].path.parent
        default_path = input_dir / "scan-cropper-batch.json"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前批次配置",
            str(default_path),
            "JSON 配置 (*.json)",
        )
        if not filename:
            return
        selected_profile = self.color_profile_combo.currentText()
        payload = {
            "format": "scan-cropper-batch",
            "version": BATCH_CONFIG_VERSION,
            "input_dir": str(input_dir.resolve()),
            "output_dir": self.output_edit.text().strip(),
            "current_index": self.current_index,
            "auto_crop": self.smart_config(),
            "crop_settings": {
                "inner_mode": self.applied_inner_mode(),
                "preview_background": str(self.background_combo.currentData()),
                "top": self.top_spin.value(),
                "bottom": self.bottom_spin.value(),
                "outer": self.outer_spin.value(),
                "inner": self.inner_spin.value(),
            },
            "color": {
                "enabled": self.color_correction_check.isChecked(),
                "profile_name": selected_profile,
                "profile": self.color_profiles.get(selected_profile, {}).copy(),
            },
            "pages": [],
        }
        pages: list[dict[str, object]] = []
        for item in self.detections:
            offset_x, offset_y = self.page_offsets.get(item.path, (0, 0))
            pages.append(
                {
                    "name": item.path.name,
                    "detection": {
                        "source_width": item.source_width,
                        "source_height": item.source_height,
                        "left": item.left,
                        "right": item.right,
                        "bottom": item.bottom,
                        "threshold": item.threshold,
                        "left_roughness": item.left_roughness,
                        "right_roughness": item.right_roughness,
                    },
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "rotation": self.page_rotations.get(item.path, 0.0),
                    "perspective": self.perspectives.get(item.path),
                    "raw_analysis": (
                        self.raw_smart_pages[item.path].to_dict()
                        if item.path in self.raw_smart_pages
                        else None
                    ),
                    "width_delta": self.page_size_deltas.get(item.path, (0, 0))[0],
                    "height_delta": self.page_size_deltas.get(item.path, (0, 0))[1],
                    "inferred": item.path in self.inferred_paths,
                }
            )
        payload["pages"] = pages
        destination = Path(filename)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.status_label.setText(f"批次配置已导出：{destination}")

    @staticmethod
    def _finite_config_number(value: object, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} 不是有效数字") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label} 不是有限数字")
        return number

    def import_batch_config(self) -> None:
        if self.detections:
            answer = QMessageBox.question(
                self,
                "替换当前批次？",
                "导入会替换当前窗口中的检测框和逐页微调。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "导入批次配置",
            self.input_edit.text(),
            "JSON 配置 (*.json)",
        )
        if not filename:
            return
        try:
            raw = json.loads(Path(filename).read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("配置顶层必须是对象")
            if raw.get("format") != "scan-cropper-batch":
                raise ValueError("不是扫描裁切器批次配置")
            config_version = int(raw.get("version", 0))
            if config_version not in (
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                BATCH_CONFIG_VERSION,
            ):
                raise ValueError("不支持的批次配置版本")

            input_dir = Path(str(raw.get("input_dir", "")))
            if not input_dir.is_dir():
                replacement = QFileDialog.getExistingDirectory(
                    self,
                    "原扫描目录不存在，请重新选择",
                    self.input_edit.text(),
                )
                if not replacement:
                    return
                input_dir = Path(replacement)

            raw_pages = raw.get("pages")
            if not isinstance(raw_pages, list) or not raw_pages:
                raise ValueError("配置中没有页面")
            detections: list[Detection] = []
            offsets: dict[Path, tuple[int, int]] = {}
            rotations: dict[Path, float] = {}
            size_deltas: dict[Path, tuple[int, int]] = {}
            perspectives = {}
            raw_analyses = {}
            inferred_paths: set[Path] = set()
            for index, page in enumerate(raw_pages):
                if not isinstance(page, dict):
                    raise ValueError(f"第 {index + 1} 个页面配置无效")
                name = page.get("name")
                if not isinstance(name, str) or not name or Path(name).name != name:
                    raise ValueError(f"第 {index + 1} 个页面文件名无效")
                source_path = input_dir / name
                if (
                    not source_path.is_file()
                    or source_path.suffix.lower() not in IMAGE_SUFFIXES
                ):
                    raise ValueError(f"缺少扫描文件：{name}")
                detection = page.get("detection")
                if not isinstance(detection, dict):
                    raise ValueError(f"{name} 缺少检测框")
                source_width = int(detection.get("source_width", 0))
                source_height = int(detection.get("source_height", 0))
                with Image.open(source_path) as source:
                    if source.size != (source_width, source_height):
                        raise ValueError(
                            f"{name} 尺寸已变化：配置为 "
                            f"{source_width}×{source_height}，当前为 "
                            f"{source.width}×{source.height}"
                        )
                left = self._finite_config_number(detection.get("left"), f"{name} left")
                right = self._finite_config_number(
                    detection.get("right"), f"{name} right"
                )
                bottom = self._finite_config_number(
                    detection.get("bottom"), f"{name} bottom"
                )
                if (
                    not -source_width <= left < right <= 2 * source_width
                    or not 0 < bottom <= 2 * source_height
                ):
                    raise ValueError(f"{name} 的检测框无效")
                item = Detection(
                    path=source_path,
                    source_width=source_width,
                    source_height=source_height,
                    left=left,
                    right=right,
                    bottom=bottom,
                    threshold=self._finite_config_number(
                        detection.get("threshold", 0.0),
                        f"{name} threshold",
                    ),
                    left_roughness=self._finite_config_number(
                        detection.get("left_roughness", 0.0),
                        f"{name} left_roughness",
                    ),
                    right_roughness=self._finite_config_number(
                        detection.get("right_roughness", 0.0),
                        f"{name} right_roughness",
                    ),
                )
                offset_x = int(page.get("offset_x", 0))
                offset_y = int(page.get("offset_y", 0))
                rotation = self._finite_config_number(
                    page.get("rotation", 0.0), f"{name} rotation"
                )
                width_delta = int(page.get("width_delta", 0))
                height_delta = int(page.get("height_delta", 0))
                if not all(
                    SPIN_INT_MIN <= value <= SPIN_INT_MAX
                    for value in (offset_x, offset_y, width_delta, height_delta)
                ):
                    raise ValueError(f"{name} 的整数值超出控件可表示范围")
                if config_version == 1:
                    offset_x, offset_y, rotation = -offset_x, -offset_y, -rotation
                detections.append(item)
                offsets[source_path] = (offset_x, offset_y)
                rotations[source_path] = rotation
                size_deltas[source_path] = (width_delta, height_delta)
                if page.get("perspective") is not None:
                    perspectives[source_path] = validate_state(page["perspective"])
                if page.get("raw_analysis") is not None:
                    raw_analyses[source_path] = PageAnalysis.from_dict(
                        page["raw_analysis"]
                    )
                if bool(page.get("inferred", False)):
                    inferred_paths.add(source_path)

            settings = raw.get("crop_settings", {})
            if not isinstance(settings, dict):
                raise ValueError("crop_settings 无效")
            spin_settings = (
                (self.top_spin, "top"),
                (self.bottom_spin, "bottom"),
                (self.outer_spin, "outer"),
                (self.inner_spin, "inner"),
            )
            for spin, key in spin_settings:
                value = int(settings.get(key, spin.value()))
                if not spin.minimum() <= value <= spin.maximum():
                    raise ValueError(f"{key} 超出范围")
                with QSignalBlocker(spin):
                    spin.setValue(value)
            inner_mode = str(settings.get("inner_mode", "alternate-right"))
            inner_index = self.inner_mode.findData(inner_mode)
            if inner_index < 0:
                raise ValueError("装订内侧模式无效")
            with QSignalBlocker(self.inner_mode):
                self.inner_mode.setCurrentIndex(inner_index)
            background = str(
                settings.get("preview_background", PREVIEW_BACKGROUNDS[0][1])
            )
            background_index = self.background_combo.findData(background)
            if background_index >= 0:
                with QSignalBlocker(self.background_combo):
                    self.background_combo.setCurrentIndex(background_index)
                self.update_preview_background()

            color = raw.get("color", {})
            if isinstance(color, dict):
                profile_name = str(
                    color.get("profile_name", DEFAULT_COLOR_PROFILE_NAME)
                ).strip()
                profile = normalized_color_profile(color.get("profile", {}))
                if profile_name and profile is not None:
                    self.color_profiles[profile_name] = profile
                    with QSignalBlocker(self.color_profile_combo):
                        self.color_profile_combo.clear()
                        self.color_profile_combo.addItems(list(self.color_profiles))
                        self.color_profile_combo.setCurrentText(profile_name)
                with QSignalBlocker(self.color_correction_check):
                    self.color_correction_check.setChecked(
                        bool(color.get("enabled", True))
                    )
                self._persist_color_profiles()

            self.input_edit.setText(str(input_dir))
            self.output_edit.setText(str(raw.get("output_dir", "")))
            requested_index = int(raw.get("current_index", 0))
            current_index = max(0, min(requested_index, len(detections) - 1))
            shifts = self.restore_smart_config(raw.get("auto_crop"), detections) or {}
            offsets = {
                p: (dx + shifts.get(p, (0, 0))[0], dy + shifts.get(p, (0, 0))[1])
                for p, (dx, dy) in offsets.items()
            }
            self.perspectives = perspectives
            self.raw_smart_pages = raw_analyses
            # An imported batch records no analysis switches; infer them for
            # re-analyzing single pages.
            self.applied_analysis_flags = (
                any(plan.cost or plan.confidence for plan in self.smart_plans.values()),
                any(
                    isinstance(state, dict) and state.get("automatic")
                    for state in perspectives.values()
                ),
            )
            self._activate_imported_batch(
                detections,
                offsets,
                rotations,
                size_deltas,
                inferred_paths,
                current_index,
            )
            if config_version < 4:
                self.refresh_legacy_boundaries(rotations)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.critical(self, "导入批次配置失败", str(exc))
            return
        self.status_label.setText(
            f"批次配置已导入：{filename}"
            + ("；正在后台更新旋转后边界" if config_version < 4 else "")
        )

    def _activate_imported_batch(
        self,
        detections: list[Detection],
        offsets: dict[Path, tuple[int, int]],
        rotations: dict[Path, float],
        size_deltas: dict[Path, tuple[int, int]],
        inferred_paths: set[Path],
        current_index: int,
    ) -> None:
        self.clear_crop_undo()
        if self.preview_cache_cancel_event is not None:
            self.preview_cache_cancel_event.set()
        if self.current_rotation_cancel_event is not None:
            self.current_rotation_cancel_event.set()
        self.cancel_full_previews()
        for cancel in self.draft_inflight.values():
            cancel.set()
        self.draft_inflight.clear()
        self.preview_drafts.clear()
        self.displayed_preview = None
        self.preview_cache_generation += 1
        self.current_rotation_generation += 1
        self.rotation_preview_timer.stop()
        self.full_preview_timer.stop()
        self.detections = detections
        self.boundary_generation += 1
        self.import_boundary_generation += 1
        self.boundary_worker = None
        self.import_boundary_worker = None
        self.inferred_paths = inferred_paths
        self.page_offsets = offsets
        self.page_rotations = rotations
        self.page_size_deltas = size_deltas
        self.current_index = current_index
        self.preview_image = None
        self.preview_path = None
        self.preview_transform_cache_key = None
        self.preview_transform_cache_box = None
        self.preview_transform_cache_image = None
        self.prefetched_rotation_cache.clear()
        self.full_preview_cache.clear()
        self.full_preview_cache_key = None
        self.full_preview_image = None
        self.full_preview_box = None
        self.pending_full_preview_keys = ()
        self.pending_full_toggle = False
        self.show_full_preview = self.prefer_full_preview
        self.reset_view_on_refresh = True
        with QSignalBlocker(self.page_jump_spin):
            self.page_jump_spin.setRange(1, len(detections))
            self.page_jump_spin.setSuffix(f" / {len(detections)}")
            self.page_jump_spin.setValue(current_index + 1)
        self._set_controls_enabled(True)
        self.export_batch_config_action.setEnabled(True)
        self.detect_color_shift_action.setEnabled(True)
        self.edit_color_profiles_action.setEnabled(True)
        self.load_current_preview()

    def refresh_legacy_boundaries(self, rotations: dict[Path, float]) -> None:
        token = self.import_boundary_generation
        angles = rotations.copy()
        worker = DetectWorker([item.path for item in self.detections], angles)
        worker.signals.finished.connect(
            lambda result: self.legacy_boundaries_ready(token, angles, result)
        )
        worker.signals.error.connect(
            lambda details: self.legacy_boundaries_error(token, details)
        )
        self.import_boundary_worker = worker
        self.status_label.setText("正在后台更新旧配置的旋转后边界…")
        self.thread_pool.start(worker, -1)

    def legacy_boundaries_ready(
        self,
        token: int,
        angles: dict[Path, float],
        result: object,
    ) -> None:
        if token != self.import_boundary_generation:
            return
        detections, fallback_details = result  # type: ignore[misc]
        updated = {item.path: item for item in detections}
        inferred = {path for path, _ in fallback_details}
        for index, item in enumerate(self.detections):
            if self.page_rotations.get(item.path) == angles.get(item.path):
                replacement = updated.get(item.path)
                if replacement is not None:
                    self.detections[index] = replacement
                    if item.path in inferred:
                        self.inferred_paths.add(item.path)
                    else:
                        self.inferred_paths.discard(item.path)
        self.import_boundary_worker = None
        self.refresh_preview()
        self.status_label.setText("旧配置的旋转后边界已更新")

    def legacy_boundaries_error(self, token: int, details: str) -> None:
        if token == self.import_boundary_generation:
            self.import_boundary_worker = None
            self.status_label.setText(f"旧配置边界更新失败：{details.splitlines()[-1]}")

    def start_mouse_rotation(self) -> None:
        if not self.detections or not self.rotation_spin.isEnabled():
            return
        self.stop_edge_sweep()
        self.mouse_rotation_base = self.rotation_spin.value()

    def preview_mouse_rotation(self, delta: float) -> None:
        """While dragging, only turn the pixels on screen; nothing is rendered."""
        if self.mouse_rotation_base is None or not self.detections:
            return
        target = self.mouse_rotation_base + delta
        shown = self.displayed_preview
        path = self.detections[self.current_index].path
        pivot = self.preview_view.rotate_pivot
        if shown is not None and shown[0] == path and pivot is not None:
            self.preview_view.set_provisional_rotation(
                target - shown[1], (pivot.x(), pivot.y())
            )
        self.status_label.setText(
            f"鼠标旋转：{target:+.4f}°（本次 {delta:+.4f}°），松开后重新计算"
        )

    def finish_mouse_rotation(self, delta: float) -> None:
        base, self.mouse_rotation_base = self.mouse_rotation_base, None
        if base is None:
            return
        target = round(base + delta, 4)
        if abs(target - base) < 1e-4:
            # A click without turning: drop the provisional display.
            self.refresh_preview()
            return
        # One spin change: one undo step, then the exact render as usual.
        self.rotation_spin.setValue(target)

    def toggle_level_mode(self) -> None:
        self.set_level_mode(not self.level_mode_active)

    def set_level_mode(self, enabled: bool) -> None:
        if enabled and (not self.detections or not self.rotation_spin.isEnabled()):
            return
        self.level_mode_active = enabled
        self.preview_view.set_level_mode(enabled)
        self.level_button.setText("取消" if enabled else "参考线")
        if enabled:
            self.preview_view.setFocus(Qt.FocusReason.ShortcutFocusReason)
            self.status_label.setText(
                "两点校正：Ctrl+左键依次点击应水平或竖直的两点；Esc 或 Ctrl+R 取消"
            )
        else:
            self.status_label.setText("已退出两点校正")

    @staticmethod
    def level_correction(first: QPointF, second: QPointF) -> float | None:
        """Rotation (degrees, clockwise) that makes the line level or plumb.

        The nearer of horizontal/vertical is the target, whatever order the
        points were picked in. Scene coordinates already include the current
        page rotation, so the result is added to it.
        """
        dx, dy = second.x() - first.x(), second.y() - first.y()
        if math.hypot(dx, dy) < 3:
            return None
        direction = math.degrees(math.atan2(dy, dx))
        target = 0.0 if abs(dx) >= abs(dy) else 90.0
        residual = (direction - target + 90.0) % 180.0 - 90.0
        return -residual

    def apply_level_line(self, first: QPointF, second: QPointF) -> None:
        if not self.level_mode_active or not self.detections:
            return
        correction = self.level_correction(first, second)
        if correction is None:
            self.preview_view.set_level_mode(True)
            self.status_label.setText("两点距离太近，请重新点击两点")
            return
        self.set_level_mode(False)
        self.rotation_spin.setValue(round(self.rotation_spin.value() + correction, 4))
        self.status_label.setText(f"已按两点连线校正旋转 {correction:+.4f}°")

    def _install_shortcuts(self) -> None:
        level_action = QAction(self)
        level_action.setShortcut(QKeySequence("Ctrl+R"))
        level_action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        level_action.triggered.connect(self.toggle_level_mode)
        self.addAction(level_action)
        for key, callback in (
            (QKeySequence.StandardKey.MoveToPreviousPage, self.previous_page),
            (QKeySequence.StandardKey.MoveToNextPage, self.next_page),
            (QKeySequence("Q"), self.previous_page),
            (QKeySequence("E"), self.next_page),
        ):
            action = QAction(self)
            action.setShortcut(key)
            action.triggered.connect(
                lambda checked=False, callback=callback: self.run_navigation_shortcut(
                    callback
                )
            )
            self.addAction(action)

    @staticmethod
    def is_input_widget(widget):
        while isinstance(widget, QWidget):
            if isinstance(
                widget,
                (QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit),
            ):
                return True
            widget = widget.parentWidget()
        return False

    def text_input_active(self, watched=None):
        return self.is_input_widget(QApplication.focusWidget()) or self.is_input_widget(
            watched
        )

    def finish_edit_for_preview_shortcut(self):
        if self.is_input_widget(QApplication.focusWidget()):
            # Normal focus-out commits pending spin-box text and emits the
            # editor's completion signals before the viewport action starts.
            self.preview_view.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def run_navigation_shortcut(self, callback):
        if not self.text_input_active():
            callback()

    def clear_crop_undo(self):
        self.crop_undo_history.clear()
        self.crop_redo_history.clear()
        self.undo_batch_values = self.smart_crop_values() if hasattr(self,'crop_inner_spin') else None
        if hasattr(self, 'undo_crop_action'):
            self.undo_crop_action.setEnabled(False)
            self.redo_crop_action.setEnabled(False)

    def push_crop_undo(self, entry):
        if self.undo_restoring:
            return
        self.crop_redo_history.clear()
        self.redo_crop_action.setEnabled(False)
        gesture = entry[2]
        if gesture is not None and self.crop_undo_history and self.crop_undo_history[-1][2] == gesture:
            return
        self.crop_undo_history.append(entry)
        del self.crop_undo_history[:-300]
        self.undo_crop_action.setEnabled(True)

    def record_page_undo(self, path):
        if self.undo_restoring:
            return
        view = self.preview_view
        gesture = ((path, view.edit_gesture)
                   if view.handle_drag is not None or view.crop_drag_start is not None else None)
        self.push_crop_undo(('page', self.crop_page_snapshot(path), gesture))

    def crop_page_snapshot(self, path):
        return dict(path=path, offset=self.page_offsets.get(path,(0,0)),
                     angle=self.page_rotations.get(path,0.), size=self.page_size_deltas.get(path,(0,0)),
                     perspective=deepcopy(self.perspectives.get(path)), page=self.smart_pages.get(path),
                     detection=next(item for item in self.detections if item.path == path))

    def undo_crop_adjustment(self):
        self.restore_crop_history(redo=False)

    def redo_crop_adjustment(self):
        self.restore_crop_history(redo=True)

    def restore_crop_history(self, redo=False):
        history = self.crop_redo_history if redo else self.crop_undo_history
        opposite = self.crop_undo_history if redo else self.crop_redo_history
        if (not history or not self.offset_x_spin.isEnabled() or self.smart_worker is not None
                or self.smart_timer.isActive() or self.corner_live is not None
                or self.preview_view.handle_drag is not None or self.preview_view.crop_drag_start is not None):
            return
        kind, saved, _ = history.pop()
        current = self.smart_crop_values() if kind == 'batch' else self.crop_page_snapshot(saved['path'])
        self.undo_restoring = True
        try:
            self.stop_edge_sweep()
            self.position_jump = None
            if kind == 'batch':
                for spin,value in zip(self.smart_crop_spins(),saved):
                    with QSignalBlocker(spin):
                        spin.setValue(value)
                self.update_smart_crop()
            else:
                path = saved['path']
                index = next(i for i,item in enumerate(self.detections) if item.path == path)
                self.current_index = index
                self.page_offsets[path] = saved['offset']
                self.page_rotations[path] = saved['angle']
                self.page_size_deltas[path] = saved['size']
                if saved['perspective'] is None:
                    self.perspectives.pop(path,None)
                else:
                    self.perspectives[path] = saved['perspective']
                if saved['page'] is not None:
                    self.smart_pages[path] = saved['page']
                self.detections[index] = saved['detection']
                self.boundary_generation += 1
                self.invalidate_full_preview_cache(path)
                self.load_current_preview()
            opposite.append((kind,current,None))
            self.status_label.setText('已重做裁剪调整' if redo else '已撤销上次裁剪调整')
        finally:
            self.undo_restoring = False
            self.undo_crop_action.setEnabled(bool(self.crop_undo_history))
            self.redo_crop_action.setEnabled(bool(self.crop_redo_history))

    def _set_controls_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.stop_edge_sweep()
            if self.level_mode_active:
                self.set_level_mode(False)
        self.preview_view.edit_enabled = enabled
        self.level_button.setEnabled(enabled)
        for widget in (
            self.inner_mode,
            self.background_combo,
            self.top_spin,
            self.bottom_spin,
            self.outer_spin,
            self.inner_spin,
            self.offset_x_spin,
            self.offset_y_spin,
            self.rotation_spin,
            self.width_delta_spin,
            self.height_delta_spin,
            self.previous_button,
            self.page_jump_spin,
            self.next_button,
            self.export_button,
            self.export_selected_button,
            self.color_correction_check,
            self.color_profile_combo,
        ):
            widget.setEnabled(enabled)
        self.smart_group.setEnabled(
            (enabled and bool(self.smart_pages))
            or (
                str(self.inner_mode.currentData()).startswith("bound-")
                and not self.detections
                and self.smart_worker is None
            )
        )
        for widget in (
            self.top_spin,
            self.bottom_spin,
            self.outer_spin,
            self.inner_spin,
            self.legacy_crop_note,
        ):
            self.legacy_crop_layout.setRowVisible(widget, not bool(self.smart_pages))
        self.page_buttons.setVisible(bool(self.smart_pages))
        self.reset_pose_button.setEnabled(enabled and bool(self.smart_pages))
        self.reanalyze_page_button.setEnabled(
            enabled and bool(self.smart_pages) and self.page_reanalysis_worker is None
        )
        self.legacy_crop_group.setVisible(
            bool(self.detections) and not bool(self.smart_pages)
        )
        self.smart_group.setVisible(
            bool(self.smart_pages)
            or not self.detections
            or str(self.inner_mode.currentData()).startswith("bound-")
        )
        self.recalculate_action.setEnabled(enabled and bool(self.smart_pages))
        mode_enabled = enabled or (not self.detections and self.smart_worker is None)
        for control in (
            self.inner_mode,
            self.mode_side_combo,
            self.alternate_check,
            self.fixed_width_check,
        ):
            control.setEnabled(mode_enabled)
        self.width_delta_spin.setEnabled(enabled)
        self.page_perspective_check.setEnabled(enabled)
        self.perspective_panel.setEnabled(enabled)
        self.height_delta_spin.setEnabled(enabled)
        if enabled and not self.color_correction_check.isChecked():
            self.color_profile_combo.setEnabled(False)

    def update_preview_background(self) -> None:
        color = str(self.background_combo.currentData() or "#4F7774")
        self.preview_view.set_background_color(color)
        self.preview_view.setStyleSheet(
            f"QGraphicsView {{ border: none; background-color: {color}; }}"
        )
        self.refresh_preview()

    def invalidate_full_preview_cache(self, path: Path | None = None) -> None:
        # Only renders for the affected page are abandoned; neighbors keep
        # rendering. What is on screen stays until something replaces it.
        self.cancel_full_previews(path)
        if path is None:
            self.full_preview_cache.clear()
        else:
            self.full_preview_cache = {
                key: value
                for key, value in self.full_preview_cache.items()
                if key[0] != path
            }
        if path is None or (
            self.full_preview_cache_key and self.full_preview_cache_key[0] == path
        ):
            self.full_preview_cache_key = None
            self.full_preview_image = None
            self.full_preview_box = None
        self.pending_full_preview_keys = ()

    def update_crop_settings(self) -> None:
        if self.smart_pages:
            # Legacy manual trims do not apply to analyzed batches.
            return
        if self.preview_cache_cancel_event is not None:
            self.preview_cache_cancel_event.set()
        if self.current_rotation_cancel_event is not None:
            self.current_rotation_cancel_event.set()
        self.preview_cache_generation += 1
        self.current_rotation_generation += 1
        self.pending_rotation_request = None
        self.rotation_preview_timer.stop()
        self.prefetched_rotation_cache.clear()
        self.refresh_preview()
        self.full_preview_timer.start()

    def choose_input(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择扫描原图目录", self.input_edit.text()
        )
        if directory:
            self.set_input_folder(directory)

    def set_input_folder(self, directory: str) -> None:
        """Chosen or dropped scan folder; output defaults to its cropped/."""
        self.input_edit.setText(directory)
        self.output_edit.setText(str(Path(directory) / "cropped"))

    def choose_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.output_edit.text()
        )
        if directory:
            self.output_edit.setText(directory)

    def analyze_folder(self) -> None:
        input_dir = Path(self.input_edit.text().strip())
        if not input_dir.is_dir():
            QMessageBox.warning(self, "目录无效", "请选择存在的扫描原图目录。")
            return
        paths = sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=natural_key,
        )
        if not paths:
            QMessageBox.warning(self, "没有图片", "目录中没有支持的图片文件。")
            return

        if self.smart_worker is not None:
            return
        # A pending mode change takes effect now: the batch spins switch to
        # the meaning of the selected mode before its values are read.
        self.mode_follows_ui = True
        self.update_mode_controls()
        self.start_smart(paths)
        if self.smart_worker is None:
            self.mode_follows_ui = False
            self.update_mode_controls()

    def worker_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status_label.setText(message)

    def analysis_finished(self, result: object) -> None:
        detections, fallback_details = result  # type: ignore[misc]
        self.detections = list(detections)
        self.inferred_paths = {path for path, _ in fallback_details}
        self.page_offsets = {item.path: (0, 0) for item in self.detections}
        self.page_rotations = {item.path: 0.1 for item in self.detections}
        self.page_size_deltas = {item.path: (0, 0) for item in self.detections}
        self.current_index = 0
        with QSignalBlocker(self.page_jump_spin):
            self.page_jump_spin.setRange(1, len(self.detections))
            self.page_jump_spin.setSuffix(f" / {len(self.detections)}")
            self.page_jump_spin.setValue(1)
        self.active_worker = None
        self.analyze_button.setEnabled(True)
        self._set_controls_enabled(True)
        self.edit_color_profiles_action.setEnabled(True)
        self.detect_color_shift_action.setEnabled(bool(self.detections))
        self.export_batch_config_action.setEnabled(bool(self.detections))
        self.import_batch_config_action.setEnabled(True)
        self.load_current_preview()
        if fallback_details:
            names = "\n".join(f"• {path.name}" for path, _ in fallback_details)
            self.status_label.setText(
                f"完成：{len(self.detections)} 张图片；"
                f"{len(fallback_details)} 张需手动确认"
            )
            QMessageBox.warning(
                self,
                "部分页面已使用推定裁切框",
                "以下页面无法自动识别，已根据同批页面生成初始裁切框。\n"
                "请切换到这些页面，用 X/Y 偏移和旋转进行确认：\n\n"
                f"{names}",
            )
        else:
            self.status_label.setText(f"完成：检测到 {len(self.detections)} 张图片")

    def worker_error(self, details: str) -> None:
        self.active_worker = None
        self.analyze_button.setEnabled(True)
        self._set_controls_enabled(bool(self.detections))
        self.edit_color_profiles_action.setEnabled(True)
        self.detect_color_shift_action.setEnabled(bool(self.detections))
        self.export_batch_config_action.setEnabled(bool(self.detections))
        self.import_batch_config_action.setEnabled(True)
        self.status_label.setText("处理失败")
        QMessageBox.critical(self, "处理失败", details)

    def detect_current_color_shift(self) -> None:
        if not self.detections or self.channel_shift_worker is not None:
            return
        text, ok = QInputDialog.getText(
            self,
            "整批 RGB 偏移标定",
            "标定页码（按当前文件顺序，如 3-12 或 3,5-12；留空用全部）：",
            text="",
        )
        if not ok:
            return
        try:
            numbers = parse_page_numbers(text, len(self.detections)) or set(
                range(1, len(self.detections) + 1)
            )
        except ValueError as exc:
            QMessageBox.warning(self, "标定页码无效", str(exc))
            return
        paths = [self.detections[i - 1].path for i in sorted(numbers)]
        self.pending_color_profile_name = self.color_profile_combo.currentText()
        self.detect_color_shift_action.setEnabled(False)
        self.edit_color_profiles_action.setEnabled(False)
        self.color_profile_combo.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status_label.setText(f"正在测量 {len(paths)} 张图片的共同 RGB 偏移")
        worker = ChannelShiftDetectWorker(paths)
        worker.signals.progress.connect(self.worker_progress)
        worker.signals.finished.connect(self.color_shift_detection_finished)
        worker.signals.error.connect(self.color_shift_detection_error)
        self.channel_shift_worker = worker
        self.thread_pool.start(worker)

    @staticmethod
    def practical_shift(value: float) -> float:
        return round(value, 3)

    def color_shift_detection_finished(self, result: object) -> None:
        paths, measurement, elapsed, accepted, rejected = result  # type: ignore[misc]
        assert isinstance(measurement, ChannelShiftMeasurement)
        self.channel_shift_worker = None
        profile_name = (
            self.pending_color_profile_name or self.color_profile_combo.currentText()
        )
        self.pending_color_profile_name = None
        self.edit_color_profiles_action.setEnabled(True)
        self.color_profile_combo.setEnabled(self.color_correction_check.isChecked())
        self.detect_color_shift_action.setEnabled(bool(self.detections))
        self.progress.setRange(0, 1)
        self.progress.setValue(1)

        neutral_enough = measurement.neutral_fraction >= 0.55
        stable = (
            measurement.red_response >= 0.10
            and measurement.blue_response >= 0.10
            and measurement.red_mad <= 0.35
            and measurement.blue_mad <= 0.35
        )
        reliable = neutral_enough and stable
        message = (
            f"共同标定：{len(paths)} 张，可靠 {accepted} 张，未采用 {len(rejected)} 张\n\n"
            "以下数值表示为对齐绿通道，需要移动各通道的距离：\n"
            f"红：X {measurement.red_x:+.3f} px，"
            f"Y {measurement.red_y:+.3f} px\n"
            "绿：X +0.000 px，Y +0.000 px（基准）\n"
            f"蓝：X {measurement.blue_x:+.3f} px，"
            f"Y {measurement.blue_y:+.3f} px\n\n"
            f"有效区域：红 {measurement.red_samples} / "
            f"蓝 {measurement.blue_samples}；耗时 {elapsed:.1f} 秒。"
        )
        if rejected:
            message += "\n\n未采用的图片：\n" + "\n".join(
                f"{name}：{reason}" for name, reason in rejected
            )
        if not neutral_enough:
            message += (
                "\n\n⚠ 当前图片彩色区域较多，通道本身的颜色差异会影响测量。"
                "请切换到黑白内页后重新检测。"
            )
        elif not stable:
            message += (
                "\n\n⚠ 各区域结果不够一致，本次结果只适合参考，" "不建议写入 Profile。"
            )
        else:
            message += (
                f"\n\n将以上三位小数原样写入 Profile：{profile_name}，"
                "应用于整批导出。"
            )

        box = QMessageBox(self)
        box.setWindowTitle("整批共同 RGB 偏移标定")
        box.setIcon(
            QMessageBox.Icon.Information if reliable else QMessageBox.Icon.Warning
        )
        box.setText(message)
        apply_button = None
        if reliable:
            apply_button = box.addButton(
                "写入当前 Profile，应用整批",
                QMessageBox.ButtonRole.AcceptRole,
            )
        box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
        box.exec()

        if apply_button is not None and box.clickedButton() is apply_button:
            self.color_profiles[profile_name] = {
                "red_x": self.practical_shift(measurement.red_x),
                "red_y": self.practical_shift(measurement.red_y),
                "green_x": 0.0,
                "green_y": 0.0,
                "blue_x": self.practical_shift(measurement.blue_x),
                "blue_y": self.practical_shift(measurement.blue_y),
            }
            # Select the measured profile explicitly; enabling correction must
            # not trigger an intermediate write through the toggled signal.
            with QSignalBlocker(self.color_profile_combo), QSignalBlocker(
                self.color_correction_check
            ):
                self.color_profile_combo.setCurrentText(profile_name)
                self.color_correction_check.setChecked(True)
            self.color_profile_combo.setEnabled(True)
            if self._persist_color_profiles():
                self.status_label.setText(f"检测结果已写入颜色 Profile：{profile_name}")
            else:
                self.status_label.setText(
                    f"颜色 Profile 写入失败：{profile_name}；当前仅在内存中生效"
                )
            self.refresh_color_preview()
        else:
            self.status_label.setText(
                f"共同 RGB 偏移标定完成：{accepted}/{len(paths)} 张可靠"
            )

    def color_shift_detection_error(self, details: str) -> None:
        self.channel_shift_worker = None
        self.pending_color_profile_name = None
        self.edit_color_profiles_action.setEnabled(True)
        self.color_profile_combo.setEnabled(self.color_correction_check.isChecked())
        self.detect_color_shift_action.setEnabled(bool(self.detections))
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("颜色偏移检测失败")
        QMessageBox.critical(self, "颜色偏移检测失败", details)

    def current_geometry(
        self,
        apply_page_offsets: bool = True,
    ) -> tuple[int, int, list[tuple[int, int, int, int]], list[str], list[str]]:
        if not self.detections:
            raise ValueError("尚未分析图片")
        if self.smart_size and self.smart_plans:
            width, height = self.smart_size
            if min(width, height) < 16 or max(width, height) > 100000:
                raise ValueError("裁剪宽高需在 16～100000 像素之间")
            boxes, sides, notes = [], [], []
            for item in self.detections:
                plan = self.smart_plans[item.path]
                dx, dy = (
                    self.page_offsets.get(item.path, (0, 0))
                    if apply_page_offsets
                    else (0, 0)
                )
                dw, dh = self.page_size_deltas.get(item.path, (0, 0))
                if width + dw <= 0 or height + dh <= 0:
                    raise ValueError("本页宽高增量使尺寸无效")
                base_x, base_y = self.smart_base(item.path)
                auto_dx, auto_dy = self.smart_default_offsets[item.path]
                x, y = base_x + auto_dx + dx, base_y - auto_dy - dy
                boxes.append((x, y, x + width + dw, y + height + dh))
                sides.append(plan.inner)
                notes.append(
                    "排除页手调"
                    if item.path in self.smart_manual_pages
                    else "自动推荐＋手动微调"
                )
            return width, height, boxes, sides, notes
        page_width, page_height = choose_common_size(self.detections)
        top = self.top_spin.value()
        bottom = self.bottom_spin.value()
        outer = self.outer_spin.value()
        inner = self.inner_spin.value()
        mode = str(self.inner_mode.currentData())
        height = page_height - top - bottom
        if height <= 0:
            raise ValueError("顶部与底部裁量之和超过纸张高度")

        boxes: list[tuple[int, int, int, int]] = []
        sides: list[str] = []
        notes: list[str] = []
        if mode == "none":
            width = page_width - outer * 2
            if width <= 0:
                raise ValueError("左右裁量超过纸张宽度")
            for item in self.detections:
                delta_width, delta_height = self.page_size_deltas.get(item.path, (0, 0))
                item_width = width + delta_width
                item_height = height + delta_height
                if item_width <= 0 or item_height <= 0:
                    raise ValueError(f"{item.path.name} 的宽高增量使裁切尺寸无效")
                boxes.append(centered_crop_box(item, item_width, item_height, top))
                sides.append("none")
                notes.append("左右对称")
        else:
            width = page_width - outer - inner
            if width <= 0:
                raise ValueError("内外侧裁量超过纸张宽度")
            for index, item in enumerate(self.detections):
                side, note = inner_side_for(item, mode, index)
                delta_width, delta_height = self.page_size_deltas.get(item.path, (0, 0))
                item_width = width + delta_width
                item_height = height + delta_height
                if item_width <= 0 or item_height <= 0:
                    raise ValueError(f"{item.path.name} 的宽高增量使裁切尺寸无效")
                boxes.append(
                    outer_anchored_crop_box(
                        item, item_width, item_height, top, side, outer
                    )
                )
                sides.append(side)
                notes.append(note)
        if apply_page_offsets:
            boxes = [
                (
                    box[0] + self.page_offsets.get(item.path, (0, 0))[0],
                    box[1] - self.page_offsets.get(item.path, (0, 0))[1],
                    box[2] + self.page_offsets.get(item.path, (0, 0))[0],
                    box[3] - self.page_offsets.get(item.path, (0, 0))[1],
                )
                for item, box in zip(self.detections, boxes)
            ]
        return width, height, boxes, sides, notes

    def update_current_adjustment(self) -> None:
        if not self.detections:
            return
        path = self.detections[self.current_index].path
        previous_rotation = self.page_rotations.get(path, 0.0)
        if (self.page_offsets.get(path,(0,0)) != (self.offset_x_spin.value(),self.offset_y_spin.value())
                or previous_rotation != self.rotation_spin.value()
                or self.page_size_deltas.get(path,(0,0)) != (self.width_delta_spin.value(),self.height_delta_spin.value())):
            self.record_page_undo(path)
        self.page_offsets[path] = (
            self.offset_x_spin.value(),
            self.offset_y_spin.value(),
        )
        self.page_rotations[path] = self.rotation_spin.value()
        self.page_size_deltas[path] = (
            self.width_delta_spin.value(),
            self.height_delta_spin.value(),
        )
        if self.smart_pages:
            if self.page_rotations[path] != previous_rotation:
                self.invalidate_full_preview_cache(path)
                self.full_preview_timer.start()
            self.refresh_preview()
            return
        if self.page_rotations[path] != previous_rotation:
            self.invalidate_full_preview_cache(path)
            self.full_preview_timer.start()
            self.boundary_generation += 1
            worker = RotatedBoundaryWorker(
                self.boundary_generation, path, self.page_rotations[path]
            )
            worker.signals.ready.connect(self.rotated_boundary_ready)
            worker.signals.finished.connect(self.rotated_boundary_finished)
            self.boundary_worker = worker
            self.rotation_thread_pool.start(worker)
        self.refresh_preview()

    def rotated_boundary_ready(self, result: object) -> None:
        generation, path, angle, outcome = result  # type: ignore[misc]
        if (
            generation != self.boundary_generation
            or self.page_rotations.get(path) != angle
        ):
            return
        if isinstance(outcome, Exception):
            self.status_label.setText(f"旋转后的边界检测失败：{path.name}：{outcome}")
            return
        for index, item in enumerate(self.detections):
            if item.path == path:
                self.detections[index] = outcome
                self.inferred_paths.discard(path)
                self.refresh_preview()
                break

    def rotated_boundary_finished(self, generation: int) -> None:
        if generation == self.boundary_generation:
            self.boundary_worker = None

    def drag_current_crop(self, dx: int, dy: int) -> None:
        if not self.offset_x_spin.isEnabled():
            return
        if not self.detections:
            return
        with QSignalBlocker(self.offset_x_spin), QSignalBlocker(self.offset_y_spin):
            self.offset_x_spin.setValue(self.offset_x_spin.value() + dx)
            self.offset_y_spin.setValue(self.offset_y_spin.value() + dy)
        self.update_current_adjustment()

    def drag_current_edge(self, edge, dx, dy):
        if not self.detections or not self.width_delta_spin.isEnabled():
            return
        try:
            box = self.current_geometry()[2][self.current_index]
        except ValueError:
            return
        item = self.detections[self.current_index]
        h = state_matrix(self.perspectives.get(item.path))
        if h is not None:
            source, _, rotation = self.crop_source_quad(item)
            quad = project(source, rotation)
            i = {"top": 0, "right": 1, "bottom": 2, "left": 3}[edge]
            middle = (quad[i] + quad[(i + 1) % 4]) / 2
            mapping = rotation @ h @ np.linalg.inv(rotation)
            moved = project([middle, middle + [dx, dy]], mapping)
            dx, dy = map(lambda x: round(float(x)), moved[1] - moved[0])
        # Keep the opposite edge fixed. Y offset is positive upwards.
        dx = (
            max(16 - (box[2] - box[0]), dx)
            if edge == "right"
            else min(box[2] - box[0] - 16, dx)
        )
        dy = (
            max(16 - (box[3] - box[1]), dy)
            if edge == "bottom"
            else min(box[3] - box[1] - 16, dy)
        )
        with ExitStack() as blocks:
            for spin in (
                self.offset_x_spin,
                self.offset_y_spin,
                self.width_delta_spin,
                self.height_delta_spin,
            ):
                blocks.enter_context(QSignalBlocker(spin))
            if edge == "left":
                self.offset_x_spin.setValue(self.offset_x_spin.value() + dx)
                self.width_delta_spin.setValue(self.width_delta_spin.value() - dx)
            elif edge == "right":
                self.width_delta_spin.setValue(self.width_delta_spin.value() + dx)
            elif edge == "top":
                self.offset_y_spin.setValue(self.offset_y_spin.value() - dy)
                self.height_delta_spin.setValue(self.height_delta_spin.value() - dy)
            elif edge == "bottom":
                self.height_delta_spin.setValue(self.height_delta_spin.value() + dy)
        self.update_current_adjustment()

    def ensure_perspective(self):
        item = self.detections[self.current_index]
        if item.path not in self.perspectives:
            box = self.current_geometry()[2][self.current_index]
            self.perspectives[item.path] = manual_state(
                box,
                self.page_rotations.get(item.path, 0),
                item.source_width,
                item.source_height,
            )
        return self.perspectives[item.path]

    def sync_perspective_controls(self):
        if not self.detections:
            return
        state = self.perspectives.get(self.detections[self.current_index].path)
        with QSignalBlocker(self.page_perspective_check):
            self.page_perspective_check.setChecked(bool(state and state.get("enabled")))
        self.perspective_panel.setVisible(state is not None)
        if state:
            self.perspective_note.setText(state.get("reason", ""))
            for pair, values in zip(self.corner_spins, state["offsets"]):
                for j, (spin, value) in enumerate(zip(pair, values)):
                    with QSignalBlocker(spin):
                        spin.setValue(-value if j == 1 else value)

    def apply_perspective_state(self, candidate, record_undo=True):
        item = self.detections[self.current_index]
        candidate = validate_state(candidate)
        matrix = state_matrix(candidate)
        transformed_bounds(
            item.source_width,
            item.source_height,
            combined_matrix(
                item.source_width,
                item.source_height,
                self.page_rotations.get(item.path, 0),
                perspective=matrix,
            ),
        )
        raw = self.raw_smart_pages.get(item.path)
        page = warp_analysis(raw, candidate) if raw is not None else None
        if record_undo and candidate != self.perspectives.get(item.path):
            self.record_page_undo(item.path)
        self.perspectives[item.path] = candidate
        if page is not None:
            self.smart_pages[item.path] = page
        self.sync_perspective_controls()
        self.refresh_preview()

    def update_page_perspective(self, *_):
        if not self.detections:
            return
        previous = deepcopy(
            self.perspectives.get(self.detections[self.current_index].path)
        )
        candidate = deepcopy(self.ensure_perspective())
        candidate["enabled"] = self.page_perspective_check.isChecked()
        for i, pair in enumerate(self.corner_spins):
            for j, spin in enumerate(pair):
                if self.sender() is spin:
                    candidate["offsets"][i][j] = (
                        -spin.value() if j == 1 else spin.value()
                    )
        try:
            self.apply_perspective_state(candidate)
        except (ValueError, np.linalg.LinAlgError) as exc:
            path = self.detections[self.current_index].path
            if previous is None:
                self.perspectives.pop(path, None)
            else:
                self.perspectives[path] = previous
            self.sync_perspective_controls()
            self.status_label.setText(str(exc))

    def crop_source_quad(self, item, state=None):
        box = self.current_geometry()[2][self.current_index]
        rect = np.array(
            [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
            float,
        )
        rotation = rotation_matrix(
            self.page_rotations.get(item.path, 0), item.source_width, item.source_height
        )
        matrix = state_matrix(
            self.perspectives.get(item.path) if state is None else state
        )
        forward = rotation if matrix is None else rotation @ matrix
        return (
            project(rect, np.linalg.inv(forward)),
            project(rect, np.linalg.inv(rotation)),
            rotation,
        )

    def corner_candidate(self, live, dx, dy):
        candidate = deepcopy(live["state"])
        source = live["source"].copy()
        # Pointer movement is measured on the rotation-only preview.
        delta = live["rotation"][:2, :2].T @ np.array([dx, dy])
        source[live["corner"]] += delta
        new_matrix = homography(source, live["target"])
        # Preserve the original parameter baseline while moving the visible
        # crop corner, which need not coincide with a detected paper corner.
        effective = project(candidate["target"], np.linalg.inv(new_matrix))
        candidate["offsets"] = (effective - np.asarray(candidate["source"])).tolist()
        candidate["enabled"] = True
        candidate["automatic"] = False
        candidate["reason"] = "手动四角补正；仅导出时应用梯形校正"
        return validate_state(candidate)

    def drag_current_corner(self, corner, dx, dy):
        live = self.corner_live
        self.corner_live = None
        if (
            not self.detections
            or not self.page_perspective_check.isEnabled()
            or max(abs(dx), abs(dy)) < 0.01
            or (
                live is not None
                and live["path"] != self.detections[self.current_index].path
            )
        ):
            self.refresh_preview()
            return
        if live is None:
            self.start_corner_drag(corner)
            live = self.corner_live
            self.corner_live = None
        if live is None:
            return
        try:
            self.apply_perspective_state(self.corner_candidate(live, dx, dy))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.refresh_preview()
            self.status_label.setText(str(exc))

    def start_corner_drag(self, corner):
        if not self.detections:
            return
        item = self.detections[self.current_index]
        state = deepcopy(self.ensure_perspective())
        source, target, rotation = self.crop_source_quad(item, state)
        self.corner_live = dict(
            state=state,
            source=source,
            target=target,
            rotation=rotation,
            corner=corner,
            dx=0.0,
            dy=0.0,
            path=item.path,
        )
        # Only expose the already loaded image; never rebuild its pixmaps.
        if self.show_full_preview:
            self.preview_view.set_image_clip()
        self.render_corner_drag()

    def preview_corner_drag(self, corner, dx, dy):
        if self.corner_live is None:
            return
        self.corner_live.update(dx=dx, dy=dy)
        self.render_corner_drag()

    def render_corner_drag(self):
        live = self.corner_live
        if (
            live is None
            or not self.detections
            or self.detections[self.current_index].path != live["path"]
        ):
            return
        try:
            self.corner_candidate(live, live["dx"], live["dy"])
            quad = project(live["source"], live["rotation"])
            quad[live["corner"]] += [live["dx"], live["dy"]]
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in quad])
            self.preview_view.set_crop_outline(polygon, visible=self.show_full_preview)
            if not self.show_full_preview:
                self.preview_view.set_image_clip(polygon)
            self.preview_view.edit_polygon = polygon
            self.status_label.setText("正在调整四边形裁剪框；梯形校正仅在导出时应用")
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.status_label.setText(str(exc))

    def load_current_preview(self) -> None:
        self.position_jump = None
        self.edge_jump_pending = None
        self.stop_edge_sweep()
        if not self.detections:
            return
        item = self.detections[self.current_index]
        self.sync_perspective_controls()
        offset_x, offset_y = self.page_offsets.get(item.path, (0, 0))
        rotation = self.page_rotations.get(item.path, 0.0)
        delta_width, delta_height = self.page_size_deltas.get(item.path, (0, 0))
        with (
            QSignalBlocker(self.offset_x_spin),
            QSignalBlocker(self.offset_y_spin),
            QSignalBlocker(self.rotation_spin),
            QSignalBlocker(self.width_delta_spin),
            QSignalBlocker(self.height_delta_spin),
            QSignalBlocker(self.page_jump_spin),
        ):
            self.offset_x_spin.setValue(offset_x)
            self.offset_y_spin.setValue(offset_y)
            self.rotation_spin.setValue(rotation)
            self.width_delta_spin.setValue(delta_width)
            self.height_delta_spin.setValue(delta_height)
            self.page_jump_spin.setValue(self.current_index + 1)
        if self.preview_path != item.path or self.preview_image is None:
            if self.level_mode_active:
                # Picked points belong to the previous page.
                self.set_level_mode(False)
            if self.current_rotation_cancel_event is not None:
                self.current_rotation_cancel_event.set()
            self.current_rotation_generation += 1
            self.pending_rotation_request = None
            self.rotation_preview_timer.stop()
            # Only the header is read here; decoding and rendering run in
            # background workers so flipping pages never blocks the UI.
            with Image.open(item.path) as source:
                self.preview_image = SourceInfo(source.size)
            self.preview_path = item.path
            self.preview_transform_cache_key = None
            self.preview_transform_cache_box = None
            self.preview_transform_cache_image = None
            self.full_preview_cache_key = None
            self.full_preview_image = None
            self.full_preview_box = None
            self.pending_full_toggle = False
            self.current_preview_box = None
            self.reset_view_on_refresh = True
        self.refresh_preview()
        self.full_preview_timer.start()

    def stop_edge_sweep(self):
        self.edge_sweep_timer.stop()
        self.edge_sweep = None
        self.edge_sweep_pending = None
        saved = self.edge_sweep_saved
        self.edge_sweep_saved = None
        if (
            saved is not None
            and self.detections
            and self.detections[self.current_index].path == saved[0]
        ):
            self.preview_view.restore_view_state(saved[1])
            self.preview_view._select_preview_level()

    def start_edge_sweep(self, key):
        self.edge_jump_pending = None
        if (
            not self.detections
            or self.preview_image is None
            or self.corner_live is not None
            or not self.offset_x_spin.isEnabled()
        ):
            return
        if self.edge_sweep is not None and self.edge_sweep["key"] == key:
            return
        self.stop_edge_sweep()
        self.edge_sweep_saved = (
            self.detections[self.current_index].path,
            self.preview_view.capture_view_state(),
        )
        if not self.full_preview_is_ready():
            self.edge_sweep_pending = key
            self.prepare_current_full_preview()
            self.status_label.setText("正在准备原图预览，松开按键可取消扫边")
            return
        self.refresh_preview()
        polygon = self.preview_view.edit_polygon
        if len(polygon) != 4:
            return
        pairs = {
            Qt.Key.Key_W: (0, 1, "上"),
            Qt.Key.Key_A: (3, 0, "左"),
            Qt.Key.Key_X: (2, 3, "下"),
            Qt.Key.Key_D: (1, 2, "右"),
            Qt.Key.Key_S: (0, 0, "四条"),
        }
        if key not in pairs:
            return
        lengths = [
            math.hypot(
                (polygon[(i + 1) % 4] - polygon[i]).x(),
                (polygon[(i + 1) % 4] - polygon[i]).y(),
            )
            for i in range(4)
        ]
        short = min(lengths)
        if short <= 0:
            return
        first, last, label = pairs[key]
        start, end = QPointF(polygon[first]), QPointF(polygon[last])
        points = (
            [QPointF(polygon[i]) for i in (0, 1, 2, 3, 0)]
            if key == Qt.Key.Key_S
            else [start, end]
        )
        segment_lengths = [
            math.hypot((b - a).x(), (b - a).y()) for a, b in zip(points, points[1:])
        ]
        duration = self.sweep_short_duration * sum(segment_lengths) / short
        view = self.preview_view
        self.set_inspection_scale()
        view.centerOn(start)
        self.edge_sweep = dict(
            key=key,
            start=start,
            end=end,
            points=points,
            segment_lengths=segment_lengths,
            duration=duration,
            delay=self.sweep_start_delay,
            started=time.perf_counter(),
            path=self.detections[self.current_index].path,
        )
        self.edge_sweep_timer.start()
        pause_text = "每个角停留" if key == Qt.Key.Key_S else "起点停留"
        self.status_label.setText(
            f"{pause_text} {self.sweep_start_delay:g} 秒，{self.inspection_scale:g} 倍显示扫过{label}边；移动用时 {duration:.2f} 秒；松开恢复原视角"
        )

    def set_inspection_scale(self):
        view = self.preview_view
        # Qt view coordinates are logical pixels; account for Windows DPI so
        # the configured scale is measured in physical display pixels.
        ratio = max(view.devicePixelRatioF(), 1e-9)
        view.setTransform(
            QTransform.fromScale(
                self.inspection_scale / ratio, self.inspection_scale / ratio
            )
        )
        view._select_preview_level()
        view._update_pan_bounds()
        view._update_sampling_mode()

    def toggle_position_jump(self, key):
        if (
            not self.detections
            or self.preview_image is None
            or self.corner_live is not None
            or not self.offset_x_spin.isEnabled()
        ):
            return
        self.stop_edge_sweep()
        path = self.detections[self.current_index].path
        saved = self.position_jump
        if saved is not None and saved["path"] == path and saved["key"] == key:
            self.edge_jump_pending = None
            self.position_jump = None
            self.show_full_preview = saved["show_full"]
            self.prefer_full_preview = saved["prefer_full"]
            self.reset_view_on_refresh = False
            self.refresh_preview()
            self.preview_view.restore_view_state(saved["view"])
            self.preview_view._select_preview_level()
            self.status_label.setText("已恢复本轮定位前的视角")
            return
        if saved is None or saved["path"] != path:
            self.position_jump = dict(
                path=path,
                key=key,
                view=self.preview_view.capture_view_state(),
                show_full=self.show_full_preview,
                prefer_full=self.prefer_full_preview,
            )
        else:
            saved["key"] = key
        self.jump_to_crop_point(key)

    def jump_to_crop_point(self, key):
        self.edge_jump_pending = None
        if (
            not self.detections
            or self.preview_image is None
            or self.corner_live is not None
            or not self.offset_x_spin.isEnabled()
        ):
            return
        self.stop_edge_sweep()
        if not self.full_preview_is_ready():
            self.edge_jump_pending = (self.detections[self.current_index].path, key)
            self.prepare_current_full_preview()
            self.status_label.setText("正在准备预览，完成后跳转到裁剪框指定位置")
            return
        if key == Qt.Key.Key_S:
            self.show_full_preview = True
            self.refresh_preview()
            self.preview_view.fit_preview()
            self.status_label.setText("完整图像：适合窗口")
            return
        self.refresh_preview()
        p = self.preview_view.edit_polygon
        if len(p) != 4:
            return
        points = {
            Qt.Key.Key_Q: (p[0], "左上角"),
            Qt.Key.Key_W: ((p[0] + p[1]) / 2, "上边中点"),
            Qt.Key.Key_E: (p[1], "右上角"),
            Qt.Key.Key_A: ((p[0] + p[3]) / 2, "左边中点"),
            Qt.Key.Key_X: ((p[3] + p[2]) / 2, "下边中点"),
            Qt.Key.Key_D: ((p[1] + p[2]) / 2, "右边中点"),
            Qt.Key.Key_Z: (p[3], "左下角"),
            Qt.Key.Key_C: (p[2], "右下角"),
        }
        if key not in points:
            return
        point, label = points[key]
        self.set_inspection_scale()
        self.preview_view.centerOn(point)
        self.status_label.setText(f"{self.inspection_scale:g} 倍显示：{label}")

    def advance_edge_sweep(self):
        state = self.edge_sweep
        if state is None:
            return
        if (
            not self.detections
            or self.detections[self.current_index].path != state["path"]
        ):
            self.stop_edge_sweep()
            return
        elapsed = max(0.0, time.perf_counter() - state["started"])
        total_length = sum(state["segment_lengths"])
        for i, length in enumerate(state["segment_lengths"]):
            a, b = state["points"][i : i + 2]
            if elapsed < state["delay"]:
                self.preview_view.centerOn(a)
                return
            elapsed -= state["delay"]
            duration = state["duration"] * length / total_length
            if elapsed < duration:
                self.preview_view.centerOn(a + (b - a) * (elapsed / duration))
                return
            elapsed -= duration
        self.preview_view.centerOn(state["points"][-1])
        self.edge_sweep_timer.stop()
        self.status_label.setText("扫边完成；松开按键恢复原视角")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            self.level_mode_active
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
            and QApplication.activeModalWidget() is None
        ):
            self.set_level_mode(False)
            return True
        if (event.type() in (QEvent.Type.ShortcutOverride,QEvent.Type.KeyPress)
                and event.key() == Qt.Key.Key_Z
                and event.modifiers() in (Qt.KeyboardModifier.ControlModifier,
                    Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
                and QApplication.activeModalWidget() is None and not self.text_input_active(watched)):
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()
            elif not event.isAutoRepeat():
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.redo_crop_adjustment()
                else:
                    self.undo_crop_adjustment()
            return True
        jump_keys = (
            Qt.Key.Key_Q,
            Qt.Key.Key_W,
            Qt.Key.Key_E,
            Qt.Key.Key_A,
            Qt.Key.Key_S,
            Qt.Key.Key_X,
            Qt.Key.Key_D,
            Qt.Key.Key_Z,
            Qt.Key.Key_C,
        )
        sweep_keys = (
            Qt.Key.Key_W,
            Qt.Key.Key_A,
            Qt.Key.Key_X,
            Qt.Key.Key_D,
            Qt.Key.Key_S,
        )
        if event.type() == QEvent.Type.FocusIn and self.is_input_widget(watched):
            self.edge_jump_pending = None
            self.stop_edge_sweep()
        if event.type() in (
            QEvent.Type.ShortcutOverride,
            QEvent.Type.KeyPress,
            QEvent.Type.KeyRelease,
        ) and self.text_input_active(watched):
            modifiers = event.modifiers()
            shift_jump = (
                event.key() in jump_keys
                and modifiers & Qt.KeyboardModifier.ShiftModifier
                and not modifiers
                & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
            )
            alt_sweep = (
                event.key() in sweep_keys
                and modifiers & Qt.KeyboardModifier.AltModifier
                and not modifiers
                & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
            )
            active_key = (
                self.edge_sweep["key"] if self.edge_sweep else self.edge_sweep_pending
            )
            sweep_release = (
                event.type() == QEvent.Type.KeyRelease
                and active_key is not None
                and event.key() in (active_key, Qt.Key.Key_Alt)
            )
            if not (shift_jump or alt_sweep or sweep_release):
                # Plain typing and editor shortcuts keep their native behavior;
                # explicit Shift navigation and Alt inspection remain global.
                if event.type() == QEvent.Type.ShortcutOverride:
                    event.accept()
                    return True
                return super().eventFilter(watched, event)
        if (
            event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
            and event.key() in jump_keys
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and not event.modifiers()
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.MetaModifier
            )
            and QApplication.activeModalWidget() is None
        ):
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()
            elif not event.isAutoRepeat():
                self.finish_edit_for_preview_shortcut()
                self.toggle_position_jump(event.key())
            return True
        if (
            event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
            and event.key() in sweep_keys
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
            and not event.modifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            and QApplication.activeModalWidget() is None
        ):
            if event.type() == QEvent.Type.ShortcutOverride:
                event.accept()
            elif not event.isAutoRepeat():
                self.finish_edit_for_preview_shortcut()
                self.start_edge_sweep(event.key())
            return True
        if event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            active_key = (
                self.edge_sweep["key"] if self.edge_sweep else self.edge_sweep_pending
            )
            if event.key() == Qt.Key.Key_Alt or event.key() == active_key:
                self.stop_edge_sweep()
                if active_key is not None:
                    self.status_label.setText("扫边已停止；已恢复原视角")
                if event.key() in sweep_keys:
                    return True
        if event.type() == QEvent.Type.WindowDeactivate and watched is self:
            self.edge_jump_pending = None
            self.stop_edge_sweep()
        if watched is self.preview_view.viewport() and event.type() in (
            QEvent.Type.Wheel,
            QEvent.Type.MouseButtonPress,
        ):
            self.edge_jump_pending = None
            self.stop_edge_sweep()
        if (
            event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
            and event.key() == Qt.Key.Key_Control
        ):
            self.preview_view.update_edit_cursor(
                control=event.type() == QEvent.Type.KeyPress
            )
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_A,
            Qt.Key.Key_D,
            Qt.Key.Key_W,
            Qt.Key.Key_X,
        ):
            if (
                self.detections
                and self.preview_image is not None
                and self.offset_x_spin.isEnabled()
                and not isinstance(QApplication.focusWidget(), QLineEdit)
                and not event.modifiers()
                & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
            ):
                dx, dy = {
                    Qt.Key.Key_Left: (-1, 0),
                    Qt.Key.Key_Right: (1, 0),
                    Qt.Key.Key_Up: (0, 1),
                    Qt.Key.Key_Down: (0, -1),
                    Qt.Key.Key_A: (-1, 0),
                    Qt.Key.Key_D: (1, 0),
                    Qt.Key.Key_W: (0, 1),
                    Qt.Key.Key_X: (0, -1),
                }[event.key()]
                self.drag_current_crop(dx, dy)
                return True
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Space:
            if event.isAutoRepeat():
                return True
            if isinstance(QApplication.focusWidget(), QLineEdit):
                return super().eventFilter(watched, event)
            if self.detections and self.preview_image is not None:
                self.toggle_full_preview()
                return True
        elif event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Space:
            if isinstance(QApplication.focusWidget(), QLineEdit):
                return super().eventFilter(watched, event)
            return True
        return super().eventFilter(watched, event)

    def toggle_full_preview(self) -> None:
        if self.detections:
            self.show_full_preview = not self.show_full_preview
            self.prefer_full_preview = self.show_full_preview
            self.settings_save_timer.start()
            self.refresh_preview()
            return

    def current_full_preview_parameters(
        self,
    ) -> (
        tuple[
            Path,
            float,
            tuple[float, float],
            tuple[object, ...],
        ]
        | None
    ):
        if not self.detections or self.preview_image is None:
            return None
        item = self.detections[self.current_index]
        rotation_center = (
            self.preview_image.width / 2,
            self.preview_image.height / 2,
        )
        rotation_angle = self.page_rotations.get(item.path, 0.0)
        cache_key = perspective_cache_key(
            item.path,
            self.preview_image.size,
            rotation_angle,
            rotation_center,
            channel_shifts=self.current_channel_shifts(),
        )
        return item.path, rotation_angle, rotation_center, cache_key

    def full_preview_is_ready(self) -> bool:
        parameters = self.current_full_preview_parameters()
        return bool(parameters is not None and parameters[3] in self.full_preview_cache)

    def prepare_current_full_preview(self) -> None:
        parameters = self.current_full_preview_parameters()
        if parameters is None:
            return
        channel_shifts = self.current_channel_shifts()
        # Exact renders near the current page; RGB-aligned ones are larger.
        ahead, behind = (3, 2) if channel_shifts else (6, 3)
        count = len(self.detections)
        indices = [self.current_index]
        for distance in range(1, max(ahead, behind) + 1):
            if distance <= ahead and self.current_index + distance < count:
                indices.append(self.current_index + distance)
            if distance <= behind and self.current_index - distance >= 0:
                indices.append(self.current_index - distance)
        keys: list[tuple[object, ...]] = []
        jobs = []
        for index in indices:
            item = self.detections[index]
            angle = self.page_rotations.get(item.path, 0.0)
            center = (item.source_width / 2, item.source_height / 2)
            key = perspective_cache_key(
                item.path,
                (item.source_width, item.source_height),
                angle,
                center,
                channel_shifts=channel_shifts,
            )
            keys.append(key)
            if key not in self.full_preview_cache:
                jobs.append((key, (item.path, angle, center, None, channel_shifts)))
        wanted = set(keys)
        self.full_preview_cache = {
            key: value
            for key, value in self.full_preview_cache.items()
            if key in wanted
        }
        for key in [k for k in self.full_preview_inflight if k not in wanted]:
            self.full_preview_inflight.pop(key)[1].cancel_event.set()
        for rank, (key, job) in enumerate(jobs):
            if key in self.full_preview_inflight:
                continue
            self.full_preview_serial += 1
            serial = self.full_preview_serial
            worker = FullPreviewWorker(serial, [job], threading.Event())
            worker.signals.ready.connect(self.full_preview_ready)
            worker.signals.draft.connect(self.preview_draft_ready)
            worker.signals.finished.connect(
                lambda _serial, key=key, serial=serial: self.full_preview_finished_for(
                    key, serial
                )
            )
            self.full_preview_inflight[key] = (serial, worker)
            # Nearer pages first; the current page always leads the queue.
            self.full_preview_thread_pool.start(worker, len(jobs) - rank)
        self.schedule_preview_drafts(ahead, behind)
        if not jobs:
            if self.pending_full_toggle and not self.show_full_preview:
                self.pending_full_toggle = False
                self.toggle_full_preview()

    def schedule_preview_drafts(self, ahead: int, behind: int) -> None:
        """Quick drafts further out, so pages appear at once when flipping."""
        count = len(self.detections)
        keep_lo, keep_hi = self.current_index - 15, self.current_index + 15
        paths = [d.path for d in self.detections]
        for index, path in enumerate(paths):
            if not keep_lo <= index <= keep_hi:
                self.preview_drafts.pop(path, None)
        wanted = [
            self.current_index + d
            for d in list(range(ahead + 1, 13)) + [-d for d in range(behind + 1, 6)]
            if 0 <= self.current_index + d < count
        ]
        wanted_paths = {paths[i] for i in wanted}
        for path in [p for p in self.draft_inflight if p not in wanted_paths]:
            self.draft_inflight.pop(path).set()
        for index in wanted:
            path = paths[index]
            if path in self.preview_drafts or path in self.draft_inflight:
                continue
            cancel = threading.Event()
            worker = DraftWorker(0, path, cancel)
            worker.signals.ready.connect(self.preview_draft_ready)
            worker.signals.finished.connect(
                lambda _g, path=path, cancel=cancel: (
                    self.draft_inflight.pop(path, None)
                    if self.draft_inflight.get(path) is cancel
                    else None
                )
            )
            self.draft_inflight[path] = cancel
            self.draft_thread_pool.start(worker)

    def preview_draft_ready(self, result: object) -> None:
        _, path, draft = result  # type: ignore[misc]
        if not any(d.path == path for d in self.detections):
            return
        self.preview_drafts[path] = draft
        if (
            self.detections
            and self.detections[self.current_index].path == path
            and (self.displayed_preview is None or self.displayed_preview[0] != path)
        ):
            self.refresh_preview()

    def full_preview_finished_for(self, key, serial: int) -> None:
        entry = self.full_preview_inflight.get(key)
        if entry is not None and entry[0] == serial:
            del self.full_preview_inflight[key]

    def cancel_full_previews(self, path: Path | None = None) -> None:
        for key in [
            k for k in self.full_preview_inflight if path is None or k[0] == path
        ]:
            self.full_preview_inflight.pop(key)[1].cancel_event.set()

    def full_preview_ready(self, result: object) -> None:
        serial, key, full_box, image = result  # type: ignore[misc]
        entry = self.full_preview_inflight.get(key)
        if entry is None or entry[0] != serial:
            return
        levels = getattr(image, "preview_levels", None) or build_preview_qimages(image)
        raster = PreviewRaster(levels, image.size)
        self.full_preview_cache[key] = (full_box, raster)
        parameters = self.current_full_preview_parameters()
        if parameters is not None and parameters[3] == key:
            self.full_preview_cache_key = key
            self.full_preview_box = full_box
            self.full_preview_image = raster
            if self.pending_full_toggle and not self.show_full_preview:
                self.pending_full_toggle = False
                self.toggle_full_preview()
            else:
                self.refresh_preview()
            if self.edge_sweep_pending is not None:
                self.start_edge_sweep(self.edge_sweep_pending)
            if self.edge_jump_pending is not None:
                path, key = self.edge_jump_pending
                self.edge_jump_pending = None
                if self.detections[self.current_index].path == path:
                    self.jump_to_crop_point(key)

    @staticmethod
    def crop_outline_polygon(
        preview_box: tuple[int, int, int, int],
    ) -> QPolygonF:
        left, top, right, bottom = preview_box
        return QPolygonF(
            [
                QPointF(left, top),
                QPointF(right, top),
                QPointF(right, bottom),
                QPointF(left, bottom),
            ]
        )

    def cached_preview_crop(
        self,
        preview_box: tuple[int, int, int, int],
        rotation_angle: float,
        rotation_center: tuple[float, float],
        allow_render: bool = True,
    ) -> Image.Image | None:
        """Reuse a rotated overscan region while the user adjusts X/Y."""
        if self.preview_image is None:
            raise ValueError("尚未加载预览图片")
        if abs(rotation_angle) < 1e-9:
            return self.preview_image.crop(preview_box)

        cache_key = (
            self.preview_path,
            self.preview_image.size,
            round(rotation_angle, 4),
            rotation_center,
        )
        full_preview = self.full_preview_cache.get(cache_key)
        if full_preview is not None:
            full_box, full_image = full_preview
            return full_image.crop(
                (
                    preview_box[0] - full_box[0],
                    preview_box[1] - full_box[1],
                    preview_box[2] - full_box[0],
                    preview_box[3] - full_box[1],
                )
            )
        cache_box = self.preview_transform_cache_box
        cache_covers_crop = (
            cache_key == self.preview_transform_cache_key
            and cache_box is not None
            and self.preview_transform_cache_image is not None
            and cache_box[0] <= preview_box[0]
            and cache_box[1] <= preview_box[1]
            and cache_box[2] >= preview_box[2]
            and cache_box[3] >= preview_box[3]
        )
        if not cache_covers_crop:
            prefetched = self.prefetched_rotation_cache.get(cache_key)
            if prefetched is not None:
                prefetched_box, prefetched_image = prefetched
                if (
                    prefetched_box[0] <= preview_box[0]
                    and prefetched_box[1] <= preview_box[1]
                    and prefetched_box[2] >= preview_box[2]
                    and prefetched_box[3] >= preview_box[3]
                ):
                    self.preview_transform_cache_key = cache_key
                    self.preview_transform_cache_box = prefetched_box
                    self.preview_transform_cache_image = prefetched_image
                    cache_box = prefetched_box
                    cache_covers_crop = True
        if not cache_covers_crop:
            if not allow_render:
                return None
            left, top, right, bottom = preview_box
            cache_box = (
                left - ROTATION_CACHE_MARGIN,
                top - ROTATION_CACHE_MARGIN,
                right + ROTATION_CACHE_MARGIN,
                bottom + ROTATION_CACHE_MARGIN,
            )
            self.preview_transform_cache_image = transformed_crop(
                self.preview_image,
                cache_box,
                rotation_angle,
                rotation_center,
            )
            self.preview_transform_cache_key = cache_key
            self.preview_transform_cache_box = cache_box
            self.prefetched_rotation_cache[cache_key] = (
                cache_box,
                self.preview_transform_cache_image,
            )

        assert cache_box is not None
        assert self.preview_transform_cache_image is not None
        return self.preview_transform_cache_image.crop(
            (
                preview_box[0] - cache_box[0],
                preview_box[1] - cache_box[1],
                preview_box[2] - cache_box[0],
                preview_box[3] - cache_box[1],
            )
        )

    def request_current_rotation_cache(
        self,
        path: Path,
        preview_box: tuple[int, int, int, int],
        rotation_angle: float,
        rotation_center: tuple[float, float],
    ) -> None:
        request = (
            path,
            preview_box,
            round(rotation_angle, 4),
            rotation_center,
        )
        if request == self.pending_rotation_request and (
            self.rotation_preview_timer.isActive()
            or self.current_rotation_worker is not None
        ):
            return
        if self.current_rotation_cancel_event is not None:
            self.current_rotation_cancel_event.set()
        self.current_rotation_generation += 1
        self.pending_rotation_request = request
        self.rotation_preview_timer.start()

    def start_current_rotation_worker(self) -> None:
        if self.pending_rotation_request is None:
            return
        path, preview_box, rotation_angle, rotation_center = (
            self.pending_rotation_request
        )
        cancel_event = threading.Event()
        self.current_rotation_cancel_event = cancel_event
        worker = CurrentRotationWorker(
            self.current_rotation_generation,
            path,
            preview_box,
            rotation_angle,
            rotation_center,
            cancel_event,
        )
        worker.signals.ready.connect(self.current_rotation_ready)
        worker.signals.finished.connect(self.current_rotation_finished)
        self.current_rotation_worker = worker
        self.rotation_thread_pool.start(worker)

    def current_rotation_ready(self, result: object) -> None:
        generation, key, cache_box, image = result  # type: ignore[misc]
        if generation != self.current_rotation_generation:
            return
        self.prefetched_rotation_cache[key] = (cache_box, image)
        self.preview_transform_cache_key = key
        self.preview_transform_cache_box = cache_box
        self.preview_transform_cache_image = image
        self.pending_rotation_request = None
        self.refresh_preview()

    def current_rotation_finished(self, generation: int) -> None:
        if generation == self.current_rotation_generation:
            self.current_rotation_worker = None

    def schedule_preview_cache(self) -> None:
        if self.preview_cache_cancel_event is not None:
            self.preview_cache_cancel_event.set()
        self.preview_cache_generation += 1
        generation = self.preview_cache_generation
        if not self.detections:
            self.prefetched_rotation_cache.clear()
            return
        current_path = self.detections[self.current_index].path
        self.prefetched_rotation_cache = {
            key: value
            for key, value in self.prefetched_rotation_cache.items()
            if key[0] == current_path
        }

        try:
            _, _, boxes, _, _ = self.current_geometry()
        except Exception:
            return

        jobs: list[
            tuple[
                Path,
                tuple[int, int, int, int],
                tuple[float, float],
            ]
        ] = []
        stop_index = min(len(self.detections), self.current_index + 4)
        for index in range(self.current_index, stop_index):
            item = self.detections[index]
            box = boxes[index]
            rotation_center = (
                item.source_width / 2,
                item.source_height / 2,
            )
            expected_size = (item.source_width, item.source_height)
            already_cached = True
            for step in (-1, 1, -2, 2):
                cache_key = (
                    item.path,
                    expected_size,
                    round(step / 10, 4),
                    rotation_center,
                )
                cached = self.prefetched_rotation_cache.get(cache_key)
                if cached is None:
                    already_cached = False
                    break
                cached_box, _ = cached
                if not (
                    cached_box[0] <= box[0]
                    and cached_box[1] <= box[1]
                    and cached_box[2] >= box[2]
                    and cached_box[3] >= box[3]
                ):
                    already_cached = False
                    break
            if already_cached:
                continue
            jobs.append((item.path, box, rotation_center))
        if not jobs:
            return

        cancel_event = threading.Event()
        self.preview_cache_cancel_event = cancel_event
        worker = PreviewCacheWorker(
            generation,
            jobs,
            cancel_event,
        )
        worker.signals.ready.connect(self.preview_cache_ready)
        worker.signals.finished.connect(self.preview_cache_finished)
        self.preview_cache_worker = worker
        self.thread_pool.start(worker, -1)

    def preview_cache_ready(self, result: object) -> None:
        generation, key, cache_box, image = result  # type: ignore[misc]
        if generation != self.preview_cache_generation:
            return
        self.prefetched_rotation_cache[key] = (cache_box, image)
        if self.pending_rotation_request is not None:
            path, preview_box, angle, rotation_center = self.pending_rotation_request
            if (
                key[0] == path
                and key[2] == angle
                and key[3] == rotation_center
                and cache_box[0] <= preview_box[0]
                and cache_box[1] <= preview_box[1]
                and cache_box[2] >= preview_box[2]
                and cache_box[3] >= preview_box[3]
            ):
                if self.current_rotation_cancel_event is not None:
                    self.current_rotation_cancel_event.set()
                self.current_rotation_generation += 1
                self.rotation_preview_timer.stop()
                self.pending_rotation_request = None
                self.current_rotation_worker = None
                self.preview_transform_cache_key = key
                self.preview_transform_cache_box = cache_box
                self.preview_transform_cache_image = image
                self.refresh_preview()

    def preview_cache_finished(self, generation: int) -> None:
        if generation == self.preview_cache_generation:
            self.preview_cache_worker = None

    def refresh_preview(self) -> None:
        if self.corner_live is not None:
            return
        if not self.detections:
            self.page_output_size_label.setText("—")
            return
        try:
            width, height, boxes, sides, notes = self.current_geometry()
            item = self.detections[self.current_index]
            box = boxes[self.current_index]
            self.page_output_size_label.setText(f"{box[2] - box[0]} × {box[3] - box[1]} px")
            if self.preview_image is None:
                return
            scale_x = self.preview_image.width / item.source_width
            scale_y = self.preview_image.height / item.source_height
            x = round(box[0] * scale_x)
            y = round(box[1] * scale_y)
            crop_width = max(1, round((box[2] - box[0]) * scale_x))
            crop_height = max(1, round((box[3] - box[1]) * scale_y))
            preview_box = (x, y, x + crop_width, y + crop_height)
            self.current_preview_box = preview_box
            rotation_center = (
                self.preview_image.width / 2,
                self.preview_image.height / 2,
            )
            rotation_angle = self.page_rotations.get(item.path, 0.0)
            perspective = state_matrix(self.perspectives.get(item.path))
            source, _, rotation = self.crop_source_quad(item)
            points = project(source, rotation)
            quad = QPolygonF(
                [QPointF(float(px * scale_x), float(py * scale_y)) for px, py in points]
            )
            perspective = None
            channel_shifts = self.current_channel_shifts()
            expected_full_key = perspective_cache_key(
                item.path,
                self.preview_image.size,
                rotation_angle,
                rotation_center,
                perspective,
                channel_shifts,
            )
            if (
                abs(rotation_angle) < 1e-9
                and perspective is None
                and not channel_shifts
                and isinstance(self.preview_image, Image.Image)
            ):
                self.full_preview_cache[expected_full_key] = (
                    (0, 0, self.preview_image.width, self.preview_image.height),
                    self.preview_image,
                )
            # Everything about a render except the rotation angle.
            signature = (expected_full_key[1], expected_full_key[3]) + expected_full_key[4:]
            cached_full = self.full_preview_cache.get(expected_full_key)
            provisional = cached_full is None
            if provisional:
                self.prepare_current_full_preview()
                shown = self.displayed_preview
                draft = self.preview_drafts.get(item.path)
                if shown is not None and shown[0] == item.path and shown[2] == signature:
                    # Same page, new angle: rotate the exact pixels on screen
                    # until the exact render for this angle arrives.
                    self.preview_view.set_provisional_rotation(
                        rotation_angle - shown[1], rotation_center
                    )
                elif draft is not None and draft.source_size == self.preview_image.size:
                    if shown is None or shown[0] != item.path or shown[3] != "draft":
                        self.preview_view.set_preview(
                            draft,
                            reset_view=self.reset_view_on_refresh,
                            rotation=rotation_angle,
                            rotation_center=rotation_center,
                        )
                        self.reset_view_on_refresh = False
                        self.displayed_preview = (item.path, 0.0, None, "draft")
                    else:
                        self.preview_view.set_provisional_rotation(
                            rotation_angle, rotation_center
                        )
                else:
                    self.displayed_preview = None
                    self.preview_view.show_message("正在载入本页预览…")
                    self.page_label.setText(
                        f"{self.current_index + 1} / {len(self.detections)}　{item.path.name}"
                    )
                    self.status_label.setText("正在后台准备预览…")
                    return
            else:
                self.full_preview_cache_key = expected_full_key
                self.full_preview_box, self.full_preview_image = cached_full
                self.preview_view.set_preview(
                    self.full_preview_image,
                    reset_view=self.reset_view_on_refresh,
                    offset=QPointF(
                        self.full_preview_box[0],
                        self.full_preview_box[1],
                    ),
                )
                self.reset_view_on_refresh = False
                self.displayed_preview = (item.path, rotation_angle, signature, "full")
            if provisional:
                self.preview_view.set_image_clip()
                self.preview_view.set_detected_edges([])
            self.preview_view.set_crop_outline(quad, visible=self.show_full_preview)
            if not self.show_full_preview:
                self.preview_view.set_image_clip(quad)
            self.preview_view.full_preview_mode = True

            curves = []
            if (
                self.show_full_preview
                and self.show_edges_check.isChecked()
                and item.path in self.smart_pages
                and item.path not in self.smart_manual_pages
            ):
                page = self.raw_smart_pages.get(item.path, self.smart_pages[item.path])
                for side, points in page.edges.items():
                    if self.smart_bound and side == self.smart_plans[item.path].inner:
                        continue
                    transformed = rotated(
                        points, rotation_angle, item.source_width, item.source_height
                    )
                    curves.append([(x * scale_x, y * scale_y) for x, y in transformed])
            self.preview_view.set_detected_edges(curves)
            handle_box = (
                preview_box
                if self.show_full_preview
                else (0, 0, crop_width, crop_height)
            )
            self.preview_view.edit_polygon = (
                quad if quad is not None else self.crop_outline_polygon(handle_box)
            )
            self.preview_view.edit_enabled = self.offset_x_spin.isEnabled()
            # The page turns about the same center the exact render uses.
            self.preview_view.rotate_enabled = self.rotation_spin.isEnabled()
            self.preview_view.rotate_pivot = QPointF(*rotation_center)
            side_text = {"left": "左", "right": "右", "none": "无"}[
                sides[self.current_index]
            ]
            inferred = item.path in self.inferred_paths
            verification_note = (
                "\n⚠ 自动检测失败：当前为推定裁切框，请手动确认" if inferred else ""
            )
            self.page_label.setText(
                f"{self.current_index + 1} / {len(self.detections)}　"
                f"{item.path.name}"
                f"{'　⚠ 需确认' if inferred else ''}"
            )
            self.geometry_label.setText(
                f"统一输出：{width} × {height} px\n"
                f"本页输出：{box[2] - box[0]} × {box[3] - box[1]} px\n"
                f"本页裁切框：{box}\n"
                f"本页偏移：X {self.offset_x_spin.value():+d} / "
                f"Y {self.offset_y_spin.value():+d} px\n"
                f"本页旋转：{self.rotation_spin.value():+.4f}°\n"
                f"内侧：{side_text}（{notes[self.current_index]}）\n"
                f"边缘起伏：左 {item.left_roughness:.0f} / 右 "
                f"{item.right_roughness:.0f} px"
                f"{verification_note}{self.smart_note(item.path)}"
            )
            self.status_label.setText(
                "快速预览；正在后台生成预览…" if provisional else "预览已更新"
            )
        except Exception as exc:
            self.page_output_size_label.setText("尺寸无效")
            self.preview_view.show_message(str(exc))
            self.geometry_label.setText(str(exc))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "settings_save_timer"):
            self.settings_save_timer.start()

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "settings_save_timer"):
            self.settings_save_timer.start()

    def closeEvent(self, event):
        self.edge_jump_pending = None
        self.stop_edge_sweep()
        self.settings_save_timer.stop()
        self.save_user_settings()
        self.smart_timer.stop()
        if self.smart_worker:
            self.smart_worker.cancel.set()
        for cancellation in (
            self.preview_cache_cancel_event,
            self.current_rotation_cancel_event,
            *self.draft_inflight.values(),
        ):
            if cancellation:
                cancellation.set()
        self.cancel_full_previews()
        self.thread_pool.waitForDone()
        self.rotation_thread_pool.waitForDone()
        self.full_preview_thread_pool.waitForDone()
        self.draft_thread_pool.waitForDone()
        event.accept()

    def previous_page(self) -> None:
        if not self.detections:
            return
        self.current_index = (self.current_index - 1) % len(self.detections)
        self.load_current_preview()

    def next_page(self) -> None:
        if not self.detections:
            return
        self.current_index = (self.current_index + 1) % len(self.detections)
        self.load_current_preview()

    def jump_to_page(self, page: int) -> None:
        if not self.detections:
            return
        index = max(0, min(page - 1, len(self.detections) - 1))
        if index == self.current_index:
            return
        self.current_index = index
        self.load_current_preview()

    def export_all(self) -> None:
        self.export_pages()

    def export_selected(self) -> None:
        if not self.detections or self.smart_worker is not None or self.smart_timer.isActive():
            return
        text, ok = QInputDialog.getText(self, "导出指定页码",
            "按当前图片顺序填写页码，例如 1,3,5-8：", text=str(self.current_index+1))
        if not ok:
            return
        try:
            pages = parse_page_numbers(text,len(self.detections))
            if not pages:
                raise ValueError("请填写需要导出的页码。")
        except ValueError as exc:
            QMessageBox.warning(self,"页码无效",str(exc))
            return
        self.export_pages(pages)

    def export_pages(self, page_numbers=None) -> None:
        if self.smart_worker is not None or self.smart_timer.isActive():
            return
        if not self.detections:
            return
        output_text = self.output_edit.text().strip()
        if not output_text:
            QMessageBox.warning(self, "缺少输出目录", "请选择输出目录。")
            return
        output_dir = Path(output_text)
        input_dir = self.detections[0].path.parent.resolve()
        if output_dir.resolve() == input_dir:
            QMessageBox.warning(self, "输出目录无效", "输出目录不能等于原图目录。")
            return
        try:
            _, _, boxes, _, _ = self.current_geometry()
        except Exception as exc:
            QMessageBox.warning(self, "裁切参数无效", str(exc))
            return

        selected = [i for i in range(len(self.detections))
                    if page_numbers is None or i+1 in page_numbers]
        detections = [self.detections[i] for i in selected]
        boxes = [boxes[i] for i in selected]

        conflicts = [
            output_dir / item.path.name
            for item in detections
            if (output_dir / item.path.name).exists()
        ]
        if conflicts:
            answer = QMessageBox.question(
                self,
                "覆盖已有文件？",
                f"输出目录中已有 {len(conflicts)} 个同名文件，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._set_controls_enabled(False)
        self.analyze_button.setEnabled(False)
        self.edit_color_profiles_action.setEnabled(False)
        self.detect_color_shift_action.setEnabled(False)
        self.export_batch_config_action.setEnabled(False)
        self.import_batch_config_action.setEnabled(False)
        rotation_centers = [
            (item.source_width / 2, item.source_height / 2) for item in detections
        ]
        rotation_angles = [
            self.page_rotations.get(item.path, 0.0) for item in detections
        ]
        worker = ExportWorker(
            detections,
            boxes,
            rotation_centers,
            rotation_angles,
            output_dir,
            self.current_channel_shifts(),
        )
        worker.perspectives = {
            d.path: state_matrix(self.perspectives.get(d.path)) for d in detections
        }
        worker.signals.progress.connect(self.worker_progress)
        worker.signals.finished.connect(self.export_finished)
        worker.signals.error.connect(self.worker_error)
        self.active_worker = worker
        self.thread_pool.start(worker)

    def export_finished(self, result: object) -> None:
        self.active_worker = None
        self.analyze_button.setEnabled(True)
        self._set_controls_enabled(True)
        self.edit_color_profiles_action.setEnabled(True)
        self.detect_color_shift_action.setEnabled(bool(self.detections))
        self.export_batch_config_action.setEnabled(bool(self.detections))
        self.import_batch_config_action.setEnabled(True)
        output_dir = Path(result)  # type: ignore[arg-type]
        self.status_label.setText(f"导出完成：{output_dir}")
        QMessageBox.information(self, "导出完成", f"裁切图片已写入：\n{output_dir}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("漫画扫描自动裁切")
    initial: Path | None = None
    if len(sys.argv) > 1:
        initial = Path(sys.argv[1])
    window = CropperWindow(initial)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
