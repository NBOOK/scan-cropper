#!/usr/bin/env python3
"""Batch-crop top-aligned scanned pages while detecting horizontal placement.

The detector uses the dark scanner bed as its background model.  It votes over
most rows of each page instead of following a single contour, so a local torn
notch does not pull the crop edge inward.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import cv2
except ImportError:  # Pillow remains a compatible fallback.
    cv2 = None
else:
    # Batch parallelism is managed by the GUI; avoid 8 workers each creating
    # another full OpenCV thread pool.
    cv2.setNumThreads(1)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class Detection:
    path: Path
    source_width: int
    source_height: int
    left: float
    right: float
    bottom: float
    threshold: float
    left_roughness: float
    right_roughness: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0


@dataclass(frozen=True)
class ChannelShiftMeasurement:
    red_x: float
    red_y: float
    blue_x: float
    blue_y: float
    red_samples: int
    blue_samples: int
    red_response: float
    blue_response: float
    red_mad: float
    blue_mad: float
    neutral_fraction: float


def smooth(values: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    kernel = np.ones(radius * 2 + 1, dtype=np.float32)
    kernel /= kernel.size
    return np.convolve(values, kernel, mode="same")


def longest_run(indices: np.ndarray) -> tuple[int, int]:
    if indices.size == 0:
        raise ValueError("没有找到纸张区域")
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, indices.size - 1]
    lengths = indices[ends] - indices[starts] + 1
    best = int(np.argmax(lengths))
    return int(indices[starts[best]]), int(indices[ends[best]])


def rotated_full_box(
    width: int,
    height: int,
    rotation_degrees: float,
    center: tuple[float, float] | None = None,
) -> tuple[int, int, int, int]:
    """Bounds of an image rotated clockwise in its original coordinates."""
    if abs(rotation_degrees) < 1e-9:
        return (0, 0, width, height)
    center_x, center_y = center or (width / 2, height / 2)
    radians = math.radians(rotation_degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    corners = [
        (
            center_x + cosine * (x - center_x) - sine * (y - center_y),
            center_y + sine * (x - center_x) + cosine * (y - center_y),
        )
        for x, y in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    return (
        math.floor(min(x for x, _ in corners)),
        math.floor(min(y for _, y in corners)),
        math.ceil(max(x for x, _ in corners)),
        math.ceil(max(y for _, y in corners)),
    )


def detect_page(
    path: Path,
    analysis_width: int,
    min_bright_fraction: float,
    rotation_degrees: float = 0.0,
) -> Detection:
    with Image.open(path) as source:
        source_width, source_height = source.size
        gray = ImageOps.grayscale(source)

    full_box = rotated_full_box(source_width, source_height, rotation_degrees)
    if abs(rotation_degrees) >= 1e-9:
        gray = transformed_crop(
            gray,
            full_box,
            rotation_degrees,
            (source_width / 2, source_height / 2),
        )
    full_width, full_height = gray.size
    work_width = min(full_width, analysis_width)
    work_height = max(1, round(full_height * work_width / full_width))
    if (work_width, work_height) != gray.size:
        gray = gray.resize((work_width, work_height), Image.Resampling.BILINEAR)
    scale_x = full_width / work_width
    scale_y = full_height / work_height

    pixels = np.asarray(gray, dtype=np.float32)

    # The scan may be cropped close to the page on any one side.  At least one
    # of bottom/left/right normally still contains scanner-bed background, so
    # select the darkest robust side estimate instead of assuming the bottom.
    side_fraction = 0.045
    side_height = max(4, round(work_height * side_fraction))
    side_width = max(4, round(work_width * side_fraction))
    background_candidates = (
        pixels[-side_height:, :],
        pixels[:, :side_width],
        pixels[:, -side_width:],
    )
    background_level = min(
        float(np.percentile(strip, 80)) for strip in background_candidates
    )
    # A darker side can otherwise set a threshold low enough that textured
    # scanner-bed pixels on the opposite side are classified as paper.
    threshold = min(150.0, max(90.0, background_level + 14.0))

    usable_rows = pixels[: round(work_height * 0.88), :]
    raw_column_fraction = np.mean(usable_rows > threshold, axis=0)
    column_fraction = smooth(raw_column_fraction, max(2, round(work_width * 0.006)))
    paper_columns = np.flatnonzero(column_fraction > min_bright_fraction)
    left_i, right_i = longest_run(paper_columns)

    # The wide smoothing window finds the page reliably but moves its edges
    # outward by several source pixels. Refine each edge at the steepest local
    # paper/background transition before converting back to full resolution.
    edge_gradient = np.diff(smooth(raw_column_fraction, 2))
    search_radius = max(4, round(work_width * 0.015))
    left_lo = max(0, left_i - search_radius)
    left_hi = min(edge_gradient.size, left_i + search_radius + 1)
    right_lo = max(0, right_i - search_radius)
    right_hi = min(edge_gradient.size, right_i + search_radius + 1)
    if left_lo < left_hi:
        left_gradient = edge_gradient[left_lo:left_hi]
        if float(left_gradient.max()) > 0.01:
            left_i = left_lo + int(np.argmax(left_gradient)) + 1
    if right_lo < right_hi:
        right_gradient = edge_gradient[right_lo:right_hi]
        if float(right_gradient.min()) < -0.01:
            right_i = right_lo + int(np.argmin(right_gradient))

    if right_i - left_i < work_width * 0.45:
        raise ValueError("检测到的纸张过窄；请检查扫描背景或调低 --bright-fraction")

    # Convert the inclusive run to continuous source-image edges.
    left = full_box[0] + left_i * scale_x
    right = full_box[0] + (right_i + 1) * scale_x

    inner_margin = max(2, round((right_i - left_i) * 0.015))
    x0 = min(right_i, left_i + inner_margin)
    x1 = max(x0 + 1, right_i - inner_margin)
    row_fraction = np.mean(pixels[:, x0:x1] > threshold, axis=1)
    row_fraction = smooth(row_fraction, max(2, round(work_height * 0.004)))

    paper_rows = np.flatnonzero(row_fraction > min_bright_fraction * 0.9)
    _, rough_bottom = longest_run(paper_rows)
    search_radius = max(8, round(work_height * 0.018))
    derivative_radius = max(2, round(work_height * 0.003))
    lo = max(derivative_radius, rough_bottom - search_radius)
    hi = min(
        work_height - derivative_radius - 1,
        rough_bottom + search_radius,
    )
    if hi <= lo:
        bottom_i = rough_bottom
    else:
        centers = np.arange(lo, hi + 1)
        falling = (
            row_fraction[centers + derivative_radius]
            - row_fraction[centers - derivative_radius]
        )
        bottom_i = int(centers[int(np.argmin(falling))])
    bottom = full_box[1] + bottom_i * scale_y

    if bottom < source_height * 0.45:
        raise ValueError("检测到的纸张高度过短；请检查扫描背景或调低 --bright-fraction")

    def edge_roughness(side: str) -> float:
        band_height = max(5, round(work_height * 0.006))
        search_radius = max(12, round(work_width * 0.025))
        positions: list[float] = []
        y_start = max(band_height, round(work_height * 0.02))
        y_stop = min(bottom_i - band_height, round(work_height * 0.94))
        for y in range(y_start, y_stop, band_height):
            band = pixels[y : y + band_height]
            if side == "left":
                start = max(0, left_i - search_radius)
                stop = min(work_width, left_i + search_radius + 1)
                profile = np.mean(band[:, start:stop] > threshold, axis=0)
                profile = smooth(profile, 2)
                candidates = np.flatnonzero(profile > min_bright_fraction)
                if candidates.size:
                    positions.append(float(start + candidates[0]))
            else:
                start = max(0, right_i - search_radius)
                stop = min(work_width, right_i + search_radius + 1)
                profile = np.mean(band[:, start:stop] > threshold, axis=0)
                profile = smooth(profile, 2)
                candidates = np.flatnonzero(profile > min_bright_fraction)
                if candidates.size:
                    positions.append(float(start + candidates[-1]))

        if len(positions) < 12:
            return 0.0
        values = np.asarray(positions, dtype=np.float32)
        median = float(np.median(values))
        # Reject large shifts caused by a dark full-bleed drawing rather than
        # the physical page edge, while keeping ordinary tears and burrs.
        values = values[np.abs(values - median) <= search_radius * 0.7]
        if values.size < 10:
            return 0.0
        q10, q90 = np.percentile(values, [10, 90])
        return float((q90 - q10) * scale_x)

    left_roughness = edge_roughness("left")
    right_roughness = edge_roughness("right")

    return Detection(
        path=path,
        source_width=source_width,
        source_height=source_height,
        left=left,
        right=right,
        bottom=bottom,
        threshold=threshold,
        left_roughness=left_roughness,
        right_roughness=right_roughness,
    )


def _channel_detail(channel: np.ndarray) -> np.ndarray:
    values = channel.astype(np.float32) / 255.0
    return values - cv2.GaussianBlur(values, (0, 0), 2.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    midpoint = sorted_weights.sum() / 2
    index = int(np.searchsorted(np.cumsum(sorted_weights), midpoint))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _summarize_channel_measurements(
    samples: list[tuple[float, float, float, float]],
) -> tuple[float, float, int, float, float]:
    valid = [
        item
        for item in samples
        if item[2] >= 0.06 and abs(item[0]) <= 6 and abs(item[1]) <= 6
    ]
    if len(valid) < 6:
        raise ValueError("共享边缘不足，无法可靠测量通道偏移")
    dx = np.asarray([item[0] for item in valid], dtype=np.float64)
    dy = np.asarray([item[1] for item in valid], dtype=np.float64)
    weights = np.asarray(
        [item[2] * max(0.1, item[3]) for item in valid],
        dtype=np.float64,
    )
    median_x = _weighted_median(dx, weights)
    median_y = _weighted_median(dy, weights)
    radial_error = np.hypot(dx - median_x, dy - median_y)
    return (
        median_x,
        median_y,
        len(valid),
        float(np.median([item[2] for item in valid])),
        float(np.median(radial_error)),
    )


def detect_rgb_channel_shifts(
    path: Path,
    tile_size: int = 768,
) -> ChannelShiftMeasurement:
    """Measure R/B translations needed to align them to the green channel.

    Neutral monochrome pages are the best targets.  Multiple detailed tiles
    are measured independently so printed content or a local defect cannot
    dominate the result.
    """
    if cv2 is None:
        raise RuntimeError("检测颜色通道偏移需要 OpenCV")

    red_samples: list[tuple[float, float, float, float]] = []
    blue_samples: list[tuple[float, float, float, float]] = []
    neutral_fractions: list[float] = []
    with Image.open(path) as source:
        if source.mode in {"1", "L", "LA", "I", "F"} or source.mode.startswith("I;16"):
            raise ValueError(
                "灰度图不含独立 RGB 通道，已跳过；请选择以 RGB 模式扫描的黑白／灰度内容页"
            )
        rgb_source = source.convert("RGB")
        width, height = rgb_source.size
        size = min(tile_size, width, height)
        x_centers = np.linspace(0.14 * width, 0.86 * width, 5)
        y_centers = np.linspace(0.14 * height, 0.86 * height, 5)
        window = cv2.createHanningWindow((size, size), cv2.CV_32F)
        for center_y in y_centers:
            for center_x in x_centers:
                left = int(np.clip(round(center_x - size / 2), 0, width - size))
                top = int(np.clip(round(center_y - size / 2), 0, height - size))
                tile = np.asarray(rgb_source.crop((left, top, left + size, top + size)))
                gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
                if float(cv2.Laplacian(gray, cv2.CV_32F).std()) < 3.0:
                    continue
                chroma = tile.max(axis=2).astype(np.int16) - tile.min(axis=2).astype(
                    np.int16
                )
                neutral_fraction = float(np.mean(chroma <= 20))
                neutral_fractions.append(neutral_fraction)
                red, green, blue = cv2.split(tile)
                red_shift, red_response = cv2.phaseCorrelate(
                    _channel_detail(red),
                    _channel_detail(green),
                    window,
                )
                blue_shift, blue_response = cv2.phaseCorrelate(
                    _channel_detail(blue),
                    _channel_detail(green),
                    window,
                )
                red_samples.append(
                    (
                        float(red_shift[0]),
                        float(red_shift[1]),
                        float(red_response),
                        neutral_fraction,
                    )
                )
                blue_samples.append(
                    (
                        float(blue_shift[0]),
                        float(blue_shift[1]),
                        float(blue_response),
                        neutral_fraction,
                    )
                )

    red_x, red_y, red_count, red_response, red_mad = _summarize_channel_measurements(
        red_samples
    )
    blue_x, blue_y, blue_count, blue_response, blue_mad = (
        _summarize_channel_measurements(blue_samples)
    )
    return ChannelShiftMeasurement(
        red_x=red_x,
        red_y=red_y,
        blue_x=blue_x,
        blue_y=blue_y,
        red_samples=red_count,
        blue_samples=blue_count,
        red_response=red_response,
        blue_response=blue_response,
        red_mad=red_mad,
        blue_mad=blue_mad,
        neutral_fraction=(
            float(np.median(neutral_fractions)) if neutral_fractions else 0.0
        ),
    )


def even_round(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def choose_common_size(detections: list[Detection]) -> tuple[int, int]:
    widths = [item.width for item in detections]
    bottoms = [item.bottom for item in detections]
    width = even_round(statistics.median(widths))
    height = even_round(statistics.median(bottoms))
    return width, height


def centered_crop_box(
    item: Detection, width: int, height: int, top: int = 0
) -> tuple[int, int, int, int]:
    left = int(round(item.center_x - width / 2))
    if width <= 0 or height <= 0:
        raise ValueError("裁切参数产生了非正数尺寸")
    return left, top, left + width, top + height


def outer_anchored_crop_box(
    item: Detection,
    output_width: int,
    height: int,
    top: int,
    inner_side: str,
    outer_inset: int,
) -> tuple[int, int, int, int]:
    if inner_side == "left":
        # The smooth right edge is the anchor; all remaining width extends inward.
        right = int(round(item.right)) - outer_inset
        left = right - output_width
    elif inner_side == "right":
        # The smooth left edge is the anchor.
        left = int(round(item.left)) + outer_inset
        right = left + output_width
    else:
        raise ValueError(f"未知内侧方向：{inner_side}")

    if output_width <= 0 or height <= 0 or right - left != output_width:
        raise ValueError("裁切参数产生了无效尺寸")
    return left, top, right, top + height


def inner_side_for(item: Detection, mode: str, index: int) -> tuple[str, str]:
    if mode == "left":
        return "left", "指定"
    if mode == "right":
        return "right", "指定"
    if mode == "alternate-left":
        return ("left" if index % 2 == 0 else "right"), "奇偶交替"
    if mode == "alternate-right":
        return ("right" if index % 2 == 0 else "left"), "奇偶交替"
    if mode == "auto":
        side = "left" if item.left_roughness >= item.right_roughness else "right"
        larger = max(item.left_roughness, item.right_roughness)
        smaller = min(item.left_roughness, item.right_roughness)
        confidence = (larger - smaller) / max(larger, 1.0)
        return side, f"自动，置信差={confidence:.0%}"
    return "none", "对称"


def transformed_crop(
    source: Image.Image,
    box: tuple[int, int, int, int],
    rotation_degrees: float = 0.0,
    rotation_center: tuple[float, float] | None = None,
    source_shift: tuple[float, float] = (0.0, 0.0),
    perspective=None,
    pixels: np.ndarray | None = None,
) -> Image.Image:
    """Translate in source coordinates, then rotate/crop in one resampling.

    ``pixels`` may carry ``np.asarray(source)`` when the caller renders many
    regions of one source, so the array is not copied for every region.
    """
    shift_x, shift_y = source_shift
    if perspective is not None:
        from scan_cropper_perspective import combined_matrix

        forward = combined_matrix(
            source.width, source.height, rotation_degrees, rotation_center, perspective
        )
        inverse = np.linalg.inv(forward)
        inverse = (
            np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y], [0.0, 0.0, 1.0]])
            @ inverse
        )
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            raise ValueError("裁切参数产生了非正数尺寸")
        inverse = inverse @ np.array(
            [[1.0, 0.0, left], [0.0, 1.0, top], [0.0, 0.0, 1.0]]
        )
        inverse /= inverse[2, 2]
        if cv2 is not None:
            result = cv2.warpPerspective(
                np.asarray(source) if pixels is None else pixels,
                inverse,
                (right - left, bottom - top),
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            return Image.fromarray(result, mode=source.mode)
        return source.transform(
            (right - left, bottom - top),
            Image.Transform.PERSPECTIVE,
            tuple(inverse.ravel()[:8]),
            resample=Image.Resampling.BICUBIC,
            fillcolor=0,
        )
    if abs(rotation_degrees) < 1e-9 and abs(shift_x) < 1e-9 and abs(shift_y) < 1e-9:
        return source.crop(box)

    left, top, right, bottom = box
    output_width = right - left
    output_height = bottom - top
    if output_width <= 0 or output_height <= 0:
        raise ValueError("裁切参数产生了非正数尺寸")

    center_x, center_y = rotation_center or (
        (left + right) / 2,
        (top + bottom) / 2,
    )
    # A positive angle rotates the page clockwise in image coordinates.
    # Affine transforms map output pixels back to source pixels.
    radians = math.radians(-rotation_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)

    # Pillow's affine transform maps output coordinates back into the source.
    # This is equivalent to expanding the rotated canvas and then taking the
    # requested fixed-size crop, without allocating the large intermediate.
    matrix = (
        cosine,
        -sine,
        center_x + cosine * (left - center_x) - sine * (top - center_y) - shift_x,
        sine,
        cosine,
        center_y + sine * (left - center_x) + cosine * (top - center_y) - shift_y,
    )
    if cv2 is not None:
        pixels = np.asarray(source) if pixels is None else pixels
        if pixels.dtype in (np.uint8, np.uint16, np.float32) and pixels.ndim in (
            2,
            3,
        ):
            affine = np.asarray(
                (matrix[:3], matrix[3:]),
                dtype=np.float64,
            )
            transformed = cv2.warpAffine(
                pixels,
                affine,
                (output_width, output_height),
                flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                borderMode=(
                    cv2.BORDER_REFLECT_101
                    if abs(shift_x) >= 1e-9 or abs(shift_y) >= 1e-9
                    else cv2.BORDER_CONSTANT
                ),
                borderValue=0,
            )
            return Image.fromarray(transformed, mode=source.mode)

    transform_source = source
    transform_matrix = matrix
    if abs(shift_x) >= 1e-9 or abs(shift_y) >= 1e-9:
        pad = max(4, math.ceil(abs(shift_x)), math.ceil(abs(shift_y))) + 2
        pixels = np.asarray(source)
        pad_width = (
            ((pad, pad), (pad, pad))
            if pixels.ndim == 2
            else ((pad, pad), (pad, pad), (0, 0))
        )
        transform_source = Image.fromarray(np.pad(pixels, pad_width, mode="reflect"))
        transform_matrix = (
            matrix[0],
            matrix[1],
            matrix[2] + pad,
            matrix[3],
            matrix[4],
            matrix[5] + pad,
        )
    return transform_source.transform(
        (output_width, output_height),
        Image.Transform.AFFINE,
        transform_matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )


def transformed_crop_with_channel_shifts(
    image: Image.Image,
    box: tuple[int, int, int, int],
    rotation_degrees: float,
    rotation_center: tuple[float, float] | None,
    channel_shifts: dict[str, tuple[float, float]] | None,
    perspective=None,
) -> Image.Image:
    """Align RGB in unrotated scanner coordinates, then rotate and crop."""
    if not channel_shifts or image.mode not in {"RGB", "RGBA"}:
        return transformed_crop(
            image,
            box,
            rotation_degrees,
            rotation_center,
            perspective=perspective,
        )

    bands = image.split()
    corrected_bands: list[Image.Image] = []
    for band, channel in zip(bands[:3], ("red", "green", "blue")):
        # The source shift is applied before the rotation mapping; its X/Y
        # components must stay in the original scanner coordinate system.
        corrected_bands.append(
            transformed_crop(
                band,
                box,
                rotation_degrees,
                rotation_center,
                channel_shifts.get(channel, (0.0, 0.0)),
                perspective=perspective,
            )
        )
    if image.mode == "RGBA":
        corrected_bands.append(
            transformed_crop(
                bands[3],
                box,
                rotation_degrees,
                rotation_center,
                perspective=perspective,
            )
        )
    return Image.merge(image.mode, corrected_bands)


class RenderCancelled(Exception):
    """A preview render was abandoned because its result is no longer wanted."""


def render_transformed(
    image: Image.Image,
    box: tuple[int, int, int, int],
    rotation_degrees: float,
    rotation_center: tuple[float, float] | None,
    channel_shifts: dict[str, tuple[float, float]] | None = None,
    perspective=None,
    executor=None,
    cancel=None,
    stripe_height: int = 512,
) -> Image.Image:
    """Exactly ``transformed_crop_with_channel_shifts``, rendered in parallel.

    Every output pixel is resampled from the source with the same mapping, so
    splitting the output into horizontal stripes (and RGB planes) changes only
    the scheduling, never a pixel. OpenCV releases the GIL during warps.
    """
    if executor is None or cv2 is None or image.mode not in {"L", "LA", "RGB", "RGBA"}:
        return transformed_crop_with_channel_shifts(
            image, box, rotation_degrees, rotation_center, channel_shifts, perspective
        )
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise ValueError("裁切参数产生了非正数尺寸")
    shifted = bool(channel_shifts) and image.mode in {"RGB", "RGBA"}
    if not shifted and perspective is None and abs(rotation_degrees) < 1e-9:
        return image.crop(box)
    if shifted:
        bands = image.split()
        planes = [
            (band, tuple(channel_shifts.get(channel, (0.0, 0.0))))
            for band, channel in zip(bands[:3], ("red", "green", "blue"))
        ]
        if image.mode == "RGBA":
            planes.append((bands[3], (0.0, 0.0)))
    else:
        planes = [(image, (0.0, 0.0))]
    arrays = [np.asarray(plane) for plane, _ in planes]
    height = bottom - top
    stripes = [
        (y, min(height, y + stripe_height)) for y in range(0, height, stripe_height)
    ]

    def render(index, y0, y1):
        if cancel is not None and cancel.is_set():
            raise RenderCancelled()
        plane, shift = planes[index]
        return np.asarray(
            transformed_crop(
                plane,
                (left, top + y0, right, top + y1),
                rotation_degrees,
                rotation_center,
                shift,
                perspective,
                pixels=arrays[index],
            )
        )

    futures = [
        (index, y0, y1, executor.submit(render, index, y0, y1))
        for index in range(len(planes))
        for y0, y1 in stripes
    ]
    try:
        if shifted:
            output = np.empty((height, right - left, len(planes)), arrays[0].dtype)
            for index, y0, y1, future in futures:
                output[y0:y1, :, index] = future.result()
            return Image.fromarray(output, image.mode)
        parts = [future.result() for _, _, _, future in futures]
        return Image.fromarray(np.concatenate(parts, axis=0), image.mode)
    except BaseException:
        for *_, future in futures:
            future.cancel()
        raise


def save_crop(
    source_path: Path,
    destination: Path,
    box: tuple[int, int, int, int],
    rotation_degrees: float = 0.0,
    rotation_center: tuple[float, float] | None = None,
    channel_shifts: dict[str, tuple[float, float]] | None = None,
    perspective=None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        metadata: dict[str, object] = {}
        for key in ("dpi", "icc_profile", "exif"):
            if source.info.get(key) is not None:
                metadata[key] = source.info[key]
        cropped = transformed_crop_with_channel_shifts(
            source,
            box,
            rotation_degrees,
            rotation_center,
            channel_shifts,
            perspective,
        )
        if destination.suffix.lower() in {".jpg", ".jpeg"}:
            metadata.update(quality=96, subsampling=0)
        cropped.save(destination, **metadata)


def make_preview(
    detections: list[Detection],
    boxes: list[tuple[int, int, int, int]],
    destination: Path,
) -> None:
    cell_width = 420
    cell_height = 620
    label_height = 28
    columns = min(4, len(detections))
    rows = math.ceil(len(detections) / columns)
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * (cell_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (item, box) in enumerate(zip(detections, boxes)):
        with Image.open(item.path) as source:
            preview = source.crop(box).convert("RGB")
            preview.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        paste_x = x + (cell_width - preview.width) // 2
        sheet.paste(preview, (paste_x, y + label_height))
        draw.text((x + 6, y + 7), item.path.name, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动寻找扫描页左右位置，并以统一尺寸批量裁切。"
    )
    parser.add_argument("input", type=Path, help="扫描原图目录")
    parser.add_argument(
        "-o", "--output", type=Path, help="输出目录（默认：输入目录/cropped）"
    )
    parser.add_argument(
        "--analysis-width",
        type=int,
        default=1275,
        help="检测时缩小到的宽度；越大越精确但越慢（默认：1275）",
    )
    parser.add_argument(
        "--bright-fraction",
        type=float,
        default=0.28,
        help="一列中判为纸张所需的亮像素比例（默认：0.28）",
    )
    parser.add_argument("--width", type=int, help="手动指定统一输出宽度")
    parser.add_argument("--height", type=int, help="手动指定统一输出高度")
    parser.add_argument(
        "--horizontal-inset",
        type=int,
        default=None,
        help=("强制左右对称内缩；指定后会覆盖内外侧模式"),
    )
    parser.add_argument(
        "--inner-side",
        choices=(
            "none",
            "left",
            "right",
            "alternate-left",
            "alternate-right",
            "auto",
        ),
        default="none",
        help=(
            "装订内侧方向：none=对称，left/right=固定，"
            "alternate-left/right=第一张从该侧开始交替，auto=按边缘粗糙度判断"
        ),
    )
    parser.add_argument(
        "--outer-inset",
        type=int,
        default=15,
        help="装订外侧裁切像素数，负数表示向外扩展（默认：15）",
    )
    parser.add_argument(
        "--inner-inset",
        type=int,
        default=25,
        help="装订内侧裁切像素数，负数表示向外扩展（默认：25）",
    )
    parser.add_argument(
        "--top-inset",
        type=int,
        default=15,
        help="顶部裁切像素数，负数表示向外扩展（默认：15）",
    )
    parser.add_argument(
        "--bottom-inset",
        type=int,
        default=0,
        help="底部裁切像素数，负数表示向外扩展（默认：0）",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="另外生成一张裁切结果联系表（JPEG）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只检测和报告，不写裁切图"
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = (args.output or input_dir / "cropped").resolve()
    if not input_dir.is_dir():
        print(f"错误：输入目录不存在：{input_dir}", file=sys.stderr)
        return 2
    if output_dir == input_dir:
        print("错误：输出目录不能与输入目录相同。", file=sys.stderr)
        return 2
    if not 0.05 <= args.bright_fraction <= 0.8:
        print("错误：--bright-fraction 应在 0.05 到 0.8 之间。", file=sys.stderr)
        return 2

    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        print(f"错误：{input_dir} 中没有支持的图片。", file=sys.stderr)
        return 2

    detections: list[Detection] = []
    for path in paths:
        try:
            item = detect_page(path, args.analysis_width, args.bright_fraction)
        except Exception as exc:
            print(f"检测失败：{path.name}：{exc}", file=sys.stderr)
            return 1
        detections.append(item)
        print(
            f"检测 {path.name}: 左={item.left:.0f}, 右={item.right:.0f}, "
            f"宽={item.width:.0f}, 底={item.bottom:.0f}, 阈值={item.threshold:.0f}, "
            f"边缘起伏=左{item.left_roughness:.0f}/右{item.right_roughness:.0f}"
        )

    auto_width, auto_height = choose_common_size(detections)
    height = args.height or (auto_height - args.top_inset - args.bottom_inset)
    if height <= 0:
        print("错误：顶部与底部裁量产生了非正数高度。", file=sys.stderr)
        return 2
    top = args.top_inset

    sides: list[str] = []
    side_notes: list[str] = []
    if args.width:
        width = args.width
        boxes = [centered_crop_box(item, width, height, top) for item in detections]
        sides = ["manual"] * len(detections)
        side_notes = ["手动宽度，居中"] * len(detections)
    elif args.horizontal_inset is not None:
        raw_width = auto_width - 2 * args.horizontal_inset
        if raw_width <= 0:
            print("错误：左右裁量产生了非正数宽度。", file=sys.stderr)
            return 2
        width = even_round(raw_width)
        boxes = [centered_crop_box(item, width, height, top) for item in detections]
        sides = ["none"] * len(detections)
        side_notes = [f"对称内缩 {args.horizontal_inset}px"] * len(detections)
    elif args.inner_side == "none":
        symmetric_inset = 12
        width = even_round(auto_width - 2 * symmetric_inset)
        boxes = [centered_crop_box(item, width, height, top) for item in detections]
        sides = ["none"] * len(detections)
        side_notes = [f"对称内缩 {symmetric_inset}px"] * len(detections)
    else:
        width = auto_width - args.outer_inset - args.inner_inset
        if width <= 0:
            print("错误：内外侧裁量产生了非正数宽度。", file=sys.stderr)
            return 2
        boxes = []
        for index, item in enumerate(detections):
            side, note = inner_side_for(item, args.inner_side, index)
            sides.append(side)
            side_notes.append(note)
            boxes.append(
                outer_anchored_crop_box(
                    item,
                    width,
                    height,
                    top,
                    side,
                    args.outer_inset,
                )
            )

    print(f"\n统一输出尺寸：{width} x {height}")
    for item, box, side, note in zip(detections, boxes, sides, side_notes):
        side_text = {"left": "左", "right": "右", "none": "无", "manual": "手动"}[side]
        print(f"裁切 {item.path.name}: {box}, 内侧={side_text}（{note}）")

    if args.dry_run:
        if args.preview:
            make_preview(detections, boxes, args.preview.resolve())
            print(f"预览联系表：{args.preview.resolve()}")
        return 0

    conflicts = [
        output_dir / item.path.name
        for item in detections
        if (output_dir / item.path.name).exists()
    ]
    if conflicts and not args.overwrite:
        print(
            f"错误：输出目录中已有 {len(conflicts)} 个同名文件；"
            "请换目录或加 --overwrite。",
            file=sys.stderr,
        )
        return 2

    for item, box in zip(detections, boxes):
        destination = output_dir / item.path.name
        save_crop(item.path, destination, box)
        print(f"已写入 {destination}")

    if args.preview:
        make_preview(detections, boxes, args.preview.resolve())
        print(f"预览联系表：{args.preview.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
