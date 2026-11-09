"""Scanner adaptations: every detection step written for one specific setup.

The generic detection in scan_cropper_auto assumes nothing about a
particular scanner. This file adds steps for the author's scanner and way of
placing pages:

* top rim: a bright rim along the top of the bed, often followed by a dark
  groove, which would otherwise be taken as the paper's top edge
  (unsuitable when pages are placed flush against the top of the scan area);
* ruler: the scanner's ruler or frame at the top in fixed-width scans;
* hidden print: dark print at a page edge as dark as the grainy black bed,
  found by texture and accepted only when it brings the page to the batch's
  typical height.

While this file exists all of it is used. Deleting it is safe: the core and
GUI fall back to the generic detection with no other change.

Hooks called by scan_cropper_auto:
    top_rows, fixed_width_top_band, fixed_width_frame  — during page analysis
    after_batch, after_single_page                     — after analysis
    CACHE_KEY                                          — per-page cache key
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from scan_cropper_auto import SIDE_LABELS, check_cancel

# Adapted and generic per-page analyses of one file are cached separately.
# Change this when a per-page step here changes its results.
CACHE_KEY = '|adapters:1'


# --- top rim ---------------------------------------------------------------

def top_rows(blur, left, right, threshold, level, noise, factor):
    """Hook before thresholding a dark-backed scan: (blur, rim, scan_start)."""
    return _top_rim_correction(blur, left, right, threshold, level, noise, factor)


# --- ruler -----------------------------------------------------------------

def fixed_width_top_band(smoothed, background, noise, factor):
    """Hook for fixed-width scans on a light backing: (smoothed, rim).

    Remove scanner-wide ruler/groove illumination from the coarse copy. A
    narrow outer-background band distinguishes it from real page ink.
    """
    rim = 1
    sh, sw = smoothed.shape
    band = max(12, round(sh*.035)); outer = max(3, sw//100)
    outside = np.median(smoothed[:band, -outer:], axis=1)
    expected = np.median(background[:band, -outer:], axis=1)
    ruler = outside-expected > max(30., noise*8)
    changes = np.diff(np.r_[False, ruler, False].astype(np.int8))
    runs = [(a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if a < band and b-a < sh*.02]
    if runs:
        smoothed[:band] -= (outside-expected)[:, None]
        rim = max(1, round(max(b for a, b in runs)*factor)+8)
    return smoothed, rim


def fixed_width_frame(gray, region, right, margin, left, x0, level, bg):
    """Hook for fixed-width scans on a dark backing: (region, frame_end).

    A scanner ruler can touch the tilted page, so row components alone do not
    separate it. Identify it in the exposed bed to the right of the page.
    """
    h, w = gray.shape
    outside_x = min(w-3, right+max(12, margin//3))
    band_height = max(24, round(h*.035))
    outside_rows = np.median(gray[:band_height, outside_x:], axis=1)
    ruler = outside_rows > level+max(30., 8*float(np.median(np.abs(bg-level))))
    changes = np.diff(np.r_[False, ruler, False].astype(np.int8))
    runs = [(start, end) for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
            if start < h*.02 and end-start < h*.02]
    frame_end = 0
    if runs:
        # Cancel the shared scanner-row brightness pattern, including the dark
        # groove beneath the bright ruler. Blanking a horizontal band would
        # erase real paper corners where a tilted page overlaps that band.
        region[:band_height] = np.clip(region[:band_height].astype(np.float32)
                                       - outside_rows[:, None]+level, 0, 255).astype(np.uint8)
        for start, end in runs:
            region[max(0, start-4):min(h, end+4)] = round(level)
            frame_end = max(frame_end, end+8)
        region[:, :max(0, left-x0)] = round(level)
    return region, frame_end


# --- hidden print ----------------------------------------------------------

def after_batch(paths, pages, cancel=None):
    """Hook after a whole batch has been analyzed."""
    expected = typical_height(pages)
    return [recover_hidden_print(path, page, expected, cancel) for path, page in zip(paths, pages)]


def after_single_page(path, page, batch_pages):
    """Hook after one page is re-analyzed, checked against the rest of its batch."""
    return recover_hidden_print(path, page, typical_height(batch_pages))


TEXTURE_WINDOW=15


def _bed_texture(gray, page):
    """Typical row-wise grain of the exposed backing beside the page.

    Only sides with a reliable edge are used (in fixed-width mode the inner
    side is the neighbouring page, not backing). Returns None when there is
    no usable strip or the backing is too smooth to tell apart from ink.
    """
    h,w=gray.shape
    y0,y1=int(h*.25),int(h*.75)
    strips=[]
    if page.confidence['left']>=.6:
        left=int(np.median(np.asarray(page.edges['left'])[:,0]))-30
        if left>=2*TEXTURE_WINDOW+10:
            strips.append(gray[y0:y1,left-2*TEXTURE_WINDOW:left])
    if page.confidence['right']>=.6:
        right=int(np.median(np.asarray(page.edges['right'])[:,0]))+30
        if w-right>=2*TEXTURE_WINDOW+10:
            strips.append(gray[y0:y1,right:right+2*TEXTURE_WINDOW])
    if not strips:
        return None
    grain=float(np.median([np.median(s.astype(np.float32).std(axis=1)) for s in strips]))
    return grain if grain>=4 else None


def _extend_hidden_edge(gray, side, points, bed_grain, limit=1):
    """Move an edge outward across dark print that matches the bed's tone.

    Brightness cannot separate a dark printed band at the page edge from the
    scanner bed, but texture can: flat ink is smooth, the bed is grainy. If
    the strip just outside the traced edge is clearly smoother than the bed
    at most stations, and the bed's grain starts at a consistent distance,
    the real paper edge is there. Returns (points, shift) or None.
    """
    h,w=gray.shape
    vertical=side in ('left','right')
    outward=-1 if side in ('left','top') else 1
    length=w if vertical else h
    reach=max(60,int((h if not vertical else w)*.04))
    half=TEXTURE_WINDOW
    shifts=[]; inner=[]
    for x,y in np.asarray(points)[::3]:
        along=int(round(y if vertical else x)); edge=int(round(x if vertical else y))
        if along-half<0 or along+half>(h if vertical else w):
            continue
        start=edge+outward*2
        stop=start+outward*reach
        stop=min(max(stop,max(limit,0)),length-1) if outward<0 else min(stop,length-1)
        if (stop-start)*outward<2*half:
            continue
        positions=np.arange(start,stop,outward)
        block=(gray[along-half:along+half,positions] if vertical
               else gray[positions,along-half:along+half])
        grain=block.astype(np.float32).std(axis=0 if vertical else 1)
        # Median over ~15 px rejects single lines of print or bed specks.
        grain=np.median(np.lib.stride_tricks.sliding_window_view(
            np.pad(grain,7,mode='edge'),15),axis=1)
        inner.append(float(np.median(grain[:12])))
        found=np.flatnonzero(grain>=.7*bed_grain)
        shifts.append(int(found[0])+2 if len(found) else -1)
    if len(shifts)<8:
        return None
    shifts=np.asarray(shifts); inner=np.asarray(inner)
    hidden=(shifts>8)&(inner<.6*bed_grain)
    if hidden.mean()<.6:
        return None
    shift=float(np.median(shifts[hidden]))
    spread=float(np.percentile(shifts[hidden],75)-np.percentile(shifts[hidden],25))
    if spread>max(10.,shift*.25):
        return None
    moved=np.asarray(points,dtype=float).copy()
    moved[:,0 if vertical else 1]+=outward*shift
    return moved,shift


HIDDEN_PRINT_TOLERANCE=12


def page_height(page):
    return (float(np.median(np.asarray(page.edges['bottom'])[:,1]))
            -float(np.median(np.asarray(page.edges['top'])[:,1])))


def typical_height(pages):
    """Robust typical paper height of a batch, or None if too few pages."""
    heights=np.array([page_height(p) for p in pages
                      if min(p.confidence['top'],p.confidence['bottom'])>=.6])
    if len(heights)<4:
        return None
    center=float(np.median(heights))
    spread=max(4.,1.4826*float(np.median(np.abs(heights-center))))
    inliers=heights[np.abs(heights-center)<=3*spread]
    return float(np.median(inliers)) if len(inliers)>=3 else None


def recover_hidden_print(path, page, expected_height, cancel=None):
    """Hidden print, verified against the batch.

    Only a page clearly shorter than the batch's typical height is examined.
    A texture-based extension of its top, bottom or both is accepted only if
    it brings the page to the typical height, so a smooth pressure board or
    other non-print region beside the page cannot pull an edge outward.
    """
    if expected_height is None:
        return page
    height=page_height(page)
    if height>=expected_height-HIDDEN_PRINT_TOLERANCE:
        return page
    check_cancel(cancel)
    with Image.open(path) as image:
        gray=np.asarray(image.convert('L'))
    if gray.shape!=(page.height,page.width):
        return page
    grain=_bed_texture(gray,page)
    if grain is None:
        return page
    found={}
    for side in ('top','bottom'):
        extended=_extend_hidden_edge(gray,side,page.edges[side],grain)
        if extended is not None:
            found[side]=extended
    best=None
    for sides in (('top',),('bottom',),('top','bottom')):
        if not all(side in found for side in sides):
            continue
        gained=sum(found[side][1] for side in sides)
        error=abs(height+gained-expected_height)
        if error<=HIDDEN_PRINT_TOLERANCE and (best is None or error<best[0]):
            best=(error,sides)
    if best is None:
        return page
    edges=dict(page.edges)
    warnings=list(page.warnings)
    for side in best[1]:
        points,shift=found[side]
        edges[side]=points.tolist()
        warnings.append(f"{SIDE_LABELS[side]}边外侧为与背景同色的深色印刷，已按纹理外推约 {shift:.0f}px（与整批高度一致）")
    return replace(page,edges=edges,warnings=warnings)


# --- top rim helpers -------------------------------------------------------

def _top_rim_correction(blur, left, right, threshold, level, noise, factor):
    """Top rim: a bright rim along the top of the scanner bed,
    often followed by a dark groove.

    The rim's row-wise excess, measured on the exposed bed beside the page,
    is removed before thresholding so neither it nor its blurred tail becomes
    the top edge (paper overlapping the rim stays brighter than the bed).
    Returns the corrected blur, the lowest row the top trace may use, and the
    row from which the first-step top search starts (below any groove).
    """
    sh,sw=blur.shape
    left_space=max(2,left//2)
    right_space=min(sw-2,right+(sw-right)//2)
    band=max(12,sh//40)
    outside=np.median(np.column_stack((blur[:band,:left_space],
                                     blur[:band,right_space:])),axis=1)
    bright=outside>threshold
    rim_rows=int(np.argmin(bright)) if not bright.all() else band
    if rim_rows:
        # Include the blurred falling edge until the bed returns to its level.
        settle=np.flatnonzero(outside[rim_rows:]<=level+max(6.,noise*3))
        rim_rows=min(band,rim_rows+(int(settle[0]) if len(settle) else band))
        blur=blur.copy()
        blur[:rim_rows]-=np.clip(outside[:rim_rows]-level,0,None)[:,None]
    # The full-resolution trace may still search into the rim's falling edge:
    # the physical paper can begin before the background strip becomes dark.
    rim = max(1,round(rim_rows*factor)-round(6*factor)) if rim_rows else 1
    # The groove's rise back to the bed level must not be taken as the paper
    # edge: the top search starts once the bed has settled at its brightness.
    scan_start = rim
    if rim_rows and rim_rows < band:
        bed = float(np.median(outside[rim_rows:]))
        near = np.abs(outside[rim_rows:]-bed) <= max(6., noise*3)
        # A short run: the falling rim edge crosses the bed level briefly on
        # its way into the groove.
        run = 3
        settled = (np.flatnonzero(np.convolve(near, np.ones(run), 'valid') == run)
                   if len(near) >= run else np.flatnonzero(near))
        if len(settled):
            scan_start = max(rim, round((rim_rows+int(settled[0]))*factor))
    return blur, rim, scan_start
