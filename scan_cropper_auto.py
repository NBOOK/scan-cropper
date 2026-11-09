"""Geometry-first crop recommendations; no resampling or edits to source files.

All coordinates are source pixels. Positive angles are clockwise, around the
source centre, matching scan_cropper.transformed_crop. Border complexity is a
heuristic, not face/hand recognition or a guarantee that filling is possible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import hashlib
import json
import math
import re
import threading

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

VERSION = 6
SIDES = ('left', 'right', 'top', 'bottom')
POLICIES = {'less_fill': (1.0, 4.0), 'balanced': (1.0, 1.6), 'preserve': (2.5, 1.0)}


_ADAPTERS = None


def adapters():
    """The optional scanner-adaptation module, or None when it is absent.

    scan_cropper_adapters holds every step written for one particular
    scanner setup. When the file exists its steps run at the few hook points
    below; deleting it leaves the generic detection.
    """
    global _ADAPTERS
    if _ADAPTERS is None:
        try:
            import scan_cropper_adapters as module
        except ImportError:
            module = False
        _ADAPTERS = module
    return _ADAPTERS or None


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise Cancelled()


def natural_key(path):
    return [int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)', Path(path).name)]


@dataclass
class PageAnalysis:
    name: str
    width: int
    height: int
    edges: dict  # side -> list of source (x,y) pairs
    complexity: dict  # side -> per-sample [0,1] texture/line cost
    confidence: dict
    angles: dict
    roughness: dict
    warnings: list[str]
    fingerprint: str = ''

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        result = cls(**raw)
        if not (1 <= result.width <= 100000 and 1 <= result.height <= 100000):
            raise ValueError('自动分析的原图尺寸无效')
        if Path(result.name).name != result.name:
            raise ValueError('自动分析的文件名无效')
        for side in SIDES:
            p = np.asarray(result.edges[side], dtype=float)
            c = np.asarray(result.complexity[side], dtype=float)
            if (p.ndim != 2 or p.shape[1] != 2 or not 4 <= len(p) <= 1024
                    or c.shape != (len(p),) or not np.isfinite(p).all()
                    or not np.isfinite(c).all() or np.any((c < 0) | (c > 1))
                    or np.any(np.abs(p) > max(result.width, result.height)*2)):
                raise ValueError('自动边界数据无效')
            for values in (result.confidence, result.angles, result.roughness):
                if not math.isfinite(float(values[side])):
                    raise ValueError('自动分析中含非有限数值')
        return result


@dataclass
class CropPlan:
    box: tuple[int, int, int, int]
    angle: float
    inner: str
    cost: float
    confidence: float
    max_outside: float
    max_trim: float
    warnings: list[str]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        result = cls(**raw)
        result.box = tuple(int(v) for v in result.box)
        if (len(result.box) != 4 or result.box[2] <= result.box[0]
                or result.box[3] <= result.box[1]
                or any(abs(v) > 300000 for v in result.box)
                or not all(math.isfinite(float(v)) for v in
                           (result.angle, result.cost, result.confidence, result.max_outside, result.max_trim))
                or result.inner not in ('left', 'right', 'none')):
            raise ValueError('自动裁剪方案无效')
        return result


def rotated(points, angle, width, height):
    p = np.asarray(points, dtype=float)
    center = np.array([width/2, height/2])
    t = math.radians(angle)
    m = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (p-center) @ m.T + center


def _robust_line(points, vertical, tolerance):
    """Deterministic consensus avoids locking onto occasional panel/text edges."""
    p = np.asarray(points, dtype=float)
    axis, dep = (1, 0) if vertical else (0, 1)
    t, v = p[:, axis], p[:, dep]
    span = max(float(np.ptp(t)), 1.)
    rng = np.random.default_rng(19)
    best = None
    for _ in range(220):
        i, j = rng.choice(len(p), 2, replace=False)
        if abs(t[j]-t[i]) < span*.35:
            continue
        a = (v[j]-v[i])/(t[j]-t[i])
        b = v[i]-a*t[i]
        residual = np.abs(v-(a*t+b))
        mask = residual < tolerance
        score = np.minimum(1., tolerance/np.maximum(residual, .01)).sum()
        if best is None or score > best[0]:
            best = score, mask
    mask = best[1] if best is not None else np.ones(len(p), bool)
    if mask.sum() < 4:
        mask[:] = True
    a, b = np.polyfit(t[mask], v[mask], 1)
    residual = v-(a*t+b)
    confidence = float(mask.mean()) * min(1., float(np.ptp(t[mask]))/span)
    return a, b, confidence, residual


def _sample_edge(gray, side, seed, scale, rim, coarse=None, initial_radius=None, bidirectional=False, prefer_near=False, scan_start=None):
    """Trace near a coarse physical boundary, refine in full source resolution."""
    h, w = gray.shape
    vertical = side in ('left', 'right')
    axis_len = h if vertical else w
    ortho_len = w if vertical else h
    start, end = seed['top'], seed['bottom']
    if not vertical:
        start, end = seed['left'], seed['right']
    stations = np.linspace(start+(end-start)*.025, end-(end-start)*.025, 144)
    expected = seed[side]
    radius = max(28, int(ortho_len*.012))
    sign = 1 if side in ('left','top') else -1

    def candidates(center_positions, search_radius, first_strong=False):
        points, strengths = [], []
        for station, position in zip(stations, center_positions):
            q = int(np.clip(round(station), 3, axis_len-4))
            lo = max(1, int(position-search_radius))
            hi = min(ortho_len-2, int(position+search_radius))
            if side == 'top':
                lo = max(lo, rim)
            if first_strong:
                # An uneven bed can pass the coarse threshold, putting the seed
                # just below the rim. Cover the whole band below the rim.
                lo = max(1, rim, scan_start)
                hi = max(hi, min(ortho_len-2, int(lo+ortho_len*.06)))
            if hi <= lo+3:
                lo, hi = max(1,min(lo,ortho_len-6)), max(5,min(ortho_len-2,lo+6))
            strip = gray[q-3:q+4,lo:hi+1] if vertical else gray[lo:hi+1,q-3:q+4].T
            profile = strip.mean(axis=0).astype(np.float32)
            profile = cv2.GaussianBlur(profile.reshape(1,-1),(5,1),0).ravel()
            gradient = np.diff(profile)*sign
            if bidirectional:
                gradient = np.abs(gradient)
            # Very small preference for nearby transitions, allowing real curvature.
            locations = np.arange(lo,lo+len(gradient))+.5
            quality = (gradient/(1+(np.abs(locations-position)/max(3.,scale*3))**2)
                       if prefer_near else gradient-.015*np.abs(locations-position))
            j = int(np.argmax(quality))
            span = max(6, int(scale*3))
            if first_strong and len(profile) > 2*span+2:
                # Bed-to-paper is the first large rise below the rim; ink
                # transitions inside the page come later and must not win.
                # Compare window means, not single pixels: bed noise averages
                # out while a dark printed margin still forms a clear step.
                c = np.r_[0., np.cumsum(profile, dtype=np.float64)]
                i = np.arange(span-1, len(profile)-span)
                step = np.full(len(gradient), -np.inf)
                step[i] = ((c[i+1+span]-c[i+1])-(c[i+1]-c[i+1-span]))/span
                peak = float(step.max())
                if peak >= 20:
                    k = int(np.flatnonzero(step >= .5*peak)[0])
                    j = k+int(np.argmax(step[k:k+2*span]))
            v = locations[j]
            points.append((v,q) if vertical else (q,v))
            strengths.append(max(0.,float(gradient[j])))
        return np.asarray(points), np.asarray(strengths)

    positions = np.full(len(stations),expected)
    if coarse is not None and len(coarse) >= 4:
        axis, dependent_axis = (1,0) if vertical else (0,1)
        positions = np.interp(stations,coarse[:,axis],coarse[:,dependent_axis])
    initial, strength = candidates(positions,radius if initial_radius is None else initial_radius,
                                   scan_start is not None and side == 'top' and not bidirectional)
    strong = strength > 1.5
    use = initial[strong] if strong.sum()>=12 else initial
    a,b,confidence,_ = _robust_line(use,vertical,max(3.,scale*2))
    predicted = a*stations+b
    points,strength = candidates(predicted,max(12,int(scale*5)))
    dependent = points[:,0 if vertical else 1]
    valid = (strength>1.5) & (np.abs(dependent-predicted)<max(10,scale*4))
    if valid.sum()>=12:
        a,b,confidence,residual = _robust_line(points[valid],vertical,max(3.,scale*2))
        line = a*stations+b
        # Keep local curvature; interpolate only unreliable observations.
        filtered = dependent.copy()
        valid &= np.abs(dependent-line)<max(12,scale*5)
        if valid.sum()>=4:
            filtered[~valid] = np.interp(stations[~valid],stations[valid],dependent[valid])
        confidence *= float(valid.mean())
    else:
        filtered = predicted
        confidence = .1
        residual = np.zeros(1)
    # Median filter removes isolated ink-line jumps, not smooth bows.
    padded = np.pad(filtered,2,mode='edge')
    filtered = np.median(np.lib.stride_tricks.sliding_window_view(padded,5),axis=1)
    points[:,0 if vertical else 1] = filtered
    correction = math.degrees(math.atan(a))*(1 if vertical else -1)
    roughness = float(np.percentile(residual,90)-np.percentile(residual,10))
    return points, min(1.,max(0.,confidence)), correction, roughness


def _border_background(small, sides=('left','right','bottom')):
    """Fit a smooth backing from exposed side/bottom pixels, not page ink."""
    h,w=small.shape
    ys=np.linspace(h*.06,h*.94,160).astype(int)
    xs=np.linspace(w*.06,w*.94,160).astype(int)
    if sides==('right','bottom'):
        # In bound scans the neighboring page can cover the left of the bottom
        # strip. It is not backing and must not drive an extrapolated plane.
        xs=xs[xs>=w*.65]
    strips={'left':np.column_stack((np.ones(len(ys)),ys)),
            'right':np.column_stack((np.full(len(ys),w-2),ys)),
            'bottom':np.column_stack((xs,np.full(len(xs),h-2)))}
    points=np.vstack([strips[side] for side in sides]).astype(int)
    values=small[points[:,1],points[:,0]].astype(float)
    matrix=np.column_stack((np.ones(len(points)),points[:,0]/w,points[:,1]/h))
    keep=np.ones(len(values),bool)
    for _ in range(5):
        coeff=np.linalg.lstsq(matrix[keep],values[keep],rcond=None)[0]
        residual=values-matrix@coeff
        noise=float(np.median(np.abs(residual[keep])))
        keep=np.abs(residual)<=max(4.,4*noise)
        if keep.sum()<40:
            return None
    if noise>9 or keep.mean()<.55:
        return None
    background=coeff[0]+coeff[1]*np.arange(w)[None,:]/w+coeff[2]*np.arange(h)[:,None]/h
    if sides==('right','bottom') and float(np.median(values))<110:
        # Only the right backing is guaranteed exposed. Extrapolating the
        # bottom/right plane far left can classify a dark pressure board as
        # paper, especially when it covers a differently shaded scanner bed.
        profile=np.median(small[:,-max(3,w//100):],axis=1).astype(np.float32)
        middle=float(np.median(profile[h//10:h*9//10]))
        limit=max(12.,noise*6)
        profile=np.clip(profile,middle-limit,middle+limit)
        profile=cv2.GaussianBlur(profile[:,None],(1,31),0)
        background=np.broadcast_to(profile,(h,w))
    return background,noise,float(np.median(values))


def _analyze_light_background(gray, small, background, noise, name, fingerprint, fixed_width=None):
    """Separate paper of either polarity from a smooth gray/white backing."""
    h,w=gray.shape; sh,sw=small.shape
    factor=w/sw
    smoothed=cv2.GaussianBlur(small.astype(np.float32),(9,9),0)
    rim=1
    if fixed_width is not None and adapters():
        smoothed, rim = adapters().fixed_width_top_band(smoothed, background, noise, factor)
    contrast=smoothed-background
    # A dark pressure board can be darker than the exposed scanner backing.
    # Do not turn that whole board into a foreground "page" component.
    dark_backing=fixed_width is not None and float(np.median(background))<110
    if dark_backing:
        mask=(contrast>max(12.,noise*6)).astype(np.uint8)
    else:
        mask=(np.abs(contrast)>max(9.,noise*6)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    if count<2:
        raise ValueError('未找到纸张与背景的清晰分界，请手动调整')
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    mask=labels==label
    columns=np.flatnonzero(np.mean(mask,axis=0)>.25)
    if fixed_width is not None and len(columns):
        # Ignore the adjoining page. Only the target's outer edge determines X.
        left=max(0,round(columns[-1]+1-fixed_width/factor))
        mask[:,:left]=False
        rows=np.flatnonzero(np.mean(mask[:,left:columns[-1]+1],axis=1)>.25)
        columns=np.arange(left,columns[-1]+1)
    else:
        rows=np.flatnonzero(np.mean(mask,axis=1)>.25)
    if len(columns)<(4 if fixed_width is not None else sw*.4) or len(rows)<sh*.4:
        raise ValueError('纸张与浅色背景的分界不够可靠，请手动调整')
    seed=dict(left=float(columns[0]*factor),right=float((columns[-1]+1)*factor),
              top=float(rows[0]*factor),bottom=float((rows[-1]+1)*factor))
    texture=cv2.GaussianBlur(small.astype(np.float32),(7,7),0)
    gradient=np.hypot(cv2.Sobel(texture,cv2.CV_32F,1,0)/8,cv2.Sobel(texture,cv2.CV_32F,0,1)/8)
    edges={}; confidence={}; angles={}; roughness={}; complexity={}; warnings=[]
    for side in SIDES:
        vertical=side in ('left','right')
        axis=1 if vertical else 0
        valid=mask.any(axis=axis)
        locations=np.argmax(mask,axis=axis)
        if side in ('right','bottom'):
            locations=mask.shape[axis]-1-np.argmax(np.flip(mask,axis=axis),axis=axis)
        stations=np.flatnonzero(valid)
        coarse=(np.column_stack((locations[valid],stations)) if vertical else
                np.column_stack((stations,locations[valid])))*factor
        if fixed_width is not None and side=='left':
            p=np.column_stack((np.full(144,seed['left']),np.linspace(seed['top'],seed['bottom'],144)))
            c,angle,rough=0.,0.,0.
        else:
            p,c,angle,rough=_sample_edge(gray,side,seed,factor,rim,coarse,
                                       bidirectional=True,prefer_near=dark_backing)
        if side=='top' and seed['top']<factor*2 and c<.6:
            # A clipped page has no physical transition above it. Do not use
            # an uncertain printed line inside the page as its missing edge.
            p[:,1]=0.
            c,angle,rough=0.,0.,0.
            warnings.append('上纸边不完整或缺少背景，按图像边界保留')
        edges[side]=p.tolist(); confidence[side]=c; angles[side]=angle; roughness[side]=rough
        values=[]
        direction={'left':(1,0),'right':(-1,0),'top':(0,1),'bottom':(0,-1)}[side]
        for x,y in p/factor:
            x=int(round(x+direction[0]*12)); y=int(round(y+direction[1]*12))
            patch=gradient[max(0,y-7):min(sh,y+8),max(0,x-7):min(sw,x+8)]
            values.append(float(np.clip(float(np.percentile(patch,80))/12.,0,1)) if patch.size else 0.)
        complexity[side]=values
        if c<.6 and not (fixed_width is not None and side=='left'):
            warnings.append({'left':'左','right':'右','top':'上','bottom':'下'}[side]+'边界不确定')
    if seed['top']<factor*2 or seed['bottom']>h-factor*2:
        warnings.append('纸边可能触及扫描画布，需检查原图是否完整')
    if fixed_width is not None:
        warnings.append('未拆书：内侧由指定宽度确定')
    return PageAnalysis(name,w,h,edges,complexity,confidence,angles,roughness,warnings,fingerprint)


SIDE_LABELS={'left':'左','right':'右','top':'上','bottom':'下'}
def analyze_array(gray, name='image.png', fingerprint=''):
    if cv2 is None:
        raise RuntimeError('自动推荐需要 OpenCV。请安装 requirements.txt 中的依赖。')
    h,w = gray.shape
    if min(h,w)<128:
        raise ValueError('扫描图太小，无法可靠分析边界')
    factor = max(1.,w/1600)
    small = cv2.resize(gray,(round(w/factor),round(h/factor)),interpolation=cv2.INTER_AREA)
    backing=_border_background(small)
    if backing is not None and backing[2]>=110:
        return _analyze_light_background(gray,small,*backing[:2],name,fingerprint)
    sh,sw = small.shape
    blur = cv2.GaussianBlur(small.astype(np.float32),(9,9),0)
    strips = [blur[int(sh*.05):int(sh*.95),:max(3,sw//40)],
              blur[int(sh*.05):int(sh*.95),-max(3,sw//40):],
              blur[-max(3,sh//80):,int(sw*.1):int(sw*.9)]]
    bg = min(strips,key=lambda s:float(np.median(s)))
    level = float(np.median(bg))
    noise = float(np.median(np.abs(bg-level)))
    threshold = level+max(12.,noise*5)
    if threshold>220:
        raise ValueError('未找到纸张与扫描背景的可靠分界，请手动调整')
    # Majority vote keeps a dark printed margin (often ~90) separate from bed (~60).
    column = np.mean(blur[int(sh*.04):int(sh*.94)]>threshold,axis=0)
    columns = np.flatnonzero(column>.28)
    if len(columns)<sw*.4:
        raise ValueError('纸张边界置信度不足，请使用手动分析')
    left,right = int(columns[0]),int(columns[-1])
    x0,x1=left+int((right-left)*.08),right-int((right-left)*.08)
    rim, scan_start = 1, None
    if adapters():
        blur, rim, scan_start = adapters().top_rows(
            blur, left, right, threshold, level, noise, factor)
    row = np.mean(blur[:,x0:x1]>threshold,axis=1)
    rows = np.flatnonzero(row>.26)
    if len(rows)<sh*.4:
        raise ValueError('未找到完整的纸张上下边界')
    top=int(rows[0])
    bottom=int(rows[-1])
    seed={'left':left*factor,'right':(right+1)*factor,'top':top*factor,'bottom':(bottom+1)*factor}
    # Follow coarse physical boundaries along the entire side. A fixed x/y seed
    # only found the middle of a tilted edge and implicitly restricted rotation.
    paper = blur > threshold
    top_paper = paper
    coarse = {}
    for side in SIDES:
        mask = top_paper if side == 'top' else paper
        vertical = side in ('left','right')
        axis = 1 if vertical else 0
        valid = mask.any(axis=axis)
        locations = np.argmax(mask,axis=axis)
        if side in ('right','bottom'):
            locations = mask.shape[axis]-1-np.argmax(np.flip(mask,axis=axis),axis=axis)
        stations = np.flatnonzero(valid)
        values = locations[valid]
        coarse[side] = (np.column_stack((values,stations)) if vertical else
                        np.column_stack((stations,values))) * factor
    edges,confidence,angles,roughness,complexity={},{},{},{},{}
    # Suppress screen dots before assessing structures near the boundary.
    texture=cv2.GaussianBlur(small.astype(np.float32),(7,7),0)
    gx=cv2.Sobel(texture,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(texture,cv2.CV_32F,0,1,ksize=3)/8
    gradient=np.hypot(gx,gy)
    for side in SIDES:
        p,c,angle,rough=_sample_edge(gray,side,seed,factor,rim,coarse[side],scan_start=scan_start)
        edges[side]=p.tolist(); confidence[side]=c; angles[side]=angle; roughness[side]=rough
        values=[]
        for x,y in p/factor:
            # Sample inside, not scanner texture or paper/background transition.
            direction={'left':(1,0),'right':(-1,0),'top':(0,1),'bottom':(0,-1)}[side]
            x=int(round(x+direction[0]*12)); y=int(round(y+direction[1]*12))
            patch=gradient[max(0,y-7):min(sh,y+8),max(0,x-7):min(sw,x+8)]
            value=float(np.percentile(patch,80)) if patch.size else 8.
            values.append(float(np.clip(value/12.,0,1)))
        complexity[side]=values
    warnings=[]
    for side in SIDES:
        if confidence[side]<.6:
            warnings.append(SIDE_LABELS[side]+'边界不确定')
    if seed['top']<2 or seed['bottom']>h-2:
        warnings.append('纸边可能触及扫描画布，需检查原图是否完整')
    return PageAnalysis(name,w,h,edges,complexity,confidence,angles,roughness,warnings,fingerprint)


def analyze_bound_array(gray, width, name='image.png', fingerprint=''):
    """Detect the target page from the outer right edge, without a left gap."""
    if cv2 is None:
        raise RuntimeError('自动推荐需要 OpenCV。请安装 requirements.txt 中的依赖。')
    h, w = gray.shape
    width = int(width)
    if width < 16 or width > w*2:
        raise ValueError('图像宽度需至少 16 像素且不超过原图宽度的两倍')
    factor = max(1., w/1600)
    small = cv2.resize(gray, (round(w/factor), round(h/factor)), interpolation=cv2.INTER_AREA)
    backing=_border_background(small,('right','bottom'))
    if backing is not None:
        try:
            return _analyze_light_background(gray,small,*backing[:2],name,fingerprint,fixed_width=width)
        except ValueError:
            pass  # Preserve the original fallback when no coherent page is found.
    blur = cv2.GaussianBlur(small.astype(np.float32), (9,9), 0)
    sh, sw = blur.shape
    # Only the right and bottom scanner background are relevant in this mode.
    strips = [blur[sh//10:sh*9//10, -max(3,sw//40):],
              blur[-max(3,sh//80):, sw//2:]]
    bg = min(strips, key=lambda a: float(np.median(a)))
    level = float(np.median(bg))
    threshold = level + max(12., 5*float(np.median(np.abs(bg-level))))
    columns = np.flatnonzero(np.mean(blur[sh//20:sh*19//20] > threshold, axis=0) > .28)
    if not len(columns) or threshold > 220:
        raise ValueError('未拆书模式未找到右侧纸张与扫描背景的分界')
    right = min(w, round((int(columns[-1])+1)*factor))
    left = max(0, right-width)
    margin = max(48, round(min(width,w)*.08))
    x0, x1 = max(0,left-margin), min(w,right+margin)
    region = gray[:,x0:x1].copy()
    region[:,:max(0,left-x0)] = round(level)
    frame_end = 0
    if adapters():
        region, frame_end = adapters().fixed_width_frame(
            gray, region, right, margin, left, x0, level, bg)
    # Scanner rulers can form an isolated bright horizontal strip above the
    # actual page. Keep the long paper run; remove only separate short strips
    # from this analysis copy so they cannot become the top/bottom boundary.
    row_support = (np.mean(region > threshold,axis=1) > .26).astype(np.uint8)
    gap = max(3,round(h*.005))
    joined = cv2.morphologyEx(row_support[:,None],cv2.MORPH_CLOSE,
                             np.ones((gap,1),dtype=np.uint8)).ravel().astype(bool)
    transitions = np.diff(np.r_[False,joined,False].astype(np.int8))
    starts, ends = np.flatnonzero(transitions==1), np.flatnonzero(transitions==-1)
    if len(starts):
        main = int(np.argmax(ends-starts))
        if ends[main]-starts[main] > h*.4:
            for j,(start,end) in enumerate(zip(starts,ends)):
                if j != main and end-start < h*.02:
                    region[max(0,start-4):min(h,end+4)] = round(level)
    # Masking only the analysis copy excludes the other page from top/bottom
    # detection. The resulting left line is virtual and never used to solve.
    # Limit bright print contrast so a thin dark printed margin is not mistaken
    # for the physical outer edge. Background-to-paper contrast is preserved.
    analysis_region = np.minimum(region, min(255,round(threshold+20))).astype(np.uint8)
    local = analyze_array(analysis_region, name, fingerprint)
    # Use the visible paper (before contrast clipping) to seed horizontal
    # boundaries. Otherwise a weak bed-texture transition near the ruler may
    # win the coarse threshold and prevent tracing the actual page farther in.
    white = cv2.GaussianBlur(region.astype(np.float32),(9,9),0) > level+max(40.,8*float(np.median(np.abs(bg-level))))
    seed = {side:float(np.median(np.asarray(local.edges[side])[:,0 if side in ('left','right') else 1])) for side in SIDES}
    for side in ('top','bottom'):
        valid = white.any(axis=0)
        locations = (np.argmax(white,axis=0) if side == 'top' else
                     h-1-np.argmax(white[::-1],axis=0))
        stations = np.flatnonzero(valid)
        if len(stations) < 4:
            continue
        points = np.column_stack((stations,locations[valid]))
        p,c,angle,rough = _sample_edge(analysis_region,side,seed,
                                      max(1.,region.shape[1]/1600),frame_end,points,12)
        local.edges[side] = p.tolist()
        local.confidence[side],local.angles[side],local.roughness[side] = c,angle,rough
    edges = {side:[[x+x0,y] for x,y in points] for side,points in local.edges.items()}
    confidence = dict(local.confidence); confidence['left'] = 0.
    warnings = [msg for msg in local.warnings if not msg.startswith(('左边界','上边界','下边界'))]
    for side,label in [('top','上'),('bottom','下')]:
        if local.confidence[side] < .6:
            warnings.append(label+'边界不确定')
    warnings.append('未拆书：内侧由指定宽度确定')
    return PageAnalysis(name,w,h,edges,local.complexity,confidence,local.angles,
                        local.roughness,warnings,fingerprint)


def analyze_page(path, cache_dir=None, cancel=None, bound_width=None, bound_side='left'):
    path=Path(path)
    stat=path.stat()
    mode_key = '' if bound_width is None else f'|bound-v8:{int(bound_width)}'
    if bound_width is not None and bound_side == 'right':
        mode_key += ':right'
    # Adapted and generic analyses of a file are cached separately.
    plugin_key=adapters().CACHE_KEY if adapters() else ''
    key=hashlib.sha256(f'{VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}{mode_key}{plugin_key}'.encode()).hexdigest()
    cache=Path(cache_dir)/(key+'.json') if cache_dir else None
    check_cancel(cancel)
    if cache and cache.is_file():
        try:
            page=PageAnalysis.from_dict(json.loads(cache.read_text(encoding='utf-8')))
            if page.fingerprint==key:
                return page
        except (OSError,ValueError,TypeError,KeyError):
            pass
    with Image.open(path) as image:
        gray=np.asarray(image.convert('L'))
    check_cancel(cancel)
    result=(analyze_array(gray,path.name,key) if bound_width is None else
            analyze_bound_array(gray if bound_side == 'left' else gray[:,::-1],bound_width,path.name,key))
    if bound_width is not None and bound_side == 'right':
        swap = {'left':'right','right':'left','top':'top','bottom':'bottom'}
        result.edges = {s:[[result.width-x,y] for x,y in result.edges[swap[s]]] for s in SIDES}
        result.complexity = {s:result.complexity[swap[s]] for s in SIDES}
        result.confidence = {s:result.confidence[swap[s]] for s in SIDES}
        result.roughness = {s:result.roughness[swap[s]] for s in SIDES}
        result.angles = {s:-result.angles[swap[s]] for s in SIDES}
    if cache:
        try:
            cache.parent.mkdir(parents=True,exist_ok=True)
            temporary=cache.with_suffix('.tmp')
            temporary.write_text(json.dumps(result.to_dict(),ensure_ascii=False),encoding='utf-8')
            temporary.replace(cache)
        except OSError:
            pass  # Analysis remains usable when an optional cache cannot be saved.
    return result


def clipped_side(path):
    """The side where the paper runs off the scan ('left' or 'right'), or None.

    Covers scanned as one half of a spread show only their outer edge: the
    inner part continues beyond the scan area, so paper reaches that image
    border while the outer border shows backing. The backing level comes from
    the strips along the top and bottom of the scan.
    """
    with Image.open(path) as image:
        image.draft('L',(image.width//8,image.height//8))
        gray=np.asarray(image.convert('L').reduce(4),dtype=np.float32)
    h,w=gray.shape
    rows=slice(int(h*.2),int(h*.8)); cols=slice(int(w*.25),int(w*.75))
    band=max(2,min(h,w)//100)
    backing=float(np.median(np.concatenate((gray[:band,cols].ravel(),gray[-band:,cols].ravel()))))
    left=abs(float(np.median(gray[rows,:band]))-backing)
    right=abs(float(np.median(gray[rows,-band:]))-backing)
    if max(left,right)<20 or max(left,right)<3*min(left,right)+10:
        return None
    return 'left' if left>right else 'right'


def inner_side(page,mode,index):
    if mode.startswith('bound-'):
        mode = mode[len('bound-'):]
    if mode in ('left','right','none'):
        return mode
    if mode.startswith('alternate-'):
        first=mode.split('-')[1]
        return first if index%2==0 else ('left' if first=='right' else 'right')
    return 'left' if page.roughness['left']>page.roughness['right'] else 'right'


def reference_angle(page,inner):
    outer='right' if inner=='left' else 'left'
    if inner=='none':
        outer=max(('left','right'),key=lambda s:page.confidence[s])
    sides = [s for s in SIDES if not (s == inner and '未拆书：内侧由指定宽度确定' in page.warnings)]
    if page.confidence[outer]<.6:
        outer=max(sides,key=lambda s:page.confidence[s])
    confidence = page.confidence[outer]
    # Uncertain observations must not dictate a rotation. Reliable observations
    # retain their measured angle, without clipping it to an arbitrary range.
    return (float(page.angles[outer]) if confidence >= .6 else 0.), confidence


def paper_reference(pages, mode='alternate-right', size=None, angles=None, sides=None):
    """Common physical paper dimensions and per-page top/outer-edge anchors.

    The scanner background is excluded. Curved edges are represented by their
    medians, so zero additional trim is a paper reference, not a no-fill promise.
    ``sides`` gives each page's inner side (e.g. from its plan); otherwise it
    follows the page order and ``mode``.
    """
    if sides is None:
        sides = [inner_side(page, mode, i) for i, page in enumerate(pages)]
    if angles is None:
        angles = [reference_angle(page, side)[0] for page, side in zip(pages, sides)]
    bounds = []
    for page, angle in zip(pages, angles):
        edges = {side: rotated(page.edges[side], angle, page.width, page.height) for side in SIDES}
        bounds.append([float(np.median(edges[side][:, axis]))
                       for side, axis in [('left', 0), ('right', 0), ('top', 1), ('bottom', 1)]])
    if size is None:
        dimensions = np.array([(r-l, b-t) for l, r, t, b in bounds])
        size = tuple(int(round(v)) for v in _robust_dimensions(pages, dimensions))
    width, height = size
    origins = []
    for (left, right, top, bottom), side in zip(bounds, sides):
        x = right-width if side == 'left' else left
        if side == 'none':
            x = (left+right-width)/2
        origins.append((round(x), round(top)))
    return tuple(size), origins


def prepared_edges(page,angle,inner,margins):
    top,bottom,outer,inset=margins
    offsets={'left':inset if inner=='left' else outer,
             'right':inset if inner=='right' else outer,'top':top,'bottom':bottom}
    result={}
    for side in SIDES:
        p=rotated(page.edges[side],angle,page.width,page.height)
        axis=0 if side in ('left','right') else 1
        physical=p[:,axis]
        effective=physical+(offsets[side] if side in ('left','top') else -offsets[side])
        result[side]=(effective,np.asarray(page.complexity[side]),physical)
    return result


def _edge_weights(page,first,second):
    """Let a reliable edge anchor the crop when its opposite edge is doubtful."""
    ca,cb=float(page.confidence[first]),float(page.confidence[second])
    if ca>=.6 and cb<.6:
        return 1.,.15
    if cb>=.6 and ca<.6:
        return .15,1.
    return 1.,1.


def _axis_best(first,second,length,policy,weights=(1.,1.)):
    a,ca,_=first; b,cb,_=second
    wa,wb=weights
    center=((float(np.median(a))+float(np.median(b))-length)/2 if wa==wb else
            float(np.median(a)) if wa>wb else float(np.median(b))-length)
    low=min(float(np.percentile(a,5)),float(np.percentile(b,5))-length,center)-12
    high=max(float(np.percentile(a,95)),float(np.percentile(b,95))-length,center)+12
    # Keep requests far outside plausible paper size finite and inexpensive.
    low=max(low,center-600); high=min(high,center+600)
    candidates=np.linspace(low,high,min(101,max(3,int(high-low)+1)))
    clip_w,fill_w=POLICIES[policy]

    def scores(locations):
        d1=locations[:,None]-a[None,:]
        d2=b[None,:]-(locations[:,None]+length)
        score=np.zeros(len(locations))
        for d,c,weight in ((d1,ca,wa),(d2,cb,wb)):
            trim=np.maximum(d,0); fill=np.maximum(-d,0)
            score+=weight*np.mean(clip_w*(.10+2.5*c)*(trim+trim**2/80)
                                  +fill_w*(.15+4*c)*(fill+fill**2/25),axis=1)
        # Prefer centred loss when candidates have nearly equivalent cost.
        score+=.0005*(locations-center)**2
        return score
    values=scores(candidates)
    best=float(candidates[int(values.argmin())])
    fine=np.arange(round(best)-3,round(best)+4,dtype=float)
    values=scores(fine); j=int(values.argmin())
    return int(fine[j]),float(values[j])


def solve_page(page,size,inner,margins=(15,0,15,25),policy='balanced',refine_angle=True):
    width,height=map(int,size)
    if width<16 or height<16 or width>page.width*2 or height>page.height*2:
        raise ValueError('输出宽高需为正数且不超过原图对应尺寸的两倍')
    if policy not in POLICIES:
        raise ValueError('未知裁剪偏好')
    base,confidence=reference_angle(page,inner)
    angles=[base] if not refine_angle or confidence<.6 else [base,base-.015,base+.015,base-.03,base+.03]
    best=None
    for angle in angles:
        edges=prepared_edges(page,angle,inner,margins)
        x,cx=_axis_best(edges['left'],edges['right'],width,policy,
                        _edge_weights(page,'left','right'))
        y,cy=_axis_best(edges['top'],edges['bottom'],height,policy,
                        _edge_weights(page,'top','bottom'))
        cost=cx+cy+4*((angle-base)/.03)**2
        if best is None or cost<best[0]:
            best=(cost,x,y,angle,edges)
    cost,x,y,angle,edges=best
    physical_outside=[]; trim=[]
    for side,location,sign in [('left',x,1),('right',x+width,-1),('top',y,1),('bottom',y+height,-1)]:
        effective,_,physical=edges[side]
        physical_outside.append(float(np.max(np.maximum((physical-location)*sign,0))))
        trim.append(float(np.max(np.maximum((location-effective)*sign,0))))
    warnings=list(page.warnings)
    if confidence<.6:
        warnings.append('旋转参考边不可靠')
    if max(physical_outside)>3:
        warnings.append(f'预计纸外缺边最宽 {max(physical_outside):.0f}px')
    if max(trim)>25:
        warnings.append(f'超过预留裁量的页内裁入最深 {max(trim):.0f}px')
    # Heuristic only: do not pretend a low-texture edge is semantically safe.
    if cost>65:
        warnings.append('边缘取舍较大，检查文字和人物')
    return CropPlan((x,y,x+width,y+height),round(angle,4),inner,float(cost),
                    min(confidence,min(page.confidence.values())),
                    max(physical_outside),max(trim),list(dict.fromkeys(warnings)))


def _typical_cost(costs):
    """Batch score from the best-fitting 80% of pages.

    The common size must suit typical pages. A few unusual pages (a shorter
    insert, a torn or folded edge) keep their own warnings and per-page
    adjustment instead of shrinking or growing every page; scoring the
    expensive tail once cut ~80 px from a whole batch for one short page.
    """
    ordered=np.sort(np.asarray(costs,dtype=float))
    return float(np.mean(ordered[:max(1,math.ceil(len(ordered)*.8))]))


def _robust_dimensions(pages,dimensions):
    """Median width/height from pages whose two edges on that axis are reliable.

    One misdetected edge only affects its own axis, and gross outliers (for
    example a scanner rim taken as the top edge) cannot drag the batch size.
    """
    result=[]
    for column,(first,second) in enumerate((('left','right'),('top','bottom'))):
        values=dimensions[:,column]
        reliable=np.array([min(p.confidence[first],p.confidence[second])>=.6 for p in pages])
        if reliable.sum()>=max(2,len(pages)//3):
            values=values[reliable]
        center=float(np.median(values))
        spread=max(4.,1.4826*float(np.median(np.abs(values-center))))
        inliers=values[np.abs(values-center)<=3*spread]
        result.append(float(np.median(inliers)) if len(inliers) else center)
    return np.asarray(result)


def solve_batch(pages, size=None, mode='alternate-right', margins=(15,0,15,25),
                policy='balanced', progress=None, cancel=None, page_indices=None):
    if not pages:
        raise ValueError('没有可分析的正文页面')
    indices = range(len(pages)) if page_indices is None else page_indices
    sides=[inner_side(page,mode,i) for i,page in zip(indices,pages)]
    if size is None:
        dimensions=[]
        for page,side in zip(pages,sides):
            angle,_=reference_angle(page,side)
            e=prepared_edges(page,angle,side,margins)
            dimensions.append((np.median(e['right'][0])-np.median(e['left'][0]),
                               np.median(e['bottom'][0])-np.median(e['top'][0])))
        median=_robust_dimensions(pages,np.asarray(dimensions))
        # Use all pages to evaluate a bounded neighbourhood of robust dimensions.
        sizes=[]
        radii=np.maximum(24,np.minimum(100,median*.012))
        for dw in np.linspace(-radii[0],radii[0],7):
            for dh in np.linspace(-radii[1],radii[1],7):
                sizes.append(tuple(max(16,round((v+d)/2)*2) for v,d in zip(median,(dw,dh))))
        winner=None
        for k,candidate in enumerate(sizes):
            check_cancel(cancel)
            costs=[solve_page(p,candidate,s,margins,policy,False).cost for p,s in zip(pages,sides)]
            score=_typical_cost(costs)
            if winner is None or score<winner[0]:
                winner=score,candidate
            if progress:
                progress(k+1,len(sizes),'比较整本裁剪尺寸')
        coarse=winner[1]
        for dw in (-6,0,6):
            for dh in (-6,0,6):
                check_cancel(cancel)
                candidate=(max(16,coarse[0]+dw),max(16,coarse[1]+dh))
                costs=[solve_page(p,candidate,s,margins,policy,False).cost for p,s in zip(pages,sides)]
                score=_typical_cost(costs)
                if score<winner[0]:
                    winner=score,candidate
        size=winner[1]
    plans=[]
    for i,(page,side) in enumerate(zip(pages,sides)):
        check_cancel(cancel)
        plans.append(solve_page(page,size,side,margins,policy))
        if progress:
            progress(i+1,len(pages),'推荐 '+page.name)
    return tuple(map(int,size)),plans


def solve_bound_batch(pages, width, height=None, margins=(0,0,0,0), policy='balanced', progress=None, cancel=None, mode='bound-left', page_indices=None):
    """Fixed-width placement against the outer edge; ignore the gutter."""
    width = int(width)
    if width < 16:
        raise ValueError('图像宽度至少需要 16 像素')
    prepared = []
    indices = range(len(pages)) if page_indices is None else page_indices
    sides = [inner_side(page,mode,i) for i,page in zip(indices,pages)]
    for page,side in zip(pages,sides):
        angle, confidence = reference_angle(page,side)
        edges = prepared_edges(page,angle,side,margins)
        prepared.append((angle,confidence,edges))
    if height is None:
        heights = [float(np.min(e['bottom'][0])-np.max(e['top'][0])) for _,_,e in prepared]
        reliable = [v for p,v,side in zip(pages,heights,sides)
                    if min(p.confidence[s] for s in ('right' if side=='left' else 'left','top','bottom'))>=.55]
        values = reliable if len(reliable)>=2 else heights
        # Pick a height that fits most pages rather than leaving black strips
        # on about half the batch. Low-confidence edges do not shrink all pages.
        percentile = {'less_fill':0,'balanced':10,'preserve':50}[policy]
        height = math.floor(float(np.percentile(values,percentile)))
    if height < 16:
        raise ValueError('未找到有效的纸张高度')
    plans = []
    for i,(page,(angle,confidence,edges),inner) in enumerate(zip(pages,prepared,sides)):
        check_cancel(cancel)
        if width > page.width*2 or height > page.height*2:
            raise ValueError('裁剪尺寸不得超过原图对应尺寸的两倍')
        outer = 'right' if inner == 'left' else 'left'
        position = (math.floor(float(np.min(edges[outer][0]))) if outer == 'right'
                    else math.ceil(float(np.max(edges[outer][0]))))
        x = position-width if outer == 'right' else position
        right = x+width
        y,cost = _axis_best(edges['top'],edges['bottom'],height,policy)
        inside_top = math.ceil(float(np.max(edges['top'][0])))
        inside_bottom = math.floor(float(np.min(edges['bottom'][0])))-height
        if inside_top <= inside_bottom:
            # If the requested rectangle fits, never leave avoidable paper-out
            # strips merely because an average texture cost slightly prefers it.
            y = min(max(y,inside_top),inside_bottom)
        outside, trim = [], []
        for side,position,sign in [(outer,right if outer=='right' else x,-1 if outer=='right' else 1),('top',y,1),('bottom',y+height,-1)]:
            effective,_,physical = edges[side]
            outside.append(float(np.max(np.maximum((physical-position)*sign,0))))
            trim.append(float(np.max(np.maximum((position-effective)*sign,0))))
        warnings = list(page.warnings)
        if max(outside)>3:
            warnings.append(f'预计外侧／上下缺边最宽 {max(outside):.0f}px')
        plans.append(CropPlan((x,y,right,y+height),round(angle,4),inner,cost,
                              min(confidence,*(page.confidence[s] for s in (outer,'top','bottom'))),
                              max(outside),max(trim),warnings))
        if progress:
            progress(i+1,len(pages),'定位 '+page.name)
    return (width,int(height)),plans


def analyze_batch(paths,cache_dir=None,progress=None,cancel=None,bound_width=None,bound_mode='bound-left',page_indices=None):
    pages=[]
    for i,path in enumerate(paths):
        check_cancel(cancel)
        try:
            index = i if page_indices is None else page_indices[i]
            side = inner_side(None,bound_mode,index) if bound_width is not None else 'left'
            pages.append(analyze_page(path,cache_dir,cancel,bound_width,side))
        except Cancelled:
            raise
        except Exception as exc:
            # Never silently invent a successful crop for a failed page.
            raise ValueError(f'{Path(path).name}：{exc}。可取消自动推荐，改用原有手动模式。') from exc
        if progress:
            progress(i+1,len(paths),'识别边界 '+Path(path).name)
    if adapters():
        pages=adapters().after_batch(paths,pages,cancel)
    return pages


def after_single_page(path, page, batch_pages):
    """Batch-level adapter steps for one re-analyzed page."""
    if adapters():
        return adapters().after_single_page(path, page, batch_pages)
    return page
