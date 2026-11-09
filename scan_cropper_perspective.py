"""Page-plane correction. Matrices map original scanner pixels to corrected pixels.

Corner order: top-left, top-right, bottom-right, bottom-left. A correction maps
the selected input quadrilateral to a rectangle in the deskewed coordinate
system, then maps back. A three-edge fallback removes affine skew without
straightening an uncertain fourth edge. Rotation remains downstream.
"""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
import cv2


def rotation_matrix(angle, width, height):
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    x, y = width/2, height/2
    return np.array([[c,-s,x-c*x+s*y],[s,c,y-s*x-c*y],[0,0,1.]],float)


def project(points, matrix):
    p = np.asarray(points, float)
    q = np.column_stack((p, np.ones(len(p)))) @ np.asarray(matrix,float).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:,2]) < 1e-7):
        raise ValueError('梯形参数使投影趋于无穷，请缩小调整')
    return q[:,:2] / q[:,2:3]


def validate_quad(points):
    p = np.asarray(points,float)
    if p.shape != (4,2) or not np.isfinite(p).all() or np.max(np.abs(p)) > 300000:
        raise ValueError('梯形四角坐标无效')
    e = np.roll(p,-1,axis=0)-p
    cross = e[:,0]*np.roll(e,-1,axis=0)[:,1] - e[:,1]*np.roll(e,-1,axis=0)[:,0]
    if np.min(cross) <= 1 or min(np.linalg.norm(e,axis=1)) < 4:
        raise ValueError('梯形四角不能交叉、翻转或重合')
    return p


def homography(source, target):
    a,b = validate_quad(source), validate_quad(target)
    h = cv2.getPerspectiveTransform(a.astype(np.float32), b.astype(np.float32)).astype(float)
    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-10:
        raise ValueError('梯形变换无效')
    # The complete page must stay on one side of the projective horizon.
    d = np.column_stack((a,np.ones(4))) @ h[2]
    if np.min(d)*np.max(d) <= 0:
        raise ValueError('梯形变换跨过投影极限')
    return h/h[2,2]


def state_matrix(state):
    if not state or not state.get('enabled'):
        return None
    return homography(np.asarray(state['source'])+np.asarray(state['offsets']), state['target'])


def validate_state(raw):
    if not isinstance(raw,dict):
        raise ValueError('梯形配置无效')
    state = deepcopy(raw)
    validate_quad(state['source']); validate_quad(state['target'])
    offsets = np.asarray(state['offsets'],float)
    if offsets.shape != (4,2) or not np.isfinite(offsets).all() or np.max(np.abs(offsets))>100000:
        raise ValueError('梯形角点偏移无效')
    homography(np.asarray(state['source'])+offsets,state['target'])
    state['enabled'] = bool(state.get('enabled',False))
    return state


def manual_state(box, angle, width, height):
    x,y,r,b = box
    quad = project([[x,y],[r,y],[r,b],[x,b]],np.linalg.inv(rotation_matrix(angle,width,height))).tolist()
    return dict(source=quad,target=deepcopy(quad),offsets=[[0.,0.] for _ in range(4)],
                enabled=False,automatic=False,reason='手动梯形校正；四角参数为原图像素偏移')


def _fit(points, vertical):
    p = np.asarray(points,float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p)<4 or not np.isfinite(p).all():
        raise ValueError('纸边直线拟合不足')
    axis, dep = (1,0) if vertical else (0,1)
    # Trim the ends where corner/background transitions are least trustworthy.
    # Manual/excluded pages have exactly four extent samples. Keep at least
    # four points, otherwise trimming would reject every manual correction.
    trim=min(max(1,len(p)//12),(len(p)-4)//2)
    if trim:
        p = p[trim:-trim]
    keep = np.ones(len(p),bool)
    for _ in range(4):
        k,b = np.polyfit(p[keep,axis],p[keep,dep],1)
        residual = p[:,dep]-(k*p[:,axis]+b)
        mad = np.median(np.abs(residual[keep]))
        keep = np.abs(residual) <= max(1.,3*mad)
        if keep.sum()<4:
            raise ValueError('纸边直线拟合不足')
    error = float(np.percentile(np.abs(residual),90))
    # y=k*x+b -> -kx+y=b; x=k*y+b -> x-ky=b.
    return (np.array([1.,-k]) if vertical else np.array([-k,1.])), b, error


def _shear_candidate(lines, confidence, width, height, center):
    """Infer affine skew from three reliable edges, never from a weak edge.

    Two opposite edges must agree within their fitting noise. The third edge
    establishes non-orthogonality; a rotated rectangle therefore scores zero.
    """
    candidates=[]
    for pair,others,length,other_length in [
        (('top','bottom'),('left','right'),width,height),
        (('left','right'),('top','bottom'),height,width),
    ]:
        if any(confidence[s]<.70 for s in pair):
            continue
        other=max(others,key=lambda s:confidence[s])
        if confidence[other]<.70:
            continue
        slopes={s: -lines[s][0][1 if s in ('left','right') else 0]
                for s in (*pair,other)}
        errors={s:lines[s][2] for s in (*pair,other)}
        if abs(slopes[pair[0]]-slopes[pair[1]])*length > max(2.,2*sum(errors[s] for s in pair)):
            continue
        # Require independently significant, same-direction skew at both corners.
        skew=[slopes[s]+slopes[other] for s in pair]
        if skew[0]*skew[1]<=0:
            continue
        evidence=min(abs(skew[i])*length/max(2.,2*(errors[s]+errors[other]*length/other_length))
                     for i,s in enumerate(pair))
        mean=sum(slopes[s] for s in pair)/2
        kh,kv=(mean,slopes[other]) if pair[0]=='top' else (slopes[other],mean)
        basis=np.array([[1.,kv],[kh,1.]])
        if np.linalg.det(basis)<=.5:
            continue
        linear=np.linalg.inv(basis)
        affine=np.eye(3); affine[:2,:2]=linear
        affine[:2,2]=center-linear@center
        candidates.append((evidence,affine))
    return max(candidates,key=lambda c:c[0]) if candidates else (0.,None)


def _recommend_fixed_width(page, inner, width):
    """Use only the outer, top and bottom edges; preserve requested width.

    The outer corners stay fixed after deskew. Only the inner corners move
    vertically to intersect the fitted top/bottom edges at the requested width.
    """
    from scan_cropper_auto import reference_angle
    outer='right' if inner=='left' else 'left'
    angle,_=reference_angle(page,inner)
    rot=rotation_matrix(angle,page.width,page.height)
    lines={s:_fit(project(page.edges[s],rot),s==outer) for s in (outer,'top','bottom')}
    outer_points=project(page.edges[outer],rot)
    outer_x=float(np.median(outer_points[:,0]))
    left=outer_x-width if outer=='right' else outer_x
    right=left+width
    def y_at(side,x):
        line,b,_=lines[side]
        return float(b-line[0]*x)
    quad=validate_quad([[left,y_at('top',left)],[right,y_at('top',right)],
                        [right,y_at('bottom',right)],[left,y_at('bottom',left)]])
    top=y_at('top',outer_x); bottom=y_at('bottom',outer_x)
    target=validate_quad([[left,top],[right,top],[right,bottom],[left,bottom]])
    evidence=max(abs(y_at(s,right)-y_at(s,left))/max(2.,2*(lines[s][2]+lines[outer][2]*width/(bottom-top)))
                 for s in ('top','bottom'))
    shift=float(np.max(np.linalg.norm(quad-target,axis=1)))
    enabled=False
    if min(page.confidence[s] for s in lines)<.70:
        reason='外侧／上下纸边不够可靠，未自动开启'
    elif shift>min(width,bottom-top)*.08:
        reason='估计变形过大，未自动开启，请检查纸边'
    elif evidence<1 or shift<2:
        reason='梯形／斜切趋势不显著或与纸边弯曲相近，未自动开启'
    else:
        enabled=True
        reason=f'已推荐固定宽度校正；仅调整内侧角点高度，最大补正 {shift:.1f} px'
    inv=np.linalg.inv(rot)
    return validate_state(dict(source=project(quad,inv).tolist(),target=project(target,inv).tolist(),
                               offsets=[[0.,0.] for _ in range(4)],enabled=enabled,automatic=True,
                               recommended_enabled=enabled,reason=reason,fixed_width=True))


def recommend(page, inner, width=None):
    if width is not None:
        return _recommend_fixed_width(page,inner,float(width))
    from scan_cropper_auto import reference_angle
    angle,_ = reference_angle(page,inner)
    rot = rotation_matrix(angle,page.width,page.height)
    lines = {s:_fit(project(p,rot),s in ('left','right')) for s,p in page.edges.items()}
    quad=[]
    for a,b in [('top','left'),('top','right'),('bottom','right'),('bottom','left')]:
        quad.append(np.linalg.solve(np.array([lines[a][0],lines[b][0]]),[lines[a][1],lines[b][1]]))
    quad = validate_quad(quad)
    x=(quad[0,0]+quad[3,0])/2; r=(quad[1,0]+quad[2,0])/2
    y=(quad[0,1]+quad[1,1])/2; b=(quad[2,1]+quad[3,1])/2
    target=validate_quad([[x,y],[r,y],[r,b],[x,b]])
    inv=np.linalg.inv(rot)
    source=project(quad,inv); dest=project(target,inv)
    state=dict(source=source.tolist(),target=dest.tolist(),offsets=[[0.,0.] for _ in range(4)],
               enabled=False,automatic=True,reason='')
    # Opposite-edge convergence must exceed residual bow/noise on BOTH edges.
    vertical_signal=abs((quad[1,0]-quad[0,0])-(quad[2,0]-quad[3,0]))
    horizontal_signal=abs((quad[3,1]-quad[0,1])-(quad[2,1]-quad[1,1]))
    noise_v=lines['left'][2]+lines['right'][2]
    noise_h=lines['top'][2]+lines['bottom'][2]
    evidence=max(vertical_signal/max(2.,2*noise_v), horizontal_signal/max(2.,2*noise_h))
    shear_evidence,affine=_shear_candidate(lines,page.confidence,r-x,b-y,np.array([(x+r)/2,(y+b)/2]))
    reliable= min(page.confidence.values()) >= .70
    partial=not reliable and shear_evidence>=1
    if partial:
        # Only remove the skew supported by the three strong edges. Do not
        # straighten or infer perspective convergence from the uncertain edge.
        target=validate_quad(project(quad,affine))
        state['target']=project(target,inv).tolist()
    if reliable or partial:
        evidence=max(evidence if reliable else 0.,shear_evidence)
    shift=float(np.max(np.linalg.norm(quad-target,axis=1)))
    if not reliable and not partial:
        reason='可靠纸边不足以确认梯形或斜切，未自动开启'
    elif shift > min(r-x,b-y)*.08:
        reason='估计变形过大，未自动开启，请检查纸边'
    elif evidence < 1 or shift < 2:
        reason='梯形／斜切趋势不显著或与纸边弯曲相近，未自动开启'
    else:
        kind='斜切校正（依据三条可靠纸边）' if partial else '梯形／斜切校正'
        reason=f'已推荐{kind}；角点最大补正 {shift:.1f} px'
        state['enabled']=True
    state['reason']=reason
    state['recommended_enabled']=state['enabled']
    return validate_state(state)


def warp_analysis(page, state):
    h=state_matrix(state)
    if h is None:
        return page
    edges={s:project(p,h).tolist() for s,p in page.edges.items()}
    angles={}; roughness={}
    for s,p in edges.items():
        line,b,error=_fit(p,s in ('left','right'))
        k = -line[1] if s in ('left','right') else -line[0]
        angles[s]=math.degrees(math.atan(k if s in ('left','right') else -k))
        roughness[s]=error
    return replace(page,edges=edges,angles=angles,roughness=roughness)


def combined_matrix(width,height,angle,center=None,perspective=None):
    if center is None or center==(width/2,height/2):
        rot=rotation_matrix(angle,width,height)
    else:
        x,y=center
        rot=rotation_matrix(angle,x*2,y*2)
    return rot if perspective is None else rot @ np.asarray(perspective,float)


def transformed_bounds(width,height,matrix):
    q=project([[0,0],[width,0],[width,height],[0,height]],matrix)
    d=np.column_stack(([[0,0],[width,0],[width,height],[0,height]],np.ones(4))) @ np.asarray(matrix)[2]
    if min(d)*max(d)<=0:
        raise ValueError('当前梯形变换无法显示整幅图像，请减小角点调整')
    bounds=(math.floor(min(q[:,0])),math.floor(min(q[:,1])),math.ceil(max(q[:,0])),math.ceil(max(q[:,1])))
    if (bounds[2]-bounds[0]>width*3 or bounds[3]-bounds[1]>height*3
            or (bounds[2]-bounds[0])*(bounds[3]-bounds[1])>width*height*4):
        raise ValueError('梯形展开范围过大，请减小角点调整')
    return bounds
