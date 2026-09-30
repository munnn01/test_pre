"""Outcome-free V12 low-QP policy and spatial distance."""
from itertools import product
import math
import numpy as np
import torch
import torch.nn.functional as F
from src.models.codec_search import CANDIDATES

QPS = (30,35,40,45,50)
TAUS = (0.0,0.1,0.25,0.5)
SLACKS = (0.1,0.25,0.5)
QP_MODES = {'low':(30,35),'lowmid':(30,35,40)}

def finite(v):
    return type(v) in (int,float) and math.isfinite(v)

def rgb112(clip):
    if (clip.dtype!=np.uint8 or clip.ndim!=4 or clip.shape[0]!=16 or clip.shape[-1]!=3
            or clip.shape[1]!=clip.shape[2] or clip.shape[1] not in (96,112,128)):
        raise ValueError('expected sixteen square uint8 RGB frames')
    x = torch.from_numpy(np.ascontiguousarray(clip.transpose(0,3,1,2))).float()/255
    return F.interpolate(x,size=(112,112),mode='bilinear',align_corners=False) if x.shape[-1]!=112 else x

def derivatives(x):
    return (x[...,1:]-x[...,:-1],x[...,1:,:]-x[...,:-1,:],
        x[...,1:-1,:-2]+x[...,1:-1,2:]+x[...,:-2,1:-1]+x[...,2:,1:-1]-4*x[...,1:-1,1:-1])

def spatial_distance(source,decoded):
    if source.shape!=(16,128,128,3):
        raise ValueError('source must use the frozen 128 grid')
    def prep(clip):
        x = rgb112(clip)
        return (x-x.new_tensor((.43216,.394666,.37645)).view(1,3,1,1))/x.new_tensor((.22803,.22145,.216989)).view(1,3,1,1)
    a,b = derivatives(prep(source)),derivatives(prep(decoded))
    ng = sum(float(torch.mean((a[k]-b[k])**2)) for k in (0,1))
    dg = sum(float(torch.mean(t**2)) for k in (0,1) for t in (a[k],b[k]))+1e-12
    nl = float(torch.mean((a[2]-b[2])**2))
    dl = float(torch.mean(a[2]**2)+torch.mean(b[2]**2))+1e-12
    v = (ng/dg+nl/dl)/2
    if not math.isfinite(v) or not 0<=v<=2+1e-6:
        raise ValueError('invalid spatial distance')
    return min(2.0,v)

def grid():
    return [{'tau_relative':tau,'rate_slack':slack,'qp_mode':mode} for tau,slack,mode in product(TAUS,SLACKS,QP_MODES)]

def validate_policy(p):
    if (set(p)!={'tau_relative','rate_slack','qp_mode'} or not finite(p['tau_relative'])
        or not finite(p['rate_slack']) or p['tau_relative'] not in TAUS or p['rate_slack'] not in SLACKS
        or p['qp_mode'] not in QP_MODES):
        raise ValueError('policy outside preregistered grid')

def measured_names(direction,fixed):
    if direction not in ('spatial','semantic') or fixed.get('v6') not in CANDIDATES:
        raise ValueError('invalid direction/base')
    return set(CANDIDATES) if direction=='semantic' else {'identity128','area112',fixed['v6']}

def choose(direction,qp,base,candidates,p):
    """Accept only explicit label-free fields; direct identity or feature rerank."""
    validate_policy(p)
    if direction not in ('spatial','semantic') or type(qp) is not int or qp not in QPS or base not in CANDIDATES:
        raise ValueError('invalid selector input')
    expected = measured_names(direction,{'v6':base})
    if set(candidates)!=expected:
        raise ValueError('missing or contaminated candidates')
    for item in candidates.values():
        if (set(item)!={'bpp','proxy_error'} or not finite(item['bpp']) or item['bpp']<=0
            or not finite(item['proxy_error']) or not 0<=item['proxy_error']<=2):
            raise ValueError('invalid or outcome-bearing candidate')
    if qp not in QP_MODES[p['qp_mode']]:
        return base,'high_qp_v6_retained'
    v = candidates[base]
    pool = ('identity128',) if direction=='spatial' else CANDIDATES
    valid = []
    for name in pool:
        item = candidates[name]
        gain = (v['proxy_error']-item['proxy_error'])/max(v['proxy_error'],1e-12)
        if (name!=base and gain>p['tau_relative'] and item['bpp']<=(1+p['rate_slack'])*v['bpp']
            and item['bpp']<=candidates['identity128']['bpp']):
            valid.append(name)
    if not valid:
        return base,'v6_retained'
    best = min(valid,key=lambda name:(candidates[name]['proxy_error'],candidates[name]['bpp'],CANDIDATES.index(name)))
    return best,'direct_identity_rescue' if direction=='spatial' else 'feature_rerank'

def dense_feature_distance(a,b):
    if len(a)!=2 or len(b)!=2:
        raise ValueError('exactly layer2/layer3 required')
    values = []
    for x,y in zip(a,b,strict=True):
        if x.shape!=y.shape or x.ndim!=4 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('invalid feature maps')
        x,y = F.normalize(x,dim=1,eps=1e-12),F.normalize(y,dim=1,eps=1e-12)
        values.append(float(((x-y)**2).sum(dim=1).mean()/2))
    v = sum(values)/2
    if not math.isfinite(v) or not 0<=v<=2+1e-6:
        raise ValueError('invalid feature distance')
    return min(2.0,v)
