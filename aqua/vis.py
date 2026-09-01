from matplotlib import pyplot as plt
from matplotlib.image import imsave
import torch as pt

from .op2d import Op
from .util import mag

def vis(sol,sollag,op: Op,T=None,folder='cache/',i=None):
    
    if i is None:
        istr = ''
    else:
        istr = f'{i}'.rjust(6,'0')
        
    u = sol.u
    p = sol.p
    T = sol.T

    E = pt.tensordot(op.Mb(u),u,dims=u.dim())
    
    v,q,S = op.refine(u,p,T)
#   v,q,S = u*1,p*1,T*1

    v = v.flip(dims=(1,))
    vmag = mag(v)
    
    q = q.flip(dims=(0,))
    if S is not None:
        S = S.flip(dims=(0,))
    
    vmag = vmag.cpu()
    v = v.cpu()
    q = q.cpu()
    if S is not None:
        S = S.cpu()
    
    print(f'E: {E.item():.6e}')
    print(f'umag: min,max = {vmag.min().item():.3f},{vmag.max().item():.3f}')
    print(f'u: min,max = {v[0].min().item():.3f},{v[0].max().item():.3f}')
    print(f'v: min,max = {v[1].min().item():.3f},{v[1].max().item():.3f}')
    print(f'p: min,max = {q.min().item():.3f},{q.max().item():.3f}')
    if S is not None:
        print(f'T: min,max = {S.min().item():.3f},{S.max().item():.3f}')

    if sollag is not None:
        ul = sollag.u
        pl = sollag.p
        Tl = sollag.T

#       vl,ql,Sl = ul*1,pl*1,Tl*1
        vl,ql,Sl = op.refine(ul,pl,Tl)
        
        vl = vl.flip(dims=(1,))
        ql = ql.flip(dims=(0,))

        ud = v[0] - vl[0]
        vd = v[1] - vl[1]

        qd = q - ql

        print(f'ud: min,max = {ud.min().item():.3e},{ud.max().item():.3e}')
        print(f'vd: min,max = {vd.min().item():.3e},{vd.max().item():.3e}')
        print(f'pd: min,max = {qd.min().item():.3e},{qd.max().item():.3e}')
    
    cm = plt.get_cmap('turbo',2048)
    
    imsave(folder+f'umag{istr}.png', vmag,cmap=cm)
    imsave(folder+f'u{istr}.png', v[0],cmap=cm)
    imsave(folder+f'v{istr}.png', v[1],cmap=cm)
    imsave(folder+f'p{istr}.png', q,cmap=cm)
    if S is not None:
        imsave(folder+f'T{istr}.png', S,cmap=cm)
        
