from .op2d import Op
from .setops import setops
from .problem import problem
from .advance import BDFEXT, Solution

from .util import mag
from .vis import vis
from .ns import ns
from .stokes import stokes

import matplotlib.pyplot as plt

import cProfile
import pstats
import io
import os

import numpy as np
import torch as pt

def main():
    Nx=48
    dt=0.015625

    Nx=64
    dt=0.01

    Nx=128
    dt=0.0040

    Nx=160
    dt=0.003125


    Ny=Nx
    tf=1e4
#   dt=2.5e-3
#   dt=5e-3
#   dt=2e-2

    iotime=1000
    iptime=100
    
#   nuis = pt.tensor([15500])
    nuis = pt.tensor([20000])
    nuis = pt.tensor([15000])
    nuis = pt.tensor([10000])
#   nuis = pt.tensor([5100])
    nuis = pt.tensor([15000])
    
    pt.set_default_dtype(pt.float64)

    cname='ldc_reg_ref'
    pt.set_printoptions(sci_mode=True,precision=4)

    sol_stokes,op = stokes(Nx,Ny,cname,heat=True)
    vis(sol_stokes,None,op,folder='stokes/')
    
    for nui in nuis:
        nu = 1 / nui
        nui_str = str(int(pt.round(nui)))
        dir = f'data_new/{nui_str}_/'
        os.makedirs(dir, exist_ok=True)
        sol,op = ns(Nx,Ny,tf,dt,cname,nu,u0=sol_stokes,iptime=iptime,iotime=iotime,pfx=dir)
#       vis(sol,op,folder='data/')

if __name__ == "__main__":
#   cProfile.run('main()', sort='time')
    
    profile=False
    if profile:
        pr = cProfile.Profile()
        pr.run('main()')
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        
        ps.sort_stats('time')
        ps.print_stats(20)
        
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        print(s.getvalue())
    else:
        main()
