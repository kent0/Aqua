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
    Nx=128
    Ny=Nx
    tf=2e2
    dt=2.5e-3

    nu=1/3000
    iotime=0.1
    iptime=1
    
    nuis = pt.linspace(3000,4000,21)
    
    pt.set_default_dtype(pt.float64)

    cname='ldc_reg_ref'
    pt.set_printoptions(sci_mode=True,precision=4)

    sol_stokes,op = stokes(Nx,Ny,cname,heat=True)
    vis(sol_stokes,op,folder='stokes/')
    
    for nui in nuis:
        nu = 1 / nui
        nui_str = str(int(pt.round(nui)))
        dir = f'data_new/{nui_str}/'
        os.makedirs(dir, exist_ok=True)
        sol,op = ns(Nx,Ny,tf,dt,cname,nu,u0=sol_stokes,iptime=iptime,iotime=iotime,pfx=dir)

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
