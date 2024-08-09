from utils.cropXY import cropXY
from utils.cropCoregMask import cropCoregMask
from utils.phase_structure import phase_structure
from utils.getQP import getQP
from utils.getMirroredStack import getMirroredStack
from utils.map3D import map3D
import numpy as np
from numpy import dot, pi, sin, cos, logical_and, logical_or
from tqdm import tqdm 
from numba import jit, cuda
import math


def qpmain(stack, s):
    Ny,Nx,Nz,Nt = stack.shape

    # set experimental parameters
    if Nz == 4:              # i.e. MultiPlane data
        s.optics_dz = 0.62     
        if Ny is not Nx:          # [um]
            stack=cropXY(stack)
    else:
        s.optics_dz = 0.2 
        if Ny is not Nx:   
            stack=cropXY(stack)

    #phase_structure.summarise(s)
    phase = iterate_getQP(stack, Nt, s)
    return phase
 
#@jit(nopython=True)   
def iterate_getQP(stack, Nt, s):
    phase = np.empty(stack.shape, np.float64)
    inner = tqdm(total=Nt, desc='phase - timepoint', position=0)
    mask = precompute_mask(stack, s)
    for timepoint in range(Nt):
        phase[:,:,:,timepoint], _ = getQP(stack[:,:,:,timepoint], s, mask)
        inner.update(1)
    return phase 


def precompute_mask(stack, struct):
     # mirror the data and compute adequate Fourier space grid
    kx,kz,stackM = getMirroredStack(stack,struct)

    # compute usefull stuff
    th=math.asin(struct.optics_NA / struct.optics_n)
    th_ill=math.asin(struct.optics_NA_ill / struct.optics_n)
    k0max = dot(dot(struct.optics_n,2),pi) / (struct.optics_wv - struct.optics_dlambda / 2) 
    k0min = dot(dot(struct.optics_n,2),pi) / (struct.optics_wv + struct.optics_dlambda / 2) 
    
    # compute Fourier space grid and the phase mask
    Kx,Kz=np.meshgrid(kx,kz)
    if struct.optics_kzT is None:
        mask2D = Kz >= np.dot(k0max,(1 - cos(th_ill)))
    else:
        mask2D = Kz >= struct.optics_kzT


    if struct.proc_applyFourierMask:  #  => compute the CTF mask for extra denoising
        # CTF theory
        maskCTF = logical_and(logical_and(logical_and(
            ((Kx - dot(k0max,sin(th_ill))) ** 2 + (Kz - dot(k0max,cos(th_ill))) ** 2) <= k0max ** 2, \
                ((Kx + dot(k0min,sin(th_ill))) ** 2 + (Kz - k0min) ** 2) >= k0min ** 2), Kx >= 0), \
                Kz < dot(k0max,(1 - cos(th))))            
        maskCTF = logical_or(maskCTF,maskCTF[:,::-1])
        mask2D = np.asanyarray(logical_and(mask2D,maskCTF), dtype=int)
    # since we assume a circular symetric CTF, we expand the 2Dmask in 3D
    mask=map3D(mask2D)
    return mask

    


if __name__ == '__main__':
    pass



