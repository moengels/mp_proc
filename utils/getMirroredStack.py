"""
[QP,mask] = getQP(stack,s,mask)
---------------------------------------

Mirror the data stack according to the structure s (see "setup_phase.m")
Compute corresponding reciprocal space kx,kz according to the mirroring

Inputs:
  stack        Input intensity 3D stack
  s            Structure containing all optics and processing parameters
                  see "setup_phase_default.m"
  mask         Precomputed mask for fast reconstruction

Outputs:
  QP        	Quantitative phase
  mask         Mask used to filter the intensity stack

---------------------------------------
A detailled description of the theory supporting this program can be found in : 
"Descloux, A., et al. "Combined multi-plane phase retrieval and 
  super-resolution optical fluctuation imaging for 4D cell microscopy." 
  Nature Photonics 12.3 (2018): 165."

  Copyright © 2018 Adrien Descloux - adrien.descloux@epfl.ch, 
  École Polytechnique Fédérale de Lausanne, LBEN/LOB,
  BM 5.134, Station 17, 1015 Lausanne, Switzerland.

 	This program is free software: you can redistribute it and/or modify
 	it under the terms of the GNU General Public License as published by
 	the Free Software Foundation, either version 3 of the License, or
 	(at your option) any later version.

 	This program is distributed in the hope that it will be useful,
 	but WITHOUT ANY WARRANTY; without even the implied warranty of
 	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 	GNU General Public License for more details.

 	You should have received a copy of the GNU General Public License
 	along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
 
import numpy as np
from numpy import dot
from utils.cropXY import cropXY
from copy import copy   
    
def getMirroredStack(stack=None,s=None,*args,**kwargs):

    Nx,Ny,Nz=stack.shape
    
    if Nx != Ny:                     # verify that the stack is square
        stack=cropXY(stack)          # if not, crop it
    
    # compute real space
    x=np.linspace(np.dot(- Nx,s.optics_dx) / 2,np.dot(Nx,s.optics_dx) / 2,Nx)
    z=np.linspace(np.dot(- Nz,s.optics_dz) / 2,np.dot(Nz,s.optics_dz) / 2,Nz)
    # mirror z-stack
    temp=copy(stack)
    if s.proc_mirrorZ:
        t=copy(temp)
        t = np.append(t,temp[:, :, ::-1], 2)
        temp=copy(t)
        kz=np.multiply(np.dot(2,np.pi) / (max(z) - min(z)),np.linspace(- Nz / 2,(Nz - 1) / 2,np.dot(2,Nz)))
    else:
        if np.mod(Nz,2):
            kz=np.multiply(np.dot(2,np.pi) / (max(z) - min(z)),np.linspace(- Nz / 2,Nz / 2,Nz))
        else:
            kz=np.multiply(np.dot(2,np.pi) / (max(z) - min(z)),np.linspace(- Nz / 2,Nz / 2 - 1,Nz))
            
    # mirror x dim
    if s.proc_mirrorX:
        t=copy(temp)
        t = np.concatenate((t,temp[::-1,:, :]),0)
        temp = np.concatenate((t, t[:,::-1,:]),1)
        #t[Nx:,:,:] = t[:Nx:-1,:,:]
        #t = np.concatenate((t,flip))
        #temp=copy(t)
        kx=np.multiply(np.dot(2,np.pi) / (max(x) - min(x)),np.linspace(- Nx / 2,(Nx - 1) / 2,np.dot(2,Nx)))
    else:
        if np.mod(Nz,2):
            kx=np.multiply(np.dot(2,np.pi) / (max(x) - min(x)),np.linspace(- Nx / 2,Nx / 2,Nx))
        else:
            kx=np.multiply(np.dot(2,np.pi) / (max(x) - min(x)),np.linspace(- Nx / 2,Nx / 2 - 1,Nx))
    stackM = temp
    
    return kx,kz,stackM
    
if __name__ == '__main__':
    pass
    
