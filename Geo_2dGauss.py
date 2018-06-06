
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:26:30 2017

@author: dykuang

This script calculates geodesics on the surface discribed by 2d 
Gaussian distribution. Only considering if the covariant matrix is
a diagonal one and mean at the origin. For general cases, it just needs a 
proper rotation and translation.
"""

import numpy as np
from scipy.integrate import odeint
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib as mpl
#mpl.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
import matplotlib.animation as animation

# configuration of the surface
sigma = np.array([[2,0],[0,1]]) # be careful about the 'type'
sigma_inv = inv(sigma)

f = lambda x, y: -np.exp(-x**2/sigma[0,0]-y**2/sigma[1,1])
#f = lambda x, y: np.exp(-np.dot([x,y], np.dot(sigma_inv, [x,y])))

Lf = lambda u, v: -f(u,v)*np.array([-2*u/sigma[0,0], -2*v/sigma[1,1]])
#Lf = lambda u, v: -2*f(u,v)*np.dot(sigma_inv, np.array([u,v]))

Hf = lambda u, v: (-f(u,v)*np.array([ [2/sigma[0,0]*(2*u**2/sigma[0,0]-1), 4*u*v/sigma[0,0]/sigma[1,1] ], 
                   [4*u*v/sigma[0,0]/sigma[1,1], 2/sigma[1,1]*(2*v**2/sigma[1,1]-1)]]) )
# =============================================================================
# Hf = lambda u, v: ( 2*f(u,v)*
#                    (2*np.dot(np.dot(sigma_inv, np.array([u,v])),np.dot(np.array([u,v]),sigma_inv)) 
#                    - sigma_inv 
#                    )
# 
# =============================================================================

# patch map to R3
def patch(u, v):
    X = ([u, 
         v,
         f(u,v)])
    return X

# calculate first fundamental form
def FFF(u, v):
    G = np.zeros((2,2))
    LF = Lf(u,v)
    G[0,0] =  1+LF[0]**2
    G[1,1] =  1+LF[1]**2
    G[0,1] =  LF[0]*LF[1]
    G[1,0] =  G[0,1]
    
    return G
    
# calculate the partial metrix (automatic differentiation?)
def pGpu(u, v):
    p_Gu = np.zeros((2,2))
    LF = Lf(u,v)
    HF = Hf(u,v)
    p_Gu[0,0] = 2*LF[0]*HF[0,0]
    p_Gu[0,1] = HF[0,0]*LF[1]+HF[0,1]*LF[0]
    p_Gu[1,0] = p_Gu[0,1]
    p_Gu[1,1] = 2*LF[1]*HF[0,1]
    
    return p_Gu
    
def pGpv(u, v):
    p_Gv = np.zeros((2,2))
    LF = Lf(u,v)
    HF = Hf(u,v)
    p_Gv[0,0] = 2*LF[0]*HF[0,1]
    p_Gv[0,1] = LF[0]*HF[1,1]+HF[0,1]*LF[1]
    p_Gv[1,0] = p_Gv[0,1]
    p_Gv[1,1] = 2*LF[1]*HF[1,1]
    
    return p_Gv
    
# The right side of geodesic ODE
def vf(x, t):
    u = x[0]
    v = x[1]
    G = FFF(u, v)
    G_inv = inv(G)
    p_Gu = pGpu(u, v)
    p_Gv = pGpv(u, v)
    
    res = ([(p_Gu[1,1]-2*p_Gv[0,1]) * x[3]**2 - p_Gu[0,0]*x[2]**2 - 2*p_Gv[0,0]*x[2]*x[3],
            (p_Gv[0,0]-2*p_Gu[0,1]) * x[2]**2 - p_Gv[1,1]*x[3]**2 - 2*p_Gu[1,1]*x[2]*x[3]]) 
    
    F = np.zeros(4)
    F[0] = x[2]
    F[1] = x[3]
    F[2], F[3] =  0.5*np.dot(G_inv, res) # replace with other solver?
    
    return F


# specify domain
N =201
t_end = 18
t = np.linspace(0, t_end, N)
#x = np.zeros((N, 4))

# initial condition
x0 = [[8, loc, -1, 0] for loc in np.arange(-3, 4, 1)]

# ODE solver/integrator
sol = [odeint(vf, x_ini, t) for x_ini in x0] #replace with other integrator?
track_num = len(sol)
# =============================================================================
# # visualize in local coordinates
# plt.figure()
# plt.plot(t, sol[:,0], 'b', label = 'u')
# plt.plot(t, sol[:,1], 'k', label = 'v')
# plt.legend(loc = 'best')
# plt.xlabel('t')
# plt.show()
# =============================================================================

# visualize in R3
X = np.zeros((3,N,track_num))
for j in np.arange(track_num):
    for i in np.arange(0, N):
        X[:,i, j] = patch(sol[j][i,0], sol[j][i,1])

fig = plt.figure()
ax = Axes3D(fig)
track = [ax.plot(X[0,:1,ii], X[1,:1,ii], X[2,:1,ii],color = 'k', linewidth = 0.8)[0] for ii in np.arange(track_num)]

        
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

X3 = np.zeros((len(x1), len(x2)))
for k in np.arange(len(x1)):
    for s in np.arange(len(x2)):
        X3[k,s] = f(x1[s],x2[k])  # Be careful about the orientation

# Plot the surface
ax.plot_surface(X1, X2, X3, color='cyan', alpha=1.0)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Geodesics on Gaussian surface')

#ax.view_init(elev=90., azim=0.)
#===============================================================
# animation for fun
#===============================================================
#Writer = animation.writers['ffmpeg']
#Writer = Writer(fps=30, metadata=dict(artist='D.Kuang'), bitrate=1800)

#Writer = animation.FFMpegWriter()

def animate(i):
    for x, ii in zip(track, np.arange(track_num)):
        x.set_data(X[0:2, :i, ii])
        x.set_3d_properties(X[2, :i, ii])
    
    return track

track_ani = animation.FuncAnimation(fig, animate, N, interval= 20, blit=True)

#track_ani.save('geo_2dGauss.mp4',writer= Writer)

plt.show()