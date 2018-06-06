# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:26:30 2017

@author: dykuang

This script calculates geodesics on ellipezoids with diffrent initial conditions
"""

import numpy as np
from scipy.integrate import odeint
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# configuration of the ellipsoid
a = 2.0
b = 2.0
c = 1.0

# calculate first fundamental form
def FFF(theta, phi):
    G = np.zeros((2,2))
    G[0,0] = (a**2 * np.cos(theta)**2 * np.cos(phi)**2 
             + b**2 * np.cos(theta)**2 * np.sin(phi)**2
             + c**2 * np.sin(theta)**2)
    G[1,1] = np.sin(theta)**2 * (a**2 * np.sin(phi)**2 + b**2 * np.cos(phi)**2)
    G[0,1] = (b**2 - a**2) * np.sin(2*theta)* np.sin(2*phi)/4
    G[1,0] = G[0,1] 
    
    return G
    
# calculate the partial metrix (automatic differentiation?)
def pGpt(theta, phi):
    p_Gt = np.zeros((2,2))
    p_Gt[0,0] = np.sin(2*theta)*(c**2 - a**2 * np.cos(phi)**2 - b**2 * np.sin(phi)**2)
    p_Gt[0,1] = 0.5*(b**2 - a**2)*np.cos(2*theta)**2 * np.sin(2*phi)**2
    p_Gt[1,0] = p_Gt[0,1]
    p_Gt[1,1] = np.sin(2*theta)*(a**2 * np.sin(phi)**2 + b**2 * np.cos(phi)**2)
    
    return p_Gt
    
def pGpp(theta, phi):
    p_Gp = np.zeros((2,2))
    p_Gp[0,0] = np.sin(2*phi)* (b**2 - a**2) * np.cos(theta)**2
    p_Gp[0,1] = 0.5*(b**2 - a**2)*np.sin(2*theta)**2 * np.cos(2*phi)**2
    p_Gp[1,0] = p_Gp[0,1]
    p_Gp[1,1] = np.sin(2*phi)*(a**2 - b**2) * np.sin(theta)**2
    
    return p_Gp
    
# The right side of geodesic ODE
def f(x, t):
    theta = x[0]
    phi = x[1]
    G = FFF(theta, phi)
    G_inv = inv(G)
    p_Gt = pGpt(theta, phi)
    p_Gp = pGpp(theta, phi)
    
    res = ([(p_Gt[1,1]-2*p_Gp[0,1]) * x[3]**2 - p_Gt[0,0]*x[2]**2 - 2*p_Gp[0,0]*x[2]*x[3],
            (p_Gp[0,0]-2*p_Gt[0,1]) * x[2]**2 - p_Gp[1,1]*x[3]**2 - 2*p_Gt[1,1]*x[2]*x[3]])
    
    F = np.zeros(4)
    F[0] = x[2]
    F[1] = x[3]
    F[2], F[3] = 0.5*np.dot(G_inv, res)  # replace with other solver?
    return F

# patch map to R3
def patch(theta, phi):
    X = ([a*np.sin(theta)*np.cos(phi), 
         b*np.sin(theta)*np.sin(phi),
         c*np.cos(theta)])
    return X

# specify parameter domain
N =1001
t_end = 100
t = np.linspace(0, t_end, N)
x = np.zeros((N, 4))

# initial condition
#x[0,:] = [np.pi/6, np.pi/3, 1, 1]
x[0,:] = [np.pi/15, 0, 1, 1]
# ODE solver/integrator
sol = odeint(f, x[0,:], t) #replace with other integrator?

# visualize in local coordinates
plt.figure()
plt.plot(t, sol[:,0], 'b', label = 'theta')
plt.plot(t, sol[:,1], 'k', label = 'phi')
plt.legend(loc = 'best')
plt.xlabel('t')
plt.show()

# visualize in R3
X = np.zeros((3,N))
for i in np.arange(0, N):
    X[:,i] = patch(sol[i,0], sol[i,1])

fig = plt.figure()
ax = Axes3D(fig)
#track = ax.plot(X[0,:],X[1,:],X[2,:], color= 'k')
track = ax.plot(X[0,:1],X[1,:1],X[2,:1], color= 'k',linewidth = 0.8)[0]

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x1 = a * np.outer(np.cos(u), np.sin(v))
x2 = b * np.outer(np.sin(u), np.sin(v))
x3 = c * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x1, x2, x3, color='b', alpha=0.4)
ax.quiver(0,0,0,a,0,0, color='r', linestyle = '--')
ax.quiver(0,0,0,0,b,0, color='r', linestyle = '--')
ax.quiver(0,0,0,0,0,c, color='r', linestyle = '--')
#ax.set_aspect('equal','box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Geodesics on revolutional ellilpzoid')
#===================================================#
# Add animation for fun 
#=================================================#
Writer = animation.writers['ffmpeg']
Writer = Writer(fps=30, metadata=dict(artist='D.Kuang'), bitrate=1800)

def animate(i):
    track.set_data(X[0:2,:i])
    track.set_3d_properties(X[2,:i])
    return track,

track_ani = animation.FuncAnimation(fig, animate, N, interval=20, blit=True)
track_ani.save('geo_Ellipzoid1.mp4',writer= Writer)
plt.show()
