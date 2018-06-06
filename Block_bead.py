# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:08:11 2017

@author: dykua
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from math import sin, cos, pi
#import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider


#==============================================================================
# Parameters about the block
#==============================================================================

class Block:
    def __init__(self, dim = np.array([1,1,1]), 
                 omega = np.array([1,0,0]),
                 M = 2):
       self.Dim = dim
       self.Omega = omega
       self.Mass = M
       self.Inertial =1/12* M * np.array([[dim[1]**2 + dim[2]**2, 0, 0],
                                         [0, dim[0]**2 + dim[2]**2, 0],
                                         [0, 0, dim[0]**2 + dim[1]**2]])
       self.verts = 0.5*np.array([[-dim[0], -dim[1], -dim[2]],
                                  [dim[0], -dim[1], -dim[2] ],
                                  [dim[0], dim[1], -dim[2]],
                                  [-dim[0], dim[1], -dim[2]],
                                  [-dim[0], -dim[1], dim[2]],
                                  [dim[0], -dim[1], dim[2]],
                                  [dim[0], dim[1], dim[2]],
                                  [-dim[0], dim[1], dim[2]]])

class Bead:
    def __init__(self, direction = np.array([0,0,1]),
                       m = 0.5,
                       dis = 0):
        self.Dir = direction
        self.Mass = m
        self.Dis = dis
    
    
       

dt = 0.1
t = np.arange(0, 100, dt)
block = Block(dim=[0.3,0.3,1], omega = np.array([0, -sin(3), cos(3)]))
#block = Block()
bead = Bead()
cam = 1.2*np.average(block.verts[4:8,:],0)

# =============================================================================
# Parameters about the figure
# =============================================================================
fig = plt.figure()
ax = p3.Axes3D(fig)
track = ([plt.plot(block.verts[i:i+1,0], block.verts[i:i+1,1],block.verts[i:i+1,2],
       markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)[0]
       for i in np.arange(8)])

track.append(plt.plot(bead.Dis*bead.Dir[0:1], bead.Dis*bead.Dir[1:2], bead.Dis*bead.Dir[2:3],
       markerfacecolor='r', markeredgecolor='r', marker='o', markersize=8)[0]) 
track.append(plt.plot(-bead.Dis*bead.Dir[0:1], -bead.Dis*bead.Dir[1:2], -bead.Dis*bead.Dir[2:3],
       markerfacecolor='r', markeredgecolor='r', marker='o', markersize=8)[0]) 
track.append(plt.plot([cam[0], -cam[0]], [cam[1], -cam[1]], [cam[2], -cam[2]],
             linewidth=2)[0] )
track.append(plt.plot([],[],[],
             color = 'k', linewidth=1, linestyle='--')[0] )
track.append(plt.plot([0,0], [0,0], [0,0],
             color = 'r', linewidth=1.5, linestyle='--')[0] )
pts = block.verts   
verts = [[pts[0],pts[1],pts[2],pts[3]],
         [pts[4],pts[5],pts[6],pts[7]], 
         [pts[0],pts[1],pts[5],pts[4]], 
         [pts[2],pts[3],pts[7],pts[6]], 
         [pts[1],pts[2],pts[6],pts[5]],
         [pts[4],pts[7],pts[3],pts[0]]]
 
Poly = Poly3DCollection(verts, facecolors='cyan', linewidths=.5, edgecolors='k', alpha=0.1)
ax.add_collection3d(Poly)
       
body_x = ax.quiver(0,0,0, 1, 0, 0,color='k', linestyle = '--', linewidth = 0.6)
ax.text(1, 0, 0, 'X')
body_y =ax.quiver(0,0,0, 0, 1, 0,color='k', linestyle = '--', linewidth = 0.6)
ax.text(0, 1, 0, 'Y')
body_z =ax.quiver(0,0,0, 0, 0, 1,color='k', linestyle = '--', linewidth = 0.6)
ax.text(0, 0, 1, 'Z')

body_axis = [body_x , body_y, body_z]

ax.set_xlim3d([-0.5 - block.Dim[0], 0.5 + block.Dim[0]])
ax.set_ylim3d([-0.5 - block.Dim[0], 0.5 + block.Dim[0]])
ax.set_zlim3d([-0.5 - block.Dim[0], 0.5 + block.Dim[0]])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_aspect('equal','box')
#=============================================================================
# Create sliders
# The four digits are upper left coord + length + hight
#=============================================================================
axcolor = 'lightgoldenrodyellow'
axq = plt.axes([0.02, 0.04, 0.3, 0.03], facecolor=axcolor) 
sq = Slider(axq, 'q', 0.1, 2, valinit=0)


#==============================================================================
# vector field of the rigid body equation
def f(t,x,I):
    return np.array([(I[1]-I[2])*x[1]*x[2]/I[0],
                      (I[2]-I[0])*x[2]*x[0]/I[1],
                      (I[0]-I[1])*x[0]*x[1]/I[2]]) 
    
# vector field for the reconstruction equation
def f_res(t, x, OMEGA):
     return np.dot(OMEGA, x)

 
#==============================================================================   
    
# initial condition for the reconstruction equation
Id = np.identity(3)

# angular momentum informtion, the hat map   
omega_hat = lambda omega: np.array([[0, omega[2], -omega[1]],
                                    [-omega[2], 0, omega[0]],
                                    [omega[1], -omega[0],0]])


Q = np.zeros((3,3,len(t)))
Q[:,:,0] = Id
sol_omega=np.zeros((len(t),3))
sol_omega[0,:] = block.Omega
#loc = np.zeros((8,3,len(t)))
#loc_bead = np.zeros((len(t),3))
loc_cam = np.zeros((len(t),3))
loc_cam[0,:] = 1.2*np.array([0, 0, 0.5])

momentum = np.zeros((len(t),3))
solver = ode(f).set_integrator('dopri5')
# =============================================================================
# solver_res0 = ode(f_res).set_integrator('dopri5')
# solver_res1 = ode(f_res).set_integrator('dopri5')
# solver_res2 = ode(f_res).set_integrator('dopri5')
# =============================================================================
for ii in np.arange(0,len(t)-1):
    bead.Dis = sq.val
    
    # Solve the Rigid Body equation
    solver.set_initial_value(sol_omega[ii,:],t[ii])
    Inertial = [block.Inertial[0,0]+2*bead.Mass*bead.Dis**2*(bead.Dir[1]**2+bead.Dir[2]**2),
                block.Inertial[1,1]+2*bead.Mass*bead.Dis**2*(bead.Dir[0]**2+bead.Dir[2]**2),
                block.Inertial[2,2]+2*bead.Mass*bead.Dis**2*(bead.Dir[0]**2+bead.Dir[1]**2)]
    solver.set_f_params(Inertial)
    
    sol_omega[ii+1,:] = solver.integrate(t[ii+1])
    
    # Solve the reconstruction equation, can use a different solver
# =============================================================================
#     solver_res0.set_initial_value(Q[:,0,ii], t[ii])
#     solver_res0.set_f_params(omega_hat((sol_omega[ii,:]+sol_omega[ii,:])/2))
#     solver_res1.set_initial_value(Q[:,1,ii], t[ii])
#     solver_res1.set_f_params(omega_hat((sol_omega[ii,:]+sol_omega[ii,:])/2))
#     solver_res2.set_initial_value(Q[:,2,ii], t[ii])
#     solver_res2.set_f_params(omega_hat((sol_omega[ii,:]+sol_omega[ii,:])/2))
#     
#     Q[:,:,ii+1] = np.array([solver_res0.integrate(t[ii+1]),
#                            solver_res1.integrate(t[ii+1]),
#                            solver_res2.integrate(t[ii+1])])
# =============================================================================
    Update = (np.linalg.solve(Id - dt/2*omega_hat(sol_omega[ii+1,:]), 
                              Id + dt/2*omega_hat(sol_omega[ii,:]))  )
    
    Q[:,:,ii+1] = np.dot(Update, Q[:,:,ii])    
    
    
    # realization
    loc = np.dot(block.verts, Q[:, :, ii+1])
    loc_bead = np.dot(bead.Dis*bead.Dir, Q[:,:,ii+1])
    loc_cam[ii+1,:] = 1.2*np.dot([0,0,0.5], Q[:,:,ii+1])
    momentum[ii+1,:] = np.dot(sol_omega[ii+1,:]*np.asarray(Inertial), Q[:,:,ii+1])
    
    for i in np.arange(8):
        track[i].set_data(loc[i,0],loc[i,1])
        track[i].set_3d_properties(loc[i,2])
    
    track[8].set_data(loc_bead[0],loc_bead[1])
    track[8].set_3d_properties(loc_bead[2])
    
    track[9].set_data(-loc_bead[0],-loc_bead[1])
    track[9].set_3d_properties(-loc_bead[2])
    
    track[10].set_data([loc_cam[ii+1,0], -loc_cam[ii+1,0]], [loc_cam[ii+1,1], -loc_cam[ii+1,1]])
    track[10].set_3d_properties([loc_cam[ii+1,2], -loc_cam[ii+1,2]])
    
    track[11].set_data(loc_cam[:ii+1,0], loc_cam[:ii+1,1])
    track[11].set_3d_properties(loc_cam[:ii+1,2])
    
    momentum_norm = np.linalg.norm(momentum[ii+1,:])
    track[12].set_data([0,momentum[ii+1,0]/momentum_norm], [0,momentum[ii+1,1]/momentum_norm])
    track[12].set_3d_properties([0,momentum[ii+1,2]/momentum_norm])
#    ax.quiver(0,0,0,momentum[ii+1,0],momentum[ii+1,1],momentum[ii+1,2])
    
    verts = [[loc[0],loc[1],loc[2],loc[3]],
             [loc[4],loc[5],loc[6],loc[7]], 
             [loc[0],loc[1],loc[5],loc[4]], 
             [loc[2],loc[3],loc[7],loc[6]], 
             [loc[1],loc[2],loc[6],loc[5]],
             [loc[4],loc[7],loc[3],loc[0]]]
    
    Poly.set_verts(verts)
    
    fig.canvas.draw()
    fig.canvas.flush_events()



plt.show()   
    