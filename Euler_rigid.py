# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:53:00 2017

@author: dykuang

This script test if scipy's odeint can solve Euler's rigid motion equation
"""

import numpy as np
from scipy.integrate import odeint, ode
from scipy.optimize import broyden1, fsolve, broyden2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


I = np.array([1.0, 2.0, 3.0]) # moment of inertial 

x0 = np.array([np.sin(np.pi/6), np.cos(np.pi/6), 0])

dt = 0.1
t = np.arange(0, 100, dt)

#==============================================================================
# Using odeint method
#==============================================================================
# def f(x, t, I):
#     return np.array([(I[1]-I[2])*x[1]*x[2]/I[0],
#                     (I[2]-I[0])*x[2]*x[0]/I[1],
#                     (I[0]-I[1])*x[0]*x[1]/I[2]])
#     
# sol = odeint(f, x0, t, args = (I,))
#==============================================================================

#==============================================================================
# using ode class
#==============================================================================
#==============================================================================
# def f(t,x,I):
#    return np.array([(I[1]-I[2])*x[1]*x[2]/I[0],
#                      (I[2]-I[0])*x[2]*x[0]/I[1],
#                      (I[0]-I[1])*x[0]*x[1]/I[2]]) 
# 
# solver = ode(f).set_integrator('dopri5')
# solver.set_initial_value(x0,t[0]).set_f_params(I)
# 
# i = 0
# sol = np.zeros((len(t),3))
# while solver.successful() and solver.t < t[-1]:
#      sol[i,:]=solver.integrate(solver.t + 0.1)
# #     sol[i,:] = sol[i,:]/(sol[i,0]**2 + sol[i,1]**2 + sol[i,2]**2)**0.5
#      i = i+1
#==============================================================================

#==============================================================================
# sol = np.zeros((len(t),3))
# sol[0,:] = x0
# for i in np.arange(0,len(t)-1):
#     eqn = lambda y: np.array(y)-sol[i,:]-0.1/2*(f(t[i+1], y,I)+f(t[i],sol[i,:],I))
#     sol[i+1,:] = broyden1(eqn, sol[i,:]+0.1*f(t[i],sol[i,:],I))
#==============================================================================

#==============================================================================
# Using change of coordinate
#==============================================================================
# def eqn(xp, x, dt, I):
#     F = np.zeros(3)
#     prod_x = x[0]*x[1]*x[2]
#     prod_xp = xp[0]*xp[1]*xp[2]
#     F[0] = I[0]*(xp[0]**2-x[0]**2)-dt*(I[1]-I[2])*(prod_x+prod_xp)
#     F[1] = I[1]*(xp[1]**2-x[1]**2)-dt*(I[2]-I[0])*(prod_x+prod_xp)
#     F[2] = I[2]*(xp[2]**2-x[2]**2)-dt*(I[0]-I[1])*(prod_x+prod_xp)
#     return F
# 
# def update(x, dt, I):
#     push = np.array([x[1]*x[2]*(I[1]-I[2])/I[0],
#                      x[0]*x[2]*(I[2]-I[0])/I[1],
#                      x[0]*x[1]*(I[0]-I[1])/I[2]])
#     
#     xp = fsolve(lambda y: eqn(y, x, dt, I), x + dt*push)
#     return xp
# 
# sol = np.zeros((len(t),3))
# 
# sol[0,:] = x0
# 
# for i in np.arange(0,len(t)-1):
#     sol[i+1,:] = update(sol[i,:], dt, I)
#     sol[i+1,:] = sol[i+1,:]/(sol[i+1,0]**2 + sol[i+1,1]**2 + sol[i+1,2]**2)**0.5 # a "hard" projection
#==============================================================================


#==============================================================================
# Using Euler forward to predict, change of coords + Trapzoidal to correct
# It is an explicit scheme
#==============================================================================
sol = np.zeros((len(t),3))
sol[0,:] = x0
for i in np.arange(0,len(t)-1):
    # predict
    pred = sol[i,:] + dt*np.array([sol[i,1]*sol[i,2]*(I[1]-I[2])/I[0],
                      sol[i,2]*sol[i,0]*(I[2]-I[0])/I[1],
                      sol[i,0]*sol[i,1]*(I[0]-I[1])/I[2]])
    
    # correct
    prod_x = sol[i,0]*sol[i,1]*sol[i,2]
    prod_xp = pred[0]*pred[1]*pred[2]
    sol[i+1,0] = np.sign(pred[0])*(abs(sol[i,0]**2+dt*(I[1]-I[2])/I[0]*(prod_x+prod_xp)))**0.5
    sol[i+1,1] = np.sign(pred[1])*(abs(sol[i,1]**2+dt*(I[2]-I[0])/I[1]*(prod_x+prod_xp)))**0.5
    sol[i+1,2] = np.sign(pred[2])*(abs(sol[i,2]**2+dt*(I[0]-I[1])/I[2]*(prod_x+prod_xp)))**0.5
    
#    sol[i+1,:] = sol[i+1,:]/(sol[i+1,0]**2 + sol[i+1,1]**2 + sol[i+1,2]**2)**0.5


E = 0.5*(I[0]*sol[:,0]**2 +I[1]*sol[:,1]**2 + I[2]*sol[:,2]**2 )
P = sol[:,0]**2 + sol[:,1]**2 + sol[:,2]**2
K = I[0]**2*sol[:,0]**2 +  I[1]**2*sol[:,1]**2 +  I[2]**2*sol[:,2]**2
 
fig = plt.figure()
plt.plot(t, P)
plt.plot(t, E)
plt.plot(t, K)

fig = plt.figure()
plt.plot(t, sol[:,0])
plt.plot(t, sol[:,1])
plt.plot(t, sol[:,2])

fig = plt.figure()
ax = Axes3D(fig)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x1 = np.outer(np.cos(u), np.sin(v))*(K[0])**0.5/I[0]
x2 = np.outer(np.sin(u), np.sin(v))*(K[0])**0.5/I[1]
x3 = np.outer(np.ones(np.size(u)), np.cos(v))*(K[0])**0.5/I[2]

ax.plot_surface(x1, x2, x3, color='b', alpha=0.4)
ax.quiver(0,0,0,1,0,0, color='r', linestyle = '--')
ax.quiver(0,0,0,0,1,0, color='r', linestyle = '--')
ax.quiver(0,0,0,0,0,1, color='r', linestyle = '--')
ax.set_aspect('equal','box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Solution of Euler rigid body equation using Odeint')

ax.plot(sol[:,0],sol[:,1],sol[:,2], linewidth = 1)
pt, = ax.plot(sol[0:1,0],sol[0:1,1],sol[0:1,2], Marker = 'o', Markersize = 5)

def animate(i):
    pt.set_data(sol[i,0], sol[i,1])
    pt.set_3d_properties(sol[i,2])
    
    return pt,
    
ani = animation.FuncAnimation(fig, animate, len(t), interval=20, blit=True)

plt.show()