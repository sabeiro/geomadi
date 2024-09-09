#%pylab inline
import os
import gzip
import sys
import numpy as np
import pandas as pd
import scipy as sp
import random
import matplotlib.pyplot as plt
import csv
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interpolate
from multiprocessing.dummy import Pool as ThreadPool 
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from tvtk.api import tvtk
from mayavi import mlab
import numpy as np

densM = np.load(os.environ['LAV_DIR']+"out/aggDensMat.npy")
velxM = np.load(os.environ['LAV_DIR']+"out/aggVelxMat.npy")
velyM = np.load(os.environ['LAV_DIR']+"out/aggVelyMat.npy")
(xp,yp,tp) = np.load(os.environ['LAV_DIR']+"out/aggGrid.npy")
xnp = np.array(range(xp.shape[0]))/float(xp.shape[0])
ynp = np.array(range(yp.shape[0]))/float(yp.shape[0])
tnp = np.array(range(tp.shape[0]))/float(tp.shape[0])
densT = np.array([[0,0,0]])
densS = np.array([0])
for i in range(densM.shape[0]):
    for j in range(densM.shape[1]):
        for k in range(densM.shape[2]):
            if(densM[i,j,k] > 0):
                densT = np.append(densT,[[xnp[i],ynp[j],tnp[k]]],axis=0)
                densS = np.append(densS,densM[i,j,k])

Xg, Yg = np.meshgrid(xp, yp) #np.mgrid[-1:1:30j, -1:1:30j]
x3g,y3g,z3g = np.meshgrid(xnp,ynp,tnp)
iLay = 3
Z = densM[:,:,iLay]
U = velxM[:,:,iLay]
V = velyM[:,:,iLay]
Z = ndi.filters.gaussian_filter(Z,sigma=2.)
U = ndi.filters.gaussian_filter(U,sigma=.5)
V = ndi.filters.gaussian_filter(V,sigma=.5)
# U = U/max(U.ravel())*U.shape[0]
# V = V/max(V.ravel())*V.shape[1]
M = np.hypot(U, V)
u = ndi.filters.gaussian_filter(velxM,sigma=5.)
v = ndi.filters.gaussian_filter(velyM,sigma=5.)
w = np.zeros((velxM.shape[0],velxM.shape[1],velxM.shape[2]))


mlab.clf()
ug = tvtk.UnstructuredGrid(points=densT)
ug.point_data.scalars = densS
ug.point_data.scalars.name = "value"
ds = mlab.pipeline.add_dataset(ug)
delaunay = mlab.pipeline.delaunay3d(ds)
iso = mlab.pipeline.iso_surface(delaunay)
iso.actor.property.opacity = 0.1
iso.contour.number_of_contours = 10
iso = mlab.contour3d(densM)
#mlab.quiver3d(u,v,w)
mlab.quiver3d(x3g,y3g,z3g,u,v,w,line_width=3,scale_factor=1)
mlab.flow(u,v,w)
mlab.show()


#%gui qt #for jupyter





# from numpy import pi, sin, cos, mgrid
# dphi, dtheta = pi/250.0, pi/250.0
# [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
# m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
# r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
# x = r*sin(phi)*cos(theta)
# y = r*cos(phi)
# z = r*sin(phi)*sin(theta)

# # View it.
# from mayavi import mlab
# s = mlab.mesh(x, y, z)
# mlab.show()






