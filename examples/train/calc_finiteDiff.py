#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy import signal as sg
import cv2
import matplotlib.path as mpltPath
import shapely as sh

def plog(text):
    print(text)


def kernelMatrix(weightM):
    finDif = {}
    finDif[0] = -.75*weightM[3] + 2.*weightM[4] - 1./12.*weightM[2]
    finDif[1] = -.5*weightM[1] + 2./3.*weightM[2] + 1.5*weightM[3] - 8.*weightM[4]
    finDif[2] = weightM[0] - 2.5*weightM[2] + 12.*weightM[4]
    finDif[3] = finDif[1] + weightM[1] - 3.*weightM[3]
    finDif[4] = finDif[0] + 1.5*weightM[3]
    kernelM = np.zeros((5,5))
    kernelM[2,range(5)] = [finDif[x] for x in range(5)]
    kernelM[range(5),2] = [finDif[x] for x in range(5)]
    return kernelM

def boundaryMatrix(nx,ny,nlayer=2):
    zBound = np.zeros((nx,ny)) > 0. ## boundary conditions
    for i in [0,1,zBound.shape[0]-2,zBound.shape[0]-1]:
        zBound[i] = True
        zBound[:,i] = True
    return zBound

def plotDens(x,y,BBox):
    xx, yy = np.mgrid[BBox[0]:BBox[1]:64j, BBox[2]:BBox[3]:64j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    #ax.imshow(np.rot90(f), cmap='Blues', extent=BBox)
    cset = ax.contour(xx, yy, f, colors='k')
    #ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # plt.imshow(f,extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),cmap=cm.hot, norm=LogNorm())
    # plt.imshow(f,cmap=cm.hot)
    return ax


def rayTracing(x,y,poly):
    n = len(poly)
    inside = False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def isInside(xp,yp,poly,isDebug=False):
    if isDebug:
        start_time = time()
    cellIn1 = [False for x in range(xp.shape[0])]
    if not type(poly) in [sh.geometry.Polygon,sh.geometry.MultiPolygon]:
        return cellIn1
    try:
        for l in poly.boundary:
            path = mpltPath.Path(np.array([[l.xy[0][x],l.xy[1][x]] for x in range(len(l.xy[0]))]))
            cellIn1 = cellIn1 + path.contains_points([[x,y] for x,y in zip(xp,yp)])
    except:
        return cellIn1
    if isDebug:
        print ("Ray Tracing Elapsed time: " + str(time()-start_time))
        start_time = time()
        cellIn2 = [poly.contains(sh.geometry.Point(x,y)) for x,y in zip(xp,yp)]
        print ("shapely Elapsed time: " + str(time()-start_time))
        print("sum matplotlib, sum shapely, difference")
        print(sum(cellIn1),sum(cellIn2),sum(np.array(cellIn1) ^ np.array(cellIn2)))
    return cellIn1

def moveGrad(x,y,z,nx,ny,BBox,dt=0.1):
    dz = np.gradient(z)
    vx, vy, vz = -dz[0], -dz[1], np.zeros((nx,ny))
    vmax = vx.max() if vx.max() > 0 else 1.
    vx = vx/vmax*1./nx
    vmax = vy.max() if vy.max() > 0 else 1.
    vy = vy/vmax*1./ny
    ix = x.apply(lambda x : int( (x-BBox[0])/(BBox[1]-BBox[0])*nx - 0.00000001) )
    iy = y.apply(lambda x : int( (x-BBox[2])/(BBox[3]-BBox[2])*nx - 0.00000001) )
    x_v = np.array([vx[x][y]*np.random.uniform()*dt for x,y in zip(ix,iy)])
    y_v = np.array([vy[x][y]*np.random.uniform()*dt for x,y in zip(ix,iy)])
    return x_v, y_v
