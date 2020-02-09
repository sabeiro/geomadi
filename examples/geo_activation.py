import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import shapely as sh
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import shapely.speedups
shapely.speedups.enable()
import lernia.train_viz as t_v
import albio.series_stat as s_s
import albio.series_interp as s_i
import geomadi.geo_octree as g_o
import geomadi.geo_ops as g_p
import geomadi.geo_enrich as g_e

import importlib
importlib.reload(g_o)
importlib.reload(g_e)

if False:
    print('--------------------------2d-fourier--------------------')
    chem10 = gpd.read_file(baseDir + "gis/dep/chem_pot_10.shp")
    tL = [x for x in chem10 if bool(re.search("dis_",x))]
    tL = ['urev','pot','activation','err'] + tL#[:1]
    t_v.plotFeatCorr(chem10[tL])
    plt.show()

if False:
    print('--------------------------2d-fourier--------------------')
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft2, ifft2
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import binned_statistic_2d
    
    x = stat['geometry'].apply(lambda x: x.xy[0][0])
    y = stat['geometry'].apply(lambda x: x.xy[1][0])
    z = [1 for x in range(len(x))]
    n_bin = 10
    minx = min(x)
    miny = min(y)
    deltax = max(max(x)-minx,max(y)-miny)*1.1
    dx, dy = deltax/n_bin, deltax/n_bin
    binx = np.linspace(minx,minx+deltax,n_bin)
    biny = np.linspace(miny,miny+deltax,n_bin)
    tx, ty = np.meshgrid(binx,biny)
    ret = binned_statistic_2d(x, y, None, 'count', bins=[binx,biny], expand_binnumbers=True)
    sums = np.zeros([n_bin,n_bin])
    for i,j,k in zip(x,y,z):
        r = int((i-minx)/deltax*n_bin)
        c = int((j-miny)/deltax*n_bin)
        if (r<0) | (r>=n_bin): continue
        if (c<0) | (c>=n_bin): continue
        sums[r][c] += k
    plt.imshow(sums)
    plt.show()
    
    """CREATING REAL AND MOMENTUM SPACES GRIDS"""
    N_x, N_y = 2 ** 10, 2 ** 10
    range_x, range_y = np.arange(N_x), np.arange(N_y)
    dx, dy = 0.005, 0.005
    xv, yv = dx * (range_x - 0.5 * N_x), dy * (range_y - 0.5 * N_y)
    dk_x, dk_y = np.pi / deltax, np.pi / deltax
    # momentum space grid vectors, shifted to center for zero frequency
    k_xv, k_yv = dk_x * np.append(binx[:n_bin//2]-minx, -binx[n_bin//2:0:-1]+minx), \
                 dk_y * np.append(biny[:n_bin//2]-minx, -biny[n_bin//2:0:-1]+minx)

    # create real and momentum spaces grids
    x, y = np.meshgrid(xv, yv, sparse=False, indexing='ij')
    kx, ky = np.meshgrid(k_xv, k_yv, sparse=False, indexing='ij')
    
    """FUNCTION"""
    sigma = 0.05
    f = 1/(2*np.pi*sigma**2) * np.exp(-0.5 * (kx ** 2 + ky ** 2)/sigma**2)
    F = fft2(sums)
    """PLOTTING"""
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(tx, ty, sums, cmap='viridis')
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    surf = ax2.plot_surface(tx, ty, np.abs(F)*dx*dy, cmap='viridis')
    plt.show()

print(sp.stats.pearsonr(chem10['urev'],chem10['dis']))
