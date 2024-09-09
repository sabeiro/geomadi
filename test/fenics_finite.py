#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
# from dolfin import *
# from fenics import *
import seaborn as sns
import scipy.stats as st
import osmnx as ox
import networkx as nx
import geopandas as gpd
import shapely as sh
import geomadi.geo_geohash as geohash
import geomadi.geo_octree as octree
from shapely.ops import cascaded_union
from scipy import signal as sg
import shapely as sh
import cv2
import geomadi.calc_finiteDiff as c_f
import importlib

importlib.reload(c_f)

def plog(text):
    print(text)

with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)

with open(baseDir + '/raw/basics/metrics.json') as f:
    metr = json.load(f)['metrics']

import pymongo
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
cent = pd.read_csv(baseDir + "raw/roda/fem_centroid.csv")
cent = cent.sort_values('AreaInters')
# act = pd.read_csv(baseDir + "log/roda/act_roda_11.csv.gz",compression="gzip")
# act = pd.concat([act,pd.read_csv(baseDir + "log/roda/act_roda_12.tar.gz",compression="gzip")])
# act = pd.concat([act,pd.read_csv(baseDir + "log/roda/act_roda_13.tar.gz",compression="gzip")])
act = pd.read_csv(baseDir + "log/roda/act_roda.csv.gz",compression="gzip")

act = act[['IMSI', 'cilac', 'dur', 'x', 'sx', 'y', 'sy', 't', 'st']]
act.dropna(inplace=True)
cells = pd.read_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip")
def clampF(x):
    return pd.Series({"x":np.average(x['x']),"sx":np.average(x['sx'])
                      ,"y":np.average(x['y']),"sy":np.average(x['sy'])
                      ,"t":np.average(x['t']),"st":np.average(x['st'])
                      ,"n":len(x['cilac'])
        })
gact = act.groupby('cilac').apply(clampF).reset_index()
gact.dropna(inplace=True)
gact.loc[:,"sr"] = gact["sx"] + gact["sy"]
gact = pd.merge(gact,cells,on="cilac",how="left")
gact = gpd.GeoDataFrame(gact)
gact.geometry = [sh.geometry.Point(x,y) for x,y in zip(gact['X'],gact['Y'])]
gact.geometry = [x.buffer(y,resolution=6) for x,y in zip(gact['geometry'],gact['sr']*gact['n']/1000.)]
##ue: user end device - modem, telefon,2g bts,3g nodeB,4g e-nodeB 800MHz - big
##2G6Hz mode boradbadnd smaller,condition signal to noise,thoughput 
polyS = gpd.read_file(baseDir + "gis/roda/area.shp")
poly = polyS.iloc[0]
poiC = {}
poly = polyS.iloc[0]
df = pd.DataFrame(columns=list(act.columns) + ["id_poly","x_n","y_n","inside"])
df.to_csv(baseDir + "log/roda/act_shift.csv.tar.gz",index=False,compression="gzip")

poly = polyS.iloc[59]

for j,poly in polyS.iterrows():
    print(poly['Name'])
    BBox = poly.geometry.bounds
    aroundN = 0.7
    delta = max(BBox[2] - BBox[0],BBox[3] - BBox[1])
    BBox = [BBox[0] - aroundN*delta,BBox[2] + aroundN*delta,BBox[1] - aroundN*delta,BBox[3] + aroundN*delta]
    pact = act[(act['x'] > BBox[0]) & (act['x'] < BBox[1])]
    pact = pact[(pact['y'] > BBox[2]) & (pact['y'] < BBox[3])]
    pgact = gact[(gact['X'] > BBox[0]) & (gact['X'] < BBox[1])]
    pgact = pgact[(pgact['Y'] > BBox[2]) & (pgact['Y'] < BBox[3])]
    print(pgact.shape,pact.shape)
    if(pact.shape[0] == 0):
        print("empty selection, skipping")
        continue
    
    plog("-------------------------download-network----------------------------")
    coll = client["tdg_infra"]["segments_col"]
    neiN = coll.find({'loc':{'$geoIntersects':{'$geometry':{
        "type":"Polygon"
        ,"coordinates":[
            [ [BBox[0],BBox[3]],[BBox[1],BBox[3]],[BBox[1],BBox[2]],[BBox[0],BBox[2]],[BBox[0],BBox[3]] ]
        ]}}}})
    pointL, lineL, lineS = [],[],[]
    for neii in neiN:
        lineL.append({"src":neii['src'],"trg":neii['trg'],"speed":neii['maxspeed'],"highway":neii['highway']})
        lineS.append(sh.geometry.LineString([(neii['loc']['coordinates'][0][0],neii['loc']['coordinates'][0][1]),(neii['loc']['coordinates'][1][0],neii['loc']['coordinates'][1][1])]))

    lineL = pd.DataFrame(lineL)
    colorL = ["firebrick","sienna","olivedrab","crimson","steelblue","tomato","palegoldenrod","darkgreen","limegreen","navy","darkcyan","darkorange","brown","lightcoral","blue","red","green","yellow","purple","black"]
    colorI, _ = pd.factorize(lineL['highway'])
    lineL = gpd.GeoDataFrame(lineL)
    nx, ny = (100, 100)
    lineL.loc[:,"color"] = [colorL[int(i)] for i in colorI]
    lineL.loc[:,"weight"] = lineL['speed']/max(lineL['speed'])*(BBox[3]-BBox[2])/nx
    lineL.geometry = lineS
    lineL.loc[:,"geometry"] = [x.buffer(y,resolution=2) for x,y in zip(lineL['geometry'],lineL['weight'])]
    
    xx, yy = np.mgrid[BBox[0]:BBox[1]:100j, BBox[2]:BBox[3]:100j]
    gridP = np.vstack([xx.ravel(), yy.ravel(),np.zeros(nx*ny)])
    routeN = cascaded_union(lineL.geometry)
    cilacP = cascaded_union(pgact.geometry)
    plog("-------------------------preparing-masks----------------------------")
    networkIn = c_f.isInside(gridP[0],gridP[1],routeN)
    cellIn = c_f.isInside(gridP[0],gridP[1],cilacP)
    plog("-------------------------selecting-kernel----------------------------")
    zBound  = c_f.boundaryMatrix(nx,ny)
    weightM = [0.,0.,0.1/nx,0.,1.0/nx]
    kernelM = c_f.kernelMatrix(weightM)
    nIter = 50

    gridP[2] = np.zeros(nx*ny)
    gridP[2][cellIn] = 1.
    z = gridP[2].reshape((nx,ny))
    zIn = np.array(cellIn).reshape((nx,ny))
    for i in range(nIter):
        z = cv2.filter2D(src=z,kernel=kernelM,ddepth=-1)##sg.convolve(z,kernelM)
        z[zBound] = 0.

    vact = pact.copy()
    vact.loc[:,"id_poly"] = poly['id']
    vact.dropna(inplace=True)
    x_v, y_v = c_f.moveGrad(vact['x'],vact['y'],z,nx,ny,BBox,dt=0.1)

    gridP[2] = np.zeros(nx*ny)
    gridP[2][networkIn] = 1.
    z = gridP[2].reshape((nx,ny))
    zIn = np.array(networkIn).reshape((nx,ny))
    for i in range(nIter):
        z = cv2.filter2D(src=z,kernel=kernelM,ddepth=-1)##sg.convolve(z,kernelM)
        z[zIn] = 1.
        z[zBound] = 0.

    x_v1, y_v1 = c_f.moveGrad(vact['x'],vact['y'],z,nx,ny,BBox,dt=0.1)
    x_v = x_v + x_v1
    y_y = y_v + y_v1

    vact.loc[:,"x_n"] = vact['x'] + x_v
    vact.loc[:,"y_n"] = vact['y'] + y_v
    setL = [poly.geometry.contains(sh.geometry.Point(x,y)) for x,y in zip(vact['x_n'],vact['y_n'])]
    vact.loc[:,"inside"] = False
    vact.loc[setL,"inside"]  = True
    plog('-------------------------writing-out---------------------')
    print(sum(setL),len(setL))
    vact.to_csv(baseDir + "log/roda/act_shift.csv.gz",index=False,header=False,mode="a",compression="gzip")

vact.to_csv(baseDir + "gis/roda/act_shift.csv",index=False)
vact.loc[:,"sr"] = (vact['sx']+vact['sy'])*.5
vact.loc[:,"n"] = 1.
import importlib
importlib.reload(octree)
vact.loc[:,"geohash"] = vact[['x_n','y_n']].apply(lambda x: octree.encode(x[0],x[1],precision=15),axis=1)
def clampF(x):
    return pd.Series({"n":sum(x['n']),"sr":np.mean(x['sr'])/(sum(x['n'])-1)})
lact = vact.groupby('geohash').apply(clampF).reset_index()
for i in range(5):
    setL = lact['n'] < 30.
    lact.loc[:,"geohash2"] = lact['geohash']
    lact.loc[setL,"geohash"] = lact.loc[setL,'geohash2'].apply(lambda x: x[:-1])
    lact = lact.groupby('geohash').apply(clampF).reset_index()
    
lact.to_csv(baseDir + "raw/roda/geohash.csv",index=False)
l = lact['geohash'].apply(lambda x: octree.decode(x)).values

lact.loc[:,"x"]  = [x[0] for x in l]
lact.loc[:,"y"]  = [x[1] for x in l]
lact.loc[:,"dx"] = [x[2]*.9 for x in l]
lact.loc[:,"dy"] = [x[3]*.9 for x in l]
lact.loc[:,"geometry"] = lact.apply(lambda x: sh.geometry.Polygon(
    [(x['x']-x['dx'],x['y']-x['dy']),(x['x']-x['dx'],x['y']+x['dy']),
     (x['x']+x['dx'],x['y']+x['dy']),(x['x']+x['dx'],x['y']-x['dy'])]),axis=1)
lact.loc[:,"dens"] = lact['n']/(lact['dx']*lact['dx'])
lact = gpd.GeoDataFrame(lact)
lact.sort_values(['dx','geohash'],ascending=False,inplace=True)
lact.to_file(baseDir + "gis/roda/geohash.geojson")

##bse
# coll = client["tdg_infra"]["infrastructure"]
# cellL = []
# for g in np.array(motG['geometry'][0]):
#     c = g.exterior.coords.xy
#     c1 = [[x,y] for x,y in zip(c[0],c[1])]
#     neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Polygon",'coordinates':[c1]}}}})


# act.to_csv(baseDir + "log/roda/act_roda_11_shift.csv.tar.gz",compression="gzip")


if False:
    importlib.reload(c_f)
    vact.to_csv(baseDir + "gis/roda/gradient.csv",index=False)
    poly1 = gpd.GeoDataFrame(poly)
    poly1.geometry = poly1['geometry']

    poly1.plot()
    plt.scatter(vact['x'],vact['y'],color="green",alpha=.01)
    plt.scatter(vact['x_n'],vact['y_n'],color="yellow",alpha=.01)
    plt.show()
        
    lineL.plot(color=lineL['color'])
    plt.quiver(xx,yy,vx,vy,z)
    #plt.plot(cent['x'],cent['y'],'o',color="r")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.matshow(z)
    plt.show()

    ax = c_f.plotDens(vact['x'],vact['y'],BBox)
    ax.plot(cent['x'],cent['y'],'o',color="r")
    plt.show()

    

