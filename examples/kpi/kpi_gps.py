import os, sys, gzip, random, csv, json
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import geomadi.geo_octree as g_o
import geomadi.train_viz as t_v

import importlib
importlib.reload(g_o)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])

dens = pd.read_csv(baseDir + "raw/gps/dens.csv.gz",compression="gzip",dtype={'octree':str})

if False:
    print('-----------------threshold-speed-----------------')
    speed = pd.read_csv(baseDir + "raw/gps/speed.csv.gz",compression="gzip")
    t_v.plotHist(speed['speed'])
    plt.show()
    speed.boxplot(column="speed")
    plt.show()

if False:
    print('-----------------------analyze-shopping-centers-------------------')
    ret = gpd.read_file(baseDir + "gis/retail/retail.shp")
    for j,k in ret.iterrows():
        poly = k['geometry']
        g3 = gO.boundingBox(poly.exterior.bounds)
        setL = dens['octree'].apply(lambda x: x[:len(g3)] == g3)
        densP = dens[setL]
        count = 0.
        for i,g in densP.iterrows():
            p = sh.geometry.Polygon(gO.decodePoly(g['octree']))
            a = p.intersection(poly)
            share = a.area/p.area
            count += g['n']*share
            ret.loc[j,"n"] = [count]
            
    ret.to_file(baseDir + "gis/retail/retail.shp")
    ret[['Name','n']].to_csv(baseDir + "raw/gps/retail.csv",index=False)

if False:
    polyL = [sh.geometry.Polygon(gO.decodePoly(g3))]
    polyF = gpd.GeoDataFrame({"geometry":polyL,'n':ret['name'],'octree':[g3]})
    polyF.to_file(baseDir + "gis/gps/bbox_octree.shp")

if False:
    print('-----------------------analyze-frequencies-------------------')
    freq = pd.read_csv(baseDir + "raw/gps/freq.csv.gz",compression="gzip",index_col=0)
    yt = freq.values.ravel()
    st = freq.sum(axis=1)/30
    sp = freq.sum(axis=0)/30
    importlib.reload(t_v)
    fig, ax = plt.subplots(1,2)
    t_v.plotHist(yt,threshold=5.,nBin=10,ax=ax[0])
    t_v.plotHist(st,threshold=5.,nBin=10,ax=ax[1])
    ax[0].set_title('signal density daily')
    ax[1].set_title('signal density monthly avg')
    plt.show()

    plt.plot(sp)
    plt.show()

    barF = pd.DataFrame({"user":len(st),"> 10/day":sum(st>10.),"> 20/day":sum(st>20.),"> 30/day":sum(st>30.),"> 40/day":sum(st>40.)},index=[0])
    plt.bar(barF.columns,barF.values[0])
    plt.show()
    
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
