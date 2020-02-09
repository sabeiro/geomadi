# coding: utf-8
#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
from math import sin, cos, sqrt, atan2, radians
import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import pymongo
from scipy.spatial import cKDTree
from scipy import inf
import shapely as sh
from shapely.ops import split, snap
from shapely import geometry, ops
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from shapely.geometry import Point
import time

with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
metr = {"metrics": {"gradMeter":111122.19769899677  ,"deCenter":[10.28826401,51.13341344]  ,"deBBox":[5.866,47.2704,15.0377,55.0574] }}
motG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/motorway.shp")
motL = motG.geometry.unary_union
print('starting x boost junct collider alpha^2')
nodJ = gpd.read_file(baseDir + "gis/nissan/motorway_link_nodes.shp")
junG = gpd.read_file(baseDir + "gis/nissan/de_junct.shp")
nodJ.loc[:,"node_id"] = nodJ['node_id'].astype(str)
nodJ.head()
if False:
    nodS = nodJ.loc[nodJ['id_jun'] == "A 71-24"]
    nodS = nodJ.loc[nodJ['id_jun'] == "A 4-53"]
    nodS = nodJ.loc[nodJ['id_jun'] == "A 4-39"]
else :
    nodS = nodJ.copy()
print('-----------------calculating-chirality-and-angle--------------------')
def chirality(x1,y1,x2,y2,xo,yo):
    vp = [x1 - xo,y1 - yo]
    vc = [x2 - xo,y2 - yo]
    crossP = vp[0]*vc[1] - vc[0]*vp[1]
    return 1*(crossP > 0.)

nodS.loc[:,"jun_first"] = 0
nodS.loc[:,"id_zone"] = -1
nodS.loc[:,"direction"] = "exit"
nodS.loc[nodS['dir'] > 0,"direction"] = "entry"
disTh = 6.5e-08#1.5e-08
max_d = 700./metr['metrics']['gradMeter']
setL = (nodS['m_dist'] > disTh) & (nodS['m_dist'] < disTh*10.) # removing
perfL = []
for i,g in nodS[setL].groupby("id_jun"):
    print("processed %s" % i)
    start_time = time.clock()
    for i1,g1 in g.groupby("chi"):
        if g1.shape[0] < 6:
            continue
        Z = linkage(g1[['x','y']], 'ward')
        zoneL = fcluster(Z,max_d,criterion='distance')
        nodS.loc[g1.index,'id_zone'] = zoneL
        g1.loc[:,"id_zone"] = zoneL
        for i2,g2 in g1.groupby("id_zone"):
            xyc = Point(np.mean(g2['x']),np.mean(g2['y']))
            neip = motL.interpolate(motL.project(xyc))
            nodS.loc[g2.index,"x_cross"] = neip.x
            nodS.loc[g2.index,"y_cross"] = neip.y
            for j in ["x_cross","y_cross"]:
                g2.loc[:,j] = nodS.loc[g2.index,j]
            nodS.loc[g2.index,"dir"] = g2.apply(lambda x: chirality(x['x'],x['y'],x['x_mot'],x['y_mot'],x['x_cross'],x['y_cross']),axis=1)
            g2.loc[:,"dir"] = nodS.loc[g2.index,"dir"]
            for i3,g3 in g2.groupby("dir"):
                if g3.shape[0] == 0:
                    continue
                g3 = g3.sort_values(["m_dist"])
                nodS.loc[g3.index[0],"jun_first"] = 1
    perfL.append({"id_jun":i,"calc_time":time.clock()-start_time})

perfL = pd.DataFrame(perfL)
nodS.loc[:,"direction"] = "exit"
nodS.loc[nodS['dir'] > 0,"direction"] = "entry"
junT = nodS[['id_jun','chi','dir','id_zone','x_cross','y_cross']].groupby(['id_jun','chi','dir','id_zone']).agg(np.mean).reset_index()
if False:
    nodS.loc[:,"jun_first_manual"] = nodS['jun_first']
    
if nodS.shape[0] == nodJ.shape[0]:
    print('------------writing-up---------------')
    nodS.to_file(baseDir + "gis/nissan/motorway_link_nodes.shp")
    junT.loc[:,"manual_check"] = 0
    junT = pd.read_csv(baseDir + "gis/nissan/de_junct_unique.csv")
    
if False:
    print('-------------plotting-with-folium---------------')
    mapp = folium.Map(location=[list(nodS["y"])[1],list(nodS["x"])[1]],zoom_start=18)
    nodeLay = folium.map.FeatureGroup(name="Nodes", overlay=True, control=True)
    for i,g in nodS.iterrows():
        folium.features.Circle(location=[g['y'],g['x']],
                               popup = "%s %d d %.2E c %d e %d" % (g['direction'],g['dir'],g['m_dist'],g['chi'],g['dir']),
                               radius = 6 if g['jun_first'] > 0 else 2,
                               color = "blue" if g['chi'] > 0 else 'green',
                               fill_color = "white"
        ).add_to(nodeLay)
    projLay = folium.map.FeatureGroup(name="Center of Mass", overlay=True, control=True)
    for i,g in nodS.iterrows():
        folium.features.CircleMarker(location=[g['y_mot'],g['x_mot']],
                                     popup = (str(g['id_jun'])),
                                     radius = 4 if g['dir'] > 0 else 2,
                                     color = "blue" if g['chi'] > 0 else 'green',
                                     fill_color = "white"
        ).add_to(projLay)

    massLay = folium.map.FeatureGroup(name="Center of Mass", overlay=True, control=True)
    for i,g in junT.iterrows():
        folium.features.CircleMarker(location=[g['y_cross'],g['x_cross']],
                                     popup = (str(g['id_jun'])),
                                     radius = 2,
                                     color = "red",
                                     fill_color = "white"
        ).add_to(massLay)
    colorL = ["green","blue","red","black","yellow","purple"]
    for i,g in nodS.groupby(["chi","id_zone"]):
        if i[1] < 0:
            continue
        bounds = [[g['y'].min(), g['x'].min()], [g['y'].max(), g['x'].max()]]
        folium.features.RectangleMarker(
            bounds=bounds,popup='zone %d' % i[1]
            ,fill_color = None,color=colorL[i[0]]).add_to(mapp)
        
    folium.Marker([nodS.iloc[0]['y_cross'],nodS.iloc[0]['x_cross']],popup=nodS.iloc[0]['id_jun']).add_to(mapp)
    nodeLay.add_to(mapp)
    projLay.add_to(mapp)
    massLay.add_to(mapp)
    folium.LayerControl().add_to(mapp)
    mapp.save(baseDir + "www/folium_map.html")

if False:
    plog('--------------------nodes-to-add----------------')
    nodJ = gpd.read_file(baseDir + "gis/nissan/motorway_link_nodes.shp")
    nodE = nodJ[nodJ['jun_first'] == 1]
    del nodE['geometry']
    nodE.to_csv(baseDir + "raw/nissan/node_link_add.csv.gz",index=False,compression="gzip")


if False:
    plog('-----------------merge-changes------------------')
    nodJ = gpd.read_file(baseDir + "gis/nissan/motorway_link_nodes.shp")
    nodS = gpd.read_file(baseDir + "gis/nissan/tmp/motorway_link_nodes.shp")
    junT = gpd.read_file(baseDir + "gis/nissan/de_junct_unique.shp")
    junT = junT[junT['GEN'] == "Schleswig-Holstein"]
    idL = np.unique(junT['id_jun'])
    setL = nodS['id_jun'].isin(idL)
    nodJ.loc[setL,"direction"] = nodS.loc[setL,"direction"]
    nodJ.loc[setL,"jun_first"] = nodS.loc[setL,"jun_first"]
    nodJ.to_file(baseDir + "gis/nissan/motorway_link_nodes.shp")
