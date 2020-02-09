import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.geo_octree as g_o
import geomadi.geo_motion as g_m
import lernia.train_viz as t_v
import albio.series_stat as s_s
import shapely as sh
from shapely.geometry.polygon import Polygon
from ast import literal_eval
import geomadi.geo_enrich as g_e
import pymongo

cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))['metrics']

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])

BoundBox = [5.866,47.2704,15.0377,55.0574]
idField = "id"
import importlib
importlib.reload(g_o)
importlib.reload(g_m)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

import geomadi.geo_novtree as g_o
importlib.reload(g_o)
gO = g_o.novtree(BoundBox=[5.866,47.2704,15.0377,55.0574],padding=0.1)
print(gO.decode(gO.encode(14.989551,48.218262,17)))



ags8 = gpd.read_file(baseDir + "gis/geo/ags8.shp")
ags8.loc[:,'bbox'] = ags8['geometry'].apply(lambda x: x.convex_hull.exterior.bounds)
bbox = ags8.loc[ags8['GEN'] == 'Berlin','bbox'].iloc[0]
gBerlin = str(gO.boundingBox(bbox))

if False:
    print('-------------------format-merge-retail---------------------')
    projDir = "raw/gps/retail/traj/"
    dL = os.listdir(baseDir + projDir)
    d = dL[0]
    retD = pd.DataFrame()
    for d in dL:
        day = d.split("_")[1].split(".")[0][:8]
        traj = pd.read_csv(baseDir+projDir+d,converters={"tx":literal_eval,"ret_list":literal_eval})
        retL = pd.DataFrame([x[0] for x in traj['ret_list']])
        retL.loc[:,"id"] = traj['id']
        retL.loc[:,"day"] = day
        if d == dL[0]: retD = retL
        else: retD = pd.concat([retD,retL])
    retD.loc[:,"wday"] = retD['day'].apply(lambda x: datetime.datetime.strptime(str(x),"%Y%m%d").isocalendar()[2])
    retD.to_csv(baseDir + "raw/gps/retail/unique.csv.gz",compression="gzip",index=False)

    

if False:
    print('-----------------------subset-to-berlin---------------------')
    collE = client["tdg_infra_internal"]["segments_col"]
    xc, yc, dx, dy = gO.decode(gBerlin)
    netD = g_e.localNetwork(xc,yc,dx,dy,collE)
    netJ = [x[2] for x in netD.edges(data=True)]
    geoN = gpd.GeoDataFrame(netJ)
    geoN.to_file(baseDir + "gis/gps/net_berlin.shp")

projDir = baseDir + "raw/gps/traj/"
dL = os.listdir(projDir)
dL = [x for x in dL if bool(re.search("traj",x))]
d = dL[0]
d = 'traj_' + '2019042300' + ".csv.gz"
traj = pd.read_csv(baseDir+"raw/gps/traj/"+d,compression="gzip",converters={"tx":literal_eval})
traj = traj[traj['n'] > 30]
traj = traj[ traj['bbox'].apply(lambda x: str(x)[:len(gBerlin)] == gBerlin) ]
traj = traj.sort_values('n',ascending=False)

if False:
    print('----------------cut-trajectories------------------')
    importlib.reload(g_o)
    importlib.reload(g_m)
    gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
    gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])
    threshold = 0.0002
    clustD = 1.
    motV = []
    quiver = []
    for i,g in traj.iterrows():
        X = np.array(g['tx'])
        XV = gM.motion(X,isSmooth=True,steps=10)
        speed = gO.motion(XV,precision=15)
        mot  = gM.cluster(XV,threshold=threshold)
        motV.append(mot)
        quiver.append(speed)
    motL = pd.concat(motV)

    quiver = pd.concat(quiver)
    quiver.loc[:,"n"] = 1
    quiver.loc[:,"octree"] = quiver['octree'].apply(lambda x: int(str(x).split(".")[0]))
    quiL = g_o.densGroupAv(quiver,max_iter=5,threshold=30)
    poly = gO.geoDataframe(quiL)
    poly.to_file(baseDir + "gis/gps/quiver.shp")
    dweL = motV[motV['m'] < 0.5]
    motL = motV[motV['m'] > 0.5]

    t_v.boxplotOverlap(motL,dweL,['sx','sy','ss','sa'],lab1="motion",lab2="dwelling")

    dweL = dweL.replace(float('nan'),1e-10)
    dwe = gO.dwellingBox(dweL,max_iter=3,threshold=30)
    poly = gO.geoDataframe(dwe)
    poly.to_file(baseDir + "gis/gps/dwelling_location.shp")
    
    pointL = [sh.geometry.Point(x['x'],x['y']).buffer(x['sr']) for i,x in dweL.iterrows()]
    gdf = gpd.GeoDataFrame(dweL,geometry=pointL)
    gdf = gdf.sort_values("sr",ascending=False)
    gdf.to_file(baseDir + "gis/gps/dwelling.shp")
    
if False:
    print('---------------------save-to-shapefile---------------------')
    traj.loc[:,'geometry'] = traj['tx'].apply(lambda l: sh.geometry.LineString([sh.geometry.Point(x[1],x[2]) for x in l]) )
    traS = gpd.GeoDataFrame(traj,geometry='geometry')
    setL = [np.log10(x) > 5 for x in traS['bbox']]
    traS.loc[setL,['id','n','bbox','dt','geometry']].to_file(baseDir + "gis/gps/traj_short.shp",index=False)
    setL = [np.log10(x) < 5 for x in traS['bbox']]
    traS.loc[setL,['id','n','bbox','dt','geometry']].to_file(baseDir + "gis/gps/traj.shp",index=False)

if False:
    print('----------------visualize-speed-----------------------')
    t = traj.head(1).copy()
    importlib.reload(g_m)
    gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])
    fig, ax = plt.subplots(2,1)
    XV = gM.motion(np.array(t['tx'].iloc[0]),isSmooth=False,steps=10)
    ax[0].plot(XV[:,2],label="speed")
    ax[1].plot(XV[:,3],label="angle")
    XV = gM.motion(np.array(t['tx'].iloc[0]),isSmooth=True,steps=3)
    ax[0].plot(XV[:,2],label="speed run av 5")
    ax[1].plot(XV[:,3],label="angle run av 5")
    XV = gM.motion(np.array(t['tx'].iloc[0]),isSmooth=True,steps=5)
    ax[0].plot(XV[:,2],label="speed run av 10")
    ax[1].plot(XV[:,3],label="angle run av 10")
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
    mov = t['tx'].apply(lambda x: [gO.moving(x,y,threshold=sThre) for x,y in zip(x[1:],x[:-1])])
    t.loc[:,"speed2"] = t['tx'].apply(lambda x: [gO.moving(x,y,prec) for x,y in zip(x[1:],x[:-1])])
    t = t.iloc[0]
    ts =  [(x-min(t['t']))/3600. for x in t['t']]
    plt.title('speed profile')
    plt.plot(ts,t['speed'])
    plt.plot(ts,s_s.serRunAv(t['speed'],steps=3))
    plt.show()

if False:
    print('----------------------show-bounding-octree---------------------------')
    polyL = [sh.geometry.Polygon(gO.decodePoly(g3))]
    polyF = gpd.GeoDataFrame({"geometry":polyL,'n':['berlin'],'octree':[g3]})
    polyF.to_file(baseDir + "gis/gps/bbox_octree.shp")

if False:
    print('---------------------geojson-export-------------------------')
    dens = pd.read_csv(baseDir + "raw/gps/dens.csv.gz",compression="gzip",dtype={'octree':str})
    n = len(gBerlin)
    setL = dens['octree'].apply(lambda x: x[:n] == gBerlin)
    densD = dens[setL]
    poly = gO.geoDataframe(densD)
    poly.rename(columns={"n":"z"},inplace=True)
    with open(baseDir + "gis/gps/dens_berlin.geojson", 'w') as f: f.write(poly.to_json())

    dens = pd.read_csv(baseDir + "raw/gps/motion.csv.gz",compression="gzip",dtype={'octree':str})
    n = len(gBerlin)
    setL = dens['octree'].apply(lambda x: x[:n] == gBerlin)
    densD = dens[setL]
    poly = gO.geoDataframe(densD)
    poly.rename(columns={"n":"z"},inplace=True)
    with open(baseDir + "gis/gps/dens_berlin.geojson", 'w') as f: f.write(poly.to_json())

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

