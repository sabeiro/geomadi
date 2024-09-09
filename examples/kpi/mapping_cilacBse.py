##%pylab inline
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_lib as tlib
import geomadi.train_shapeLib as shl
import geomadi.train_filter as t_f
import custom.lib_custom as l_t
import geomadi.train_execute as t_e
import geomadi.geo_enrich as g_e
import importlib
from io import StringIO
from sklearn import decomposition
from sklearn.decomposition import FastICA, PCA
import pymongo
import osmnx as ox
import shapely as sh

def plog(text):
    print(text)

plog('-------------------load/def------------------------')
ops = {"isScore":True,"lowcount":True,"p_sum":True,"isWeekday":True,"isType":False}
fSux = "20"
idField = "id_poi"
custD = "tank"
cred = json.load(open(baseDir + '/credenza/geomadi.json'))
metr = json.load(open(baseDir + '/raw/basics/metrics.json'))['metrics']
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])

#idField = "id_clust"
#custD = "tank"
##Spessart Süd, GB - 1276

if len(sys.argv) > 1:
    fSux = sys.argv[1]

poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
if custD == "tank":
    poi = poi[poi['use'] == 3]

poi.loc[:,idField] = poi[idField].astype(int)
mapL  = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL = dateL[dateL['use'] > 0]
mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
mist = mist.replace(float('nan'),0.)
mist = mist.groupby("id_poi").agg(sum).reset_index()
hL1 = mist.columns[[bool(re.search('-??T',x)) for x in mist.columns]]

if False:
    sact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac.csv.gz",compression="gzip")
else:
    sact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac/tank_cilac_t4_p11_d20.csv.gz",compression="gzip")
    
sact = pd.merge(mapL,sact,on="cilac",how="outer")
if custD == "tank":
    plog("------------------------match-chirality-----------------------")
    sact.loc[:,"chi_2"] = pd.merge(sact,poi,on="id_poi",how="left")['chirality']
    sact = sact[sact["chi"] == sact["chi_2"]]
sact = sact.replace(float('nan'),0.)
hL = sact.columns[[bool(re.search('-??T',x)) for x in sact.columns]]
del sact[hL[-1]]
hL = sact.columns[[bool(re.search('-??T',x)) for x in sact.columns]]
hL = sorted(list(set(hL) & set(hL1)))
hL1 = [x + "T" for x in set(dateL['day'])]
hL = sorted(list(set(hL) & set(hL1)))

if custD == "mc":
    poi.loc[:,"type1"] = poi[['type','subtype']].apply(lambda x: "%s-%s" % (x[0],x[1]),axis=1)
    mist = pd.merge(mist,poi[[idField,"type1"]],on=idField,how="left")
    gist = mist.groupby("type1").agg(np.mean)
    sact = pd.merge(sact,poi[[idField,"type1"]],on=idField,how="left")

if False:
    plog('-----------------------weekday-correction--------------------')
    
import importlib
importlib.reload(t_e)
importlib.reload(g_e)

if ops["isScore"]:
    act = t_e.joinSource(sact[[idField] + hL],mist[[idField] + hL],how="inner",idField=idField)
    scorM1 = t_e.scorPerf(act,step="etl",idField=idField)

p = poi.head(1)#[poi[idField] == 1061]
geoL = []
netL = []
colE = []
for k,p in poi.iterrows():
    print(p[idField])
    sumL = pd.DataFrame({"cilac":g['cilac'],"count":g[hL].sum(axis=1)/len(hL)})
    dx, dy = 0.012, 0.006
    xc, yc = p['x'], p['y']
    collE = client["tdg_infra_internal"]["segments_col"]
    netD = g_e.localNetwork(xc,yc,dx,dy,collE)
    netJ = [x[2] for x in netD.edges(data=True)]
    geoN = gpd.GeoDataFrame(netJ)
    collE = client["tdg_infra"]["infrastructure"]
    geoV = g_e.cellPolygon(g['cilac'].values,collE)
    geoD = g_e.localPolygon(xc,yc,dx,dy,collE,geo_idx="geom")
    geoD.loc[:,"cilac"] = geoD.apply(lambda x: "%d-%d" % (x['cell_ci'],x['cell_lac']),axis=1 )
    geoD.loc[:,"type"]  = geoD.apply(lambda x: "%s-%s" % (x['broadcast_method'],x['frequency']),axis=1 )
    geoD = pd.merge(geoD,sumL,on="cilac",how="left")
    geoD.loc[:,idField] = p[idField]
    geoL.append(geoD)
    netL.append(geoN)

geoL = pd.concat(geoL)
netL = pd.concat(netL)

if False:
    tL = ["cilac","type",'broadcast_method','cell_ci','cell_lac','cell_node_id','estimated_radius','frequency','geometry','id','state','zip_code']
    geoL.loc[:,tL].to_file(baseDir + "gis/"+custD+"/coverage.shp")
    netL.to_file(baseDir + "gis/"+custD+"/network.shp")
    p.to_csv(baseDir + "gis/"+custD+"/poi.csv",index=False)
    xL = [x[0] for x in geoL['centroid']]
    yL = [x[1] for x in geoL['centroid']]
    g = pd.DataFrame({"cilac":geoL['cilac'],"type":geoL['type'],"x_cell":xL,"y_cell":yL,"radius":geoL['estimated_radius'],idField:geoL[idField]})
    g.to_csv(baseDir + "gis/"+custD+"/centroid.csv",index=False)

if False:
    plog('----------------displacement-deviation--------------')
    from descartes import PolygonPatch
    xd, yd = dy*.5, dy*.5
    xySum = []
    xG = np.linspace(-xd*.5,xd*.5,10)
    yG = np.linspace(-yd*.5,yd*.5,10)
    for i in xG:
        x = xc + i
        for j in yG:
            y = yc + j
            poly = sh.geometry.Polygon(((x-xd,y-yd),(x-xd,y+yd),(x+xd,y+yd),(x+xd,y-yd)))
            fract = 0
            for k,g in geoD.iterrows():
                if poly.intersects(g['geometry']):
                    fract += poly.intersection(g['geometry']).area/g['geometry'].area*g['count']/poly.area
            xySum.append({"x":i*metr['gradMeter'],"y":j*metr['gradMeter'],"z":fract})
    xySum = pd.DataFrame(xySum)
    xG = np.reshape(xySum['x'].values,(10,10))
    yG = np.reshape(xySum['y'].values,(10,10))
    zG = np.reshape(xySum['z'].values,(10,10))
    zG = 1. - zG / zG.mean()
    poly = sh.geometry.Polygon(((xc-xd,yc-yd),(xc-xd,yc+yd),(xc+xd,yc+yd),(xc+xd,yc-yd)))
    xV, yV, zV = [x for x in poly.exterior.xy[0]], [x for x in poly.exterior.xy[1]], np.zeros((len(poly.exterior.xy[0])))
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    #ax.plot_surface(xySum['x'],xySum['y'],xySum['z'], label='parametric curve')
    surf = ax.plot_surface(xG,yG,zG,rstride=10,cstride=10,alpha=0.3)
    surf = ax.plot_surface(xG,yG,zG,cmap=cm.coolwarm,linewidth=2,antialiased=False,alpha=.5)
    ax.contour(xG,yG,zG,zdir='z', offset=zG.min(), cmap=cm.coolwarm)
    ax.contour(xG,yG,zG,zdir='x', offset=xG.min(), cmap=cm.coolwarm)
    ax.contour(xG,yG,zG,zdir='y', offset=yG.min(), cmap=cm.coolwarm)
    ax.scatter(0,0,0,c="b",marker="o")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if False:
    plog('------------------area-difference------------------')
    xySum = []
    xG = yG = np.linspace(0,dx,100)
    for i in xG:
        xd = yd = i
        poly = sh.geometry.Polygon(((xc-xd,yc-yd),(xc-xd,yc+yd),(xc+xd,yc+yd),(xc+xd,yc-yd)))
        fract = 0
        for k,g in geoD.iterrows():
            if poly.intersects(g['geometry']):
                fract += poly.intersection(g['geometry']).area/g['geometry'].area*g['count']/poly.area
        xySum.append({"x":i*metr['gradMeter'],"y":j*metr['gradMeter'],"z":fract})
    xySum = pd.DataFrame(xySum)
    zG = 1. - xySum['z'] / xySum['z'].mean()
    fig, ax = plt.subplots(1,1)
    ax.plot(xG[1:]*metr['gradMeter'],zG[1:],label="area stability")
    ax.set_xlabel("square edge")
    ax.set_ylabel("count variation")
    plt.legend()
    plt.show()
    
if False:
    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_title("n cell %d" % (geoD.shape[0]))
    geoD.plot(column="broadcast_method",ax=ax)
    geoN.plot(column="highway",ax=ax)
    ax.scatter(xc,yc,s=500,linewidths=40,color="red",marker="o")
    ax.plot(xV,yV)
    plt.show()


mapN = mapL.groupby(idField).agg(len)


