#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import modin.pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from tzlocal import get_localzone
tz = get_localzone()
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import geopandas as gpd
def plog(text):
    print(text)

import pymongo
from scipy.spatial import cKDTree
from scipy import inf
import shapely as sh
import geomadi.train_shapeLib as shl
import geomadi.geo_octree as octree
from shapely.ops import split, snap
from shapely import geometry, ops

metr = {"metrics": {"gradMeter":111122.19769899677  ,"deCenter":[10.28826401,51.13341344]  ,"deBBox":[5.866,47.2704,15.0377,55.0574] }}

plog('starting x boost junct collider alpha^2')

junG = gpd.GeoDataFrame.from_file(baseDir + "gis/motorway/de_junct.shp")
nodJ = gpd.GeoDataFrame.from_file(baseDir + "gis/motorway/motorway_link_nodes.shp")
motG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/motorway.shp")
if False:
    plog("---------------junction - motorway - index-------------")
    motG.loc[:,"x"] = [x.geometry.xy[0][0] for i,x in motG.iterrows()]
    motG.loc[:,"y"] = [x.geometry.xy[1][0] for i,x in motG.iterrows()]
    junG.loc[:,"id_motorway"] = "xx"
    for i,c in junG.iterrows():
        disk = np.sqrt(((motG['x']-c.geometry.x)**2 + (motG['y']-c.geometry.y)**2))
        disk = disk.sort_values()
        junG.loc[i,"id_motorway"] = motG.loc[disk.index[0]]['ref']
        
    setL = junG['ref'] == ""
    for i,c in junG[setL].iterrows():
        disk = np.sqrt( ((junG['x']-c.geometry.x)**2 + (junG['y']-c.geometry.y)**2) )
        disk = disk.sort_values()
        junG.loc[i,"ref"] = junG.loc[disk.index[0]]['ref']
    # junG.loc[:,"id_mot"] = junG.apply(lambda x: "%s" % (x['id_motorway'] if x['id_motorway'] != "" else ,x['TMC:cid_58:tabcd_1:TypeName:loc']),axis=1)
    # junG.loc[:,"id_jun"] = junG.apply(lambda x: "%s-%s" % (x['id_motorway'],x['id_mot']),axis=1)
    junG.loc[:,"id_jun"] = junG.apply(lambda x: "%s-%s" % (x['id_motorway'],x['ref']),axis=1)
    junG.loc[:,"x"] = [x.geometry.xy[0][0] for i,x in junG.iterrows()]
    junG.loc[:,"y"] = [x.geometry.xy[1][0] for i,x in junG.iterrows()]
    for i,c in nodJ.iterrows():
        disk = np.sqrt( ((junG['x']-c.geometry.x)**2 + (junG['y']-c.geometry.y)**2) )
        disk = disk.sort_values()
        nodJ.loc[i,"id_jun"] = junG.loc[disk.index[0]]['id_jun']

    junG.to_file(baseDir + "gis/motorway/de_junct.shp")
    nodJ.to_file(baseDir + "gis/motorway/motorway_link_nodes.shp")
    
if True:
    junG = gpd.GeoDataFrame.from_file(baseDir + "gis/motorway/de_junct.shp")
    junG.loc[:,"x"] = [x.geometry.xy[0][0] for i,x in junG.iterrows()]
    junG.loc[:,"y"] = [x.geometry.xy[1][0] for i,x in junG.iterrows()]    
    def clampF(x):
        return pd.Series({"x_cross":np.mean(x['x']),'y_cross':np.mean(x['y'])})
    junT1 = junG.groupby("id_jun").apply(clampF).reset_index()
    junT = gpd.GeoDataFrame.from_file(baseDir + "gis/motorway/de_junct_unique.shp")
    junT

    junT.to_file(baseDir + "gis/motorway/de_junct_unique.shp")

if False:
    nodB = gpd.GeoDataFrame.from_file(baseDir + "gis/destatis/junct_bundesland.shp")
    nodc = nodB[['id_jun','GEN']].groupby('GEN').agg(len).reset_index()

if False:
    junT = pd.read_csv(baseDir + "gis/motorway/de_junct_unique.csv")
    nodJ = gpd.GeoDataFrame.from_file(baseDir + "gis/motorway/motorway_link_nodes.shp")
    nodJ.columns = ['id_jun', 'y', 'node_id', 'x', 'geometry']
    nodJ = pd.merge(nodJ,junT,left_on=["id_jun"],right_on=["id_jun"],how="left")
    nodJ.to_file(baseDir + "gis/motorway/motorway_link_nodes.shp")
    
if False:
    motL = motG.geometry.unary_union
    if True:
        nodS = nodJ.copy()
    else:
        nodS = nodJ.loc[nodJ['id_jun'] == "A 4-53"]
    print('--------------------projection-on-the-motorway----------------------')
    neip = nodS.apply(lambda x: motL.interpolate(motL.project(x['geometry'])),axis=1)
    nodS.loc[:,"x_mot"] = [x.xy[0][0] for x in neip]
    nodS.loc[:,"y_mot"] = [x.xy[1][0] for x in neip]
    nodS.loc[:,"m_dist"] = nodS.apply(lambda x: (x['x']-x['x_mot'])**2 + (x['y']-x['y_mot'])**2,axis=1)
    def chirality(x1,y1,x2,y2,xo,yo):
        vp = [x1 - xo,y1 - yo]
        vc = [x2 - xo,y2 - yo]
        crossP = vp[0]*vc[1] - vc[0]*vp[1]
        return 1*(crossP > 0.)
    dec = metr['metrics']['deCenter']
    nodS.loc[:,"chi"] = nodS.apply(lambda x: chirality(x['x'],x['y'],x['x_mot'],x['y_mot'],dec[0],dec[1]),axis=1)
    nodS.loc[:,"dir"] = nodS.apply(lambda x: chirality(x['x'],x['y'],x['x_mot'],x['y_mot'],x['x_cross'],x['y_cross']),axis=1)
    
    nodS.loc[:,"jun_first"] = 0
    nodS.loc[:,"direction"] = "exit"
    nodS.loc[nodS['dir'] > 0,"direction"] = "entry"

    nodS.to_file(baseDir + "gis/motorway/motorway_link_nodes.shp")
    nodS.to_csv(baseDir + "raw/motorway/motorway_link_nodes.csv.gz",compression="gzip",index=False)

if False:
    plog('-------------prepare-log-file----------------')
#    junT = gpd.read_file(baseDir + 'gis/motorway/de_junct_unique.shp')
    junT = gpd.read_file(baseDir + 'gis/motorway/motorway_link_nodes.shp')
    junT = junT[junT['jun_first']==2]
    inLoc = []
    for i,g in junT.groupby("node_id"):
        c = g.iloc[0]
        name = c['id_jun'] + "_" + c["direction"] + "_" + str(c['chi'])
        inLoc.append({"location_id":name,"node_list":[int(float(i))]})

    job = json.load(open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_via.json"))
    job["odm_via_conf"]["input_locations"] = inLoc
    json.dump(job,open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_via_thuering.json","w"),sort_keys=True,indent=4)
    junT.to_csv(baseDir + "raw/motorway/nodes_job.csv",index=False)
    
    
    #    junT = junT.loc[junT['GEN'].isin(['Thüringen'])]
    JunL = np.unique(junT['id_jun'])
    nodeL = pd.read_csv(baseDir + "raw/motorway/node_link_add.csv.gz",compression="gzip")
    nodeL = nodeL.loc[nodeL['id_jun'].isin(JunL)] 
    for i,g in nodeL.groupby("id_jun"):
        nodeL.loc[g.index,"location_id"] = g.apply(lambda x: str(x["id_jun"]) +"_"+ str(x["direction"])+"_"+ str(x["chi"]) , axis=1)
    

if False:
    plog('--------------------check-motorway-nodes-in-graph---------------------')
    job = json.load(open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_via_thuering.json"))
    nodV = pd.DataFrame(job['odm_via_conf']['input_locations'])
    nodV.loc[:,"node_id"] = nodV['node_list'].apply(lambda x: x[0])
    client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])        
    coll = client["celery_t_tdg_infra_18b09"]["nodes"]
    junctL = []
    nodeQ = coll.find({"node_id":{"$in":list(nodV['node_id'])}})
    for n in nodeQ:
        junctL.append({"x":n["loc"]["coordinates"][0],
                       "y":n["loc"]["coordinates"][1],
                       "node_id":n["node_id"]})
    junctL = pd.DataFrame(junctL)
    nodV = pd.merge(nodV,junctL,on="node_id",how="left")
    print(nodV[nodV['x'] != nodV['x']].shape)
    nodV.to_csv(baseDir + "raw/motorway/nodes_job.csv",index=False)
