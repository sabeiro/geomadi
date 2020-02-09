#%pylab inline
import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import geopandas as gpd
import shapely as sh
from datetime import datetime
import scipy.spatial as spatial
from collections import OrderedDict as odict
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import fiona
def plog(text):
    print(text)

import pymongo

with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)

with open(baseDir + '/raw/metrics.json') as f:
    metr = json.load(f)['metrics']
    
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
#client.database_names()

print('-----------------dir-count-tank-poi---------------------------------')

poi = pd.read_csv(baseDir + "raw/poi_tank.csv")
coll = client["tdg_grid"]["grid_250"]
tileL = []
for i in range(poi.shape[0]):
    poii = poi.loc[i]    
    poi_coord = [x for x in poii.ix[['x','y']]]
    neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Point",'coordinates':poi_coord}}}})
    for neii in neiN:
        del neii['_id']
        neii['id'] = poii['id_poi']
        neii['geometry'] = sh.geometry.shape(neii['geom'])
        del neii['geom']
        tileL.append(neii)

tile1 = tileL
tileL = pd.DataFrame(tile1)
tileL = gpd.GeoDataFrame(tile1)#,geometry="geom")

with open(baseDir + "gis/tank/tile_dir.geojson","w") as fo:
    fo.write(tileL.to_json())

tileU = np.unique(tileL['tile_id'])
query = "select * from tdg.tile_direction_daily_hours_sum_20170703 where tile_id in ("
for i in tileU:
    query += "" + str(i) + ","
query = query[:-1] + ");"

print('---------------------bse----------------------------------')

coll = client["tdg_infra"]["nodes"]
neiDist = 400.
nodeL = []
nodeD = []

fs = pd.read_csv(baseDir + "raw/visits_se.csv")
fs = fs.groupby(['Location ID']).agg(np.mean)
coll = client["telia_se_grid"]["grid_250"]
tileL = []
for i in range(fs.shape[0]):
    poii = fs.iloc[i]
    poi_coord = [poii['Longitude'],poii['Latitude']]
    neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Point",'coordinates':poi_coord}}}})
    for neii in neiN:
        tileL.append(neii)

tileL = pd.DataFrame(tileL)
tileL.index = fs.index
fs = pd.concat([fs,tileL],axis=1)
fs.to_csv(baseDir + "raw/visits_se_tile.csv")

print('---------------------bse----------------------------------')

coll = client["tdg_infra"]["infrastructure"]
poi = pd.read_csv(baseDir + "raw/tr_cilac_sel1.csv")
colL = list(poi.columns)
colL[0] = 'domcell'
poi.columns = colL
poi.loc[:,'ci']  = [re.sub("-.*","",x) for x in poi['domcell']]
poi.loc[:,'lac'] = [re.sub(".*-","",x) for x in poi['domcell']]
queryL = []
for i,p in poi.iterrows():
    queryL.append({"cell_ci":p['ci']})
    queryL.append({"cell_lac":p['lac']})

#bse = coll.find_one({"$or":queryL})
bse = coll.find({"$or":queryL})
bL = []
for b in bse:
    bL.append(b)

    neii['poi_id'] = poii['@id']
    neii['poi_coord'] = poi_coord
    nodeId.append(neii['node_id'])
    convL.append(neii)
    nodeL.append(neii['loc']['coordinates']+[poii['@id']])
nodeD.append({"location_id":poii['@id'],"node_list":nodeId})
    
print('------------geo-json----------')

if False:
    odm = pd.read_csv(os.environ['LAV_DIR']+"/log/ODM20170901.csv.tar.gz",compression='gzip',sep=',',quotechar='"',names=["count","orig","dest","h_orig","h_dest"],header=0)
    odm = odm.replace(np.nan,0)
    odm['orig'] = odm['orig'].apply(lambda x: int(x))
    odm['dest'] = odm['dest'].apply(lambda x: int(x))
    odm['count'] = odm['count'].apply(lambda x: int(x))
    tileL = odm['orig'].unique()

gradMeter = 111122.19769899677

centL = pd.read_csv(os.environ['LAV_DIR']+"/raw/centroids.csv.tar",compression='gzip',sep=',',index_col=0)
print(centL.head())
centL.to_csv(baseDir + "gis/centroid.csv")
with open(os.environ['LAV_DIR']+'gis/mvg/wider_munich.geojson') as f:
    data = json.load(f)

import matplotlib.path as mplPath

bbPath = mplPath.Path(data['features'][0]['geometry']['coordinates'][0])
inside = []
for i in range(centL.shape[0]):
    if bbPath.contains_point((centL.iloc[i]['X'],centL.iloc[i]['Y'])):
        inside.append(centL.iloc[i])

print('done')
pd.DataFrame(inside,columns=centL.columns).to_csv("out/munich_area.csv")


import scipy.spatial as spatial
point_tree = spatial.cKDTree(centL[['X','Y']])
centi = centL.iloc[0][['X','Y']]
neiDist = 3000./gradMeter
centNei = pd.DataFrame()
for i in range(poi.shape[0]):
    poii = poi.iloc[i][['x','y']]
    centI = point_tree.query_ball_point(poii,neiDist)
    ##dd, centI = point_tree.query(poii, k=5)
    if centL.iloc[centI].shape[0] == 0:
        print("empty " + str(i))
        continue
    neiI = centL.iloc[centI]
    neiI.loc[:,'poi'] = i
    neiI.loc[:,'x_poi'] = poii['x']
    neiI.loc[:,'y_poi'] = poii['y']
    centNei = centNei.append(neiI)

centNei.to_csv(os.environ['LAV_DIR'] + "raw/tr_cilac_sel2.csv")

print('-----------------odm------------------------')

infra_conn_dev = client["tdg_17d08"]["infrastructure"]
infra_conn_subway = client["subway_graph"]["munich_cilac_nodes"]
client.database_names()

gridSe = client["telia_se_grid"]['grid_250']
cur = gridSe.find()  
odm = pd.read_csv(os.environ['LAV_DIR']+"/log/ODM20170901.csv.tar.gz",compression='gzip',sep=',',quotechar='"',names=["count","orig","dest","h_orig","h_dest"],header=0)
odm = odm.replace(np.nan,0)
odm['orig'] = odm['orig'].apply(lambda x: int(x))
odm['dest'] = odm['dest'].apply(lambda x: int(x))
odm['count'] = odm['count'].apply(lambda x: int(x))
tileL = odm['orig'].unique()

with open(baseDir + 'gis/geo/TDG_cilac_to_MS.json') as f:
    cilac = json.load(f)

cilac = pd.DataFrame(cilac).transpose()
cilac.to_csv(baseDir + 'gis/geo/TDG_cilac_to_MS.csv')

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')



