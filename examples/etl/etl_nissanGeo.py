#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geopandas as gpd


if False:
    print('---------------------nodes----------------------------------')
    poi = pd.read_csv(baseDir + "raw/nissan/nodeExits.csv")
    poi = gpd.GeoDataFrame(poi)
    st = poi['wkt_geom'].apply(lambda x: re.sub('POINT\(','',x))
    st = st.apply(lambda x: re.sub('\)','',x))
    poi.loc[:,'x'] = st.apply(lambda x: float(x.split(" ")[0]))
    poi.loc[:,'y'] = st.apply(lambda x: float(x.split(" ")[1]))
    coll = client["tdg_infra"]["nodes"]
    coll = client["tdg_infra"]["segments_col"]
    neiDist = 500.
    nodeD = []
    for i,poii in poi.iterrows():
        poii = poi.loc[i]
        poi_coord = [x for x in poii.ix[['x','y']]]
        neiN = coll.find({'loc':{'$nearSphere':{'$geometry':{'type':"Point",'coordinates':poi_coord},'$minDistance':0,'$maxDistance':neiDist}}})
        nodeId = []
        for neii in neiN:
            if not neii['highway'] == "motorway_link":
                continue
            nodeD.append({"highway":neii["highway"],"src":neii["src"],"trg":neii["trg"],"x":neii["loc"]['coordinates'][0][0],"y":neii["loc"]['coordinates'][0][1],"grp":poii['ref'],"id_poi":poii['@id']})

    nodeD = pd.DataFrame(nodeD)
    nodeD.loc[:,"id_junct"] = nodeD[['src','trg']].apply(lambda x: str(x[0]) + "-" + str(x[1]),axis=1)
    nodeD = nodeD.groupby('id_junct').first().reset_index()
    nodeD.to_csv(baseDir + "raw/nissan/nodeList.csv",index=False)


    
coll = client["tdg_grid"]["grid_250"]
tileL = []
for i in range(nodeL.shape[0]):
    poii = nodeL.loc[i]
    #poii = poi.loc[i]    
    poi_coord = [x for x in poii.ix[['x','y']]]
    neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Point",'coordinates':poi_coord}}}})
    for neii in neiN:
        tileL.append(neii)
    
tileL = pd.DataFrame(tileL)
tileL = tileL.groupby(['tile_id']).head(1)
#tileL.to_csv(baseDir + "raw/tank_tiles.csv")
# use tdg;
#CAPTURE 'capture.csv';
tileU = np.unique(tileL['tile_id'])
query = "select * from tile_direction_daily_hours_sum_20170703 where tile_id in ("
for i in tileU:
    query += "" + str(i) + ","
query = query[:-1] + ");"

with open(baseDir + "raw/nodeList.json","w") as fo:
    fo.write(json.dumps({"input_locations":nodeD},separators=(',',':')))

countL = pd.read_csv(baseDir + "raw/dir_count3.csv")
countL = pd.concat([countL,pd.read_csv(baseDir + "raw/dir_count4.csv")],axis=0)
countL = pd.concat([countL,pd.read_csv(baseDir + "raw/dir_count5.csv")],axis=0)
countL = pd.concat([countL,pd.read_csv(baseDir + "raw/dir_count6.csv")],axis=0)
countL = pd.merge(countL,tileL,left_on="tile_id",right_on="tile_id",how="left")
featL = []
for i in range(countL.shape[0]):
    cou = countL.loc[i]
    couD = cou['geom']['coordinates']
    del cou['geom']
    del cou['_id']
    featL.append({"type":"Feature","geometry":{"type":"Polygon",'coordinates':couD},"properties":dict(cou)})

#featL = featL[0:4]
with open(baseDir + "gis/nissan/count_dir.geojson","w") as fo:
    fo.write('{"type":"FeatureCollection","features":'+pd.Series(featL).to_json(orient='values')+'}')




zipN = zipN.groupby("nearest_graph_node").head(1)
convT.index = pd.merge(convT,zipN,left_on="origin_id",right_on="nearest_graph_node",how="left")['PLZ']
del convT['origin_id']
conTmp = pd.DataFrame({"node":[int(x) for x in convT.columns]})
conTmp.loc[:,"ciccia"] = 1
zipN.loc[:,"nearest_graph_node"] = zipN["nearest_graph_node"].astype(int)
convT.columns = pd.merge(conTmp,zipN,left_on="node",right_on="nearest_graph_node",how="left")['PLZ']
convT.to_csv(baseDir + "log/nissan/zip2zip_motor.csv.tar.gz",compression="gzip")

    
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
coll = client["tdg_infra"]["nodes"]
zipN = pd.read_csv(baseDir + "log/nissan/zip_node.csv")
queryL = []
for i,p in zipN.iterrows():
    queryL.append({"node_id":int(p['nearest_graph_node'])})

bse = coll.find({"$or":queryL})
bL = []
for b in bse:
    bL.append(b)

bL = pd.DataFrame(bL)

zip5 = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/zip5.shp")
zip5.loc[:,"centroid"] = zip5.geometry.apply(lambda x: x.centroid)
coll = client["tdg_infra"]["segments_col"]
neiDist = 700.
nodeL = []
for i,p in enumerate(zip5['centroid']):
    poiL = [p.xy[0][0],p.xy[1][0]]
    neiN = coll.find({'loc':{'$near':{'$geometry':{'type':'Point','coordinates': poiL},'$maxDistance':neiDist,'$minDistance':0}}})
    nodeId = []
    for neii in neiN:
        if neii['highway'] == 'motorway':
            continue
        nodeL.append({'src':neii['src'],"maxspeed":neii['maxspeed'],'street':neii['highway']
                ,"x_node":neii['loc']['coordinates'][0][0],"y_node":neii['loc']['coordinates'][0][1]
                ,"zip":zip5.loc[i]["PLZ"]
                ,"x_zip":poiL[0],"y_zip":poiL[1]
        })
        break

nodeL = pd.DataFrame(nodeL)
nodeL = nodeL.groupby("zip").head(1).reset_index()
print(zip5.shape)
print(nodeL.shape)
nodeL.to_csv(baseDir + "gis/graph/zip2node.csv",index=False)

zipN = pd.read_csv("zip_node.csv")
zipN = zipN.groupby("nearest_graph_node").head(1)
zipSel = pd.read_csv("zip_sel.csv")
zipSel.dropna(inplace=True)
zipN.loc[:,"nearest_graph_node"] = zipN['nearest_graph_node'].apply(lambda x: int(x))
zipSel.loc[:,"PLZ"] = zipN['PLZ'].apply(lambda x: int(x))
zipSel = pd.merge(zipSel,zipN,left_on="PLZ",right_on="PLZ",how="left")
node2 = pd.read_csv("node2node.csv.tar.gz",compression="gzip")
cL = [x for x in node2.columns]
cL[0] = "origin_id"
node2.columns = cL
node2.dropna(inplace=True)
node2['origin_id'] = node2['origin_id'].astype(int)
zipSel['nearest_graph_node'] = zipSel['nearest_graph_node'].astype(int)
node2 = node2[node2['zip2zip.csv'].isin(zipSel['nearest_graph_node'].values)]
node2['destination_id'] = node2['destination_id'].astype(int)
node2.loc[:,"origin"] = pd.merge(node2,zipSel,left_on="origin_id",right_on="nearest_graph_node",how="left")['PLZ']
node2.loc[:,"destination"] = pd.merge(node2,zipN,left_on="destination_id",right_on="nearest_graph_node",how="left")['PLZ']
node2.to_csv('zip2zip_nissan.csv',index=False)

