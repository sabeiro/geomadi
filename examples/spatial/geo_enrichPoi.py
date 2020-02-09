#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geopandas as gpd
import pymongo

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import cKDTree
from scipy import inf
import shapely as sh
import geomadi.train_shape as shl
import geomadi.geo_octree as octree
import geomadi.geo_enrich as g_e
import shapely as sh
import shapely.speedups
shapely.speedups.enable()

import importlib
importlib.reload(g_e)

colorL = ["firebrick","sienna","olivedrab","crimson","steelblue","tomato","palegoldenrod","darkgreen","limegreen","navy","darkcyan","darkorange","brown","lightcoral","blue","red","green","yellow","purple","black"]

cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))['metrics']

custD = "tank"
idField = "id_clust"
custD = "mc"
custD = "bast"
idField = "id_poi"

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")

if False:
    plog('----------------------classify-subtypes-------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"submask"] = shl.binMask(poi['type'])
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    plog('---------------------------add-pois---------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi = poi.groupby('id_poi').first().reset_index()
    poiR = pd.read_csv(baseDir + "raw/"+custD+"/poi_raw.csv")
    poiR = poiR[poiR['x'] == poiR['x']]
    poiN = poiR[~poiR['id_poi'].isin(poi['id_poi'])]
    poiN.to_csv(baseDir + "raw/"+custD+"/poi_new.csv",index=False)

if False:
    plog('-----------------enrich-id-------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    idL = pd.read_csv(baseDir + "raw/"+custD+"/poi_id_comp.csv")
    poi.loc[:,"id_tile"] = pd.merge(poi,idL,left_on="id_poi",right_on="Sitenummer",how="left")["Tile_ID"]
    poi.loc[:,"id_mtc"] = pd.merge(poi,idL,left_on="id_poi",right_on="Sitenummer",how="left")["id_MTC"]
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('-----------------enrich-id-------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    setL = poi['competitor'] == 1
    poi.loc[setL,idField] = poi.loc[setL,:].apply(lambda x: "%d-%s" % (x[idField],"c"),axis=1)
    seen = set(poi[idField])
    uniq = [x for x in poi[idField] if x not in seen and not seen.add(x)]
    import collections
    repeat = [item for item, count in collections.Counter(poi[idField]).items() if count > 1]
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    plog('---------------enrich-act-foot---------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    gact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_weighted.csv.gz",compression="gzip")
    hL = [x for x in gact.columns if bool(re.search("T",x))]
    poiS = pd.DataFrame({idField:gact[idField],"act":gact[hL].sum(axis=1)/len(hL)})
    dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount_d.csv.gz",compression="gzip")
    hL = [x for x in dirc.columns if bool(re.search("T",x))]
    poiD = pd.DataFrame({idField:dirc[idField],"foot":dirc[hL].sum(axis=1)/len(hL)})
    poi = pd.merge(poi,poiS,on=idField,how="left")
    poi = pd.merge(poi,poiD,on=idField,how="left")
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:    
    plog('----------------------------region------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"region"] = gen.addRegion(poi,baseDir + "gis/geo/bundesland.shp")
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    plog('---------------------spatial-resolution--------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,'id_zone'] = gen.addZone(poi,500./metr['gradMeter'])
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    print('---------------------find-neighboring-osm-nodes--------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    coll = client["tdg_infra_internal"]["segments_col"]
    neiDist = 400.
    nodeL = []
    for i,poii in poi.iterrows():
        poi_coord = [x for x in poii.ix[['x','y']]]
        neiN = coll.find({'loc':{'$nearSphere':{'$geometry':{'type':"Point",'coordinates':poi_coord},'$minDistance':0,'$maxDistance':neiDist}}}) 
        nodeId = []
        for neii in neiN:
            # if bool(re.search("motorway",neii['highway'])):
            #     continue
            nodeL.append({'id_poi':poii['id_poi'],'src':neii['src'],'trg':neii['trg'],"maxspeed":neii['maxspeed'],'street':neii['highway'],idField:poii[idField]
                          ,"x_poi":poii['x'],"y_poi":poii['y']
                          ,"x1":neii['loc']['coordinates'][0][0],"y1":neii['loc']['coordinates'][0][1]
                          ,"x2":neii['loc']['coordinates'][1][0],"y2":neii['loc']['coordinates'][1][1]
            })

    nodeL = pd.DataFrame(nodeL)
    nodeL.loc[:,'north'] = np.arctan2((nodeL['y2']-nodeL['y1']),(nodeL['x2']-nodeL['x1']))*180./np.pi
    nodeL.loc[:,'north'] = 90.-nodeL.loc[:,'north']
    nodeL1 = nodeL
    if False:
        plt.hist(nodeL['north'],bins=40)
        plt.xlabel("north angle")
        plt.ylabel("counts")
        plt.title("orientation of osm segments @motorway")
        plt.show()

    sorter = ['motorway','motorway_link','trunk','primary','secondary','secondary_link','tertiary','trunk_link','residential','unclassified','living_street']
    dist1 = np.sqrt((nodeL['x1']-nodeL['x_poi'])**2+(nodeL['y1']-nodeL['y_poi'])**2)
    dist2 = np.sqrt((nodeL['x2']-nodeL['x_poi'])**2+(nodeL['y2']-nodeL['y_poi'])**2)
    nodeL.loc[:,'dist'] = np.min([dist1,dist2],axis=0)
    nodeL.loc[:,"orth_dist"] = np.abs((nodeL['x2']-nodeL['x1'])*(nodeL['y1']-nodeL['y_poi'])-(nodeL['y2']-nodeL['y1'])*(nodeL['x1']-nodeL['x_poi']))
    nodeL.loc[:,"orth_dist"] = nodeL["orth_dist"]/(np.abs((nodeL['x2']-nodeL['x1'])) + np.abs((nodeL['y2']-nodeL['y1'])))
    #nodeL.loc[:,"dist"] = nodeL['orth_dist']*nodeL['dist']
    v1 = [nodeL['x1'] - metr['deCenter'][0],nodeL['y1'] - metr['deCenter'][1]]
    v2 = [nodeL['x2'] - nodeL['x1'],nodeL['y2'] - nodeL['y1']] 
    crossP = v1[0]*v2[1] - v2[0]*v1[1]
    nodeL.loc[:,'chirality'] = 1.*(crossP>0.)

    sorterIndex = dict(zip(sorter,range(len(sorter))))
    nodeL['rank'] = nodeL['street'].map(sorterIndex)
    nodeL.sort_values(['rank','orth_dist'],inplace=True)
    nodeJ = nodeL[nodeL['street']=="motorway_link"].groupby(idField).first().reset_index()
    nodeJ.loc[:,"jun_dist"] = np.sqrt((nodeL['x2']-nodeL['x_poi'])**2+(nodeL['y2']-nodeL['y_poi'])**2)
    nodeJ.sort_values(['rank','jun_dist'],inplace=True)
    nodeG = nodeL.groupby([idField,'chirality']).first().reset_index()
    # nodeG.sort_values(idField,inplace=True)
    nodeG.loc[:,"mot_dist"] = nodeG['orth_dist']
    nodeG.to_csv(baseDir + "gis/"+custD+"/nodeList.csv",index=False)
    nodeG = nodeG.groupby([idField,'chirality']).first().reset_index()
    nodeG.loc[:,"chirality"] = nodeG['chirality'].astype(int)
    poi.loc[:,"id_node"] = pd.merge(poi,nodeG[['id_poi','src','chirality']],on=[idField,'chirality'],how="left")["src"].values
    nodeJ.to_csv(baseDir + "gis/"+custD+"/nodeListJunct.csv",index=False)
    nodeG = nodeG.rename(columns={"street":"highway"})
    for i in poi.columns[poi.columns.isin(['north','highway','mot_dist',"maxspeed",'jun_dist'])]:
        del poi[i]
    poi = pd.merge(poi,nodeG[[idField,'north','highway','mot_dist',"maxspeed","chirality"]],how="left",on=[idField,'chirality'],suffixes=["","_y"])
    poi = pd.merge(poi,nodeJ[[idField,'jun_dist']],how="left",on=idField)
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    nodeL = nodeL.sort_values(['orth_dist'])
    nodeL = nodeL.sort_values([idField,'rank','orth_dist'],ascending=True)
    nodeL.to_csv(baseDir + "raw/"+custD+"/nei_node.csv.gz",compression="gzip",index=False)
    nodeP = nodeL.groupby([idField,"chirality"]).first().reset_index()
    nodeP.to_csv(baseDir + "gis/"+custD+"/nei_node.csv",index=False)
    #nodeL.to_csv(baseDir + "gis/"+custD+"/nodeList.csv",index=False)

if False:
    print('-------------check-via-node-position---------------')
    custD = "tank"
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")

    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])
    coll = client["tdg_infra_internal"]["nodes"]
    nodeL = coll.find({"node_id":{"$in":[int(x) for x in poi['id_node'].values]}})
    coord = [] 
    for n in nodeL:
        coord.append({"x":n["loc"]["coordinates"][0],
                      "y":n["loc"]["coordinates"][1],
                      "id_node":n["node_id"]})
    coorD = pd.DataFrame(coord)
    inLoc = poi[[idField,"id_node"]].merge(coorD,on="id_node",how="left")
    inLoc.to_csv(baseDir + "gis/"+custD+"/node_job.csv",index=False)
    
if False:
    plog('--------------------------------calculate-tangent-from-motorway------------------')
    from shapely.ops import split, snap
    from shapely import geometry, ops
    motG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/motorway.shp")
    motG = motG[motG['geometry'].apply(lambda x: x.is_valid).values]
    line = motG.geometry.unary_union
    for i,poii in poi.iterrows():
        p = geometry.Point(poi.loc[i][['x','y']])
        neip = line.interpolate(line.project(p))
        #snap(coords, line, tolerance)
        poi.loc[i,"x_mot"] = neip.x
        poi.loc[i,"y_mot"] = neip.y

    poi.loc[:,'angle'] = np.arctan2((poi['y']-poi['y_mot']),(poi['x']-poi['x_mot']))*180./np.pi
    poi.loc[:,'angle'] = - poi.loc[:,'angle']
    poi.loc[:,'tang']  = np.arctan2(metr['deCenter'][1]-poi['y'],metr['deCenter'][0]-poi['x'])*180./np.pi
    poi.loc[:,'tang']  = 90. - poi.loc[:,'tang']
    poi.loc[poi['tang']>180.,'tang'] -= 180.
    t = np.abs(poi['tang']-poi['angle'])
    t[t>180.] = 360.-t
    poi.loc[:,'chirality'] = 1*(t>90)
    for i,p in poi.iterrows():
        cSet = poi[(poi['id_poi'] == p['id_poi'])].index
#        cSet = poi[(poi['id_poi'] == p['id_poi']) & (poi['competitor'] == 0)].index
        if len(cSet) > 0:
            poi.loc[i,'chirality'] = poi.loc[cSet[0],'chirality']
            poi.loc[i,'angle'] = poi.loc[cSet[0],'angle']
    if False:
        poi.loc[:,idField] = poi[['id_zone','chirality']].astype(str).apply(lambda x: "-".join(x),axis=1)
        poi.loc[:,idField] = poi[['id_poi','id_zone','chirality']].astype(str).apply(lambda x: "-".join(x),axis=1)
        poi.loc[poi['competitor']==1,idField] = poi.loc[poi['competitor']==1,idField] + "-c"
    print(len(set(poi[idField])))
    del poi['x_mot']
    del poi['y_mot']
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    plog('-------------------isochrone-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    dx, dy = 0.4, 0.2
    collE = client["tdg_infra_internal"]["segments_col"]
    import importlib
    importlib.reload(gen)
    polyL = []
    for i,g in poi.iterrows():
        xc, yc = g['x'], g['y']
        G = gen.localNetwork(xc,yc,dx,dy,collE)
        poly = gen.isochrone(G,isPlot=True)
        polyL.append({"id_poi":g['id_poi'],"poly":poly})
    
if False:
    print('-------------------closest-density-----------------')
    #densG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/pop_dens_2km.shp")
    densG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/pop_dens.shp")
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    BBox = [0.05,0.05]
    P = [sh.geometry.Polygon([[x-BBox[0],y-BBox[1]],[x+BBox[0],y-BBox[1]],[x+BBox[0],y+BBox[1]],[x-BBox[0],y+BBox[1]]]) for x,y in zip(poi['x'],poi['y'])]
    poiG = gpd.GeoDataFrame(P,columns=["geometry"])
    for i,p in poiG.iterrows():
        pMask = densG.intersects(p['geometry'])
        dens = densG[pMask]
        if dens.shape[0] == 0:
            continue
        x1 = p.geometry.centroid.x
        y1 = p.geometry.centroid.y
        poi.loc[i,'pop_dens'] = gen.interp2D(dens,x1,y1,z_col="Einwohner")
        poi.loc[i,'women'] = gen.interp2D(dens,x1,y1,z_col="Frauen_A")
        poi.loc[i,'foreign'] = gen.interp2D(dens,x1,y1,z_col="Auslaender")
        poi.loc[i,'flat_dens'] = gen.interp2D(dens,x1,y1,z_col="Wohnfl_Whg")
        poi.loc[i,'land_use'] = gen.interp2D(dens,x1,y1,z_col="Leerstands")
        poi.loc[i,'elder'] = gen.interp2D(dens,x1,y1,z_col="ab65_A")
    for i in ['pop_dens','women','foreign','flat_dens','land_use','elder']:
        poi.loc[:,i] = poi[i].abs()
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('----------------spatial-degeneracy----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    degT = []
    for i,g in poi.iterrows():
        r = np.sqrt((poi['x'] - g['x'])**2 + (poi['y'] - g['y'])**2)
        deg = degeneracy(r)
        degT.append(deg)
    poi.loc[:,"degeneracy"] = degT
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('-------------------bast-counts-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    bast = pd.read_csv(baseDir + "raw/bast/poi.csv")
    bast = bast[bast['Str_Kl'] == 'A']
    poi.loc[:,'bast'] = 0
    for i,c in poi.iterrows():
        x_c, y_c = c['x'],c['y']
        disk = ((bast['x']-x_c)**2 + (bast['y']-y_c)**2)
        disk = disk.sort_values()
        disk = disk[~np.isnan(bast["DTV_Kfz_MobisSo_Q"])]
        disk = disk.head(1)
        poi.ix[i,'id_bast'] = bast.loc[disk.index]["id_poi"].values[0]
        poi.ix[i,'bast'] = bast.loc[disk.index]["DTV_Kfz_MobisSo_Q"].values[0]
        poi.ix[i,'bast_fr'] = bast.loc[disk.index]["bFr"].values[0]
        poi.ix[i,'bast_su'] = bast.loc[disk.index]["bSo"].values[0]
        # poi.ix[i,'bast_r1'] = bast.loc[disk.index]["DTV_Kfz_MobisSo_Ri1"].values[0]
        # poi.ix[i,'bast_r2'] = bast.loc[disk.index]["DTV_Kfz_MobisSo_Ri2"].values[0]
        # poi.ix[i,'x_bast'] = bast.loc[disk.index]["x"].values[0]
        # poi.ix[i,'y_bast'] = bast.loc[disk.index]["y"].values[0]
        # poi.loc[:,"bast"] = poi.apply(lambda x: max(x['bast'],x['bast_heavy']+x['bast_light']))
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('---------------------------monthly-vists---------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    vist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",index_col=0)
    poi_m = pd.DataFrame({"id_poi":vist.index,"sum":vist.sum(axis=1)})
    dayL = np.unique([x[:10] for x in vist.columns])
    dayN = vist.shape[1] - vist.isnull().sum(axis=1)
    poi_m.loc[:,"sum"] = poi_m['sum']/dayN.values
    poi_m = poi_m.groupby([idField]).agg(np.sum).reset_index()
    poi_m.loc[:,idField] = poi_m.astype(str)
    poi.loc[:,'daily_visit'] = pd.merge(poi,poi_m,on=idField,how="left")['sum']
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('---------------------------tile-id-poi--------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    tileG = []
    for i,poii in poi.iterrows():
        poi_coord = [x for x in poii.ix[['x','y']]]
        if True: # pick from closer node
            coll = client["tdg_infra_internal"]["nodes"]
            neiN = coll.find({"node_id":int(poii["id_node"])})
            for n in neiN:
                poi_coord = n['loc']['coordinates']
        coll = client["tdg_infra_internal"]["grid_250"]
        neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Point",'coordinates':poi_coord}}}})
        for neii in neiN:
            # if poii['id_tile'] == np.nan:
            poi.loc[i,"id_tile_autom"] = neii['tile_id']
            feat = {}
            feat['type'] = "Feature"
            feat['geometry'] = neii['geom']
            del neii['geom'], neii['_id']
            feat['properties'] = neii
            tileG.append(feat)
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    tG = gpd.GeoDataFrame.from_features(tileG)
    tG.to_file(baseDir + "gis/"+custD+"/tile_geom.shp")
    
if False:
    plog('---------------------------tile-id-node--------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    nodeL = pd.read_csv(baseDir + "raw/"+custD+"/nei_node.csv")
    coll = client["tdg_grid"]["grid_250"]
    tileL = []
    for i,poii in nodeL.iterrows():
        poi_coord = [x for x in poii.ix[['x1','y1']]]
        neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Point",'coordinates':poi_coord}}}})
        for neii in neiN:
            neii['id_poi'] = poii['id_poi']
            neii[idField] = poii[idField]
            poly = sh.geometry.Polygon(neii['geom']['coordinates'][0])
            neii['x'], neii['y'] = poly.centroid.xy[0][0], poly.centroid.xy[1][0]
            tileL.append(neii)
            
    tileL = pd.DataFrame(tileL)
    tileL.to_csv(baseDir + "raw/"+custD+"/tileList.csv.gz",compression="gzip",index=False)
    tileL.rename(columns={'geom':'geometry'},inplace=True)
    del tileL['_id']
    tileG = gpd.GeoDataFrame(tileL)
    tileG.to_file(baseDir+"gis/"+custD+"/tileList.shp")
    
    if False:
        plog('-------------------shape-file-with-edge-counts-------------------')
        dirCount = pd.read_csv(baseDir + "raw/"+custD+"/tileCounts.csv")
        dirCount = pd.read_csv(baseDir + "raw/"+custD+"/dir_count.csv")
        dirCount = dirCount.replace(np.nan,0)
        dirCount.loc[:,'in'] = dirCount['north_in'] + dirCount['east_in'] + dirCount['south_in'] + dirCount['west_in'] 
        dirCount.loc[:,'out'] = dirCount['north_out'] + dirCount['east_out'] + dirCount['south_out'] + dirCount['west_out']
        dirCount = dirCount.groupby(['tile_id']).agg(np.nansum).reset_index()

        tileS = pd.merge(tileL,dirCount,left_on="tile_id",right_on="tile_id",how="left")
        #tileS = pd.merge(tileL,poi[['id_poi','name',"type"]],on="id_poi",how="left")
        tileS.to_csv(baseDir + "raw/"+custD+"/tileGeom.csv",index=False)
        tileS = tileS.replace(np.nan,0)
        g = tileS.geometry[0]
        tileG = gpd.GeoDataFrame(tileS)#.set_geometry('geometry')
        tileG.loc[:,"geometry"] = tileG['geometry'].apply(lambda x: sh.geometry.Polygon(x['coordinates'][0]))
        with open(baseDir + "gis/"+custD+"/tileList.geojson","w") as fo:
            fo.write(tileG.to_json())

    poi.loc[:,'id_tile_auto'] = pd.merge(poi,tileS[[idField,'tile_id']],on=idField,how="left")['tile_id']
    if False:
        plog('-------------------------------------tile-orientation---------------------------')
        poi.loc[:,'dir_count'] = pd.merge(poi,tileS[[idField,'in']],on=idField,how="left")['in']
        poi.loc[(poi['angle'] > -45.) & (poi['angle'] <= 45. ),"orientation"] = "north"
        poi.loc[(poi['angle'] >  45.) & (poi['angle'] <= 135.),"orientation"] = "east"
        poi.loc[(poi['angle'] > 135.) | (poi['angle'] <=-135.),"orientation"] = "south"
        poi.loc[(poi['angle'] >-135.) & (poi['angle'] <= -45.),"orientation"] = "west"
        # print(np.bincount(poi['orientation'].astype(str)))
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    plog('----------------------------tile-edge------------------------')
    tileG = gpd.read_file(baseDir + "gis/"+custD+"/tileList.geojson")
    tileG.loc[:,'sum'] = tileG.loc[:,'north_in'] + tileG.loc[:,'south_in'] + tileG.loc[:,'east_in'] + tileG.loc[:,'west_in']
    dirg = tileG[['tile_id','sum','north_in','south_in','east_in','west_in','north_out','south_out','east_out','west_out']].groupby(['tile_id']).agg(sum)
    dirg = dirg.reset_index()
    tileL = tileG[['tile_id','col_id','row_id','geometry']].groupby(['tile_id']).head(1)
    dirg = pd.merge(dirg,tileL,left_on="tile_id",right_on="tile_id",how="left")
    dirg = gpd.GeoDataFrame(dirg)
    with open(baseDir + "gis/"+custD+"/junction_tile.geojson","w") as fo:
        fo.write(dirg.to_json())
    
    dirl = gpd.GeoDataFrame(columns=["in","out","dir","geometry"])
    for i,a in dirg.iterrows():
        l = a['geometry'].boundary
        ll = LineString([(l.xy[0][2],l.xy[1][2]),(l.xy[0][3],l.xy[1][3])])
        dirl.loc[str(a['tile_id']) + 'a'] = [a['east_in'],a['east_out'],"e",ll]
        ll = LineString([(l.xy[0][1],l.xy[1][1]),(l.xy[0][2],l.xy[1][2])])
        dirl.loc[str(a['tile_id']) + 'b'] = [a['north_in'],a['north_out'],"n",ll]
        ll = LineString([(l.xy[0][0],l.xy[1][0]),(l.xy[0][1],l.xy[1][1])])
        dirl.loc[str(a['tile_id']) + 'c'] = [a['west_in'],a['west_out'],"w",ll]
        ll = LineString([(l.xy[0][3],l.xy[1][3]),(l.xy[0][4],l.xy[1][4])])
        dirl.loc[str(a['tile_id']) + 'd'] = [a['south_in'],a['south_out'],"s",ll]
                                         
    dirl = gpd.GeoDataFrame(dirl)
    with open(baseDir + "gis/"+custD+"/junction_edge.geojson","w") as fo:
        fo.write(dirl.to_json())

if False:
    trajL = pd.read_csv(baseDir + "raw/"+custD+"/trajectory_count.csv")
    nodeG = pd.read_csv(baseDir + "gis/"+custD+"/nodeList.csv")
    trajL = pd.merge(trajL,nodeG,left_on="node",right_on="src",how="left")
    trajL.to_csv(baseDir + "raw/"+custD+"/node_counts.csv",index=False)

if False:
    plog('--------------------castor-tile-------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi = poi[poi['competitor'] == 0]
    poiS = poi[poi['id_poi'].isin([1020,1033,1043,1222,1289,1518,1545])]
    cGrid = gpd.GeoDataFrame.from_file(baseDir + "log/samba/Data_Science/GeoData/Deutschland_CastorGrid/Grid_with_RPs_merged.shp")
    gridSel = gpd.GeoDataFrame()
    for i,g in poiS.iterrows():
        idx = cGrid.intersects(sh.geometry.Point(list(g[['x','y']])))
        g1 = cGrid[idx]
        g1.loc[:,"id_poi"] = g['id_poi']
        gridSel = pd.concat([gridSel,g1])
    gridSel.to_csv(baseDir + "raw/"+custD+"/castor_tile.csv",index=False)
    
if False:
    plog("------------------octree------------------")
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"octree"] = poi[['x','y']].apply(lambda x:octree.encode(x['x'],x['y'],precision=10),axis=1 )
    octL = poi[['id_poi','octree']].groupby('octree').first().reset_index()
    octL.loc[:,"geometry"] = octL['octree'].apply(lambda x: sh.geometry.Polygon(octree.decodePoly(str(x))))
    octL = gpd.GeoDataFrame(octL)
    with open(baseDir + 'gis/'+custD+'/octree.geojson', 'w') as f:
        f.write(octL.to_json())
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('--------------enrich-score-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    if custD == "tank":
        scor = pd.read_csv(baseDir + "raw/"+custD+"/delivery/scor.csv")
        poi.loc[:,"score"] = pd.merge(poi,scor,on=idField,how="left")['r_20_d_l']
    if custD == "mc":
        scor = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnType.csv")        
        poi.loc[:,"score"] = pd.merge(poi,scor,on=idField,how="left")['r_weather_hybrid']
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)


if False:
    plog('----------------assign-cells-to-poi-bse-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    dx, dy = 0.012, 0.006
    collE = client["tdg_infra"]["infrastructure"]
    mapL = []
    for i,p in poi.iterrows():
        xc, yc = p['x'], p['y']
        geoD = g_e.localPolygon(xc,yc,dx,dy,collE,geo_idx="geom")
        geoD.loc[:,"cilac"] = geoD.apply(lambda x: "%d-%d" % (x['cell_ci'],x['cell_lac']),axis=1 )
        geoD.loc[:,"type"]  = geoD.apply(lambda x: "%s-%s" % (x['broadcast_method'],x['frequency']),axis=1 )
        xL = [x[0] for x in geoD['centroid']]
        yL = [x[1] for x in geoD['centroid']]
        g = pd.DataFrame({idField:p[idField],"cilac":geoD['cilac'],"type":geoD['type'],"x_cell":xL,"y_cell":yL,"radius":geoD['estimated_radius']})
        mapL.append(g)
        print(p[idField],g.shape[0])
    mapL = pd.concat(mapL)
    mapL.loc[:,"weight"] = 1.
    mapL = mapL.groupby([idField,"cilac"]).first().reset_index()
    mapL.to_csv(baseDir+"raw/"+custD+"/map_cilac.csv.gz",compression="gzip",index=False)
    
if False:
    plog('----------------assign-cells-to-poi---------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"id_poi"] = poi['id_poi'].astype(int)
    celD = pd.read_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip")
    mshare = pd.read_csv(baseDir+"raw/basics/cilacMarketshare.csv.gz",compression='gzip')
    max_d = 3000./metr['gradMeter']
    max_nei = 20
    celL = pd.DataFrame()
    def clampF(x):
        return pd.Series({"id_poi":x['id_poi'].head(1).values[0]
                          ,"poi_list":"[%s]" % ', '.join(np.unique(x['id_poi'].apply(lambda x1:str(x1))))
                          ,"x":np.average(x['x'])
                          ,"y":np.average(x['y'])
        })
    cvist = poi.groupby(['id_zone']).apply(clampF).reset_index()
    cvist.to_csv(baseDir + "raw/"+custD+"/poi_"+"id_zone"+".csv",index=False)
    hullL = []
    for i,c in cvist.iterrows():
        disk = ((celD['X']-c['x'])**2 + (celD['Y']-c['y'])**2)
        dist = max_d
        disk = disk.loc[disk <= dist**2]
        if disk.shape[0] == 0:
            print("missing id_poi: %d" % (c['id_poi']))
            continue
        disk = disk.sort_values()
        disk = disk.head(max_nei)
        cel = celD.loc[disk.index]
        cel.loc[:,"id_poi"] = int(c["id_poi"])
        cel.loc[:,"dist"] = disk
        hullL.append({"geometry":sh.geometry.MultiPoint([(x,y) for x,y in zip(cel['X'],cel['Y'])]).convex_hull,"id_poi":c['id_poi']})
        celL = pd.concat([celL,cel],axis=0)
        
    hullL = gpd.GeoDataFrame(hullL)
    with open(baseDir + "gis/"+custD+"/cilac_hull.geojson","w") as fo:
        fo.write(hullL.to_json())
    
    celL = celL.groupby('cilac').head(1)
    celL = pd.merge(celL,mshare,left_on="cilac",right_on="cilac",how="left")
    celL.loc[:,'market_share'] = celL['factor'].replace(np.nan,np.nanmean(celL['factor']))
    celL.to_csv(baseDir + "raw/"+custD+"/cilac_sel.csv",index=False)
    cellG = celL.groupby('id_poi').agg(len).reset_index()
    poi.loc[:,"n_cell"] = pd.merge(poi,cellG,on="id_poi",how="left")['structure']
    poi.to_csv(baseDir+"raw/"+custD+"/poi.csv",index=False)
    celL.loc[:,"weight"] = 1.
    celL.loc[:,"id_poi"] = celL['id_poi'].astype(int)
    celL.to_csv(baseDir+"raw/"+custD+"/map_cilac.csv.gz",compression="gzip",index=False)

if False:
    plog('-------------------enrich-poi.csv-with-orientation-----------------')
    dirD = pd.read_csv(baseDir + "raw/"+custD+"/dir_count.csv")
    dirD.loc[:,"day"] = dirD['time'].apply(lambda x:x[0:10])
    del dirD['time']
    dirG = dirD.groupby(['tile_id','day']).agg(sum).reset_index()
    dirG = pd.merge(dirG,tileL,left_on="tile_id",right_on="tile_id",how="left")
    dirG = pd.merge(dirG,poi[['id_clust','orientation_manual']],left_on="id_clust",right_on="id_clust",how="left")
    dirG = dirG[dirG['day'] != '2018-04-01']
    for i,g in dirG.iterrows():
        dirN = g['orientation_manual']
        if dirN != dirN:
            dirN = np.argmax(g[['east_out','north_out','west_out','south_out']].values)
            dirN = ['east','north','west','south'][dirN]
        dirG.loc[i,"dir_count"] = g[dirN+"_out"]

    dirG = dirG[['id_clust','day','dir_count']]
    dirG = dirG.sort_values(['dir_count'],ascending=False)
    dirG = dirG.groupby(['id_clust','day']).first().reset_index()
    dirG.to_csv(baseDir + "raw/"+custD+"/dir_count_out.csv",index=False)

if False:
    plog('-------------------------remove-nan-from-string----------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    for i,t in enumerate(poi.dtypes):
        if t == object:
            c = poi.columns[i]
            poi.loc[:,c] = poi[c].replace(float('nan'),'')
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('--------------------------produce-metadata-----------------------------')
    try :
        info = json.load(open(baseDir+"raw/"+custD+"/metadata.json"))
    except:
        info = {}
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    info['n_poi'] = len(poi['id_poi'])
    info['n_reference'] = sum(poi['daily_visit'] == poi['daily_visit'])
#    info['n_reference'] = sum(poi['daily_visit'] != 500)
    info['n_zone'] = len(np.unique(poi['id_zone']))
    info['n_cell'] = int(poi['n_cell'].sum())    
    info['n_guest'] = int(poi['daily_visit'].sum()*1.3)    
    json.dump(info,open(baseDir+"raw/"+custD+"/metadata.json","w"))
