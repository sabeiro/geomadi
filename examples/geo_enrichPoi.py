import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as sh
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
sys.path.append('/usr/local/lib/python3.6/dist-packages/')
baseDir = os.environ['LAV_DIR']
import lernia.train_viz as t_v
import albio.series_stat as s_s
import albio.series_interp as s_i
import geomadi.geo_octree as otree
import geomadi.geo_enrich as g_e
import geomadi.geo_ops as g_p

gO = otree.h3tree()
precDigit = 8
custD = "dep" 

poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")

if False:
    print('----------------------------region------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    poi.loc[:,"region"] = g_e.addRegion(poi,baseDir + "gis/geo/bundesland.shp")
    poi.to_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)

if False:    
    print('----------------------------zip------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    poi.loc[:,"zip5"] = g_e.addRegion(poi,baseDir + "gis/geo/zip5.shp",field="PLZ")
    poi.to_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)

if False:
    print('-------------------closest-density-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    densG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/pop_dens.shp")
    BBox = [0.05,0.05]
    P = g_p.boxAround(poi['x'],poi['y'],BBox=[0.02,0.02])
    poiG = gpd.GeoDataFrame(P,columns=["geometry"])
    for i,p in poiG.iterrows():
        pMask = densG.intersects(p['geometry'])
        dens = densG[pMask]
        if dens.shape[0] == 0:
            continue
        x1 = p.geometry.centroid.x
        y1 = p.geometry.centroid.y
        poi.loc[i,'pop_dens'] = g_e.interp2D(dens,x1,y1,z_col="Einwohner")
        poi.loc[i,'women'] = g_e.interp2D(dens,x1,y1,z_col="Frauen_A")
        poi.loc[i,'foreign'] = g_e.interp2D(dens,x1,y1,z_col="Wohnfl_Whg")
        poi.loc[i,'land_use'] = g_e.interp2D(dens,x1,y1,z_col="Leerstands")
        poi.loc[i,'elder'] = g_e.interp2D(dens,x1,y1,z_col="ab65_A")
    poi.to_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)

if False:
    print('-------------------auto-smooth-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    colL = ['activation','noise','urev','pot']
    P1 = g_p.boxAround(poi['x'],poi['y'],BBox=[0.01,0.01])
    P2 = [sh.geometry.Point([x,y]) for x,y in zip(poi['x'],poi['y'])]
    pointG = gpd.GeoDataFrame(P1,columns=["geometry"])
    boxG = gpd.GeoDataFrame(poi,geometry=P2)
    for i,p in pointG.iterrows():
        pMask = boxG.intersects(p['geometry'])
        dens = boxG[pMask].dropna()
        if dens.shape[0] == 0: continue
        x1 = p.geometry.centroid.x
        y1 = p.geometry.centroid.y
        for c in colL:
            poi.loc[i,'smooth_'+c] = g_e.interp2D(dens,x1,y1,z_col=c)
    poi.to_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)
    
if False:
    print('-------------------geohash-to-polygon-----------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    polyL = poi['geohash'].apply(lambda x: sh.geometry.Polygon(gO.decodePoly(x)) )
    poiG = gpd.GeoDataFrame(poi,geometry=polyL)
    poiG.to_file(baseDir + "gis/dep/poi.shp",index=False)

if False:
    print("----------------------enrich-interpolate-potential-------------------------")
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    densS = pd.read_csv(baseDir + "raw/dep/chem_pot_"+str(precDigit)+".csv.gz",compression="gzip")
    colL = ['geohash','activation','noise','urev','pot']
    poi2 = poi.merge(densS[colL],on="geohash",how="left",suffixes=["","_y"])
    poi.loc[:,colL] = poi2[colL]
    poi.to_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)
    

if False:
    print('------------------------------add-degeneracy----------------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip")
    poiX = poi.apply(lambda x: sh.geometry.Point([x['x'],x['y']]),axis=1)
    fL = os.listdir(baseDir + "gis/geo_berlin/")
    fL = [x for x in fL if bool(re.search("geojson",x))]
    fL = [x for x in fL if bool(re.search("_berlin",x))]
    for f in fL:
        print(f)
        i = f.split("_")[0]
        stat = gpd.read_file(baseDir + "gis/geo_berlin/" + f)
        poi['dis_'+i] = poiX.apply(lambda x: g_e.degeneracy(stat.distance(x),r_lim=0.03))
        
    tourL = {"name":"tourist","fclass":['archaeological','arts_centre','artwork','biergarten','hostel', 'hotel','memorial','monument', 'motel', 'museum','nightclub', 'observation_tower','park','theatre', 'theme_park','tourist_info','viewpoint','zoo']}
    resiL = {"name":"resident","fclass":['biergarten','college','community_centre','kindergarten','library','mall','nightclub','park','university']}
    for i in ['crossing']:
        print(i)
        stat = gpd.read_file(baseDir + "gis/geo_berlin/berlin_" + i + ".shp")
        if i == 'crossing': stat = stat[stat['fclass'] != 'street_lamp']
        poi['dis_'+i] = g_p.minDist(poiX,stat['geometry'])
        
    attra = gpd.read_file(baseDir + "gis/geo_berlin/berlin_" + "activities" + ".shp")
    for i in [tourL,resiL]:
        print(i['name'])
        stat = attra[attra['fclass'].isin(i['fclass'])]
        poi['dis_'+i['name']] = g_p.minDist(poiX,stat['geometry'])
        
    poi.to_csv(baseDir + "raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)

if False:
    print('----------------------------make-inital------------------------')
    densS = pd.read_csv(baseDir+"raw/"+custD+"/chem_pot_"+str(precDigit)+".csv.gz",compression="gzip")
    poi = densS[['geohash','urev']]
    lx = [gO.decode(x) for x in poi['geohash']]
    poi.loc[:,"x"] = [x[0] for x in lx]
    poi.loc[:,"y"] = [x[1] for x in lx]
    poi.to_csv(baseDir+"raw/"+custD+"/poi_"+str(precDigit)+".csv.gz",compression="gzip",index=False)


