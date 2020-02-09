import os, sys, gzip, random, csv, json, datetime, re, time
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.geo_octree as g_o
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']

callL = pd.read_csv(baseDir + "raw/app/opening.csv.gz",compression="gzip")
gO = g_o.h3tree()
for precDigit in [10,9,8]:
    print(precDigit)
    callL.loc[:,"geohash"] = callL.apply(lambda x: gO.encode(x['longitude'],x['latitude'],precision=precDigit),axis=1)
    callL.loc[:,"n"] = 1
    openG = callL.groupby('geohash').agg(sum).reset_index()
    openG.drop([0],inplace=True)
    openG = openG[openG['n'] > 4]
    openG = openG.sort_values('n',ascending=False)
    openG.loc[:,"longitude"] = openG['longitude']/openG['n']
    openG.loc[:,"latitude"] = openG['latitude']/openG['n']
    hexL = openG['geohash'].apply(lambda x: sh.geometry.Polygon(gO.decodePoly(x)))
    openG.to_csv(baseDir + "raw/app/open_dens_"+str(precDigit)+".csv.gz",compression="gzip",index=False)
    openG = gpd.GeoDataFrame(openG,geometry=hexL)
    openG.to_file(baseDir + "gis/app/opening_"+str(precDigit)+".shp")



if False:
    print('---------------------pre-format-----------------------')
    callL = pd.read_csv(baseDir + "raw/app/opening.csv.gz",compression="gzip",converters={"1":eval})
    callL.columns = ['city','property','time','id_device','id_user']
    openL = pd.DataFrame(callL['property'].tolist())
    callL.drop(columns=['property'],inplace=True)
    callL = pd.concat([callL,openL],axis=1)
    setL = callL['UserLocation'] == callL['UserLocation']
    call1 = callL[setL]
    call2 = callL[~setL]
    call1.loc[:,'UserLocation'] = call1['UserLocation'].apply(lambda x: re.sub("lat/lng: \(","",x))
    call1.loc[:,'UserLocation'] = call1['UserLocation'].apply(lambda x: re.sub("\)","",x))
    call1.loc[:,'latitude'] = call1['UserLocation'].apply(lambda x: float(x.split(",")[0]))
    call1.loc[:,'longitude'] = call1['UserLocation'].apply(lambda x: float(x.split(",")[1]))
    callL = pd.concat([call1,call2])
    callL = callL[['city','time','id_device','id_user','longitude','latitude']]
    callL.to_csv(baseDir + "raw/app/opening.csv.gz",compression="gzip",index=False)


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
