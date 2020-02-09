import os, sys, gzip, random, csv, json, re, time
import urllib3,requests
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import datetime
import http.client
import json

def routificFrame(solJ):
    fleet = pd.DataFrame.from_dict(solJ['input']['fleet'],orient='index')
    network = pd.DataFrame.from_dict(solJ['input']['network'],orient='index')
    locF = pd.DataFrame.from_dict(solJ['input']['visits'],orient='index')
    locF1 = locF['location'].apply(pd.Series)
    del locF['location']
    locF = pd.concat([locF,locF1],axis=1)
    unserved = pd.DataFrame.from_dict(solJ['output']['unserved'],orient='index',columns=["collect"])
    unserved['collect'] = False
    locF = locF.merge(network ,how="outer",left_index=True, right_index=True,suffixes=["","_y"])
    locF = locF.merge(unserved,how="outer",left_index=True, right_index=True,suffixes=["","_y"])
    solL = []
    for k in list(solJ['output']['solution'].keys()):
        solF = pd.DataFrame(solJ['output']['solution'][k])
        solF['van'] = k
        solF['sequence'] = range(solF.shape[0])
        solL.append(solF)
    solF = pd.concat(solL).reset_index()
    solF.drop(columns=["index"],inplace=True)
    locF = locF.merge(solF,how="outer",left_index=True,right_on="location_id")
    locF.loc[locF['van'] != locF['van'],'van'] = 0
    locF['van'] = [int(x) for x in locF['van']]
    locF = locF.sort_values(["van","sequence"])
    locF['y'] = locF['lat_y']
    locF['x'] = locF['lng_y']
    locF.drop(columns=['lat','lng','lat_y','lng_y'],inplace=True)
    locF['potential'] = locF['priority']*.01
    locF['potential'] = 1. + locF['potential']/locF['potential'].max()
    locF['occupancy'] = locF['load']
    locF['active'] = 1
    locF['zoneid'] = 'berlin'
    locF.loc[locF['van'] != locF['van'],'van'] = 0
    locF.loc[:,"geohash"] = locF.apply(lambda x: gO.encode(x['x'],x['y'],precision=11),axis=1)
    locF.replace(float('nan'),0,inplace=True)
    locF = locF.groupby('geohash').head(1)
    locF.index = locF['geohash']
    geomL = []
    for i,g in fleet.iterrows():
        locV = locF[locF['van'] == i]
        geomL.append(sh.geometry.LineString([[x,y] for x,y in zip(locV['x'],locV['y'])]))
    locG = gpd.GeoDataFrame({"van":fleet.index},geometry=geomL)
    return locF, locG, fleet

if False:
    import importlib
    jobN = "job_s"+str(592)+"_v"+str(9)
    solJ = json.load(open(baseDir + "raw/opt/"+jobN+"_sol.json"))
    locF, locG, vanL = routificFrame(solJ)
    opsL  = pd.DataFrame({"action":['collect','potential']})
    conf = {"cost_route":70.35,"cost_stop":.1,"max_n":50,"temp":.5,"link":5}
    locF.to_csv(baseDir + "raw/opt/sol_"+jobN+".csv",index=False)                
    locF.to_csv(baseDir + "gis/route/sol_van.csv",index=False)
    with open(baseDir + "gis/route/van.geojson", 'w') as f: f.write(locG.to_json())


