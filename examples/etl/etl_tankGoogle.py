#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.proc_text as p_t

def plog(text):
    print(text)

from urllib.parse import unquote
fileL = os.listdir(baseDir + "out/tank/")
footL = []
for f in fileL:
    if not re.search(".csv",f):
        continue
    df = pd.read_csv(baseDir + "out/tank/" + f)
    locN = unquote(df.columns[0].split("/")[5])
    locN = re.sub("\+"," ",locN)
    locN = re.sub(","," ",locN)
    curveL = df.columns[3:]
    labV = {}
    d = 0
    h1 = ""
    h2 = ""
    i1 = ""
    for i in curveL:
        c = i.split("%")[0][-2:]
        if (re.search("Currently",i) or re.search("usually",i)) :
            h = "%02d" % (int(h2[0:2]) + 1)
        else :
            h = i.split("at ")[1]
        if (re.search("AM",i) and re.search("PM",i1)):
            d = d + 1
            h1 = h
        if re.search("PM",h):
            h = "%02d" % (int(h[0:2]) + 12)
        else:
            h = "%02d" % int(h[0:2])
            h2 = h
        i1 = i
        footL.append({"wday":int(d),"hour":h,"location":locN,"footfall":c,"string":i})

footL = pd.DataFrame(footL)
footL.loc[:,"footfall"] = footL['footfall'].astype(int)
import importlib
importlib.reload(p_t)
footL = footL.pivot_table(index="location",columns="time",values="footfall").reset_index()
subD = {"South":"Süd","North":"Nord","East":"Ost","Motorway":"","[Rr]aststätte":"","[Rr]estaurant":"","Autohof":"","[Tt]ankstelle":"","Autobahn":"","Hotel":"","GmbH":"","Service [Aa]rea":"",'Rest Stop':"","Rasthof":"","Rastanlage":"","Serways":"","^ +":""," +":" "}
for k in subD.keys():
    footL.loc[:,"location"] = footL['location'].apply(lambda x: re.sub(k,subD[k],x))

poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poiId = poi[['name','id_poi','id_clust']].groupby('id_poi').first().reset_index()
seq2 = poiId['name']
seq1 = footL['location']
footL.loc[:,"name"] = p_t.text_dist(footL['location'],poiId['name'])
footL = footL[footL['name'] != '']
footL = pd.merge(footL,poiId,on="name",how="left")
hL = footL.columns[[bool(re.search('-',x)) for x in footL.columns]]
footL1 = pd.melt(footL,id_vars="id_clust",value_vars=hL)
footL1 = footL1[footL1['value'] == footL1['value']]
footL1.columns = ["id_clust","hour","popularity"]
footL1.to_csv(baseDir + "raw/tank/footfall_google.csv",index=False)


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
