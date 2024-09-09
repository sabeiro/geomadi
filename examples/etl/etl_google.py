#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geohash
import ast
import yaml
import geopandas as gpd

def plog(text):
    print(text)

refc = pd.read_csv(baseDir + "raw/roda/reference_curve.csv").T
refc.replace(np.nan,"",inplace=True)
for i in [1,2,3]: del refc[i]
for i in refc.columns: refc[i] = refc[i].str.replace("\% busy at ",",")
for i in refc.columns: refc[i] = refc[i].str.replace(" AM\.","")
for i in refc.columns: refc[i] = refc[i].str.replace("12 PM\.","12")
for i in refc.columns: refc[i] = refc[i].str.replace(" PM\.","+12")
for i in refc.columns: refc[i] = refc[i].str.replace("Currently.*","")
centL = []

print(refc.head(1))
refc.to_csv(baseDir + "raw/roda/tmp.csv",index=False)
exit

r = refc.iloc[0]
for i,r in refc.iterrows():
    cent = []
    for f in [x for x in list(r) if bool(re.search(",",str(x)))]:
        fL = f.split(",")
        try:
            cent.append({fL[1]:fL[0]})
        except:
            print(fL)
    cent = pd.DataFrame(cent)
    weekL = []
    for j in cent.columns: weekL.append({eval(j):[int(x) for x in list(cent[j].dropna())]})
    centL.append({r[0]:weekL})

refV = pd.DataFrame(columns=[''.join(x.keys()) for x in centL])
x = centL[0]
for i,x in enumerate(centL):
    for k in x.keys():
        for j,l in enumerate(x[k]):
            h = x[k][j]
            h = int(''.join(h.keys()))
            refV.loc[h,k] = x[k][h]
                
[x[0] for x in centL]


