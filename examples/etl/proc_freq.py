import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.geo_octree as g_o
from ast import literal_eval

BoundBox = [5.866,47.2704,15.0377,55.0574]
idField = "id"
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])

metric = "freq"
n_iter = 10
if(len(sys.argv) > 1): metric = sys.argv[1]
if(len(sys.argv) > 2): n_iter = sys.argv[2]
projDir = baseDir + "raw/gps/" + metric + "/"
dL = os.listdir(projDir)
uniq, freq = [], []

for d in dL:
    print(d)
    den = pd.read_csv(projDir + d,index_col=0)
    den = den.replace(float('nan'),1e-10)
    uniq = list(np.unique(list(den.index) + uniq))
    fName = d.split("_")[1].split(".")[0]
    freq.append({"day":fName,"user":den.shape[0],"event":den.values.sum()})
freq = pd.DataFrame(freq)
freq.loc[:,"day"] = freq["day"].apply(lambda x: re.sub("freq_","",x))
freq = freq.sort_values("day")
freq.loc[:,"day"] = freq["day"].apply(lambda x: int(x)/100)

fig, ax = plt.subplots(3,1)
ax[0].bar(freq['day'],freq['user'])
ax[1].bar(freq['day'],freq['event'])
ax[2].bar(freq['day'],freq['event']/freq['user'])
ax[0].set_title("unique users")
ax[1].set_title("number of events")
ax[2].set_title("number of events per user")
plt.show()

fig, ax = plt.subplots(1,2)
den.boxplot(ax=ax[0])
den.boxplot(ax=ax[1])
plt.show()

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
