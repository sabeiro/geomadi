#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)

import pymongo
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
cent = pd.read_csv(baseDir + "raw/roda/fem_centroid.csv")

lte = pd.read_csv(baseDir + "log/tracebox/tracebox_move.csv.gz",compression="gzip")
ho = pd.read_csv(baseDir + "log/tracebox/tracebox_pos.csv.gz",compression="gzip")

ho.loc[:,"r"] = (ho['x'] - ho['x'].mean())**2 + (ho['y'] - ho['y'].mean())**2

plt.hist(ho['r'],bins=20)
plt.show()



print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
