#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph

def plog(text):
    print(text)

foot = pd.read_csv(baseDir + "raw/others/castor_billboard.csv")
foot.loc[:,"rp_time"] = foot["rp_time"].apply(lambda x: "%sT%s" % (x[:10],x[11:19]) )
ts = foot["rp_time"].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") + datetime.timedelta(hours=2))
foot.loc[:,"day"] = [datetime.datetime.strftime(x,"%Y-%m-%dT") for x in ts]
foot.loc[:,"wday"] = [x.weekday() for x in ts]
#foot = foot[foot["wday"] <= 4] 
footG = foot[["day","cnt","wday"]].groupby(["day","wday"]).agg(sum).reset_index()
footG = footG[footG['cnt'] > 10000.]
footL = footG[["cnt","wday"]].groupby(["wday"]).agg(np.mean).reset_index()
print(footG['cnt'].mean()*.5)
footL.loc[:,"cnt"] = footL["cnt"]*.5
print(footL)


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
