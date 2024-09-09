#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)


fs = pd.read_csv(baseDir + "log/telia/spikes.csv.tar.gz")
fs = fs[fs.columns[[0,3,-1]]]
fs.columns = ["id","time","person"]
fs = fs.pivot_table(index="time",values="person",columns="id",aggfunc=np.sum).reset_index()
poi = pd.read_csv(baseDir + "raw/telia/id_poi.csv")
idL = dict(zip(poi["THS_id"],poi["ML_loc_id"]))
fs.columns = ['time'] + [str(idL[x]) for x in fs.columns[1:]]
dirT = pd.read_csv(baseDir + "out/telia_dir_loc.csv")

X = np.array(fs[[x for x in fs.columns if not x == "time"]].replace(np.nan,0))
X = X[1:] - X[:-1]
csum = np.abs(X.sum(axis=0))
idxL = sorted(range(len(csum)), key=lambda k: csum[k])
outL = fs[fs.columns[[x+1 for x in idxL]][:5]]
outL.replace(np.nan,0,inplace=True)
outL.index = fs['time']
ax = outL.plot()
plt.show()

gs = fs
gs.loc[:,'time'] = gs['time'].apply(lambda x:x[:10])
gs = gs.pivot_table(index="time",values=gs.columns[1:],aggfunc=np.sum).replace(np.nan,0)
X = np.array(gs)
X = X[1:] - X[:-1]
csum = np.abs(X.sum(axis=0))
idxL = sorted(range(len(csum)), key=lambda k: csum[k])
outL = gs[gs.columns[[x for x in idxL]][:5]]
ax = outL.plot()
plt.show()


outL.index = [x[:10] for x in outL.index]
outL.loc[:,"time"] = outL.index
outL = outL.pivot_table(index="time",values=outL.columns[:-1],aggfunc=np.sum)
ax = outL.plot()
plt.show()

cL = dirT.columns[dirT.columns.isin(fs.columns)]
X1 = dirT[cL]
X2 = fs[cL]
X1.index = X1['time']
X2.index = X2['time']
del X1['time']
del X2['time']
rL = X1.index[X1.index.isin(X2.index)]
X1 = np.array(X1.loc[rL][1:].replace(np.nan,0))
X2 = np.array(X2.loc[rL][1:].replace(np.nan,0))
Xd = X1 - X2
xL = Xd.reshape(Xd.shape[0]*Xd.shape[1])
plt.hist(xL,bins=20)
plt.show()
xcorr = np.corrcoef(X1,X2)
plt.imshow(xcorr)
plt.show()




print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
