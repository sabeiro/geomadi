#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats, optimize, interpolate
from tzlocal import get_localzone
from sklearn.cluster import KMeans
import viz_lib as v_l
tz = get_localzone()

def plog(text):
    print(text)

act = pd.read_csv(baseDir + "log/roda/act_roda_11.csv.tar.gz",compression="gzip")
t = (act['ts1'])# + act['ts2'])*.5
hi = t.apply(lambda x: datetime.datetime.fromtimestamp(x,tz).strftime('%H')).apply(int)

#act = act[(hi>5) & (hi<24) ]
t = (act['ts1'])# + act['ts2'])*.5
act['time'] = t.apply(lambda x: datetime.datetime.fromtimestamp(x,tz).strftime("%Y-%m-%dT%H:00:00"))
act['count'] = 1.
sact = act.pivot_table(index="cilac",columns="time",values="count",aggfunc=np.sum).fillna(0).reset_index()
hL = sact.columns[[bool(re.search(':',x)) for x in sact.columns]]
sact.loc[:,"sum_p"] = sact[hL].apply(lambda x: np.nansum(x),axis=1)
sact = sact[sact['sum_p'] > 100.]
sact.loc[:,"chisquare"] = sp.stats.chisquare(sact[hL].values.T)[1]
sact.loc[:,"kmean"] = KMeans(n_clusters=5,random_state=0).fit(sact[hL].values).labels_
sact.sort_values(["kmean",'chisquare'],inplace=True,ascending=False)
sact.sort_values(['chisquare'],inplace=True,ascending=False)

X = sact[hL].values
v_l.rawData(X)


plt.plot(sact[hL].apply(lambda x: np.nansum(x),axis=0).values)
plt.show()

cont = sp.stats.chi2_contingency(X)[3]
freq = sp.stats.contingency.expected_freq(cont)

cells = pd.read_csv(baseDir + "raw/centroids.csv.tar",compression="gzip")
cilacL = pd.read_csv(baseDir + "raw/roda/cilac_sel_unibail.csv")
cellL = np.unique(act['cilac'])

del cells['tech']
poi = pd.read_csv(baseDir + "gis/roda/poi.csv")
act.dropna(inplace=True)
act = pd.merge(act,cells,left_on="cilac",right_on="cilac",how="left",suffixes=["","_curr"])
act = pd.merge(act,cells,left_on="cilac_prev",right_on="cilac",how="left",suffixes=["","_prev"])
act = pd.merge(act,cells,left_on="cilac_next",right_on="cilac",how="left",suffixes=["","_next"])




print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
