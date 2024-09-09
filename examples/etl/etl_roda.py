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
import geohash
from scipy import signal

def plog(text):
    print(text)

with open(baseDir + '/raw/metrics.json') as f:
    metr = json.load(f)['metrics']

from shapely.geometry import Point, Polygon
import geopandas

act = pd.read_csv(baseDir + "log/roda/act_shift.csv.tar.gz",index_col=False,compression="gzip")
cells = pd.read_csv(baseDir + "raw/centroid_angle.csv.tar.gz",compression="gzip")
act.sample(n=int(act.shape[0]/10)).to_csv(baseDir + "gis/roda/act.csv",index=False)

def clampF(x):
    return pd.Series({"x":np.average(x['x']),"sx":np.average(x['sx'])
                      ,"y":np.average(x['y']),"sy":np.average(x['sx'])
                      ,"t":np.average(x['t']),"st":np.average(x['st'])
                      ,"n":len(x['cilac'])
                      ,"n_in":sum(x['inside'])
        })
gact = act.groupby(["cilac","id_poly"]).apply(clampF).reset_index()
gact.loc[:,"weight"] = gact['n_in']/gact['n']
gact = pd.merge(gact,cells,on="cilac",how="left")
gact.to_csv(baseDir + "raw/roda/cilac_weight.csv",index=False)

plog('-------------------create-mapping------------------------')

fileL = ["raw/roda/cilac_poi_intersection.csv","raw/roda/cilac_weight_manual.csv"]
poi = pd.read_csv(baseDir + "raw/roda/poi.csv")
polyS = gpd.GeoDataFrame.from_file(baseDir + "gis/roda/area.shp")
with open(baseDir + '/gis/roda/unibailshoppingcenterbalanced.json') as f:
    mapD = json.load(f)['mapping']

for f in fileL:
    gact = pd.read_csv(baseDir + f)
    gact = gact[gact['weight'] > 0]
    marketS = 0
    cilacW = []
    for i,p in gact.iterrows():
        id_poly = str(p['id_poly'])
        try :
            cilac = mapD[p['cilac']]
        except :
            cilac = {id_poly:{"weight":p['weight'],"market_share":marketS}}
        j = [x for x in cilac.keys()]
        marketS = cilac[j[0]]['market_share']
        weigthI = p['weight']
        if id_poly in j:
            weightI = max(p['weight'],cilac[id_poly]['weight'])
            cilac[id_poly]['weight'] = weightI
        else :
            cilac.update({id_poly:{'weight':p['weight'],"market_share":marketS}})
            cilacW.append({"cilac":p['cilac'],"id_poly":id_poly,"weight":weightI})
            mapD.update({p['cilac']:cilac})
            ##mapD[p['cilac']].update(cilac)

    cilacW = pd.DataFrame(cilacW)
    
cilacW.to_csv(baseDir + "raw/roda/cilac_weight_prod.csv",index=False)
with open(baseDir + '/gis/roda/unibailweighted.json',"w") as f:
    json.dump({"mapping":mapD},f)

if False:
    plog('--------------------select-cells------------------------')
    max_d = 15000./metr['gradMeter']
    max_nei = 30
    cellL = pd.DataFrame()
    mshare = pd.read_csv(baseDir+"raw/cilacMarketshare.csv.tar.gz",compression='gzip',sep=',',quotechar='"',names=["cilac","factor"],header=0)
    hullL = []
    for i,c in poi.iterrows():
        x_c, y_c = c['x'],c['y']
        disk = ((cells['X']-x_c)**2 + (cells['Y']-y_c)**2)
        disk = disk.loc[disk <= max_d**2]
        if(disk.shape[0] <= 0) :
            continue
        if disk.shape[0] > max_nei:
            disk = disk.sort_values()
            disk = disk.head(max_nei)
        tmp = cells.loc[disk.index]
        tmp.loc[:,"id_zone"] = c["id"]
        cellL = pd.concat([cellL,tmp],axis=0)

    cellL = cellL.groupby('cilac').head(1)
    cellL = pd.merge(cellL,mshare,left_on="cilac",right_on="cilac",how="left")
    cellL['factor'] = cellL['factor'].replace(np.nan,np.nanmean(cellL['factor']))
    cellL.to_csv(baseDir + "raw/roda/cilac_sel_roda.csv",index=False)
    print(cellL.shape)
    
    coll = client["tdg_infra"]["infrastructure"]
    neiDist = 200.
    nodeL = []
    cellL = []
    for i,poii in poi.iterrows():
        poii = poi.loc[i]
        poi_coord = [x for x in poii.ix[['x','y']]]
        neiN = coll.find({'geom':{'$nearSphere':{'$geometry':{'type':"Point",'coordinates':poi_coord},'$minDistance':0,'$maxDistance':neiDist}}}) 
        nodeId = []
        for neii in neiN:
            cellL.append({"centroid":str(neii['cell_ci']) + '-' + str(neii['cell_lac'])})


plog('-----------------curves----------------------')
act = act[act['dur'] > 0.25]
act = act[act['dur'] < 4]
act.loc[:,"sr"] = np.sqrt(act['sx']**2 + act['sy']**2)
act.loc[:,"time"] = act['t'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%H'))
def clampF(x):
    return pd.Series({"n":len(x['cilac'])
        })
hact = act.groupby(['cilac','time']).apply(clampF).reset_index()
hact.to_csv(baseDir + "raw/roda/activity_melt.csv",index=False)


if False:
    nBin = 100
    dayL = np.linspace(min(act['t']),max(act['t']),num=nBin)
    def make_gauss(N, sigma, mu):
        k = N / (sigma * np.sqrt(2*np.pi))
        s = -1.0 / (2 * sigma * sigma)
        def f(x):
            return k * np.exp(s * (x - mu)*(x - mu))
        return f
    
    cilacN = len(set(act['cilac']))
    cilacL = np.unique(act['cilac'])
    cilacM = np.zeros((cilacN,nBin))
    delta = 1./(max(act['t']) - min(act['t']))
    tmin = min(act['t'])
    for i,p in act.iterrows():
        if i%1000 == 0:
            print("processing %.2f" % (i/act.shape[0]))
        rowN = [j for j,x in enumerate(cilacL) if x == p['cilac']][0]
        gaussian = make_gauss(nBin,p['t'],p['st'])
        cilacM[rowN] += gaussian(dayL)

    cilacM = pd.DataFrame(cilacM,columns=dayL,index=cilacL)
    cilacM.loc[:,"cilac"] = cilacM.index
    meltM = pd.melt(cilacM,id_vars=["cilac"])
    meltM.loc[:,"time"] = meltM['variable'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%S'))
    meltM.to_csv(baseDir + "raw/roda/activity_melt.csv",index=False)











pact = pd.merge(act,cells,on="cilac",how="left")
x = pact['X']
y = pact['Y']

plt.hist(x,bins=20)
plt.show()

vact.loc[:,"geohash"] = vact[['x_n','y_n']].apply(lambda x: geohash.encode(x[0],x[1],precision=5))
vact.loc[:,"geohash"] = vact[['x','y']].apply(lambda x: geohash.encode(x['x'],x['y'],precision=5))
vact[['x','y']].apply(lambda x: x)
    



poi = pd.read_csv(baseDir + "raw/roda/poi.csv")
if False:
    polyS = gpd.GeoDataFrame.from_file(baseDir + "gis/roda/area.shp")
    for i,px in polyS.iterrows():
        print(i)
        inTile = [px.geometry.contains(Point(x,y)) for x,y in zip(act['x'],act['y'])]
        act.loc[inTile,"inside"] = px["id"]

    tact = act[~np.isnan(act['inside'])]
    tact.loc[:,"inside"] = tact['inside'].apply(lambda x:int(x))
    del tact['cilac']
    tact = tact.groupby('inside').agg(np.nanmean).reset_index()
    poi = pd.merge(poi,tact,left_on="id",right_on="inside",how="left")
    polyS = pd.merge(polyS,tact,left_on="id",right_on="inside",how="left")
    polyS.loc[:,"buffer"] = polyS['sr']*metr['gradMeter']/2.
    polyS.to_file(baseDir + "gis/roda/area.shp")
    poi.to_csv(baseDir + "raw/roda/poi.csv",index=False)

actD = act[['x','y','sr','dur']].dropna()
actD.loc[actD['sr']<=0.,'sr'] = min(actD['sr'][actD['sr']>0])
actD.loc[:,"sr"] = actD['sr']
actD.to_csv(baseDir + "gis/roda/current.csv")

antD = pd.read_csv(baseDir + "raw/antenna_spec.csv.tar.gz",compression="gzip")
cells = pd.read_csv(baseDir + "raw/centroids_angle.csv.tar.gz",compression="gzip")

poi = pd.read_csv(baseDir + "gis/roda/poi.csv")
fs.dropna(inplace=True)

fs = pd.merge(fs,cells,left_on="cilac",right_on="cilac",how="left",suffixes=["","_curr"])
fs = pd.merge(fs,cells,left_on="cilac_prev",right_on="cilac",how="left",suffixes=["","_prev"])
fs = pd.merge(fs,cells,left_on="cilac_next",right_on="cilac",how="left",suffixes=["","_next"])

def  hashId(x):
    try:
        return geohash.encode(x[0],x[1],precision=5)
    except:
        return "-1"

fs.loc[:,"hash_curr"] = fs[['X','Y']].apply(lambda x: hashId(x),axis=1)
fs.loc[:,"hash_prev"] = fs[['X_prev','Y_prev']].apply(lambda x: hashId(x),axis=1)
fs.loc[:,"hash_next"] = fs[['X_next','Y_next']].apply(lambda x: hashId(x),axis=1)

#gs = pd.crosstab(index=fs['hash_curr'].values,columns=fs['hash_prev'].values)
gs = fs.pivot_table(index='hash_prev',values='cilac',aggfunc=len).reset_index()
def  hashId(x):
    try:
        return geohash.decode(x)
    except:
        return (-1,-1)

coorL = gs['hash_prev'].apply(lambda x: hashId(x))
gs.loc[:,"x"] = [x[0] for x in coorL]
gs.loc[:,"y"] = [x[1] for x in coorL]
gs = gs[gs['cilac']>30]
gs[1:].to_csv(baseDir + "gis/roda/density.csv")

fs[['X_prev','Y_prev','dur']].to_csv(baseDir + "gis/roda/previous.csv")


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
