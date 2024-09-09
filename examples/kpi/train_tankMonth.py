import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import geomadi.train_shapeLib as shl
import custom.lib_custom as t_l
import geomadi.series_lib as s_l
import geomadi.train_lib as tlib
import geopandas as gpd
import etl.etl_mapping as e_m

import importlib
importlib.reload(t_l)
importlib.reload(s_l)
importlib.reload(e_m)
importlib.reload(tlib)
gradMeter = 111122.19769899677

def plog(text):
    print(text)

poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi.loc[:,"bast_sum"] = poi[['bast_light','bast_heavy']].apply(lambda x: np.nansum([x[0],x[1]]),axis=1)
poi.loc[:,"bast"] = poi[['bast','bast_sum']].apply(lambda x: np.max(x),axis=1)
dTh = '20'
if len(sys.argv) > 1:
    dTh = str(sys.argv[1])

plog('--------------------------------------activities-----------------------------')
mist = pd.read_csv(baseDir + "raw/tank/visit_march_melted.csv.tar.gz",compression="gzip")
mist = mist[mist['time'] < '2018-03-28T00:00:00']##Easter
mist.loc[:,"day"]  = mist['time'].apply(lambda x:x[0:10])
mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
mist.loc[:,"wday"] = mist['day'].apply( lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
mist.loc[:,"y_dif"] = mist["ref"]/mist['dist_20']
mist.replace(float("Inf"),1.,inplace=True)
mist.replace(float("-Inf"),1.,inplace=True)
def clampF(x):
    return pd.Series({"y_dif":np.mean(x['y_dif']),"s_dif":np.std(x['y_dif'])})
gist = mist[['id_clust','wday','y_dif']].groupby(['id_clust','wday']).apply(clampF).reset_index()
del mist['y_dif']
mist = pd.merge(mist,gist,on=["id_clust","wday"],how="left")
mist.loc[:,"dist_20"] = mist['dist_20']*mist['y_dif']
mist.loc[:,"ref_raw"] = mist['ref']

mistW = mist[mist['wday'] < 4]
dist  = mist[ ['id_clust','day','ref','dist_20']].groupby(['id_clust','day']).agg(sum).reset_index()
distW = mistW[['id_clust','day','ref','dist_20']].groupby(['id_clust','day']).agg(sum).reset_index()
mist.loc[: ,"corr_20"] = mist['dist_20'] *t_l.regressorTank(mist ,poi,idFloat="dist_20")
mistW.loc[:,"corr_20"] = mistW['dist_20']*t_l.regressorTank(mistW,poi,idFloat="dist_20")
dist.loc[: ,"corr_20"] = dist['dist_20'] *t_l.regressorTank(dist ,poi,idFloat="dist_20")
distW.loc[:,"corr_20"] = distW['dist_20']*t_l.regressorTank(distW,poi,idFloat="dist_20")
isSingle = False
mist.loc[: ,"corr_20"] = t_l.smoothClust(mist['corr_20'] ,mist['id_clust'] ,beta=5,r=9,isSingle=isSingle)
mistW.loc[:,"corr_20"] = t_l.smoothClust(mistW['corr_20'],mistW['id_clust'],beta=5,r=9,isSingle=isSingle)
mist.loc[:,"ref"]      = t_l.smoothClust(mist['ref']     ,mist['id_clust'] ,beta=5,r=9,isSingle=isSingle)
mistW.loc[:,"ref"]     = t_l.smoothClust(mistW['ref']    ,mist['id_clust'] ,beta=5,r=9,isSingle=isSingle)
dist.loc[: ,"corr_20"] = t_l.smoothClust(dist['corr_20'] ,dist['id_clust'] ,beta=3,r=3,isSingle=isSingle)
distW.loc[:,"corr_20"] = t_l.smoothClust(distW['corr_20'],distW['id_clust'],beta=3,r=3,isSingle=isSingle)
dist.loc[:,"ref"]      = t_l.smoothClust(dist['ref']     ,dist['id_clust'] ,beta=3,r=3,isSingle=isSingle)
distW.loc[:,"ref"]     = t_l.smoothClust(distW['ref']    ,dist['id_clust'] ,beta=3,r=3,isSingle=isSingle)
distW.loc[:,"ref_raw"] = distW['ref']
distW.loc[:,"dist_20"] = distW['dist_20']*sum(distW['ref'])/sum(distW['dist_20'])
plog('--------------------------------------scores-------------------------------')
def clampF(x):
    return pd.Series({"r_20"  :sp.stats.pearsonr(x['corr_20'] ,x['ref'])[0]
                      ,"s_ref":sum(x['ref']),"s_20" :sum(x['corr_20'])
    })

scorHL = mist.groupby( 'id_clust').apply(clampF).reset_index()
scorHW = mistW.groupby('id_clust').apply(clampF).reset_index()
scorDL = dist.groupby( 'id_clust').apply(clampF).reset_index()
scorDW = distW.groupby('id_clust').apply(clampF).reset_index()

for g in [scorHL,scorHW,scorDL,scorDW]:
    g.loc[:,"d_20"] = (g['s_20'] - g['s_ref'])/g['s_ref']

dH = scorHL.columns[[bool(re.search('^d_',x)) for x in scorHL.columns]]

scor = pd.DataFrame({"id_clust":scorHL['id_clust']
                     ,"r_20_h_l":scorHL['r_20'],"r_20_h_w":scorHW['r_20']
                     ,"r_20_d_l":scorDL['r_20'],"r_20_d_w":scorDW['r_20']
                     ,"d_20_h_l":scorHL['d_20'],"d_20_h_w":scorHW['d_20']
                     ,"d_20_d_l":scorDL['d_20'],"d_20_d_w":scorDW['d_20']
                     ,"s_ref_h_l":scorHL['s_ref'],"s_ref_h_w":scorHW['s_ref']
                     ,"s_ref_d_l":scorDL['s_ref'],"s_ref_d_w":scorDW['s_ref']})

if False:
    shl.kpiDis(scor,tLab="all days (h)",col_cor="r_20_h_l",col_dif="d_20_h_l",col_sum="s_ref_h_l")
    shl.kpiDis(scor,tLab="working  (h)",col_cor="r_20_h_w",col_dif="d_20_h_w",col_sum="s_ref_h_w")
    shl.kpiDis(scor,tLab="all days (d)",col_cor="r_20_d_l",col_dif="d_20_d_l",col_sum="s_ref_d_l")
    shl.kpiDis(scor,tLab="working  (d)",col_cor="r_20_d_w",col_dif="d_20_d_w",col_sum="s_ref_d_w")

if False:
    tist = distW[['day','ref','dist_20','corr_20']].groupby('day').agg(sum).reset_index()
    plt.plot(tist['ref'],label="ref")
    plt.plot(tist['dist_20'],label="raw")
    plt.plot(tist['corr_20'],label="corr_20")
    plt.legend()
    plt.show()
    print(sp.stats.pearsonr(tist['ref'],tist['corr_20']))

cellL = pd.read_csv(baseDir + "raw/tank/cilac_weight.csv")
plog('-------------------------------------direction-count-------------------------------')
##check joins!!!!
dirG = pd.read_csv(baseDir + "raw/tank/dir_count_out.csv")
if True: ##correction factor
    dirD = dirG[['id_clust','dir_count']].groupby(["id_clust"]).agg(np.mean).reset_index()
    dirD.loc[:,"id_poi"] = pd.merge(dirD,poi[['id_clust','id_poi']],on="id_clust",how="left")['id_poi']
    histD = pd.read_csv(baseDir + "raw/tank/capture_rate.csv")
    histD = histD[histD['id_poi_x'] == histD['id_poi_x']]
    histD = histD.groupby('id_poi_x').first().reset_index()
    histD.loc[:,"dir_count"] = pd.merge(histD,dirD,left_on="id_clust",right_on="id_clust",how="left")['dir_count']
    corrF = np.median(histD['count']/histD['dir_count'])
    dirG.loc[:,"dir_count"] = dirG['dir_count']*corrF
distC = distW[['id_clust','corr_20','ref','ref_raw']].groupby("id_clust").agg(np.mean).reset_index()
distT = pd.merge(dist,dirG,on=["id_clust","day"],how="left")
distT = pd.merge(distT,poi[['id_clust','bast']],on="id_clust",how="left")
distT = distT[distT['ref'] == distT['ref']]
distT.loc[:,"people_car"] = distT['dir_count']/distT['bast']*0.5
distT.loc[:,"capture_rate"] = distT['corr_20']/distT['dir_count']
distT.replace(float("Inf"),1.,inplace=True)
distT.replace(float("Nan"),0.,inplace=True)
distT.to_csv(baseDir + "raw/tank/visit_march_workday.csv",index=False)
distT.loc[:,'wday'] = distT['day'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
distW = distT[['id_clust','wday','dir_count']].groupby(["id_clust",'wday']).agg(np.mean).reset_index()
distD = distT[['id_clust','dir_count','capture_rate']].groupby("id_clust").agg(np.mean).reset_index()
distW = pd.merge(distW,distD,how="left",on="id_clust")
distW.loc[:,"day_fact"] = distW['dir_count_x'] / distW['dir_count_y']
distW = distW.pivot_table(index="id_clust",columns="wday",values="day_fact",aggfunc=np.mean)
distW.columns = ["fact_mon","fact_tue","fact_wed","fact_thu","fact_fri","fact_sat","fact_sun"]
distW = distW.reset_index()
distW = pd.merge(distW,distD,on="id_clust",how="left")
distW = pd.merge(distW,poi[['id_clust','bast','bast_we','bast_su','bast_fr']],on="id_clust",how="left")

mist.loc[:,"y_dif"] = (mist["corr_20"]-mist["ref"])/mist['corr_20']
mist.loc[:,"y_dif"] = mist['corr_20']/mist["ref"]
mist.replace(float("Inf"),1.,inplace=True)
mist.replace(float("-Inf"),1.,inplace=True)
def clampF(x):
    return pd.Series({"y_dif":np.mean(x['y_dif']),"s_dif":np.std(x['y_dif'])})
gist = mist[['id_clust','wday','y_dif']].groupby(['id_clust','wday']).apply(clampF).reset_index()
gist.loc[:,"b_sect"], _ = tlib.binVector(gist['y_dif'],nBin=5,threshold=5)
clit = gist[['id_clust','b_sect']].groupby(["b_sect"]).first().reset_index()
distW = pd.merge(distW,clit,on="id_clust",how="left")
distW.to_csv(baseDir + "raw/tank/weekday_dir_correction.csv",index=False)
gist.to_csv(baseDir + "raw/tank/weekday_act_correction.csv",index=False)


if False:
    plt.hist(clit['b_sect'],bins=6)
    plt.show()

    gist.boxplot(column="y_dif",by="wday")
    gist.boxplot(column="y_dif",by="b_sect")
    plt.ylim(0,3)
    plt.show()


if False:
    distW.boxplot()
    plt.title("Direction counts vs weekday")
    plt.xlabel("weekday")
    plt.ylabel("correction factor")
    plt.show()

scor = pd.merge(scor,poi[['id_clust','bast','bast_light','bast_heavy']],on="id_clust",how="left")
scor.loc[:,"dir_count"] = pd.merge(scor,dirD,on="id_clust",how="left")['dir_count']
scor = pd.merge(scor,distC,on="id_clust",how="left")
scor.loc[:,"people_car"] = scor['dir_count']/scor['bast']
scor.loc[:,"capture_rate"] = scor['corr_20']/scor['dir_count']
scor.replace(float("Inf"),1.,inplace=True)
scor.replace(float("Nan"),0.,inplace=True)
scor.to_csv(baseDir + "raw/tank/kpi_march.csv",index=False)

print("kpi summary")
print("corr > 0.6 hourly - all days : %f" % (scor[scor['r_20_h_l']>0.6].shape[0]/scor.shape[0]))
print("corr > 0.6 hourly - work days: %f" % (scor[scor['r_20_h_w']>0.6].shape[0]/scor.shape[0]))
print("corr > 0.6 daily  - all days : %f" % (scor[scor['r_20_d_l']>0.6].shape[0]/scor.shape[0]))
print("corr > 0.6 daily  - work days: %f" % (scor[scor['r_20_d_w']>0.6].shape[0]/scor.shape[0]))
print("diff < 0.2 hourly - all days : %f" % (scor[scor['d_20_h_l'].abs()<0.2].shape[0]/scor.shape[0]))
print("diff < 0.2 hourly - work days: %f" % (scor[scor['d_20_h_w'].abs()<0.2].shape[0]/scor.shape[0]))
print("diff < 0.2 daily  - all days : %f" % (scor[scor['d_20_d_l'].abs()<0.2].shape[0]/scor.shape[0]))
print("diff < 0.2 daily  - work days: %f" % (scor[scor['d_20_d_w'].abs()<0.2].shape[0]/scor.shape[0]))
print("people_car 1<>2 work days: %f" % (scor[(scor['people_car']>1.0)&(scor['people_car']<2.0)].shape[0]/scor.shape[0]))
print("capture rate 0.1%%<>9%% work days: %f" % (scor[(scor['capture_rate']>0.001)&(scor['capture_rate']<0.07)].shape[0]/scor.shape[0]))

if True: #correct weights
    corrF = mist
    corrF.loc[:,"corr_reg"] = corrF['dist_20']/corrF['corr_20']
    corrF.loc[:,"corr_ref"] = corrF['dist_20']/corrF['ref']
    corrF.replace(float("Inf"),1,inplace=True)
    corrF = corrF[['id_clust','id_poi','corr_reg','corr_ref']].groupby(['id_clust','id_poi']).agg(np.nanmean).reset_index()
    corrF.loc[:,"zone"] = corrF["id_clust"].apply(lambda x: x.split("-")[0])
    map10 = e_m.map2df(json.load(open(baseDir + "/raw/tank/tank_und_rast_"+dTh+".json")))
    map10 = pd.merge(map10,corrF,on="zone",how="left")
    map10.replace(float("Nan"),1,inplace=True)
    map10.loc[:,"weight_bkp"] = map10['weight']
    map10.loc[:,"weight"] = map10['weight']*map10['corr_reg']
    if True:
        poiR = pd.read_csv(baseDir + "raw/tank/poi_new.csv")
        cells = pd.read_csv(baseDir + "raw/centroids.csv.tar",compression="gzip")
        max_d = 5000./gradMeter
        max_nei = 20
        cellL = pd.DataFrame()
        id_zone = max(poi['id_zone'])
        for i,c in poiR.iterrows():
            x_c, y_c = c['x'],c['y']
            disk = ((cells['X']-x_c)**2 + (cells['Y']-y_c)**2)
            dist = max_d
            disk = disk.loc[disk <= dist**2]
            if disk.shape[0] > max_nei:
                disk = disk.sort_values()
                disk = disk.head(max_nei)
            tmp = cells.loc[disk.index]
            tmp.loc[:,"zone"] = id_zone
            tmp.loc[:,"id_poi"] = c['id_poi']
            id_zone = id_zone + 1
            cellL = pd.concat([cellL,tmp],axis=0)
        cellL = cellL.groupby('cilac').head(1)
        cellL = e_m.getMarketShare(cellL)
        cellL.loc[:,"weight"] = 1.
        del cellL['X'], cellL['Y'], cellL['tech']
        map10 = pd.merge(map10,cellL,on=["cilac","zone","market_share","weight"],how="outer")
    
    e_m.writeJson(e_m.df2Map(map10),baseDir + "/raw/tank/tank_"+dTh+"_reweighted.json")
    #map10.loc[:,"zone"] = map10[['zone','id_poi']].apply(lambda x: str(x[0]) + "-" + str(int(x[1])),axis=1)

if False:
    plt.plot(map10['corr_reg'],label="reg")
    plt.plot(map10['corr_ref'],label="ref")
    plt.plot(map10['weight'],label="weight")
    plt.legend()
    plt.show()

    plt.hist(map10['weight'],bins=20)
    plt.show()
    
        
