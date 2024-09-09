#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as t_l
import geomadi.series_lib as s_l
import geomadi.train_lib as tlib
import etl.etl_mapping as e_m
import geomadi.train_filter as t_f
import geomadi.train_shapeLib as t_s
import geomadi.train_keras as t_k
import custom.lib_custom as l_c

import importlib
importlib.reload(s_l)
importlib.reload(t_l)
importlib.reload(t_s)
importlib.reload(l_c)

isPseudo = False
poiM = pd.read_csv(baseDir + "raw/mc/poi_mc.csv")
poiM = poiM[poiM['Business T'] == 'Autobahnraststaette']
poiT = pd.read_csv(baseDir + "raw/tank/poi.csv")
poiT = poiT[poiT['competitor'] == 0]

mist = pd.read_csv(baseDir + "raw/tank/ref_vist_h.csv.gz")
sact = mist.pivot_table(index=["id_poi"],columns="time",values="ref",aggfunc=np.sum).reset_index()
colL = list(sact.columns[1:])
if isPseudo: #pseudo day
    colL = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in colL]
    colL = ["%02d-%02d-%02d" % (x.isocalendar()[1],x.weekday(),x.hour) for x in colL]
sact.columns = ['id_tank'] + colL
sact = sact.groupby("id_tank").first().reset_index()

cast = pd.read_csv(baseDir + "raw/tank/castor_tile_count.csv")
cast.loc[:,"time"] = cast['rp_time'].apply(lambda x: "%sT%02d:00:00" % (x[:10],(int(x[11:13])+2)%24))
casT = pd.read_csv(baseDir + "raw/tank/castor_tile.csv")
cast = pd.merge(cast,casT,left_on="intersection_id",right_on="tile_id",how="left")
cast = cast[['id_poi','time','cnt']]
cact = cast.pivot_table(index="id_poi",columns="time",values="cnt",aggfunc=np.sum).reset_index()

act = pd.read_csv(baseDir + "raw/tank/act_compare.csv.gz",compression="gzip")
act.loc[:,"id_poi"] = act['id_clust'].apply(lambda x: int(x.split("-")[0]))
act.loc[:,"chirality"] = act['id_clust'].apply(lambda x: int(x.split("-")[1]))
act = pd.merge(act,poiT[['id_poi','id_clust','chirality']],on="id_poi",how="left")
act = act[act['chirality_x'] == act['chirality_y']]
act.sort_values(['id_poi','time'],inplace=True)
pact = act.pivot_table(index="id_poi",columns="time",values="count",aggfunc=np.sum).reset_index()

if False:
    bast = pd.read_csv(baseDir + "log/bast_2016.csv.tar.gz",compression="gzip")
    bast.loc[:,"count"] = bast["vehicle_r1"] + bast["vehicle_r2"]
    bast = bast[bast['id'].isin(poiT['id_bast'])]
    bast.loc[:,"time"] = bast[['day','hour']].apply(lambda x: "%sT%02d:00:00" % (x[0],x[1]-1),axis=1)
    bact = bast.pivot_table(index="id",columns="time",values="count",aggfunc=np.sum).reset_index()
    colL = list(bact.columns[1:])
    if isPseudo: #pseudo day
        colL = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in colL]
        colL = ["%02d-%02d-%02d" % (x.isocalendar()[1],x.weekday(),x.hour) for x in colL]
    mact = bact.copy()
    mact = pd.merge(mact,poiT[['id_poi','id_bast']],left_on="id",right_on="id_bast",how="left")
    hL = [x for x in mact.columns if bool(re.search("T",x))]
    mact = mact[ ['id_poi'] + hL ]
    mact.columns = ['id_tank'] + list(mact.columns[1:])
    
else:
    gradMeter = 111122.19769899677
    max_d = 50000./gradMeter
    matchL = []
    for i,c in poiM.iterrows():
        disk = ((poiT['x']-c['X-Position'])**2 + (poiT['y']-c['Y-Position'])**2)
        diskI = [(x==disk.min()) for x in disk]
        matchL.append({"id_mc":c['Sitenummer'],"id_tank":poiT[diskI]['id_poi'].values[0],"dist":disk.min()})

    matchL = pd.DataFrame(matchL)
    matchL.loc[:,"dist"] = matchL['dist']/gradMeter
    matchL.sort_values("dist",inplace=True)
#    matchL = matchL[matchL['dist'] < 1e-10 ]
    matchL = matchL.groupby(["id_tank","id_mc"]).first().reset_index()

    mact = pd.read_csv(baseDir + "raw/mc/act_mc_visit.csv.tar.gz",compression="gzip")
    colL = list(mact.columns[1:])
    if isPseudo: #speudo day
        colL = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in colL]
        colL = ["%02d-%02d-%02d" % (x.isocalendar()[1],x.weekday(),x.hour) for x in colL]
    mact.columns = ['id_mc'] + colL
    mact = pd.merge(mact,matchL,on="id_mc",how="left")
    mact = mact[mact['id_tank'].isin(sact['id_tank'])]
    mact = mact.groupby("id_tank").first().reset_index()
    sact = sact[sact['id_tank'].isin(matchL['id_tank'])]

hL = list(mact.columns[mact.columns.isin(sact.columns)])[1:]
if True:
    hL = list(cact.columns[cact.columns.isin(hL)])
    hL = list(pact.columns[pact.columns.isin(hL)])
    hL = [x for x in hL if not bool(re.search('2017-05-06',x))]
    hL = [x for x in hL if not bool(re.search('2017-05-22',x))]
dL = list(np.unique([x[:11] for x in hL]))
    
mact = mact[ ['id_tank'] + hL ]
sact = sact[ ['id_tank'] + hL ]
mact.loc[:,"id_tank"] = mact['id_tank'].astype(int)
sact.loc[:,"id_tank"] = sact['id_tank'].astype(int)
sact.index = range(sact.shape[0])
mact.index = range(mact.shape[0])
mact = mact.groupby('id_tank').first().reset_index()

print(mact.shape)
print(sact.shape)

nact = pd.read_csv(baseDir + "raw/tank/mc_neutral.csv.gz",compression="gzip")
nact = nact.pivot_table(index="id_zone",columns="time",values="count",aggfunc=np.sum).reset_index()
nact.replace(float('nan'),0,inplace=True)

corrL = []
for i,g in sact.iterrows():
    if sum(np.isnan(g[hL])) > 10:
        continue
    ts1 = pd.DataFrame({"ref":g[hL]})
    ts2 = pd.DataFrame({"rest":mact.loc[i,hL]})
    ts3 = pd.DataFrame({"cast":cact.loc[i,hL]})
    ts4 = pd.DataFrame({"pact":pact.loc[i,hL]})
    ts5 = pd.DataFrame({"nact":nact.loc[i,dL]})
    ts1.loc[:,"time"] = ts1.index
    ts2.loc[:,"time"] = ts2.index
    ts3.loc[:,"time"] = ts2.index
    ts4.loc[:,"time"] = ts4.index
    ts5.loc[:,"time"] = [x[:10] for x in ts5.index]
    if isPseudo:
        for t in [ts1,ts2,ts3,ts4]:
            t.loc[:,"day"] = t['time'].apply(lambda x: x[:5])
            t.loc[:,"hour"] = t['time'].apply(lambda x: x[6:8])
    else:
        for t in [ts1,ts2,ts3,ts4]:
            t.loc[:,"day"] = t['time'].apply(lambda x: x[:10])
            t.loc[:,"hour"] = t['time'].apply(lambda x: x[11:13])
    if False: ##6 - 20
        for t in [ts1,ts2,ts3,ts4]:
            t = t[(t['hour'] >= '06') & (t['hour'] <= "20")]
    tp1 = ts1.pivot_table(index=["day"],columns="hour",values="ref",aggfunc=np.sum).reset_index()
    missL = tp1.isnull().sum(axis=1) <= 9
    tp1 = tp1[missL]
    tp1.replace(float('nan'),0,inplace=True)
    tp2 = ts2.pivot_table(index=["day"],columns="hour",values="rest",aggfunc=np.sum).reset_index()
    tp2 = tp2[missL]
    tp2.replace(float('nan'),0,inplace=True)
    tp3 = ts3.pivot_table(index=["day"],columns="hour",values="cast",aggfunc=np.sum).reset_index()
    tp3 = tp3[missL]
    tp3.replace(float('nan'),0,inplace=True)
    tp4 = ts4.pivot_table(index=["day"],columns="hour",values="pact",aggfunc=np.sum).reset_index()
    tp4 = tp4[missL]
    tp4.replace(float('nan'),0,inplace=True)
    hL1 = tp1.columns[1:]
    tpA = pd.DataFrame({"tank":tp1[hL1].values.ravel(),"mc":tp2[hL1].values.ravel(),"cast":tp3[hL1].values.ravel(),"act":tp4[hL1].values.ravel()}).corr()
    tpAd = pd.DataFrame({"tank":tp1[hL1].sum(axis=1).values,"mc":tp2[hL1].sum(axis=1).values,"cast":tp3[hL1].sum(axis=1).values,"act":tp4[hL1].sum(axis=1).values,"neut":ts5['nact'].values[missL]}).corr()
    corrL.append({"id_poi":g['id_tank']
                  ,"r_d_cast_mc":tpAd.iloc[1,2],"r_d_cast_tank":tpAd.iloc[1,4],"r_d_mc_tank":tpAd.iloc[2,4],"r_d_act_mc":tpAd.iloc[3,2],"r_d_act_tank":tpAd.iloc[3,4]
                  ,"r_h_cast_mc":tpA.iloc[1,2],"r_h_cast_tank":tpA.iloc[1,3],"r_h_mc_tank":tpA.iloc[2,3],"r_h_act_mc":tpA.iloc[0,2],"r_h_act_tank":tpA.iloc[0,3]
    })               
    if True:
        fig, ax = plt.subplots(1,2)
        ax[0].plot(tp1[hL1].values.ravel()/tp1[hL1].values.ravel().max(),label="service")
        ax[0].plot(tp2[hL1].values.ravel()/tp2[hL1].values.ravel().max(),label="restaurant")
        ax[0].plot(tp3[hL1].values.ravel()/tp3[hL1].values.ravel().max(),label="castor")
        ax[0].plot(tp4[hL1].values.ravel()/tp4[hL1].values.ravel().max(),label="pollux")
        ax[0].set_xlabel("hours")
        ax[0].set_ylabel("visits")
        ax[0].set_title("%d: hourly correlation %.2f" % (g['id_tank'],sp.stats.pearsonr(tp1[hL1].values.ravel(),tp4[hL1].values.ravel())[0]) )

        ax[1].plot(tp1['day'],tp1[hL1].sum(axis=1).values/tp1[hL1].sum().max(),label="service")
        ax[1].plot(tp2['day'],tp2[hL1].sum(axis=1).values/tp2[hL1].sum().max(),label="restaurant")
        ax[1].plot(tp3['day'],tp3[hL1].sum(axis=1).values/tp3[hL1].sum().max(),label="castor")
        ax[1].plot(ts5['time'].values[missL],ts5['nact'].values[missL]/ts5['nact'].values[missL].max(),label="pollux")
        # ax[1].plot(t1/t1.max(),label="service")
        # ax[1].plot(t2/t2.max(),label="restaurant")
        plt.xticks(rotation=45)
        ax[1].set_title("%d: daily correlation %.2f" % (g['id_tank'],sp.stats.pearsonr(tp1[hL1].sum(axis=1).values,ts5['nact'].values[missL])[0]) )
        plt.legend()
        plt.show()
    
corrL = pd.DataFrame(corrL)
corrL.to_csv(baseDir + "raw/tank/compare.csv",index=False)
print("corr > 0.6 day   - all days : %f" % (corrL[corrL['r_d_act_tank']>0.6].shape[0]/corrL.shape[0]))
print("corr > 0.6 hour  - all days : %f" % (corrL[corrL['r_h_act_tank']>0.6].shape[0]/corrL.shape[0]))

corrL = pd.merge(corrL,matchL,left_on="id_poi",right_on="id_tank",how="left")
corrL = pd.merge(corrL,poiT[['id_poi','type']],on="id_poi",how="left")
corrL = pd.merge(corrL,poiM[['Sitenummer','Siteart']],left_on="id_mc",right_on="Sitenummer",how="left")
corrL.to_csv(baseDir + "raw/mc/compare.csv")


if False:
    for j in range(10):
        i = sact['id_tank'].values[j]
        y1 = dist[dist['id_poi'] == i]['act']
        y2 = dist[dist['id_poi'] == i]['ref']
        plt.title("corr %f" % sp.stats.pearsonr(y1,y2)[0])
        plt.plot(y1)
        plt.plot(y2)
        plt.show()




print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
