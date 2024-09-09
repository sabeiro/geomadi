#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as l_c
import geomadi.series_lib as s_l
import geomadi.train_shapeLib as shl
import geomadi.train_lib as tlib
import etl.etl_mapping as e_m
import seaborn as sns
import geomadi.train_filter as t_f

def plog(text):
    print(text)

import importlib
importlib.reload(s_l)
importlib.reload(l_c)
##spread with cumulative number of guests
opsS = json.load(open(baseDir + "src/conf/tank_ops.json"))
ops = opsS['ops_start']
for i in opsS['ops']['play_blind'].keys():
    ops[i] = opsS['ops']['play_blind'][i]

dTh = '20'
if len(sys.argv) > 1:
    dTh = str(sys.argv[1])

if ops['f_etl']:  # etl_tankYear.py
    mist = pd.read_csv(baseDir + "raw/tank/act_vist_year.csv.gz",compression="gzip") 
    mist.loc[:,"day"]  = mist['time'].apply(lambda x:x[0:10])
    mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
elif ops['f_timeSeries']: # etl_tankTimeSeries.py
    mist = pd.read_csv(baseDir + "raw/tank/act_gauss_year.csv.gz",compression="gzip")
    mist.loc[:,"time"] = mist[['day','hour']].apply(lambda x: "%sT%02d:00:00" %(x[0],int(x[1])),axis=1)
elif ops['f_delivery']:
    mist = pd.read_csv(baseDir + "raw/tank/delivery/act_vist_final.csv.zip",compression="zip")
    mist.loc[:,"day"]  = mist['time'].apply(lambda x:x[0:10])
    mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
    mist = pd.merge(mist,poi[['id_poi','id_clust']],on="id_poi",how="left")
elif ops['f_play']: # etl_tankTimeSeries.py
    mist = pd.read_csv(baseDir + "raw/tank/act_vist_play.csv.gz",compression="gzip")
    mist.loc[:,"time"] = mist[['day','hour']].apply(lambda x: "%sT%02d:00:00" %(x[0],int(x[1])),axis=1)
else:  # regression_tankYear.py
    mist = pd.read_csv(baseDir + "raw/tank/act_vist_reg_year.csv.gz",compression="gzip")
    mist.loc[:,"time"] = mist[['day','hour']].apply(lambda x: "%sT%02d:00:00" %(x[0],int(x[1])),axis=1)

mist.loc[:,"wday"] = mist['day'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").weekday())

poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi = poi[poi['competitor'] == 0]
poi = poi.groupby('id_clust').first().reset_index()
map10 = e_m.map2df(json.load(open(baseDir + "/raw/tank/tank_und_rast_"+str(dTh)+".json")))
zmap = map10[['zone','weight']].groupby('zone').agg(len).reset_index()
zmap.loc[:,"zone"] = zmap['zone'].astype(int)
poi.loc[:,"n_cell"] = pd.merge(poi,zmap,left_on="id_zone",right_on="zone",how="left")['weight']
gvist = mist[['id_clust','act']].groupby('id_clust').agg(len).reset_index()
poi.loc[:,"c_source"] = pd.merge(poi,gvist,on="id_clust",how="left")['act']

if ops['time_filter']:
    dateL = pd.read_csv(baseDir + "raw/tank/dateList.csv")
    mist = mist[mist['day'].isin(dateL['day'][dateL['use']==2])]

if ops['rem_anomaly']:
    poiAr = pd.read_csv(baseDir + "raw/tank/poi_anomaly_ref.csv")
    poiAa = pd.read_csv(baseDir + "raw/tank/poi_anomaly_act.csv")
    mist = mist[~mist['id_clust'].isin(poiAr['id'])]
    mist = mist[~mist['id_clust'].isin(poiAa['id'])]

if ops['weekday_correction']:
    plog("weekday-poi correction")
    mist.loc[:,"act_raw"] = mist['act']
    dateT = dateL[dateL['use'] == 1]
    mist.loc[:,"y_dif"] = mist["ref"]/mist['act']
    mist.replace(float("Inf"),1.,inplace=True)
    mist.replace(float("-Inf"),1.,inplace=True)
    def clampF(x):
        return pd.Series({"y_dif":np.mean(x['y_dif']),"s_dif":np.std(x['y_dif'])})
    mist1 = mist[mist['day'].isin(dateT['day'])]
    gist = mist1[['id_clust','wday','y_dif']].groupby(['id_clust','wday']).apply(clampF).reset_index()
    #gist = pd.read_csv(baseDir + "raw/tank/weekday_act_correction.csv")
    del mist['y_dif']
    mist = pd.merge(mist,gist,on=["id_clust","wday"],how="left")
    mist.replace(np.nan,1.,inplace=True)
    mist.loc[:,"act"] = mist['act_raw']*mist['y_dif']
else:
    mist.loc[:,"act_raw"] = mist['act']

print("total locations %d:" % (len(set(mist['id_clust']))) )
isSingle = False
mistW = mist[mist['wday'] < 4]
if ops['hour_smoothing']:
    plog("applying smoothing")
    for g in [mist,mistW]:
        g.loc[:,"act"] = l_c.smoothClust(g['act'],g['id_clust'],width=5,steps=9,isSingle=ops['isSingle'])
        g.loc[:,"ref"] = l_c.smoothClust(g['ref'],g['id_clust'],width=5,steps=7,isSingle=ops['isSingle'])
if ops['hour_regressor']:
    plog("applying regressor")
    for g in [mist,mistW]:
        g.loc[:,"act"] = g['act_raw']*l_c.regressorTank(g,poi,idFloat="act")

dist  = mist[ ['id_clust','day','ref','act','act_raw']].groupby(['id_clust','day']).agg(sum).reset_index()
distW = mistW[['id_clust','day','ref','act','act_raw']].groupby(['id_clust','day']).agg(sum).reset_index()
for g in [mist,mistW,dist,distW]:
    g.loc[:,"ref_raw"] = g['ref']

if ops['day_smoothing']:
    plog("applying smoothing")
    for g in [dist,distW]:
        g.loc[:,"act"] = l_c.smoothClust(g['act'],g['id_clust'],width=3,steps=5,isSingle=ops['isSingle'])
        g.loc[:,"act"] = l_c.smoothClust(g['act'],g['id_clust'],width=3,steps=5,isSingle=ops['isSingle'])
        g.loc[:,"ref"] = l_c.smoothClust(g['ref'],g['id_clust'],width=3,steps=5,isSingle=ops['isSingle'])
    
if ops["day_regressor"]:
    plog("applying regressor")
    for g in [dist,distW]:
        g.loc[:,"act"] = g['act_raw']*l_c.regressorTank(g,poi,idFloat="act")
    
plog('--------------------------------------scores-------------------------------')
def clampF(x):
    return pd.Series({"r_20"  :sp.stats.pearsonr(x['act'],x['ref'])[0]
                      ,"s_ref":sum(x['ref']),"s_20" :sum(x['act'])
    })

scorHL = mist.groupby( 'id_clust').apply(clampF).reset_index()
scorHW = mistW.groupby('id_clust').apply(clampF).reset_index()
scorDL = dist.groupby( 'id_clust').apply(clampF).reset_index()
scorDW = distW.groupby('id_clust').apply(clampF).reset_index()
scorHA = mist.groupby( 'day').apply(clampF).reset_index()

cist = mist.groupby('id_clust').agg(len).reset_index()
cist.sort_values("time",inplace=True)

for g in [scorHL,scorHW,scorDL,scorDW]:
    g.loc[:,"d_20"] = (g['s_20'] - g['s_ref'])/g['s_ref']
    g.loc[:,"n_h"] = pd.merge(g,cist,on="id_clust",how="left")['time']
    g.sort_values('r_20',inplace=True)

scor = pd.DataFrame({"id_clust":scorHL['id_clust']
                     ,"r_20_h_l":scorHL['r_20'],"r_20_h_w":scorHW['r_20']
                     ,"r_20_d_l":scorDL['r_20'],"r_20_d_w":scorDW['r_20']
                     ,"d_20_h_l":scorHL['d_20'],"d_20_h_w":scorHW['d_20']
                     ,"d_20_d_l":scorDL['d_20'],"d_20_d_w":scorDW['d_20']
                     ,"s_ref_h_l":scorHL['s_ref'],"s_ref_h_w":scorHW['s_ref']
                     ,"s_ref_d_l":scorDL['s_ref'],"s_ref_d_w":scorDW['s_ref']})

print("kpi summary %d locations" % scor.shape[0])
print("corr > 0.6 hour  - all days : %f" % (scor[scor['r_20_h_l']>0.6].shape[0]/scor.shape[0]))
print("corr > 0.6 hour  - work days: %f" % (scor[scor['r_20_h_w']>0.6].shape[0]/scor.shape[0]))
print("corr > 0.6 daily - all days : %f" % (scor[scor['r_20_d_l']>0.6].shape[0]/scor.shape[0]))
print("corr > 0.6 daily - work days: %f" % (scor[scor['r_20_d_w']>0.6].shape[0]/scor.shape[0]))
print("weeks %f" % (len(np.unique(dist['day']))/7.) )
dist.loc[:,"dif_ref"] = abs((dist['ref'] - dist['ref_raw'])/(dist['ref'] + dist['ref_raw']))
print("smoothing difference: %.2f" % (100*dist['dif_ref'].mean()))
scor = pd.merge(scor,poi[['id_clust','x','y','daily_visit',"id_poi"]],on="id_clust",how="left")

if ops['f_etl']:  # etl_tankYear.py
    scor.to_csv(baseDir + "raw/tank/scor_raw.csv",index=False)
elif ops['f_timeSeries']: # etl_tankTimeSeries.py
    scor.to_csv(baseDir + "raw/tank/scor_reg.csv",index=False)
elif ops['f_delivery']:
    scor.to_csv(baseDir + "raw/tank/scor_delivery.csv",index=False)
elif ops['f_play']:
    scor.to_csv(baseDir + "raw/tank/scor_play_smooth.csv",index=False)
else:  # regression_tankYear.py
    scor.to_csv(baseDir + "raw/tank/scor_smooth.csv",index=False)

fist  = mist[['id_clust','day','ref','act','act_raw']].groupby(['id_clust','day']).agg(sum).reset_index()
fist.loc[:,"corr_f"] = fist['act']/dist['act']
mist.loc[:,"corr_f"] = pd.merge(mist,fist,on=["day","id_clust"],how="left")['corr_f']
mist.loc[:,"act"] = mist['act']*mist['corr_f']
del mist['corr_f']
fist.loc[:,"corr_f"] = fist['ref']/dist['ref']
mist.loc[:,"corr_f"] = pd.merge(mist,fist,on=["day","id_clust"],how="left")['corr_f']
mist.loc[:,"ref"] = mist['ref']*mist['corr_f']
poiId = poi[['id_poi','id_clust']]
mist = pd.merge(mist,poiId,on=["id_clust"],how="left",suffixes=["_x",""])

if False:
    print("diff < 0.2 hour  - all days : %f" % (scor[scor['d_20_h_l']<0.2].shape[0]/scor.shape[0]))
    print("diff < 0.2 hour  - work days: %f" % (scor[scor['d_20_h_w']<0.2].shape[0]/scor.shape[0]))
    print("diff < 0.2 daily - all days : %f" % (scor[scor['d_20_d_l']<0.2].shape[0]/scor.shape[0]))
    print("diff < 0.2 daily - work days: %f" % (scor[scor['d_20_d_w']<0.2].shape[0]/scor.shape[0]))

if ops['plot_corr']:
    plt.plot(scorHA['day'],scorHA['r_20'])
    plt.show()

if ops['plot_line']:
    clustL = np.unique(mist['id_clust'])
    clustL = [clustL[np.random.randint(len(clustL))]]
    i = 2
    aggF = 'time'
    timeF = "%Y-%m-%dT%H:%M:%S"
    #clustS = '266-1'
    if i == 1:
        aggF = 'day'
        timeF = "%Y-%m-%d"
        tist = dist.groupby(aggF).agg(sum).reset_index()
    else:
        tist = mist.groupby(aggF).agg(sum).reset_index()
    for clustS in clustL:
        if i == 2:
            aggF = 'day'
            tist = dist[dist['id_clust'] == clustS]
        else :
            clustS = "all"
        corV = sp.stats.pearsonr(tist['act'] ,tist['ref'])[0]
        corR = sp.stats.pearsonr(tist['act_raw'] ,tist['ref_raw'])[0]
        tist.sort_values(aggF,inplace=True)
        t = range(tist.shape[0])
        #t = [datetime.datetime.strptime(x,timeF) for x in tist[aggF]]
        plt.plot(t,tist['ref_raw'],label="vis")
        plt.plot(t,tist['act_raw'],label="act")
        plt.plot(t,tist['act'],label="act_cor")
        plt.plot(t,tist['ref'],label="vis_cor")
        plt.xlabel("time")
        plt.ylabel("count")
        plt.title("id_clust %s corr _ raw: %.2f smooth: %.2f" % (clustS,corR,corV) )
        plt.legend()
        plt.show()

if ops['remap']: #correct weights
    plog('----------------------reweight-mapping----------------------')
    gradMeter = 111122.19769899677
    corrF = mist
    corrF.loc[:,"corr_reg"] = corrF['act_raw']/corrF['act']
    corrF.loc[:,"corr_ref"] = corrF['act_raw']/corrF['ref']
    corrF.replace(float("Inf"),1,inplace=True)
    corrF = corrF[['id_clust','corr_reg','corr_ref']].groupby(['id_clust']).agg(np.nanmean).reset_index()
    corrF.loc[:,"zone"] = corrF["id_clust"].apply(lambda x: x.split("-")[0])
    map10 = e_m.map2df(json.load(open(baseDir + "/raw/tank/tank_und_rast_"+dTh+".json")))
    map10 = pd.merge(map10,corrF,on="zone",how="left")
    map10.replace(float("Nan"),1,inplace=True)
    map10.loc[:,"weight_bkp"] = map10['weight']
    map10.loc[:,"weight"] = map10['weight']*map10['corr_reg']
    e_m.writeJson(e_m.df2Map(map10),baseDir + "/raw/tank/tank_"+dTh+"_reweighted.json")
    #map10.loc[:,"zone"] = map10[['zone','id_poi']].apply(lambda x: str(x[0]) + "-" + str(int(x[1])),axis=1)

mist[['time','id_poi','act','ref']].to_csv(baseDir + "raw/tank/act_vist_final.csv.gz",index=False,compression="gzip")

#mist = pd.read_csv(baseDir + "raw/tank/act_vist_final.csv.gz",compression="gzip")
#mist[['day','hour','id_clust','act','ref']].to_csv(baseDir + "raw/tank/act_vist_final.csv",index=False)

if ops['boxplot']:
    scorHL.loc[:,"b_h"], _ = t_f.binOutlier(scorHL['n_h'],nBin=6,threshold=0.3)
    scorHL.loc[:,"b_s"], _ = t_f.binOutlier(scorHL['s_20'],nBin=6,threshold=0.3)
    print(scorHL[['b_h','id_clust']].groupby("b_h").agg(len).reset_index())
    print(scorHL[['b_s','id_clust']].groupby("b_s").agg(len).reset_index())
    scorHL.boxplot(column="r_20",by="b_h")
    plt.show()
    scorHL.boxplot(column="r_20",by="b_s")
    plt.show()

if ops['plot_kpi']: ## KPI plots
    shl.kpiDis(scor,tLab="all days (h)",col_cor="r_20_h_l",col_dif="d_20_h_l",col_sum="s_ref_h_l")
    shl.kpiDis(scor,tLab="working  (h)",col_cor="r_20_h_w",col_dif="d_20_h_w",col_sum="s_ref_h_w")
    shl.kpiDis(scor,tLab="all days (d)",col_cor="r_20_d_l",col_dif="d_20_d_l",col_sum="s_ref_d_l")
    shl.kpiDis(scor,tLab="working  (d)",col_cor="r_20_d_w",col_dif="d_20_d_w",col_sum="s_ref_d_w")

if ops['month_val']:
    poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
    mist = pd.read_csv(baseDir + "raw/tank/visit_march_melted.csv.gz",compression="gzip")
    mist.loc[:,"day"] = mist['time'].apply(lambda x:x[0:10])
    mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
    distW = pd.read_csv(baseDir + "raw/tank/weekday_correction.csv")
    mist = pd.merge(mist,distW,on="id_clust",how="left",suffixes=["","_y"])
    dist  = mist[ ['id_clust','day','ref','bon','dist_raw']].groupby(['id_clust','day']).agg(sum).reset_index()
    corMat = distW[['id_clust','bast','bast_we','bast_su','bast_fr','fact_sun','fact_fri','dir_count']].cor()
    cors = corMat.sum(axis=0)
    cor_order = cors.argsort()[::-1]
    corMat = corMat.loc[cor_order.index,cor_order.index]
    if False:
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(corMat, vmax=1, square=True,annot=True,cmap='RdYlGn')
        plt.title('Correlation matrix between the features')
        plt.show()

    dist.loc[:,"wday"] = dist['day'].apply( lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
    mist.loc[:,"wday"] = mist['day'].apply( lambda x: datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
    dist.loc[:,"y_dif"] = (dist['ref'] - dist['act'])#/dist['dist_raw']
    mist.loc[:,"y_dif"] = (mist['ref'] - mist['act'])#/mist['dist_raw']

    def clampF(x):
        return pd.Series({"y_dif":np.mean(x['y_dif']),"s_dif":np.std(x['y_dif'])})

    gist = mist[['id_clust','wday','y_dif']].groupby(['id_clust','wday']).apply(clampF).reset_index()
    Z = linkage(poi[['x','y']], 'ward')
    gradMeter = 111122.19769899677
    distW = pd.read_csv(baseDir + "raw/tank/weekday_correction.csv")
    distW = pd.merge(distW,poi[['id_clust','chirality','pop_dens','submask','type']],on="id_clust",how="left")


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
