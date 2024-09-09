import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import geomadi.series_stat as s_l
import geomadi.train_execute as t_e
import geomadi.train_score as t_s
import geomadi.train_model as tlib
import geomadi.train_shape as shl
import geomadi.train_reshape as t_r
import geomadi.train_viz as t_v
import custom.lib_custom as l_c
import pickle

custD = "mc"
custD = "tank"
idField = "id_poi"
version = "11u"

opsS = json.load(open(baseDir + "src/conf/tank_ops.json"))
ops = opsS['learn_play']['start']
for i in opsS['learn_play']['mc'].keys():
    ops[i] = opsS['learn_play']['mc'][i]

modDir = baseDir + "train/"+custD+"/"
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: str(x)+"T")
dateL.loc[:,"time"] = dateL["day"]
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
if custD == "tank":
    poi = poi[poi['competitor'] == 0]
poi = poi.groupby(idField).first().reset_index()
if custD == "tank":
    poi = poi[poi['use'] == 3]

if False:
    print('------------------radom days out-----------------------')
    dateL.loc[:,"use"] = 1
    shuffleL = random.sample(range(dateL.shape[0]),dateL.shape[0])
    dateL.loc[dateL.index[shuffleL][:30],"use"] = 2

mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip",dtype={idField:str})
ical = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",dtype={idField:str})
ical = t_r.isocal2day(ical,dateL)
gact = pd.read_csv(baseDir + "raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip",dtype={idField:str})
gact = gact[gact[idField].isin(poi.loc[poi['use'] == 3,idField])]
hL = gact.columns[1:]
#dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount/dirCount_d.csv.gz",compression="gzip",dtype={idField:str})
dirc = t_r.mergeDir(baseDir+"raw/"+custD+"/viaCount/")
tist = t_e.joinSource(gact,mist,how="outer",idField=idField,isSameTime=False)
tist = t_e.concatSource(tist,ical,how="left",idField=idField,varName="ical")
tist = t_e.concatSource(tist,dirc,how="left",idField=idField,varName="foot")
tist = tist[tist[idField].isin(poi.loc[poi['use'] == 3,idField])]
tist.loc[:,"foot"] = tist['foot']*0.4226623389831198
tist = tist[tist['time'] == tist['time']]
tist.loc[:,"day"] = tist['time'].apply(lambda x:x[:11])
tist = tist[tist["day"].isin(dateL["day"])]

tL = ['day','use','wday','cloudCover','humidity','visibility','Tmax']
tist = pd.merge(tist,dateL[tL],on="day",how="left")
tist = tist.replace(float('nan'),0)
print(tist.head(2))
print("merged locations %d" % len(set(gact[idField])))
print("sum of days %d" % (len(set(tist['day']))) )

if False:
    print('-------------check model autocompatibility-------------')
    tist.loc[:,"act"] = tist['ref']*(1.+np.random.randn(tist.shape[0])*.9)

if False: 
    print('------------small sample------------------------')
    tist = tist[tist[idField].isin(np.unique(tist[idField])[:3])]

if False:
    t_e.plotSum(tist,isLoc=False)
    plt.show()
    t_v.plotHistogram(tist['humidity'],label="correlation")
    plt.show()
    
scorM = t_s.scorPerf(tist,step="weighted",idField=idField)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
import importlib
importlib.reload(s_l)
importlib.reload(t_e)
importlib.reload(l_c)
importlib.reload(tlib)

clf = BaggingRegressor(DecisionTreeRegressor())
if custD == "mc":
    tL, sL = l_c.prepareMc(poi)
    tist.loc[:,"type"] = pd.merge(tist,poi,on=idField,how="left")['type2']
if custD == "tank":
    tL, sL = l_c.prepareTank(poi)
#tist.loc[:,"type"] = tist['type'].apply(lambda x: str(x).split(" ")[0])
if True:
    print('------------------------predict-absolute-value-----------------------------')
    y = tist[[idField,'ref']].groupby(idField).agg(np.mean)
    y.index = y.index.astype(str)
    X = poi[tL]
    X.index = poi[idField]
    X = X.loc[y.index]
    X = t_r.factorize(X)
    fit_s = clf.fit(X.values,y.values[:,0])
    fit_s, corL = tlib.regressorSingle(X.values,y.values[:,0],nXval=6,paramF="train.json")
    y_pred = fit_s.predict(X.values)
    corF = pd.DataFrame({idField:y.index.astype(str),"daily_visit":y_pred})
    print(sp.stats.pearsonr(y.values[:,0],y_pred))
    if False:
        plt.scatter(y_pred,y,marker="+")
        plt.plot(y,y,color="red")
        plt.xlabel("prediction")
        plt.ylabel("reference")
        plt.show()

#t_e.gridSearch(tist,idField,setTr,setVl)

if False:
    deli = pd.read_csv(baseDir + "raw/"+custD+"/delivery/act_ref_foot_19_02.csv")
    deli.loc[:,idField] = deli[idField].astype(str)
    scorD = t_s.scorPerf(deli,step="delivery",idField=idField)
    scorD = pd.merge(scorD,poi[[idField,'name','daily_visit']],on=idField,how="left")
    scorD.to_csv(baseDir + "raw/"+custD+"/delivery/act_ref_foot_19_02_scor.csv",index=False)

    iL = np.unique(tist[idField])
    i = iL[0]
    i = '1556'
    for i in iL:
        g = tist[tist[idField] == i]
        c = deli[deli[idField] == i]
        y1 = g.loc[[bool(re.search("-06-",x)) for x in g['day']],"ref"]
        y2 = g.loc[[bool(re.search("-02-",x)) for x in g['day']],"ref"]
        y3 = g['act']
        
        r = sp.stats.pearsonr(y3,y2)[0]
        d = 2.*np.sqrt( ((y3.values-y2.values)**2).sum())/(y3.values+y2.values).sum()
        plt.title("poi %s corr %.2f err %.2f" % (i,r,d))
        plt.plot(y3.values,label="act")
        plt.plot(range(len(y2)),y2,label="ref - february")
        #plt.plot(range(len(y1)),y1,label="ref - june")
        plt.legend()
        plt.xlabel("days")
        plt.ylabel("counts")
        plt.show()

importlib.reload(t_e)
vist, xist, kpiL, fitL = t_e.xvalRegressor(tist,idField,hL,sL)

print('-------------------save-models-and-output-------------------')
for f in fitL:
    fName = baseDir + "train/"+custD+"/prod/poi_" + f[idField] + ".pkl"
    pickle.dump(f['model'], open(fName, 'wb'))

pact = vist.pivot_table(index=idField,columns="day",values="act",aggfunc=np.sum).reset_index()
pact.to_csv(baseDir + "raw/"+custD+"/act_predict/act_predict_"+version+".csv.gz",compression="gzip",index=False)

# vist.loc[:,"time"] = vist['day']
# vist = t_e.concatSource(vist,dirc,how="left",idField=idField,varName="foot")
# vist = pd.merge(vist,poi[[idField,"daily_visit"]],on=idField,how="left")
# gist = vist.groupby(idField).agg(np.mean).reset_index()
# vist = pd.merge(vist,gist[[idField,"act"]],on=idField,how="left",suffixes=["","_y"])
# vist.loc["corr"] = vist["daily_visit"]/vist["act_y"]
# vist = t_e.concatSource(vist,gact,how="left",idField=idField,varName="weighted")
# vist.loc[:,"deli"] = vist['act']*.5 + vist['weighted']*.5
# vist.to_csv(baseDir+"raw/"+custD+"/act_ref_19_03.csv",index=False)
# scorR = t_s.scorPerf(vist,step="predict",idField=idField)
# scorR.to_csv(baseDir+"raw/"+custD+"/scor/act_ref_19_03.csv",index=False)

# rist = xist.copy()
# xist = xist.groupby([idField,"day"]).agg(np.mean).reset_index()
# gist = xist[[idField,'act']].groupby(idField).agg(np.mean).reset_index()
# gist = pd.merge(gist,corF,on=idField,how="left")
# gist = gist.replace(float('nan'),1.)
# gist.loc[:,"fact"] = gist['daily_visit']/gist['act']
# xist = pd.merge(xist,gist[[idField,'fact']],on=idField,how="left")
# xist.loc[:,"act"] = xist['act']*xist['fact']
# vist = vist[vist['day'] > '2019-02-00T']
# xist = xist[xist['day'] > '2019-02-00T']

if False:
    tlib.saveModel(fit_w,modDir+"fitDay_"+modName+".pkl")
    scorLearnAct, vist = t_e.learnPlayType(tist
                                           ,dateL[dateL['use']==1],play=False
                                           ,idField=idField,modDir=baseDir+"train/"+custD+"/act/")
    tist1 = pd.merge(vist,tist,on=["id_poi","day"],how="left",suffixes=["","_raw"])
    tist1.loc[:,"act_reg"] = tist1['act']
    tist1.loc[:,"act"] = tist1['foot']
    scorLearnFoot, vist = t_e.learnPlayType(tist1
                                            ,dateL[dateL['use']==1],play=False
                                            ,idField=idField,modDir=baseDir+"train/"+custD+"/foot/")
    tist1.loc[:,"foot"] = pd.merge(vist,tist1,on=["id_poi","day"],how="left",suffixes=["","_raw"])['act']
    scorBoost = t_s.scorPerf(tist1,step="boost",idField=idField)
    tist1.loc[:,"ref"] = tist1['act'] - tist['ref']
    tist1.loc[:,"act"] = tist1["act_raw"]
    scorLearnBoost, vist = t_e.learnPlayType(tist1
                                             ,dateL[dateL['use']==1],play=False
                                             ,idField=idField,modDir=baseDir+"train/"+custD+"/boost/")
    scorLearn = pd.merge(scorM,scorLearnAct,on="id_poi",how="outer",suffixes=["_act","_foot"])
    scorLearn = pd.merge(scorLearn,scorLearnFoot,on="id_poi",how="outer",suffixes=["_act","_foot"])
    scorLearn = pd.merge(scorLearn,scorLearnBoost,on="id_poi",how="outer",suffixes=["_act","_foot"])
    scorLearn.rename(columns={"r_weather":"r_weather_hybrid","r_country":"r_country_hybrid"},inplace=True)
    scorLearn.to_csv(baseDir + "raw/"+custD+"/scor_learnType.csv",index=False)
    vist.loc[:,"ref"] = vist['ref_raw']
    fist = vist.pivot_table(index=idField,columns="day",values="act",aggfunc=np.sum).reset_index()
    fist.to_csv(baseDir + "raw/"+custD+"/act_boost_d.csv.gz",index=False,compression="gzip")

elif ops['location']:
    test_size = .2
    iL = np.unique(tist[idField])
    N = len(iL)
    shuffleL = random.sample(list(iL),N)
    partS = [0,int(N*(1.-test_size)),int(N*(1.)),N]
    trainL = shuffleL[partS[0]:partS[1]]
    testL = shuffleL[partS[1]:partS[2]]
    t_train = tist[tist[idField].isin(trainL)]
    t_test = tist[tist[idField].isin(testL)]
    scorTrain, vist = t_e.learnPlayType(t_train,dateL,play=False,idField=idField,modDir=baseDir+"train/"+custD+"/transfer/")
    scorValid, vist = t_e.learnPlayType(t_test,dateL,play=True,idField=idField,modDir=baseDir+"train/"+custD+"/transfer/")
    scorTrain.to_csv(baseDir + "raw/"+custD+"/scor_train.csv",index=False)
    scorValid.to_csv(baseDir + "raw/"+custD+"/scor_validation.csv",index=False)
    
elif False:
    importlib.reload(t_e)
    scorLearn, vist = t_e.learnPlayDay(tist,dateL[dateL['use']==1],play=False,idField=idField,modDir=modDir)
    scorLearn.to_csv(baseDir + "raw/"+custD+"/scor_learnDay.csv",index=False)
    scorPlay , vist = t_e.learnPlayDay(tist,dateL,play=True,idField=idField,modDir=modDir)
    scorPlay.to_csv(baseDir + "raw/"+custD+"/scor_play.csv",index=False)

if ops['smooth']:
    setL = vist['day'].isin(dateL[dateL['use']==1]['day'])
    scorB = t_s.scorPerf(vist[setL],step="blind",idField=idField)
    mist = vist.copy()
    mist.loc[:,"act"] = t_e.smoothClust(mist['act'],mist[idField],width=1,steps=3)
    mist.loc[:,"ref"] = t_e.smoothClust(mist['ref'],mist[idField],width=1,steps=3)
    scorS = t_s.scorPerf(mist[setL],step="smooth",idField=idField)
    scorPlay = pd.merge(scorPlay,scorB,on=idField,how="outer")
    scorPlay = pd.merge(scorP,scorS,on=idField,how="outer")
    scorPlay.to_csv( baseDir + "raw/"+custD+"/scor_playDay.csv",index=False)

if False:
    fig, ax = plt.subplots(1,2)
    shl.kpiDis(scorLearnAct,tLab="long short term",col_cor="r_country",col_dif="d_country",col_sum="s_country",isRel=False,ax=ax[0])
    shl.kpiDis(scorLearnAct,tLab="long short term",col_cor="r_weather",col_dif="d_weather",col_sum="s_weather",isRel=False,ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1,2)
    shl.kpiDis(scorLearn,tLab="learn",col_cor="r_weather",col_dif="d_weather",col_sum="s_weather",isRel=False,ax=ax[0])
    shl.kpiDis(scorPlay,tLab="play",col_cor="r_weather",col_dif="d_weather",col_sum="s_weather",isRel=False,ax=ax[1])
    plt.show()
    
