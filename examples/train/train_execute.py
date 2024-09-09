import os, sys, gzip, random, csv, json, datetime,re
import time
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_model as tlib
import geomadi.train_shape as shl
import geomadi.train_score as t_f
import geomadi.train_metric as t_m
import geomadi.train_reshape as t_r
import importlib
import geomadi.series_stat as s_s
from sklearn.metrics import mean_squared_error
import sklearn as sk
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
import geomadi.train_score as t_s

def prepLearningSet(g,dateL):
    tL = ['wday','cloudCover','humidity','icon','precipIntensity','precipProbability','visibility','lightDur','Tmin','Tmax',"daily_sum"]
    Xg1 = g.pivot_table(index="day",columns="hour",values="act",aggfunc=np.sum).replace(float("Nan"),0)
    Xg2 = g.pivot_table(index="day",columns="hour",values="ref",aggfunc=np.sum).replace(float("Nan"),0)
    Xd1 = pd.DataFrame({"day":Xg1.index,"value":Xg1.values.sum(axis=1)})
    Xd2 = pd.DataFrame({"day":Xg2.index,"value":Xg2.values.sum(axis=1)})        
    X1 = pd.melt(Xg1.reset_index(),id_vars="day",value_vars=Xg1.columns).sort_values(["day","hour"])
    X2 = pd.melt(Xg2.reset_index(),id_vars="day",value_vars=Xg2.columns).sort_values(["day","hour"])
    Xd = Xd1.drop(columns=["value"])
    Xd = pd.merge(Xd,dateL,on="day",how="left")
    Xd = Xd[tL]
    Xd.loc[:,"icon"], _ = pd.factorize(Xd['icon'])
    Xd = np.array(Xd.values)
    Xd = np.nan_to_num(Xd)
    X = X1.drop(columns=["value"])
    X = pd.merge(X,dateL,on="day",how="left")
    X = X[tL]
    X.loc[:,"icon"], _ = pd.factorize(X['icon'])
    X = np.nan_to_num(X)
    X = shl.shapeLib(X).calcPCA()[:,:6]
    return X, Xd, X1, X2, Xg1, Xd1['value'].values, Xd2['value'].values

def twoStageRegressor(g,dateL,modName="all",modDir="train/",play=False):
    g = g.replace(float('Nan'),0)
    X, Xd, X1, X2, Xg1, Xd1, Xd2 = prepLearningSet(g,dateL)
    print("id_clust %s - corr %f" % (modName,sp.stats.pearsonr(Xd1,Xd2)[0]) )
    if not play:
        fit_w, corrLW = tlib.regressor(Xd,Xd1,Xd2,nXval=6,isShuffle=True)
        tlib.saveModel(fit_w,modDir+"fitWeather_"+modName+".pkl")
    else:
        try:
            fit_w = tlib.loadModel(modDir+"fitWeather_"+modName+".pkl")
        except:
            return X1, X2, False
    r_quot = fit_w.predict(Xd)
    corrW = Xd1*r_quot
    if play:
        corrLW = [ sp.stats.pearsonr(corrW,Xd2)[0] ]
    print("   play: %r - corr %f" % (play,np.mean(corrLW)) )
    t_M, c_M = featureShape(Xg1.values)
    if not play:
        fit_s, corrLS = tlib.regressor(c_M,corrW,Xd2,nXval=6,isShuffle=True)
        tlib.saveModel(fit_s,modDir+"fitShape_"+modName+".pkl")
    else :
        try:
            fit_s = tlib.loadModel(modDir+"fitShape_"+modName+".pkl")
        except:
            return X1, X2, [], [], False
    r_quot = fit_s.predict(c_M)
    corrS = corrW*r_quot
    if play:
        corrLS = [ sp.stats.pearsonr(corrS,Xd2)[0] ]
    print("               - corr %f" % (np.mean(corrLS)) )
    corrD = pd.DataFrame({"day":Xg1.index,"fact":corrS/Xd1})
    X1 = pd.merge(X1,corrD,on="day",how="left")
    X1.loc[:,"corr"] = X1['value']*X1['fact']
    return X1, X2, corrLW, corrLS, True

def corrPerformance(X1,X2,corrLW,corrLS,modName):
    diff = (X1['corr'].sum()-X2['value'].sum())/X2['value'].sum()
    y = X2.groupby("day").agg(sum)['value']#.values
    x1 = X1.groupby("day").agg(sum)['value']#.values
    x2 = X1.groupby("day").agg(sum)['corr']#.values
    scorP1 = {idField:modName
              ,"r_country":sp.stats.pearsonr(x1,y)[0]
              ,"r_cross":sp.stats.pearsonr(x2,y)[0]
              ,"r_weather":np.mean(corrLW)
              ,"r_shape":np.mean(corrLS)
              ,"r_max":max(corrLS)
              ,"sum":sum(X2['value'])
              ,"diff":diff
        }
    scorL1 = pd.DataFrame({"day":X1['day'].values,"hour":X1["hour"].values,"act":X1["corr"].values,"ref":X2['value'].values,"quot":X1['fact'].values})
    return scorP1, scorL1

def learnPlayHour(tist,dateL,play=False,idField="id_clust",modDir="train/"):
    modDir = modDir + idField + "/"
    t_start = time.clock()
    scorP = []
    scorL = []
    X1, X2, corrLW, corrLS, isSuccess = twoStageRegressor(tist,dateL,modName="country",modDir=modDir,play=play)
    tist = pd.merge(tist,X1,on=["day","hour"],how="left")
    tist.loc[:,"act"] = tist['act']*tist['fact']
    del tist['fact'], tist['value'], tist['corr']
    scorM1 = t_f.scorPerf(tist,"country",idField=idField)
    vist = pd.DataFrame()    
    for i,g in tist.groupby(idField):
        modName = g[idField].iloc[0]
        g = g[g['day'].isin(dateL['day'])]
        g = g.replace(float('nan'),0)
        if g.shape[0] == 0:
            continue
        X1, X2, corrLW, corrLS, isSuccess = twoStageRegressor(g,dateL,modName=modName,modDir=modDir,play=play)
        if not isSuccess:
            continue
        scorP1, scorL1 = corrPerformance(X1,X2,corrLW,corrLS,modName=modName)
        scorP.append(scorP1)
        vist1 = pd.merge(g,X1,on=["day","hour"],how="left")
        vist1.loc[:,"act"] = vist1['act']*vist1['fact']
        del vist1['fact'], vist1['value'], vist1['corr']
        vist = pd.concat([vist,vist1],axis=0)
    scorP = pd.DataFrame(scorP)
    scorP = pd.merge(scorP,scorM1,on=idField,how="outer")
    print("performance")
    for i in [x for x in scorP.columns if bool(re.search("r_",x))]:
        print("    %s: %.2f" % (i,scorP[scorP[i] > 0.6].shape[0]/scorP.shape[0]) )
    t_diff = time.clock() - t_start
    NPoi = len(set(tist[idField]))
    print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/NPoi) )
    return scorP, vist

def prepLearningSetDay(tist,dateL):
    tL = ['wday','cloudCover','humidity','icon','precipIntensity','precipProbability','visibility','lightDur','Tmin','Tmax',"daily_sum"]
    Xd1 = tist[['day','act']].groupby("day").agg(sum)
    Xd2 = tist[['day','ref']].groupby("day").agg(sum)
    Xd = Xd1.reset_index().drop(columns="act")
    Xd = pd.merge(Xd,dateL,on="day",how="left")
    Xd = Xd[tL]
    Xd.loc[:,"icon"], _ = pd.factorize(Xd['icon'])
    Xd = np.array(Xd.values)
    Xd = np.nan_to_num(Xd)
    Xd = shl.shapeLib(Xd).calcPCA()[:,:6]
    return Xd, Xd1['act'].values, Xd2['ref'].values

def oneStageRegressor(tist,dateL,modName="all",modDir="train/",play=False):
    tist = tist.replace(float('Nan'),0)
    Xd, Xd1, Xd2 = prepLearningSetDay(tist,dateL)
    if Xd.shape[0] == 0:
        return [], [], False
    if not play:
        fit_w, corrLW = tlib.regressor(Xd,Xd1,Xd2,nXval=6,isShuffle=True)
        tlib.saveModel(fit_w,modDir+"fitDay_"+modName+".pkl")
    else:
        try:
            fit_w = tlib.loadModel(modDir+"fitDay_"+modName+".pkl")
        except:
            return [], [], False
    r_quot = fit_w.predict(Xd)
    corrW = Xd1*r_quot
    corrD = pd.DataFrame({"day":np.unique(tist['day']),"fact":corrW/Xd1})
    if play:
        corrLW = [ sp.stats.pearsonr(corrW,Xd2)[0] ]
    print("%10s %5s - r: %f" % (modName,"play" if play else "learn",np.mean(corrLW)) )
    return corrD, corrLW, True

def learnPlayDay(tist,dateL,play=False,idField="id_clust",modDir="train/"):
    t_start = time.clock() 
    modDir = modDir + idField + "/"
    corrD, corrLW, isSuccess = oneStageRegressor(tist,dateL,modName="countryDay",modDir=modDir,play=play)
    tist = pd.merge(tist,corrD,on=["day"],how="left",suffixes=["","_y"])
    tist = tist.replace(float('nan'),1.)
    tist.loc[:,"act"] = tist['act']*tist['fact']
    del tist['fact'] 
    scorM1 = t_f.scorPerf(tist,"country",idField)
    vist = pd.DataFrame()
    for i,g in tist.groupby(idField):
        id_clust = str(g[idField].iloc[0])
        g = g[g['day'].isin(dateL['day'])]
        if g.shape[0] == 0:
            continue
        corrD, corrLW, isSuccess = oneStageRegressor(g,dateL,modName="day_"+id_clust,modDir=modDir,play=play)
        vist1 = pd.merge(g,corrD,on="day",how="left")
        vist1 = vist1.replace(float('nan'),1.)
        vist1.loc[:,"act"] = vist1['act']*vist1['fact']
        del vist1['fact'] 
        vist = pd.concat([vist,vist1],axis=0)
        if not isSuccess:
            continue
        
    scorM2 = t_f.scorPerf(vist,"weather",idField)
    scorP = scorM1
    scorP = pd.merge(scorP,scorM2,on=idField,how="outer")
    t_diff = time.clock() - t_start
    NPoi = len(set(tist[idField]))
    print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/NPoi) )
    return scorP, vist

def learnPlayType(tist,dateL,play=False,idField="id_clust",modDir="train/"):
    t_start = time.clock() 
    modDir = modDir + "type" + "/"
    corrD, corrLW, isSuccess = oneStageRegressor(tist,dateL,modName="countryDay",modDir=modDir,play=play)
    tist = pd.merge(tist,corrD,on=["day"],how="left",suffixes=["","_y"])
    tist = tist.replace(float('nan'),1.)
    tist.loc[:,"act"] = tist['act']*tist['fact']
    del tist['fact']
    scorM1 = t_f.scorPerf(tist,"country",idField)
    vist = pd.DataFrame()
    for i,g in tist.groupby("type"):
        id_clust = str(g["type"].iloc[0])
        g = g[g['day'].isin(dateL['day'])]
        if g.shape[0] == 0:
            continue
        corrD, corrLW, isSuccess = oneStageRegressor(g,dateL,modName="day_"+id_clust,modDir=modDir,play=play)
        vist1 = pd.merge(g,corrD,on="day",how="left")
        vist1 = vist1.replace(float('nan'),1.)
        vist1.loc[:,"act"] = vist1['act']*vist1['fact']
        del vist1['fact'] 
        vist = pd.concat([vist,vist1],axis=0)
        if not isSuccess:
            continue
        
    scorM2 = t_f.scorPerf(vist,"weather",idField)
    scorP = scorM1
    scorP = pd.merge(scorP,scorM2,on=idField,how="outer")
    t_diff = time.clock() - t_start
    NPoi = len(set(tist[idField]))
    print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/NPoi) )
    return scorP, vist

def joinSource(sact,tist,how="inner",idField="id_poi",isSameTime=True):
    hL = [x for x in tist.columns if bool(re.search("T",x))]
    hL1 = [x for x in sact.columns if bool(re.search("T",x))]
    if isSameTime:
        hL = sorted(list(set(hL) & set(hL1)))
        hL1 = sorted(list(set(hL) & set(hL1)))
    act = pd.melt(sact,value_vars=hL1,id_vars=idField)
    gact = act.groupby([idField,"variable"]).agg(np.sum).reset_index()
    gact.columns = [idField,"time","value"]
    vist = pd.melt(tist,value_vars=hL,id_vars=idField)
    vist.columns = [idField,"time","value"]
    gist = vist.groupby([idField,"time"]).agg(np.sum).reset_index()
    gact.loc[idField] = gact[idField].astype(str)
    vist.loc[idField] = vist[idField].astype(str)
    act = pd.merge(gact,gist,on=[idField,"time"],how=how)
    act.columns = [idField,"time","act","ref"]
    return act

def concatSource(pact,tist,how="inner",idField="id_poi",varName="source"):
    hL = [x for x in tist.columns if bool(re.search("T",x))]
    vist = pd.melt(tist,value_vars=hL,id_vars=idField)
    vist.columns = [idField,"time",varName]
    gist = vist.groupby([idField,"time"]).agg(np.sum).reset_index()
    act = pd.merge(pact,gist,on=[idField,"time"],how=how)
    return act

def plotSum(act,isLoc=False,colList=['act','ref'],ax=None):
    labT = "display all locations sum"
    if isLoc:
        cL = np.unique(act[idField])
        n = np.random.randint(len(cL))
        act = act[act[idField] == cL[n]]
        labT = "display %s locations sum" % (cL[n])
    try:
        ga = act[['day']+colList].groupby('day').agg(np.sum).reset_index()
        t = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in ga['day']]
    except:
        ga = act[['time']+colList].groupby('time').agg(np.sum).reset_index()
        t = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in ga['time']]
    labT = labT + " corr %.2f" % (sp.stats.pearsonr(ga[colList[0]],ga[colList[1]])[0])
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(labT)
    for i in colList:
        ax.plot(t,ga[i],label=i)
    ax.legend()
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)

def loadDf(fName,dateL,poi,mapL,isChi=False,idField="id_poi",hL1=[None]):
    sact = pd.read_csv(fName,compression="gzip",dtype={idField:str})
    hL = sact.columns[[bool(re.search('-??T',x)) for x in sact.columns]]
    sact = sact.merge(mapL,on="cilac",how="outer")
    sact[idField] = sact[idField].astype(str)
    #sact = sact.replace(float('nan'),0.)
    if isChi:
        sact = sact[sact['chi'] == sact['chi']]
        sact.loc[:,"chi"] = sact['chi'].astype(int)
        sact.loc[:,"chi_2"] = sact.merge(poi,on=idField,how="left")['chirality']
        sact = sact[sact['chi_2'] == sact['chi_2']]
        sact.loc[:,"chi_2"] = sact['chi_2'].astype(int)
        sact = sact[sact["chi"] == sact["chi_2"]]
        del sact['chi_2']
    if any(hL1):
        hL = sorted(list(set(hL) & set(hL1)))
    hL2 = [x + "T" for x in set(dateL['day'])]
    hL = sorted(list(set(hL) & set(hL2)))
    sact = sact.groupby([idField,"cilac"]).agg(np.sum).reset_index()
    sact = sact[sact[idField].isin(poi[idField])]
    return sact, hL
        
def cilacLinWeight(sact,mist,hL,idField,isPlot=False):
    keepL = []
    sact.loc[:,"weight"] = 0.
    weiL = []
    tact = []
    for i,g in sact.groupby(idField):
        G = (g[hL] <= 0).sum(axis=1) #np.linalg.det(X)
        setL = G > int(g.shape[1]*.1)
        # if not sum(setL) == 0: g = g.loc[setL]
        X = g[hL].values.T 
        norm = 1./g.shape[0]
        c = np.linspace(norm,norm,g.shape[0])
        y = mist[mist[idField] == i][hL]
        if (X.sum().sum()>0) & (y.shape[0]>0):
            y = y.values[0]
            if not any([not y.sum()>1]):
                setX = X.sum(axis=1) > 0
                setY = y > 0
                setL = setX & setY
                c, n_cell = t_m.linWeight(X[setL,:],y[setL],n_source=5)
        weiL.append(pd.DataFrame({idField:i,"cilac":g['cilac'],"weight":c}))
        tact.append(pd.DataFrame({idField:i,"day":hL,"value":np.multiply(X,c).sum(axis=1)}))
        sact.loc[g.index,"weight"] = c
        if isPlot:
            plt.plot(X.sum(axis=1))
            plt.plot(y)
            plt.show()
            M = g[hL].T.corr()*100
            M.index, M.columns = g['cilac'],g['cilac']
            sns.heatmap(M,annot=True,square=True,cmap='RdYlGn')
            plt.xticks(rotation=15)
            plt.show()
            G = g[hL]
            G.index = g['cilac']
            G.isnull().sum(axis=1)
            G.T.boxplot()
            plt.xticks(rotation=15)
            plt.ylabel("count")
            plt.show()
            #t_v.plotPairGrid(g.loc[g.index[:10],hL].T)
        if isPlot:
            x = X.sum(axis=1)
            x1 = np.multiply(X,c).sum(axis=1)
            tL = t_r.day2time(hL)
            plt.title("correlation raw: %.2f -> map: %.2f" % (sp.stats.pearsonr(x,y)[0],sp.stats.pearsonr(x1,y)[0]))
            plt.plot(tL,x/x.max(),label="raw")
            plt.plot(tL,y/y.max(),label="ref")
            plt.plot(tL,x1/x1.max(),label="mapping")
            plt.legend()
            plt.xticks(rotation=15)
            plt.show()
    return sact, pd.concat(weiL), pd.concat(tact)
    
def xvalRegressor(tist,idField,hL,sL):
    vist = []
    xist = []
    kpiL = []
    fitL = []
    for i,g in tist.groupby(idField):
        iL = np.unique(g[idField])
        iN = len(iL)
        iValid = 0#int(iN*.8)
        clf = BaggingRegressor(DecisionTreeRegressor())
        setL = g['day'] < '2019-02-15T'
        setL = ([np.random.uniform(0,1) >.1  for x in range(g.shape[0])]) & (g['day'] < '2019-03-05T')
        X, y = g[sL].values, g['ical'].values #g['ref'].values
        corL = []
        difL = []
        for l in range(10):
            fit_w = clf.fit(X[setL],y[setL])
            y_pred = fit_w.predict(X)
            post = pd.DataFrame({idField:i,"ref":g['ref'].values,"ical":g['ical'].values,"act":y_pred,"day":g['day'].values})
            setL = post['day'] > '2019-02-00T'
            setL = post['day'] < '2019-02-15T'
            cor = sp.stats.pearsonr(post.loc[setL,"ical"],post.loc[setL,"act"])[0]
            if cor > 0.6:
                break
        vist.append(pd.DataFrame(post))
        fitL.append({idField:i,"model":fit_w})
        for k in range(6):
            g1 = g[g[idField] == random.choice(iL[iValid:])]
            X, y = g1[sL].values, g1['ref'].values
            y_pred = fit_w.predict(X)
            corL.append(sp.stats.pearsonr(y,y_pred)[0])
            difL.append( (sum(y_pred) - sum(y))/(sum(y_pred) + sum(y)) )
            post = {idField:g1[idField],"ref":g1['ref'].values,"act":y_pred,"day":g1['day'].values}
            xist.append(pd.DataFrame(post))
            kpiL.append({idField:i,"cor":np.mean(corL),"dif":np.mean(difL),"type":i})
        print(i,np.mean(corL))
    vist = pd.concat(vist,axis=0)
    xist = pd.concat(xist,axis=0)    
    kpiL = pd.DataFrame(kpiL)
    return vist, xist, kpiL, fitL

def weekdayCorrection(sact,mist,hL):
    x1 = sact[hL].sum(axis=0)
    x2 = mist[hL].sum(axis=0)
    t = [datetime.datetime.strptime(x,"%Y-%m-%dT").weekday() for x in hL]
    d = x1 - x2
    dif = pd.DataFrame({"dif":d.values,"wday":t})
    dif = dif.groupby("wday").agg(sum).reset_index()
    dif.loc[:,"dif"] = (dif['dif'] - dif['dif'].mean())/dif['dif'].abs().max()
    return dif



