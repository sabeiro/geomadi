import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_model as tlib
import geomadi.train_shape as shl
import importlib
import geomadi.series_stat as s_l

def featureShape(X):
    """default feature matrix to train based on curve shapes"""
    importlib.reload(shl)
    shp = shl.shapeLib(X)
    t_M = pd.DataFrame()
    tmp = shp.periodic(period=18)
    t_M = pd.concat([t_M,tmp],axis=1)
    tmp = shp.seasonal(period=18)
    t_M = pd.concat([t_M,tmp],axis=1)
    tmp = shp.statistical()
    t_M = pd.concat([t_M,tmp],axis=1)
    tL = ['t_conv','t_trend2','t_ampl','t_std','t_median']
    t_M = t_M[tL]       
    fSel = shl.featureSel()
    tL, varL = fSel.std(t_M,2)
    p_M = t_M[tL]
    t_M.replace(float('Nan'),0,inplace=True)
    p_M = shl.shapeLib(t_M).calcPCA()[:,:5]
    #c_M, psum = tlib.binMatrix(p_M,nBin=6,threshold=2.5)
    c_M = pd.DataFrame(p_M)
    c_M.replace(float('Nan'),0,inplace=True)
    return t_M, c_M
    
def featureTank(sact,hL):
    """default feature matrix to train based on curve shapes for the project tank"""
    importlib.reload(shl)
    shp = shl.shapeLib(sact[hL])
    t_M = pd.DataFrame()
    tmp = shp.periodic(period=18)
    t_M = pd.concat([t_M,tmp],axis=1)
    tmp = shp.seasonal(period=18)
    t_M = pd.concat([t_M,tmp],axis=1)
    tmp = shp.statistical()
    t_M = pd.concat([t_M,tmp],axis=1)
    t_M.loc[:,"t_median"] = (t_M["t_median"] + 6) % 18
    t_M.loc[:,"t_dist"] = sact['t_dist']
    t_M.loc[:,"t_tech"] = pd.factorize(sact['tech'])[0]
    t_M.loc[:,"t_type"] = pd.factorize(sact['type'])[0]
    tL = ['t_conv','t_trend2','t_ampl','t_std','t_median','t_dist','t_tech','t_type']
    t_M = t_M[tL]       
    fSel = shl.featureSel()
    tL, varL = fSel.std(t_M,2)
    p_M = t_M[tL]
    t_M.replace(float('Nan'),0,inplace=True)
    p_M = shl.shapeLib(t_M).calcPCA()[:,:5]
    #c_M, psum = tlib.binMatrix(p_M,nBin=6,threshold=2.5)
    c_M = pd.DataFrame(p_M)
    return t_M, c_M

def featureMc(sact,hL,poi):
    """Feature matrix to train for the project mc"""
    shp = shl.shapeLib(sact.loc[:,hL])
    t_M = pd.DataFrame()
    tmp = shp.periodic(period=13)
    t_M = pd.concat([t_M,tmp],axis=1)
    tmp = shp.monthly(period=13)
    t_M = pd.concat([t_M,tmp],axis=1)
    tmp = shp.statistical()
    t_M = pd.concat([t_M,tmp],axis=1)

    t_M.loc[:,"t_median"] = t_M["t_median"] % 24
    t_M.loc[:,"t_loc"] = pd.factorize(poi['Region (Bu'])[0]
    t_M.loc[:,"t_type"] = pd.factorize(poi['gem_typ_'])[0]
    t_M.loc[:,"t_cafe"] = pd.factorize(poi['Mc Cafe'])[0]
    t_M.loc[:,"t_busi"] = pd.factorize(poi['Business T'])[0]
    t_M.loc[:,"t_owner"] = pd.factorize(poi['Betreiber'])[0]
    t_M.loc[:,"t_street"] = pd.factorize(poi['highway'])[0]
    #t_M.loc[:,"t_partner"] = pd.factorize(poi['Partner'])[0]
    t_M.loc[:,"t_dens"] = poi['ew_km']
    t_M.loc[:,"t_reachpois"] = poi['rp']
    t_M.loc[:,"t_act_dens"] = poi['act_qkm']
    #t_M.loc[:,"t_reachpois"] = poi['rp']
    t_M.loc[:,"t_speed"] = pd.factorize(poi['maxspeed_tile'])[0]
    t_M.loc[:,"t_crosses"] = pd.factorize(poi['anz_strabschnitte_tile'])[0]
    t_M.loc[:,"t_border"] = poi['grenzlks']
    
    p_M = shl.shapeLib(t_M).calcPCA()[:,:4]
    p_M = pd.DataFrame(p_M)
    tL = ['t_loc','t_type','t_street','t_dens','t_reachpois','t_crosses','t_border']
    tL = ['t_loc','t_type','t_street','t_dens','t_crosses','t_border',"t_act_dens"]
    p_M = t_M[tL]
    return t_M, p_M

def prepareMc(poi):
    poi = pd.read_csv(baseDir + "raw/mc/poi.csv")
    poi_cat = json.load(open(baseDir + "raw/mc/poi_cat.json"))
    typeD = []
    for i in poi_cat['type']:
        typeD.append({"type":list(i.keys())[0],"type1":list(i.values())[0]})
    typeD = pd.DataFrame(typeD)
    subD = []
    for i in poi_cat['subtype']:
        subD.append({"subtype":list(i.keys())[0],"type2":list(i.values())[0]})
    subD = pd.DataFrame(subD)
    poi.loc[:,"subtype"] = poi['subtype'].astype(str)
    poi = pd.merge(poi,typeD,on="type",how="left")
    poi = pd.merge(poi,subD,on="subtype",how="left")
    tL = ['type','region','cafe','subtype','mot_dist','pop_dens','land_use','elder','degeneracy','bast','bast_su','n_cell']
    sL = ['act','foot','wday','cloudCover','humidity','visibility','Tmax']
    if False:
        shl.plotOccurrence(poi['type3'])
        plt.show()
    if False:
        tL = ['pop_dens','women','foreign','flat_dens','land_use','elder']
        tL = ['bast','bast_fr','bast_su']
        tL = ['mot_dist','maxspeed','highway']
        tL = ['type','region','cafe','BusinessTyp']
        correlation_matrix = X[tL].corr()
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(correlation_matrix, vmax=1, square=True,annot=True,cmap='RdYlGn')
        plt.title('Correlation matrix between the features')
        plt.show()
    return tL, sL

def prepareTank(poi):
    tL = ['type','region','junct_dist','junct_time','pop_dens','flat_dens','bast','daily_visit','bast_we', 'bast_su','forecastability', 'act', 'foot']
    sL = ['act','foot','wday','cloudCover','humidity','visibility','Tmax']
    if False:
        shl.plotOccurrence(poi['type3'])
        plt.show()
    if False:
        tL = ['pop_dens','women','foreign','flat_dens','land_use','elder']
        tL = ['bast','bast_fr','bast_su']
        tL = ['mot_dist','maxspeed','highway']
        tL = ['type','region','cafe','BusinessTyp']
        correlation_matrix = X[tL].corr()
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(correlation_matrix, vmax=1, square=True,annot=True,cmap='RdYlGn')
        plt.title('Correlation matrix between the features')
        plt.show()
    return tL, sL

def regressionTank(scor1):
    """regression to match total sum"""
    t_M = pd.DataFrame()
    t_M.loc[:,"t_type"] = shl.binMask(scor1['type'])#pd.factorize(scor1['type'])[0] + 1
    # t_M.loc[:,"t_pop_dens"]   = scor1['pop_dens']
    # t_M.loc[:,"t_bast"]       = scor1['bast']
    t_M.loc[:,"t_n_cell"]     = scor1['n_cell']
    t_M.loc[:,"t_n_source"]   = scor1['c_source']
    t_M.loc[:,"t_sum"]        = scor1['sum_p']
    tL = t_M.columns
    tL = ["t_type","t_sum","t_n_source","t_n_cell"]
    if False:
        p_M = t_M.describe().transpose()
        p_M.loc[:,"rel_std"] = np.abs(p_M['std']/p_M['mean'])
        p_M = p_M.sort_values('rel_std')
    if False:
        plt.bar(p_M['rel_std'].index,p_M['rel_std'])
        plt.xticks(rotation=45)
        plt.show()
    if False:
        pd.plotting.scatter_matrix(t_M, diagonal="kde")
        plt.tight_layout()
        plt.show()
    t_M.replace(np.NaN,0,inplace=True)
    return t_M, t_M[tL]

def regressorTank(mist,poi,idField="id_clust",idFloat="dist_raw"):
    """prepare the feature matrix and apply a regressor on the total sum"""
    gmist = mist[[idField,idFloat,'ref']].groupby(idField).agg(sum).reset_index()
    gmist.loc[:,'sum'] = gmist["ref"]
    gmist.loc[:,"sum_p"] = gmist[idFloat]
    gmist.loc[:,"id_zone"] = gmist[idField].apply(lambda x: int(x.split("-")[0]))
    gmist.loc[:,"n_cell"] = pd.merge(gmist,poi,on=idField,how="left")['n_cell']
    gmist.loc[:,"type"] = pd.merge(gmist,poi,on=idField,how="left")['type']
    gmist.loc[:,"id_poi"] = pd.merge(gmist,poi,on=idField,how="left")['id_poi']
    gmist.loc[:,'c_source'] = pd.merge(gmist,poi,on=idField,how="left")['c_source']
    t_M, c_M = regressionTank(gmist)
    c_M.replace(np.nan,0,inplace=True)
    r_quot, fit_q = tlib.regressor(c_M.values,gmist[idFloat],gmist['ref'])
    corr_f = fit_q.predict(c_M)
    corr_f = pd.DataFrame({idField:gmist[idField],"corr":corr_f})
    corr_c = pd.merge(mist,corr_f,on=idField,how="left")['corr']
    return corr_c.values

def smoothClust(y,clustL,width=3,steps=5,isSingle=False):
    """smooth by cluster"""
    y1 = y.values
    if isSingle:
        clustI = set(clustL)
        for i in clustI:
            setB = (clustL == i)
            if not any(setB):
                continue
            y2 = y[setB]
            y1[setB] = s_l.serSmooth(y2,widths=width,steps=steps)
    else:
        y1 = s_l.serSmooth(y,width=width,steps=steps)
        #y1 = s_l.serRunAv(y,steps=steps)
    return y1

def weekdayFact(X1,X2,dayL):
    """weekday correction"""
    print("weekday correction")
    calL1 = X1.sum(axis=1)
    calL2 = X2.sum(axis=1)
    calL = pd.DataFrame({"day":dayL.values,"dif":calL2/calL1})
    calL.loc[:,"wday"] = calL['day'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
    gist = calL.groupby(['wday']).agg(np.mean).reset_index()
    calL = pd.merge(calL,gist,on="wday",how="left",suffixes=["","_y"])
    corrF = calL['dif_y'].values
    return np.multiply(X1,corrF[:, np.newaxis])

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

def twoStageRegressor(g,dateL,modName="all",play=False):
    g = g.replace(float('Nan'),0)
    X, Xd, X1, X2, Xg1, Xd1, Xd2 = prepLearningSet(g,dateL)
    print("id_clust %s - corr %f" % (modName,sp.stats.pearsonr(Xd1,Xd2)[0]) )
    if not play:
        fit_w, corrLW = tlib.regressor(Xd,Xd1,Xd2,nXval=6,isShuffle=True)
        tlib.saveModel(fit_w,baseDir+"train/tank/id_clust/fitWeather_"+modName+".pkl")
    else:
        try:
            fit_w = tlib.loadModel(baseDir+"train/tank/id_clust/fitWeather_"+modName+".pkl")
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
        tlib.saveModel(fit_s,baseDir+"train/tank/id_clust/fitShape_"+modName+".pkl")
    else :
        try:
            fit_s = tlib.loadModel(baseDir+"train/tank/id_clust/fitShape_"+modName+".pkl")
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

def learnPlayTank(tist,dateL,poi,play=False,idField="id_clust"):
    t_start = time.clock()
    scorP = []
    scorL = []
    X1, X2, corrLW, corrLS, isSuccess = twoStageRegressor(tist,dateL,modName="country",play=play)
    tist = pd.merge(tist,X1,on=["day","hour"],how="left")
    tist.loc[:,"act"] = tist['act']*tist['fact']
    del tist['fact'], tist['value'], tist['corr']
    scorM1 = tankPerf1(tist,"country",idField=idField)
    for i,g in tist.groupby(idField):
        g = g[g['day'].isin(dateL['day'])]
        g = g.replace(float('nan'),0)
        if g.shape[0] == 0:
            continue
        X1, X2, corrLW, corrLS, isSuccess = twoStageRegressor(g,dateL,modName=g[idField].iloc[0],play=play)
        if not isSuccess:
            continue
        scorP1, scorL1 = corrPerformance(X1,X2,corrLW,corrLS,modName=g[idField].iloc[0])
        scorP.append(scorP1)
        scorL.append((i,scorL1))
    scorP = pd.DataFrame(scorP)
    scorP = pd.merge(scorP,scorM1,on=idField,how="outer")
    print("performance")
    for i in [x for x in scorP.columns if bool(re.search("r_",x))]:
        print("    %s: %.2f" % (i,scorP[scorP[i] > 0.6].shape[0]/scorP.shape[0]) )
    t_diff = time.clock() - t_start
    NPoi = len(set(tist[idField]))
    print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/NPoi) )
    return scorP, scorL

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

def oneStageRegressor(tist,dateL,modName="all",play=False):
    tist = tist.replace(float('Nan'),0)
    Xd, Xd1, Xd2 = prepLearningSetDay(tist,dateL)
    if Xd.shape[0] == 0:
        return [], [], False
    if not play:
        fit_w, corrLW = tlib.regressor(Xd,Xd1,Xd2,nXval=6,isShuffle=True)
        tlib.saveModel(fit_w,baseDir+"train/tank/id_clust/fitDay_"+modName+".pkl")
    else:
        try:
            fit_w = tlib.loadModel(baseDir+"train/tank/id_clust/fitDay_"+modName+".pkl")
        except:
            return [], [], False
    r_quot = fit_w.predict(Xd)
    corrW = Xd1*r_quot
    corrD = pd.DataFrame({"day":np.unique(tist['day']),"fact":corrW/Xd1})
    if play:
        corrLW = [ sp.stats.pearsonr(corrW,Xd2)[0] ]
    print("%10s %5s - r: %f" % (modName,"play" if play else "learn",np.mean(corrLW)) )
    return corrD, corrLW, True

def learnPlayTankDay(tist,dateL,poi,play=False,idField="id_clust"):
    t_start = time.clock()
    corrD, corrLW, isSuccess = oneStageRegressor(tist,dateL,modName="countryDay",play=play)
    tist = pd.merge(tist,corrD,on=["day"],how="left",suffixes=["","_y"])
    tist = tist.replace(float('nan'),1.)
    tist.loc[:,"act"] = tist['act']*tist['fact']
    del tist['fact'] 
    scorM1 = tankPerf1(tist,"country",idField)
    vist = pd.DataFrame()
    for i,g in tist.groupby(idField):
        id_clust = str(g[idField].iloc[0])
        g = g[g['day'].isin(dateL['day'])]
        if g.shape[0] == 0:
            continue
        corrD, corrLW, isSuccess = oneStageRegressor(g,dateL,modName="day_"+id_clust,play=play)
        vist1 = pd.merge(g,corrD,on="day",how="left")
        vist1 = vist1.replace(float('nan'),1.)
        vist1.loc[:,"act"] = vist1['act']*vist1['fact']
        del vist1['fact'] 
        vist = pd.concat([vist,vist1],axis=0)
        if not isSuccess:
            continue
        
    scorM2 = tankPerf1(vist,"weather",idField)
    scorP = scorM1
    scorP = pd.merge(scorP,scorM2,on=idField,how="outer")
    t_diff = time.clock() - t_start
    NPoi = len(set(tist[idField]))
    print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/NPoi) )
    return scorP, vist

def tankJoinSource(sact,tist,how="inner"):
    hL = [x for x in tist.columns if bool(re.search("T",x))]
    hL1 = [x for x in sact.columns if bool(re.search("T",x))]
    hL = sorted(list(set(hL) & set(hL1)))
    act = pd.melt(sact,value_vars=hL,id_vars=idField)
    gact = act.groupby([idField,"variable"]).agg(np.sum).reset_index()
    gact.columns = [idField,"time","value"]
    vist = pd.melt(tist,value_vars=hL,id_vars=idField)
    vist.columns = [idField,"time","value"]
    gist = vist.groupby([idField,"time"]).agg(np.sum).reset_index()
    act = pd.merge(gact,gist,on=[idField,"time"],how=how)
    act.columns = [idField,"time","act","ref"]
    return act

def tankPerf1(act,step="input",idField="id_clust"):
    def clampF(x):
        return pd.Series({"r_"+step:sp.stats.pearsonr(x['act'],x['ref'])[0]})
    scorM = act.groupby([idField]).apply(clampF).reset_index()
    print("score %s: %.2f" % (step,scorM[scorM['r_'+step] > 0.6].shape[0]/scorM.shape[0]) )
    return scorM

def tankPerf(sact,tist,step="input",idField="id_clust"):
    act = tankJoinSource(sact,tist)
    def clampF(x):
        return pd.Series({"r_"+step:sp.stats.pearsonr(x['act'],x['ref'])[0]})
    scorM = act.groupby([idField]).apply(clampF).reset_index()
    print("score %s: %.2f" % (step,scorM[scorM['r_'+step] > 0.6].shape[0]/scorM.shape[0]) )
    return scorM

def plotSum(act,isLoc=False):
    labT = "display all locations sum"
    if isLoc:
        cL = np.unique(act[idField])
        n = np.random.randint(len(cL))
        act = act[act[idField] == cL[n]]
        labT = "display %s locations sum" % (cL[n])
    try:
        ga = act[['day','act','ref']].groupby('day').agg(np.sum).reset_index()
    except:
        ga = act[['time','act','ref']].groupby('time').agg(np.sum).reset_index()
    labT = labT + " corr %.2f" % (sp.stats.pearsonr(ga['act'],ga['ref'])[0])
    plt.title(labT)
    plt.plot(ga['act'],label='act')
    plt.plot(ga['ref'],label='ref')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
