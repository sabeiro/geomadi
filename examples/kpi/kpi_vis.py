#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph
import geomadi.train_shape as shl
import seaborn as sns
import geomadi.train_execute as t_e

def plog(text):
    print(text)

custD = "tank"
custD = "mc"
idField = "id_poi"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")

if False:
    print('-------------------scoring-mapping-cilacs------------------------')
    scorM = pd.read_csv(baseDir + "raw/"+custD+"/scor/scorMap.csv")
    scorM.rename(columns={'r_raw_t4_p11_d40':"0_raw","s_cor":"1_best_cilac","r_map_t4_p11_d40":"2_mapping"},inplace=True)
    scorT = pd.melt(scorM,value_vars=['0_raw','1_best_cilac','2_mapping'],id_vars=idField)
    
    scorT1 = scorT.copy()
    scorT2 = scorT.copy()
    
    fig, ax = plt.subplots(1,2)
    scorT1.boxplot(column="value",by="variable",ax=ax[0])
    scorT2.boxplot(column="value",by="variable",ax=ax[1])
    plt.title(label="")
    ax[0].set_title("all locations")
    ax[1].set_title("selected locations")
    ax[0].set_xlabel("")
    ax[1].set_xlabel("")
    ax[0].set_ylabel("correlation")
    plt.show()

if False:
    plog('---------------------scoring-evolution-heatmap--------------------')
    scorM = pd.read_csv(baseDir + "raw/"+custD+"/scorMap.csv")
    scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnType.csv")
    scorL = pd.merge(scorM,scorL,on="id_poi",how="outer")
    t_e.plotPerfHeat(scorL)
    if False:
        poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
        scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnType.csv")
        poi.loc[:,"score"] = float("nan")
        poi.loc[:,"score"] = pd.merge(poi,scorL,on=idField,how="left")['r_weather_hybrid']
        poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('---------------------correlation-difference-performance---------------------')
    tist = pd.read_csv(baseDir + "raw/"+custD+"/act_boost_d.csv.gz",compression="gzip")
    mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
    act = t_e.joinSource(tist,mist,how="inner",idField=idField)
    t_e.plotSum(act)
    scor = t_e.scorKpi(act,step="cilac",idField=idField)
    shl.kpiDis(scor,tLab="cilac mapping",col_cor="r_cilac",col_dif="d_cilac",col_sum="s_cilac")

if False:
    plog('-------------------------------double-heatmap-----------------------------')
    scorP = pd.read_csv(baseDir + "raw/"+custD+"/scor_playDay.csv")
    scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnDay.csv")
    scorM = pd.read_csv(baseDir + "raw/"+custD+"/scorMap.csv")
    scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnType.csv")
    scorL = pd.merge(scorM,scorL,on="id_poi",how="outer")
    scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_train.csv")
    t_e.plotPerfHeat(scorL)
    scorP = pd.read_csv(baseDir + "raw/"+custD+"/scor_validation.csv")
    t_e.plotPerfHeat(scorL,scorP)

if False:
    plog("-------------------feature-selection------------------------")
    importlib.reload(shl)
    y = scorP['y_cor'].values
    y = [int(10*i) for i in y]
    y, _ = t_f.binOutlier(scorP['y_cor'],nBin=10,threshold=0.1)
    X = shl.factorizeVar(scorP[tL])
    featS = shl.featureSel()
    T, _ = featS.variance(X)
    tL1 = [tL[x] for x in [y for y in range(len(tL)) if y not in _]]
    if False:
        shl.plotCorr(X,labV=tL)
        shl.plotCorr(T,labV=tL1)
        shl.plotCorr(scorP.values,list(scorP.columns))
        
    fig = plt.figure()
    for i in [0,2,3,4]:
        impD, modelN = shl.featureImportance(X,y,tL,method=i)
        plt.bar(impD['label'],impD['importance'],label=modelN,alpha=.3)
    plt.ylabel("importance")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


if False:
    plog('---------------display-score----------------------')
    act = pd.read_csv(baseDir + "raw/"+custD+"/act_vist_play.csv.gz",compression="gzip")
    scor = t_e.scorKpi(act,step="paly",idField=idField)
    shl.kpiDis(scor,tLab="visits vs footfall - daily",col_cor="r_play",col_dif="d_play",col_sum="s_play")
    act = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip")
    scor = t_e.scorKpi(act,step="cilac",idField=idField)
    shl.kpiDis(scor,tLab="cilac mapping weighted",col_cor="r_cilac",col_dif="d_cilac",col_sum="s_cilac",saveF=baseDir + "www/f_food/kpi_cilac_weight.png")

    act = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_ref.csv.gz",compression="gzip")
    scor = t_e.scorKpi(act,step="cilac",idField=idField)
    shl.kpiDis(scor,tLab="cilac mapping",col_cor="r_cilac",col_dif="d_cilac",col_sum="s_cilac",saveF=baseDir + "www/f_food/kpi_cilac_weight.png")

    shl.kpiDis(scor,tLab="visits vs footfall - daily",col_cor="r_traj",col_dif="d_traj",col_sum="s_traj")
    shl.kpiDis(scor,tLab="visits vs activities - daily",col_cor="r_act",col_dif="d_act",col_sum="s_act")
    shl.kpiDis(scor,tLab="footfall vs activities - daily",col_cor="r",col_dif="d",col_sum="s")


if False:
    fig, ax = plt.subplots(2,2)
    poi[poi['r_traj']==poi['r_traj']].boxplot(column="r_traj",by="type",ax=ax[0][0])
    poi[poi['r_traj']==poi['r_traj']].boxplot(column="r_traj",by="drive",ax=ax[0][1])
    poi[poi['r_act']==poi['r_act']].boxplot(column="r_act",by="drive",ax=ax[1][0])
    poi[poi['r']==poi['r']].boxplot(column="r",by="drive",ax=ax[1][1])
    for i,a in enumerate(ax.flat):
        for tick in a.get_xticklabels():
            tick.set_rotation(10)
    plt.show()
