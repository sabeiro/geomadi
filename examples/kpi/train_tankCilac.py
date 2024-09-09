##%pylab inline
##http://scikit-learn.org/stable/modules/ensemble.html
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_lib as tlib
import geomadi.train_shapeLib as shl
import geomadi.train_filter as t_f
import custom.lib_custom as l_t
import importlib
from io import StringIO
from sklearn import decomposition
from sklearn.decomposition import FastICA, PCA

def plog(text):
    print(text)

plog('-------------------load/def------------------------')
fSux = "20"
idField = "id_clust"
if len(sys.argv) > 1:
    fSux = sys.argv[1]

mist = pd.read_csv(baseDir + "raw/tank/visit_max.csv")
#sact = pd.read_csv(baseDir + "raw/tank/tank_activity_"+fSux+".csv.tar.gz",compression="gzip")
sact = pd.read_csv(baseDir + "raw/tank/act_test.csv.tar.gz",compression="gzip")
poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi.sort_values("competitor",inplace=True)
poi = poi.groupby("id_clust").first().reset_index()
sact = pd.merge(sact,poi[["id_clust","x","y","type"]],on="id_clust",how="left",suffixes=["","_y"])
mist = pd.merge(mist,poi[["id_clust","type"]],on="id_clust",how="left",suffixes=["","_y"])
sact.loc[:,"t_dist"] = sact.apply(lambda x: np.sqrt( (x["X"]-x["x"])**2+(x["Y"]-x["y"])**2),axis=1)
sact.drop(columns=["X","x","Y","y"],inplace=True)
hL  = sact.columns[[bool(re.search('T??:',x)) for x in sact.columns]]
hL1 = mist.columns[[bool(re.search('T??:',x)) for x in mist.columns]]
hL  = sorted(list(set(hL) & set(hL1)))

importlib.reload(shl)
scoreL = shl.scoreLib()
scoreMax = scoreL.score(sact,mist,hL,idField)
tL = list(scoreMax.columns[[bool(re.search('y_',x)) for x in scoreMax.columns]]) + ["sum"]
sact = pd.concat([sact,scoreMax[tL]],axis=1)#.reset_index()
sact.index = range(sact.shape[0])
sact.replace(float('Nan'),0,inplace=True)
plog('-----------------shape-feature-------------------')
importlib.reload(l_t)
t_M, c_M = l_t.featureTank(sact,hL)

if False:
    sact.to_csv(baseDir + "log/basics/cilac_curves.csv.gz",compression="gzip",index=False)
    t_M.loc[:,"cilac"] = sact['cilac']
    t_M.to_csv(baseDir + "log/basics/cilac_shape.csv.gz",compression="gzip",index=False)

if False:
    tlib.plotFeatCorr(t_M)
    tL = ["t_type","t_tech","t_dist","t_median","t_std","t_trend2","t_conv"]
    tlib.plotFeatCorrScatter(t_M[tL])

plog('-----------------training-correlation------------------')
importlib.reload(tlib)
metrL = ['reg','dif','cor','lin','ent','sqr']
thrL = [7,3.5,2.5,7,3.5,2.5]
binLev = {}
for i,m in enumerate(metrL):
    print("----------------------%s-------------------" % m)
    scoreI = scoreMax.loc[~np.isnan(scoreMax['y_'+m])].index
    y, _ = tlib.binVector(scoreMax['y_' + m][scoreI],nBin=7,threshold=thrL[i])
    binLev[m] = _
    tMod = tlib.trainMod(c_M.loc[scoreI],y)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/tank/tank_"+m+".json",test_size=.2)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/tank/tank_"+m+".json",test_size=.3)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/tank/tank_"+m+".json",test_size=.4)
    tMod.save(mod,baseDir + "train/tank/tank_"+m+idField+".pkl")
    sact.loc[:,"b_"+m] = mod.predict(c_M)
    print(trainR)

# scores = tlib.crossVal(mod,c_M.loc[scoreI],y,cv=5)
sact = sact.sort_values([idField,'b_cor','b_reg','t_dist'],ascending=False)
sact.to_csv(baseDir + "raw/tank/act.csv.tar.gz",compression="gzip",index=False)

if False:
    tlib.plotHist(scoreMax['y_cor'],lab="correlation",nBin=6,threshold=3.5)
    tlib.plotHist(scoreMax['y_reg'],lab="regression",nBin=6,threshold=5)
    tlib.plotHist(scoreMax['y_dif'],lab="difference",nBin=6,threshold=0.5)
    tMod.plotRoc()
    tMod.plotConfMat()
    fig, ax = plt.subplots(2,2)
    sact.boxplot(column="y_cor",by="b_cor",ax=ax[0,0])
    sact.boxplot(column="y_reg",by="b_reg",ax=ax[0,1])
    sact.boxplot(column="y_dif",by="b_dif",ax=ax[1,0])
    sact.boxplot(column="t_sum",by="b_dif",ax=ax[1,1])
    plt.show()

if False:
    scoreI = scoreMax.loc[~np.isnan(scoreMax['y_'+ metrL[i]])].index
    y, _ = tlib.binVector(scoreMax['y_' + m][scoreI],nBin=7,threshold=thrL[i])
    tMod = tlib.trainMod(c_M.loc[scoreI],y)
    tMod.tune(paramF=baseDir+"train/tank/tank.json",tuneF=baseDir+"train/tank/tank_tune.json")
    
plog('-----------------------summing-up----------------------------')
mist.loc[:,"sum"] = mist[hL].apply(lambda x: np.nansum(x),axis=1)
cvist = pd.read_csv(baseDir + "raw/tank/poi_tank_id_clust.csv")
if idField == "id_zone":
    cvist = pd.read_csv(baseDir + "raw/tank/poi_tank_id_zone.csv")

kpiL = pd.read_csv(StringIO("""suffix,variable,threshold
all,b_cor,0
b_cor-4,b_cor,4
y_cor-30,y_cor,0.30
y_reg-50,y_reg,0.50
y_lin-25,y_lin,0.25
b_lin-4,b_lin,4
y_ent-2,y_ent,2
b_ent-3,b_ent,3
y_sqr-5,y_sqr,0.5
b_sqr-3,b_sqr,3
"""))
                    
for i,k in kpiL.iterrows():
    kpiS = "_" + idField + "_" + k['suffix']
    print(kpiS)
    cilL = []
    for s in sact.groupby(idField):
        g = s[1]
        csum = g['b_cor'] > 0
        csum = g[k['variable']] > k['threshold']
        if bool(re.search('dif',k['variable'])):
            csum = np.cumsum(g[k['variable']]) < k['threshold']
        g = g[csum].sort_values('t_dist').head(10)
        cilL.append(g['cilac'].values)

    cilL  = np.unique(np.concatenate(np.array(cilL)))
    predMax = sact.loc[sact['cilac'].isin(cilL)]
    gact  = predMax.pivot_table(index=idField,values=hL,aggfunc=np.sum).replace(np.nan,0).reset_index()
    if predMax.shape[0] == 0:
        continue
    gact.loc[:,'n_cell'] = predMax.groupby(idField).apply(lambda x: len(x)).values
    scorI = scoreL.score(gact,mist,hL,idField).drop(columns=idField)
    gact = pd.concat([gact,scorI],axis=1)
    gact.loc[:,"sum_p"] = gact[hL].apply(lambda x: np.nansum(x),axis=1)
    gact = pd.merge(gact,mist[[idField,'c_source',"sum",'type']],left_on=idField,right_on=idField,how="left",suffixes=["","_v_san"])
    scor = pd.merge(cvist,gact[[idField,'y_cor','sum_p','n_cell']],left_on=idField,right_on=idField,how="left",suffixes=["","_max"])
    scor = pd.merge(scor,mist[[idField,'c_source',"sum",'type']],left_on=idField,right_on=idField,how="left",suffixes=["","_v_san"])
    scor.loc[:,"y_dif"] = np.abs(-scor["sum"]+scor["sum_p"])/(scor["sum"])
    scor = scor.sort_values("y_dif")
    importlib.reload(shl)
    print("correlation over 0.6: %f" % (scor[scor['y_cor']>0.6].shape[0]/scor[~np.isnan(scor['y_cor'])].shape[0]) )
    print("n_covered: " + str(scor[~np.isnan(scor['y_cor'])].shape[0]))
    #shl.kpiDis(scor,idField,nRef=sum(~np.isnan(scor['y_cor'])))
    shl.kpiDis(scor,idField,saveF=baseDir+"www/f_mot/kpi"+kpiS+".png")
    scor.to_csv(baseDir + "raw/tank/out/activity_cor_"+fSux+kpiS+".csv",index=False)
    gact.to_csv(baseDir + "raw/tank/out/activity_poi_"+fSux+kpiS+".csv",index=False)

if False:
    tlib.plotHist(gact.loc[:,"y_cor"],lab="correlation")
    tlib.plotHist(scor.loc[:,"y_dif"],lab="correlation")
    fig, ax = plt.subplots(1,2)
    scor.boxplot(column="sum_p",ax=ax[0])
    scor.boxplot(column="sum",ax=ax[1])
    plt.show()

if False:
    sact1 = pd.concat([sact,t_M[tL]],axis=1)
    c_M, _ = t_f.binMatrix(np.array(t_M[tL]),nBin=6)
    c_M = pd.DataFrame(c_M)
    c_M.columns = [re.sub("t_","c_",x) for x in tL]
    sact1 = pd.concat([sact1,c_M],axis=1)
    sact1 = pd.concat([sact1,scoreMax],axis=1)
    sact1.to_csv(baseDir + "raw/tank/activity_score.csv",index=False) #for scoring
    pd.melt(sact,id_vars=[idField,"cilac"],value_vars=hL).to_csv(baseDir + "raw/tank/activity_melt_dom.csv",index=False)
    
if False:
    gact = sact.pivot_table(index=idField,values=hL,aggfunc=np.sum).replace(np.nan,0).reset_index()
    gact.loc[:,'chirality'] = gact['id_clust'].apply(lambda x: int(x.split("-")[1]))
    lact1 = gact[hL][gact['chirality']==1].sum(axis=0)
    lact2 = gact[hL][gact['chirality']==0].sum(axis=0)
    plt.plot(lact1.values,label="chirality +")
    plt.plot(lact2.values,label="chirality -")
    mist.loc[:,'chirality'] = mist['id_clust'].apply(lambda x: int(x.split("-")[1]))
    mist1 = mist[hL][mist['chirality']==1].sum(axis=0)
    mist2 = mist[hL][mist['chirality']==0].sum(axis=0)
    plt.plot(mist1.values,label="chirality +")
    plt.plot(mist2.values,label="chirality -")
    plt.legend()
    plt.show()

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
