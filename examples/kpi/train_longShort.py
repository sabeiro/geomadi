#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re, time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph
import geomadi.train_keras as t_k
import geomadi.train_execute as t_e
import geomadi.train_shapeLib as shl
import importlib

def plog(text):
    print(text)

custD = "mc"
custD = "tank"
idField = "id_poi"
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
if custD == "tank":
    poi = poi[poi['competitor'] == 0]
poi.loc[:,"type"] = poi['type'].apply(lambda x: x.split(" ")[0])
if False:
    shl.plotOccurrence(poi['type'])
    plt.show()

dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: x+"T")
dateL = dateL[dateL['use'] > 0]

gact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip")
mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
bast = pd.read_csv(baseDir + "raw/"+custD+"/bast_iso.csv.gz",compression="gzip")
riso = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso.csv.gz",compression="gzip")
riso = t_e.isocal2day(riso,dateL,idField)
if custD == "mc":
    riso.loc[:,"type"] = pd.merge(riso,poi,on=idField,how="left")['type']
    gist = riso.groupby('type').agg(np.mean).reset_index()
    hL = [x for x in mist.columns if bool(re.search("T",x))]
    for i,g in gist.groupby('type'):
        setL = riso['type'] == i
        riso.loc[setL,hL] = g[hL].values
dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount_d.csv.gz",compression="gzip")
bast = t_e.isocal2day(bast,dateL,idField)

tist = t_e.joinSource(gact,mist,how="left",idField=idField)
#tist = t_e.concatSource(tist,dirc,how="outer",idField=idField,varName="foot")
tist = t_e.concatSource(tist,bast,how="outer",idField=idField,varName="bast")
tist = t_e.concatSource(tist,riso,how="outer",idField=idField,varName="hist")
tist.loc[:,"type"] = pd.merge(tist,poi,on=idField,how="left")['type']
tist.dropna(inplace=True)
#tist.replace(float('nan'),0.,inplace=True)
tist = pd.merge(tist,dateL[['day','Tmax','cloudCover','humidity']],left_on="time",right_on="day").sort_values([idField,'time'])
tist.loc[:,"time"] = tist['time'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%dT").strftime("%s"))
tist.loc[:,"time"] = tist['time'].astype(int)
tL = ['time','act','bast','hist','Tmax','cloudCover','humidity']

if False:
    plog('--------------plot-time-series-----------------')
    iL = np.unique(tist[idField])
    g = tist[tist[idField] == random.choice(iL)]
    g3 = g[['ref']+ tL]
    shl.plotTimeSeries(g3)
    plt.show()

plog('----------------init-model------------------')
importlib.reload(t_e)
importlib.reload(t_k)
iL = np.unique(tist[idField])
g = tist[tist[idField] == random.choice(iL)]
X, y = g[tL], g['ref']
tK = t_k.timeSeries(X.values)
model, kpi = tK.forecastSingle(y.values,look_back=2,ahead=30,epochs=100,isPlot=True)
print(kpi)

plog('----------------forecastability------------------')
importlib.reload(t_e)
importlib.reload(t_k)
if False:
    lPer = []
    t_start = time.clock()
    for i,g in tist.groupby(idField):
        X, y = g[tL], g['ref']
        tK = t_k.timeSeries(X.values)
        model, kpi = tK.forecastSingle(y.values,look_back=2,ahead=30,epochs=100,isPlot=False)
        kpi[idField] = i
        print(kpi)
        lPer.append(kpi)

    lPer = pd.DataFrame(lPer)
    lPer.dropna(inplace=True)
    lPer.to_csv(baseDir + "raw/"+custD+"/scor_forecastability.csv",index=False)
    t_diff = time.clock() - t_start
    print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/len(iL)))

if False:
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"forecastability"] = pd.merge(poi,lPer,on=idField,how="left")['cor']
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)


if False:
    plog('-------------------------knock-out--------------------------')
    perfL = tK.featKnockOut(X,y)
    tK.plotFeatImportance(perfL)
    tK.plotHistory()

importlib.reload(t_e)
importlib.reload(t_k)
iL = np.unique(tist[idField])
g = tist[tist[idField] == random.choice(iL)]
X, y = g[tL], g['ref']
tK = t_k.timeSeries(X.values)
model, kpi = tK.forecastLongShort(y.values,ahead=30,epoch=200,n_in=2)
tPer = []
predY = []
t_start = time.clock()
for i,g in tist.groupby(idField):
    X, y = g[tL], g['ref']
    tK.setX(X.values)
    model, kpi = tK.forecastLongShort(y.values,ahead=30,epoch=200,n_in=2)
    kpi[idField] = i
    print(kpi,"\r",end="\n",flush=True)
    tPer.append(kpi)
    # X2, y2 = prepareMatrix(g2)
    # y_pred, y_test = tK.predict(X2,y2)
    # predY.append(pd.DataFrame({idField:j,"pred":y_pred,"ref":y_test}))

# predY = pd.concat(predY,axis=0)
# tPer.to_csv(baseDir + "raw/"+custD+"/act_longShort.csv.gz",index=False,compression="gzip")
t_diff = time.clock() - t_start
print("total time %.2f min - per poi %.2f sec" % (t_diff/60.,t_diff/len(iL)))
    
tPer = pd.DataFrame(tPer)
tPer.dropna(inplace=True)
tPer.to_csv(baseDir + "raw/"+custD+"/scor_longShort.csv",index=False)

if False:
    importlib.reload(shl)    
    fig, ax = plt.subplots(1,2)
    shl.plotConfidenceInterval(tPer['cor'],ax=ax[0],label="correlation",nInt=10,color="green")
    shl.plotConfidenceInterval(tPer['rel_err'],ax=ax[1],label="relative error",nInt=10)
    plt.show()

    importlib.reload(shl)
    fig, ax = plt.subplots(1,2)
    shl.plotHistogram(tPer['cor'],label="correlation",ax=ax[0])
    shl.plotHistogram(tPer['rel_err'],label="relative error",ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1,2)
    shl.kpiDis(tPer,tLab="long short term",col_cor="cor",col_dif="rel_err",col_sum="rel_err",isRel=True,ax=ax[0])
    shl.kpiDis(tPer,tLab="long short term",col_cor="cor",col_dif="dif",col_sum="dif",isRel=False,ax=ax[1])
    plt.show()
    
    #scorM = t_e.scorPerf(predY,step="longShort",idField=idField)
 
if False:
    tPer.loc[:,"type"] = pd.merge(tPer,poi,on=idField,how="left")['type']
    tPer.loc[:,"type_test"] = pd.merge(tPer,poi,left_on="test",right_on=idField,how="left")['type_y']
    fig, ax = plt.subplots(1,2)
    tPer.boxplot(column="rel_err",by="type",ax=ax[0])
    tPer.boxplot(column="cor",by="type",ax=ax[1])
    for i, a in enumerate(ax.flat):
        for tick in a.get_xticklabels():
            tick.set_rotation(15)
    plt.show()

if False:
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"var_longShort"] = pd.merge(poi,tPer,on=idField,how="left")['var']
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    plog('-------------------------knock-out-over-all-locations-------------------------')
    tL = ['time', 'act', 'foot', 'bast', 'hist', 'Tmax', 'cloudCover','humidity']
    kFold = 40
    sL = np.unique(tist['type'])
    iL = np.unique(tist.loc[tist['type']==sL[0],idField])
    il = random.choice(iL)
    jl = random.choice(iL)
    X, y = prepareSet(tL,iL,i=il,j=jl)
    ahead = int(X.shape[0]/2)
    tK = t_k.timeSeries(X.values)
    model, kpi = tK.forecastLongShort(y.values,ahead=ahead,epoch=200,n_in=2)
    perfL = []
    for j in range(kFold):
        X, y = prepareSet(tL,iL)
        tK.setX(X.values)
        model, kpi = tK.forecastLongShort(y.values,ahead=ahead,epoch=50,n_in=2)
        kpi['feature'] = "all"
        perfL.append(kpi)
    for i in range(len(tL)):
        tL1 = tL[:i] + tL[i+1 :]
        print(tL[i])
        X, y = prepareSet(tL1,iL,i=il,j=jl)
        tK.setX(X.values)
        tK.delModel()
        model, kpi = tK.forecastLongShort(y.values,ahead=ahead,epoch=200,n_in=2)
        for j in range(kFold):
            X, y = prepareSet(tL1,iL)
            tK.setX(X.values)
            model, kpi = tK.forecastLongShort(y.values,ahead=ahead,epoch=50,n_in=2)
            kpi['feature'] = "- " + tL[i]
            perfL.append(kpi)
    perfL = pd.DataFrame(perfL)
    tK.plotFeatImportance(perfL)
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

