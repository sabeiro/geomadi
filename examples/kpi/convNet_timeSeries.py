import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as t_l
import geomadi.series_stat as s_l
import geomadi.train_model as tlib
import geomadi.train_reshape as t_r
import geomadi.train_shape as t_s
import geomadi.train_score as t_c
import geomadi.train_convNet as t_k
import geomadi.train_keras as t_K
import geomadi.train_viz as t_v
import custom.lib_custom as l_c
import geomadi as gm

custD = "bast"
idField = "id_poi"

print('----------------------------load----------------------------------')
#dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirc_via_feb_h.csv.gz",compression="gzip",dtype={idField:str})
dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount_h.csv.gz",compression="gzip",dtype={idField:str})
hL = t_r.timeCol(dirc)
iL = t_r.date2isocal(hL,date_format="%Y-%m-%dT%H:%M:%S")
dirc = dirc[[idField] + hL]
dirc.columns = [idField] + iL

refi = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_h.csv.gz",compression="gzip")
refi = refi[refi[idField] == refi[idField]]
refi.loc[:,idField] = refi[idField].apply(lambda x: str(int(x)))
hL1 = t_r.timeCol(refi)
iL1 = [x[:8] for x in hL1]
refi = refi[[idField] + hL1]
refi.columns = [idField] + iL1

if False:
    print('----------------intersect-data-frame--------------------')
    hL = t_r.overlap(iL,iL1)
    refi = refi[[idField] + hL]
    dirc = dirc[[idField] + hL]
    refi = refi[refi[idField].isin(dirc[idField])]
    dirc = dirc[dirc[idField].isin(refi[idField])]
    print(refi.shape,dirc.shape)

import importlib
importlib.reload(t_r)
dlX = t_r.isocalInWeek(dirc,isBackfold=True,roll=0)
dlY = t_r.isocalInWeek(refi,isBackfold=True)
YL = t_r.loadMnist()
XL = np.array([x['values'] for x in dlX])
YL = np.array([x['values'] for x in dlY])
ZL = np.load(baseDir+"raw/"+custD+"/dictionary.npy")
ZL = np.array([t_r.applyBackfold(x) for x in ZL])

print('----------------------train--------------------')

import importlib
importlib.reload(s_l)
importlib.reload(t_l)
importlib.reload(t_s)
importlib.reload(l_c)

#YL = np.array([x['values'] for x in dlY[:1000]])
#YL = np.array([t_r.applyInterp(x,step=3) for x in YL])
importlib.reload(t_r)
DL = np.load(baseDir+"raw/"+custD+"/dictionary.npy")
ZL = np.array([t_r.applyBackfold(x) for x in DL])
#ZL = np.array([t_r.applyInterp(x,step=3) for x in ZL])
version = ["shortConv","interp","deep","convNet","convFlat"]
version = version[0]
optL = ["sgd","adadelta","adam"]
importlib.reload(t_k)
tK = t_k.weekImg(YL,model_type=version)
if False:
    tK.plotImg(nline=4)
    tK.plotTimeSeries(nline=8)
tK.runAutoencoder(epochs=300)

tK.setOptimizer(name=optL[1],lr=0.01,decay=1e-6)
tK.setX(ZL)
tK.runAutoencoder(epochs=500,split_set=1.)

tK.setOptimizer(name=optL[1])
tK.setX(YL)
tK.runAutoencoder(epochs=300)

tK.setOptimizer(name=optL[1],lr=0.01,decay=1e-6)
tK.setX(ZL)
tK.runAutoencoder(epochs=500,split_set=1.)

tK.setOptimizer(name=optL[1])
tK.setX(YL)
tK.runAutoencoder(epochs=300)

if False:
    autoencoder = tK.getModel()
    i = random.choice(range(len(dlY)))
    t_v.showPredicted(dlY[i],autoencoder,isPlot=True)

if False:
    tK.plotHistory()
    tK.plotMorph(nline=4,n=6)

corL = []
for g in dlY:
    cor = t_v.showPredicted(g,autoencoder,isPlot=False)
    corL.append(cor)
corL = pd.DataFrame(corL)
print(corL.head(1))

if False:
    i = random.choice(range(len(dlX)))
    g = dlX[i]
    X = g['values']
    Y = list(filter(lambda x: (g[idField] == x[idField]) & (g['week'] == x['week']),dlY))
    Y = Y[0]['values']
    t_v.showPredicted(g,autoencoder,X1=Y,isPlot=True)

corX = []
for g in dlX:
    Y = list(filter(lambda x: (g[idField] == x[idField]) & (g['week'] == x['week']),dlY))
    if len(Y) <= 0 : continue
    Y = Y[0]['values']
    cor = t_v.showPredicted(g,autoencoder,X1=Y,isPlot=False)
    corX.append(cor)
corX = pd.DataFrame(corX)
print(corX.head(1))

if False:
    corL.to_csv(baseDir + "raw/"+custD+"/scor/scor_"+version+".csv",index=False)
    corX.to_csv(baseDir + "raw/"+custD+"/scor/scor_ext_"+version+".csv",index=False)
    tK.saveModel(baseDir + "train/"+custD+"/"+version)
# json.dump(autoencoder.to_json(),open(baseDir + "train/"+custD+"/"+version+".json","w"))
# autoencoder.save_weights(baseDir + "train/"+custD+"/"+version+".h5")


if False:
    print('-----------------prediction-performance-on-external-data------------------')
    corX = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_ext_"+version+".csv")
    corX.columns = ['cor_auto','cor_ext','cor_pred','err_auto','err_ext','err_pred',idField,'week']
    fig, ax = plt.subplots(1,2)
    corX.boxplot(column=['cor_auto','cor_ext','cor_pred'],ax=ax[0])
    corX.boxplot(column=['err_auto','err_ext','err_pred'],ax=ax[1])
    plt.show()

    t_v.plotJoin(corX,col_ref="cor_auto",col1="cor_ext",col2="cor_pred")


if False:
    print('-------------compare-scores-----------------')
    corN = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_noBackfold.csv")
    corB = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_backfold.csv")
    #corc = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_convNet.csv")
    corc = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_convFlat.csv")
    fig, ax = plt.subplots(3,2)
    t_v.plotConfidenceInterval(corN['cor'],ax=ax[0][0],label="correlation",nInt=10,color="green")
    t_v.plotConfidenceInterval(corN['err'],ax=ax[0][1],label="relative error",nInt=10)
    t_v.plotConfidenceInterval(corB['cor'],ax=ax[1][0],label="correlation",nInt=10,color="green")
    t_v.plotConfidenceInterval(corB['err'],ax=ax[1][1],label="relative error",nInt=10)
    t_v.plotConfidenceInterval(corc['cor'],ax=ax[2][0],label="correlation",nInt=10,color="green")
    t_v.plotConfidenceInterval(corc['err'],ax=ax[2][1],label="relative error",nInt=10)
    plt.show()

if False:
    print('--------------comapare-ranking---------------')
    corN = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_noBackfold.csv")
    corB = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_backfold.csv")
    corc = pd.read_csv(baseDir + "raw/"+custD+"/scor/scor_convNet.csv")
    scor = pd.merge(corN,corB,on=[idField,"week"],how="inner",suffixes=["_n","_b"])
    scor = scor.merge(corc,on=[idField,"week"],how="inner",suffixes=["","_c"])
    fig, ax = plt.subplots(1,2)
    scor.boxplot(column=['cor_n','cor_b','cor'],ax=ax[0])
    scor.boxplot(column=['err_n','err_b','err'],ax=ax[1])
    plt.show()

    from pySankey import sankey
    scor.loc[:,"bin_n"] = ["%.2f"%x for x in t_r.binOutlier(scor['err_n'],nBin=5,isLabel=True)[0]]
    scor.loc[:,"bin_b"] = ["%.2f"%x for x in t_r.binOutlier(scor['err_b'],nBin=5,isLabel=True)[0]]
    scor.loc[:,"bin_c"] = ["%.2f"%x for x in t_r.binOutlier(scor['err'],nBin=5,isLabel=True)[0]]
    sankey.sankey(scor['bin_b'],scor['bin_c'],aspect=20,fontsize=12)
    plt.xticks([0,1],["no","yes"])
    plt.title("relative error shift - binned")
    plt.show()

    t_v.plotParallel(scor[[idField,'err_n','err_b','err']],idField)
    plt.show()

    t_v.plotParallel(scor[[idField,'cor_n','cor_b','cor']],idField)
    plt.show()

if False:
    from vis.utils import utils
    from vis.visualization import visualize_cam
    from vis.visualization import visualize_saliency, overlay
    autoencoder = tK.getModel()
    l = [x.name for x in autoencoder.layers]
    layer_idx = utils.find_layer_idx(autoencoder,l[-1])
    X = np.reshape(YL[0],(YL.shape[1],YL.shape[2],1))
    grads = visualize_cam(autoencoder,layer_idx,filter_indices=20,seed_input=X,backprop_modifier='relu')
    img = visualize_activation(autoencoder, layer_idx, filter_indices=filter_idx)

    
if False:
    hL = [x for x in sact.columns.values if bool(re.match("^[-+]?[0-9]+$",str(x)))]
    X = sact[hL].values#[(10*7):(11*7)]
    glL = []
    redF = t_s.reduceFeature(X)
    glL.append(redF.getMatrix()[:7])
    redF.interpMissing()
    redF.fit(how="poly")
    glL.append(redF.getMatrix()[:7])
    redF.smooth(width=3,steps=7)
    glL.append(redF.getMatrix()[:7])
    dayN = redF.replaceOffChi(sact['id_poi'],sact['day'])
    sact.loc[:,hL] = redF.getMatrix()
    
if False:
    idL.loc[:,"ref"] = rawIm.sum(axis=1)
    idL.loc[:,"act"] = decIm.sum(axis=1)
    def clampF(x):
        return pd.Series({"cor":sp.stats.pearsonr(x['ref'],x['act'])[0],"s_ref":sum(x['ref']),"s_act" :sum(x['act'])})
    scor = idL.groupby('id').apply(clampF).reset_index()
    print("corr > 0.6 hour  - all days : %f" % (scor[scor['cor']>0.6].shape[0]/scor.shape[0]))
    difIm = pd.DataFrame({"id":idL['id'],"dif":(rawIm - decIm).sum(axis=1)})
    difIm.loc[:,"abs"] = difIm['dif'].abs()
    difAv = difIm.groupby("id").agg(np.mean)
    p = np.percentile(difAv['abs'],[25,75])
    
if False:
    xcorr = np.corrcoef(X1,X2)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(xcorr)
    ax[1].hist(xcorr.ravel(),bins=80,range=(0.0,1.0),fc='b',ec='b')
    ax[0].set_title("matrix correlation")
    ax[1].set_title("single correlations")
    plt.show()
    xcorr = np.correlate(X1,X2,mode='full')
    #xcorr = sp.signal.correlate2d(X1,X2)
    xcorr = sp.signal.fftconvolve(X1, X2,mode="same")

if False:
    corV = sp.stats.pearsonr(tist['cor'],tist['ref'])[0]
    t = range(tist.shape[0])
    #t = [datetime.datetime.strptime(x,timeF) for x in tist[aggF]]
    plt.plot(t,tist['ref'],label="vis")
    plt.plot(t,tist['act'],label="act")
    plt.plot(t,tist['cor'],label="act_cor")
    plt.plot(t,tist['ref'],label="vis_cor")
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("id_clust %s corr %.2f" % (clustS,corV) )
    plt.legend()
    plt.show()

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
