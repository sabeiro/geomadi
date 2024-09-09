#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
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
importlib.reload(t_k)

def plog(text):
    print(text)

custD = "tank"
idField = "id_poi"

if False:
    bast = pd.read_csv(baseDir + "log/bast/bast16.csv.gz",compression="gzip")
#    bast = pd.concat([bast,pd.read_csv(baseDir + "log/bast/bast17.csv.gz",compression="gzip")],axis=0)
    bast.loc[:,"count"] = bast["dir1"] + bast["dir2"]
    bast = bast[['id_bast','date','hour','count']]
    bast.columns = ['id_bast','day','hour','count']
    sact = bast.pivot_table(index=["id_bast","day"],columns="hour",values="count",aggfunc=np.sum).reset_index() 
    
else:
    mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_h.csv.gz",compression="gzip")
    mist.loc[:,"day"] = mist['time'].apply(lambda x: int("%s%s%s"%(x[2:4],x[5:7],x[8:10]) ))
    mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
    sact = mist.pivot_table(index=["id_poi","day"],columns="hour",values="ref",aggfunc=np.sum).reset_index()
    
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

sact.loc[:,"day"] = sact['day'].astype(str)
sact.loc[:,"day"] = sact['day'].apply(lambda x: "20%s-%s-%s" % (x[:2],x[2:4],x[4:6]))    

importlib.reload(t_k)
X, idL, den ,norm = t_k.splitInWeek(sact,idField=idField,isEven=True)
X = np.reshape(X,(X.shape[0],14,12))

X = t_k.loadMnist()

tK = t_k.weekImg(X)
if False:
    tK.plotImg()
    tK.plotTimeSeries(nline=8)

tK.simpleEncoder(epoch=25)
tK.deepEncoder(epoch=25,isEven=True)
tK.convNet(epoch=50,isEven=True)
if True:
    tK.plotMorph(nline=8)
#tK.trainCat(idL['id'],epoch=50)
encoder, decoder = tK.getEncoder(), tK.getDecoder()


rawIm = X.reshape((len(X), np.prod(X.shape[1:])))
encIm = encoder.predict(rawIm)
decIm = decoder.predict(encIm)

rawD = pd.DataFrame(rawIm.reshape(rawIm.shape[0],7,24).sum(axis=2))
decD = pd.DataFrame(decIm.reshape(decIm.shape[0],7,24).sum(axis=2))
rawD = pd.concat([idL,rawD],axis=1)
decD = pd.concat([idL,decD],axis=1)
corL = []
for i,g1 in rawD.groupby('id'):
    rawL = g1[[x for x in range(6)]].values.ravel()
    g2 = decD[decD['id'] == i]
    decL = g2[[x for x in range(6)]].values.ravel()
    corL.append({"id_poi":i,"cor":sp.stats.pearsonr(rawL,decL)[0]})

corL = pd.DataFrame(corL)
print("corr > 0.6 hour  - all days : %f" % (corL[corL['cor']>0.6].shape[0]/corL.shape[0]))
if True:
    plot('---------------------writing-down------------------------')
    wL = decIm.reshape(decIm.shape[0],7,24)*norm
    wL = wL.reshape(decIm.shape[0]*7,24)
    wL = pd.DataFrame(wL)
    wL.loc[:,"id_poi"] = den['id_poi'].values
    wL.loc[:,"day"] = den['day'].values
    vist = pd.melt(wL,value_vars=list(range(24)),id_vars=["id_poi","day"])
    vist.columns = ["id_poi","day","hour","ref"]
    if False:
        tist = vist[['day','ref']].groupby('time').agg(np.sum).reset_index()
        plt.plot(tist['ref'])
        plt.show()
    vist.loc[:,"time"] = vist.apply(lambda x:"%sT%02d:00:00" % (x['day'],int(x['hour'])),axis=1)
    vist = vist[['id_poi','time','ref']]
    vist.to_csv(baseDir + "raw/tank/ref_autoencoder.csv.gz",compression="gzip",index=False)

if False:
    x = corL['cor']
    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(x, n_bins, normed=1, histtype='step',cumulative=True, label='Empirical')
    ax.hist(x,bins=bins,normed=1,histtype='step',cumulative=-1,label='Reversed emp.')
    ax.grid(True)
    ax.legend()
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel('correlation')
    ax.set_ylabel('likelihood')
    plt.show()

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
    plt.hist(difAv['abs'],bins=20)
    plt.show()

if False:
    plt.plot(y1)
    plt.plot(y2)
    plt.show()

    plt.plot(X1.ravel())
    plt.plot(X2.ravel())
    plt.show()

    plt.imshow(X1)
    plt.show()
    
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
if False:
    ##https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Stundenwerte.html?nn=1817946
    bast = pd.read_csv(baseDir + "log/bast_2016.csv",sep=";")
    bast = bast[['Zst','Datum','Stunde','KFZ_R1','KFZ_R2','Lkw_R1','Lkw_R2']]
    bast.columns = ["id","day","hour","vehicle_r1","vehicle_r2","tir_r1","tir_r2"]
    bast.loc[:,"day"] = bast.day.astype(str)
    bast.loc[:,"day"] = bast['day'].apply(lambda x: "20%s-%s-%s" % (x[0:2],x[2:4],x[4:6]))
    bast.to_csv(baseDir + "log/bast_2016.csv.tar.gz",compression="gzip",index=False)

