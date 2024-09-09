#%pylab inline
##http://scikit-learn.org/stable/modules/ensemble.html
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src')
baseDir = os.environ['LAV_DIR']
outFile = "raw/activity_train.csv"
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import geomadi.train_lib as tlib
import geomadi.train_shapeLib as shl
import custom.lib_custom as t_l
import importlib

def plog(text):
    print(text)

idField = "id_poi"
    
sact = pd.read_csv(baseDir + "raw/mc/act_mc.csv.tar.gz",compression="gzip")
sact = sact[~np.isnan(sact[idField])]
sact[idField] = sact[idField].astype(int)
hL = sact.columns[[bool(re.search(':',x)) for x in sact.columns]]
vist = pd.read_csv(baseDir + "raw/mc/act_mc_visit.csv.tar.gz",compression="gzip")
importlib.reload(shl)
scoreL = shl.scoreLib()
scoreMax = scoreL.score(sact,vist,hL,idField)
scoreMax.loc[:,'id_poi'] = scoreMax['id_poi'].astype(int)
sact = pd.merge(sact,scoreMax,on=idField,how="left")
poi = pd.read_csv(baseDir + "raw/mc/poi_mc.csv")
poi.loc[:,'Sitenummer'] = poi['Sitenummer'].astype(int)
poi = poi.groupby('Sitenummer').head(1)
poi = pd.merge(sact[[idField]+[hL.values[0]]],poi,left_on=idField,right_on="Sitenummer",how="left")

if False:
    X1 = np.array(X1)
    X2 = np.array(X2)
    xcorr = np.corrcoef(X1,X2)
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(X1)
    ax[1].imshow(X2)    
    ax[2].imshow(xcorr)
    ax[3].hist(xcorr.ravel(),bins=80,range=(0.0, 1.0),fc='b',ec='b')
    ax[0].set_title("footfall")
    ax[1].set_title("visits")
    ax[2].set_title("matrix correlation")
    ax[3].set_title("single correlations")
    plt.show()
    xcorr = np.correlate(X1,X2,mode='full')
    #xcorr = sp.signal.correlate2d(X1,X2)
    xcorr = sp.signal.fftconvolve(X1, X2,mode="same")

if False:
    import cv
    templateNp = np.random.random( (100,100) )
    image = np.random.random( (400,400) )
    image[:100, :100] = templateNp
    resultNp = np.zeros( (301, 301) )
    templateCv = cv.fromarray(np.float32(template))
    imageCv = cv.fromarray(np.float32(image))
    resultCv =  cv.fromarray(np.float32(resultNp))
    cv.MatchTemplate(templateCv, imageCv, resultCv, cv.CV_TM_CCORR_NORMED)
    resultNp = np.asarray(resultCv)
    
plog('-----------------shape-feature-------------------')

importlib.reload(shl)
if False:
    plt.bar(p_M['rel_std'].index,p_M['rel_std'])
    plt.xticks(rotation=45)
    plt.show()

if False:
    corMat = t_M.corr()
    corrs = corMat.sum(axis=0)
    corr_order = corrs.argsort()[::-1]
    corMat = corMat.loc[corr_order.index,corr_order.index]
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(corMat, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.title('Correlation matrix between the features')
    plt.show()

if False:
    tL = ['t_loc','t_type','t_street','t_dens','t_reachpois','t_crosses','t_border']
    tL = ['t_m_trend2','t_slope','t_street','t_type','t_reachpois','t_std']
    pd.plotting.scatter_matrix(t_M[tL], diagonal="kde")
    plt.tight_layout()
    plt.show()

plog('-------------------------------------regression--------------------------------')

importlib.reload(shl)
importlib.reload(t_l)
importlib.reload(tlib)
 
hL = sact.columns[[bool(re.search(':',x)) for x in sact.columns]]
X1 = sact[hL]
X1.index = sact[idField]
X2 = vist[hL]
X2.index = vist[idField]
sact1 = sact[~np.isnan(sact['y_dif'])]
t_M, c_M = t_l.regressionMc(sact1,hL,poi)
c_M.index = sact1[idField]

if False:
    t_M.to_csv(baseDir + "raw/mc/attribute_table.csv",index=False)

vf = X1.loc[c_M.index].sum(axis=1)
vg = X2.loc[c_M.index].sum(axis=1)
r_quot, fit_q = tlib.regressor(c_M.values,vf,vg)
tlib.saveModel(fit_q,baseDir + "train/mc_correction"+idField+".pkl")
difD = pd.DataFrame({"act":vf,"ref":vg,"r_quot":vf*r_quot})
difD.loc[:,"dif"] = (difD['act'] - difD['ref'])/difD['ref']
difD.loc[:,"dif_quot"] = (difD['r_quot'] - difD['ref'])/difD['ref']
difD.loc[:,'y_cor'] = sact1['y_cor'].values
shl.kpiDis(difD,idField,baseDir+"geomadi/f_food/kpi"+"_corrected_"+idField+".png",col_cor="y_cor",col_dif="dif_quot",nRef=difD.shape[0])

t_M.index = sact[idField]
difD = pd.concat([difD,t_M],axis=1)
difD.loc[:,idField] = difD.index
difD = pd.merge(difD,poi,on=idField,how="left")
difD.loc[:,"catch_rate"] = difD['r_quot'].values/vf.values
difD.to_csv(baseDir + "raw/mc/score.csv",index=False)

sact1.loc[:,hL] = sact1[hL].multiply(r_quot,axis=0)
sact1.to_csv(baseDir + "raw/mc/corrected.csv.tar.gz",compression="gzip")

if False:
    shl.kpiDis(difD,idField,col_cor="y_cor",col_dif="dif_quot")#,baseDir+"geomadi/f_mot/kpi"+"_corrected_"+idField+".png",col_cor="y_cor",col_dif="dif_quot")
    shl.kpiDis(scoreMax,idField,col_cor="y_cor",col_dif="y_dif")

importlib.reload(tlib)
c_M, psum = tlib.binMatrix(t_M,nBin=6,threshold=2.5)
c_M.columns = t_M.columns
c_M.loc[:,"y"] = vg

tL = ['t_m_inter','t_std','t_sum','t_max',]
['t_interc','t_conv']

fig, ax = plt.subplots(3,2)
for i,c in enumerate(t_M.columns[:]):
    c_M.boxplot(column="y",by=c,ax=ax[int(i/2),i%2])
    if i == 5:
        plt.show()


c_M = c_M.reset_index()
del c_M['id_poi']
tL = ['t_loc','t_type','t_street','t_dens','t_reachpois','t_crosses','t_border']
tL = ['t_loc', 't_act_dens', 't_street', 't_dens', 't_border', 't_slope','t_type']
tL = ['t_dens','t_sum','t_crosses','t_owner','t_reachpois','t_act_density','t_street','t_loc','t_busi','t_cafe','t_type','t_speed','t_border']
c_M, psum = tlib.binMatrix(t_M[tL],nBin=6,threshold=2.5)
c_M.columns = tL

tL = t_M.columns
c_M = t_M[tL]
X = c_M.values
X[X<0] = 0.
y = (vf - vg)/vg
y, _ = tlib.binVector(y,nBin=6,threshold=3.5)

importlib.reload(shl)
impD = shl.featureImportance(X,y,tL,method=3)
imp3 = impD

impD = imp3
plt.title("Feature importances")
plt.bar(impD.index,impD['importance'],color="r", yerr=impD['std'], align="center")
plt.xticks(impD.index,impD['label'],rotation=45)
plt.show()


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
