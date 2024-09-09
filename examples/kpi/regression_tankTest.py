#%pylab inline
##http://scikit7-learn.org/stable/modules/ensemble.html
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import geomadi.train_lib as tlib
import geomadi.train_shapeLib as shl
import importlib
import custom.lib_custom as t_l

def plog(text):
    print(text)

idField = "id_clust"
#idField = "id_zone"
fSux  = "20"
kpiS  = "_" + fSux + "_" + idField + "_" + "y_cor-30"
gact  = pd.read_csv(baseDir + "raw/tank/out/activity_poi"+kpiS+".csv")
scor  = pd.read_csv(baseDir + "raw/tank/out/activity_cor"+kpiS+".csv")
poi   = pd.read_csv(baseDir + "raw/tank/poi.csv")
cvist = pd.read_csv(baseDir + "raw/tank/poi_tank_id_clust.csv")
vist  = pd.read_csv(baseDir + "raw/tank/visit_max.csv")
if idField == "id_zone":
    cvist = pd.read_csv(baseDir + "raw/tank/poi_tank_id_zone.csv")
    vist = pd.read_csv(baseDir + "raw/tank/tank_visit_max_zone.csv")
scor = pd.merge(scor,cvist,left_on=idField,right_on=idField,how="left",suffixes=["","_y"])
hL = gact.columns[[bool(re.search('T??:',x)) for x in gact.columns]]
hL1 = vist.columns[[bool(re.search('T??:',x)) for x in vist.columns]]
hL = sorted(list(set(hL) & set(hL1)))


gact.sort_values("sum_p",inplace=True,ascending=False)

if False:
    tlib.plotMatrixCorr(X1,X2)
    
if False:
    import cv2
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    res = cv2.matchTemplate(X1,X2,'cv2.TM_CCOEFF')
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
                  
    templateNp = np.random.random( (100,100) )
    image = np.random.random( (400,400) )
    image[:100, :100] = templateNp
    resultNp = np.zeros( (301, 301) )
    templateCv = cv2.fromarray(np.float32(template))
    imageCv = cv2.fromarray(np.float32(image))
    resultCv =  cv2.fromarray(np.float32(resultNp))
    cv2.MatchTemplate(templateCv, imageCv, resultCv, cv.CV_TM_CCORR_NORMED)
    resultNp = np.asarray(resultCv)
    
if False:
    c_M, psum = tlib.binMatrix(t_M,nBin=6,threshold=2.5)
    c_M.columns = t_M.columns
    c_M.loc[:,"t_type"] = t_M['t_type']
    c_M.loc[:,"t_n_source"] = t_M['t_n_source'].astype(int)
    tlib.plotBoxFeat(c_M[['t_type','t_sum','t_n_source','t_n_cell']],scor1['y_dif'])

plog('-------------------------------------regression--------------------------------')
importlib.reload(shl)
importlib.reload(t_l)
importlib.reload(tlib)
X1 = gact[hL]
X1.index = gact[idField]
X2 = vist[hL]
X2.index = vist[idField]
scor1 = scor[~np.isnan(scor['y_dif'])]
t_M, c_M = t_l.regressionTank(scor1)

c_M.index = scor1[idField]
vf = X1.loc[c_M.index].sum(axis=1)
vg = X2.loc[c_M.index].sum(axis=1)
r_quot, fit_q = tlib.regressor(c_M.values,vf,vg)
tlib.saveModel(fit_q,baseDir + "train/tank_correction"+idField+".pkl")
difD = pd.DataFrame({"act":vf,"ref":vg,"r_quot":vf*r_quot})
difD.loc[:,"dif"] = (difD['act'] - difD['ref'])/difD['ref']
difD.loc[:,"dif_quot"] = (difD['r_quot'] - difD['ref'])/difD['ref']
difD.loc[:,'y_cor'] = scor1['y_cor'].values
difD.to_csv(baseDir + "raw/tank/act_corrected.csv")
shl.kpiDis(difD,idField,baseDir+"geomadi/f_mot/kpi"+"_corrected_"+idField+".png",col_cor="y_cor",col_dif="dif_quot")
shl.kpiDis(difD,idField,col_cor="y_cor",col_dif="dif_quot")

#t_M, c_M = t_l.regressionTank(scor)
#corr_f = fit_q.predict(c_M)

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
