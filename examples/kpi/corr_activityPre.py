#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
gradMeter = 111122.19769899677
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from scipy.stats import multivariate_normal as mvnorm

from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from tzlocal import get_localzone
tz = get_localzone()

outFile = "out/activity_clust.csv"
nBin = 5
inFile = baseDir + "out/tank_visit_bon.csv"
if(len(sys.argv) > 1):
    inFile = sys.argv[1]
#vist = pd.read_csv(baseDir + "out/tank_visit_sani.csv",index_col=0)
#vist = pd.read_csv(baseDir + "out/tank_visit_max.csv",index_col=0)
vist = pd.read_csv(inFile,index_col=0)


def plog(text):
    print(text)

def binning_outlier(y,n):
    threshold = 3.5
    ybin = [threshold] + [x*100./float(n-1) for x in range(1,n-1)] + [100.-threshold]
    pbin = np.unique(np.percentile(y,ybin))
    delta = (pbin[n-1]-pbin[0])/float(n-1)
    pbin = [np.min(y)] + [x*delta + pbin[0] for x in range(n)] + [np.max(y)]
    if False:
        plt.hist(y)
        plt.hist(y,bins=pbin)
        plt.show()
        sigma = np.std(y) - np.mean(y)
    t = pd.cut(y,bins=np.unique(pbin),labels=range(len(np.unique(pbin))-1),right=True,include_lowest=True).replace(np.nan,-1)
    return t, pbin

vist.loc[:,'t_type'] = pd.factorize(vist['type'])[0]
vist.index = vist['id_poi']

hL = vist.columns[[bool(re.search(':',x)) for x in vist.columns]]
mact = vist.loc[:,hL]

tact = pd.read_csv(baseDir + "out/tank_activity.csv",index_col=0)
sact = pd.read_csv(baseDir + "out/tank_activity_h.csv",index_col=0)

clf = LinearRegression()

hL = sact.columns[[bool(re.search(':',x)) for x in sact.columns]]
sact.loc[:,"y_corr"] = 0
sact.loc[:,"y_reg"] = 0
sact.loc[:,"t_type"] = 0
for i in mact.index:
    selI = sact['id_poi'] == i
    if any(selI)==False:
        continue
    X = np.array(sact.loc[selI,hL])#.transpose()
    rsum = np.nansum(X,axis=1).astype('float')
    X = X / rsum[:,np.newaxis]
    pline = mact.loc[i] 
    if sum(pline) <= 0.:
        continue
    pline = pline / np.nansum(pline)
    ##cline = np.apply_along_axis(lambda x: np.correlate(x,pline,"valid"),axis=1,arr=msact)[:,0]
    cline = np.apply_along_axis(lambda x: sp.stats.pearsonr(x, pline),axis=1,arr=X)[:,0]
    X = scale(X.transpose())
    y = pline/pline.sum()
    fit = clf.fit(X,y) ##fit = sm.OLS(pline, msact).fit() ##fit = regressor.fit(msact,pline)
    sact.loc[selI,"y_corr"] = cline
    sact.loc[selI,"y_reg"] = fit.coef_
    sact.loc[selI,"t_type"] = int(vist.loc[i,"t_type"]) + 1
    # logit = sm.Logit(y,X).fit()
    # sact.loc[selI,"logit"] = logit.params
    # lda = LinearDiscriminantAnalysis().fit(X, y)
    # scaling = lda.scalings_[:, 0]


from scipy.optimize import leastsq as least_squares
X = np.array(tact.loc[:,[str(x) for x in range(6,24)]])
if False:
    rsum = X.sum(axis=1).astype('float')
    X = X / rsum[:,np.newaxis]
    X[np.isnan(X)] = 0

x0 = [-0.016734,0.019848,-0.000972]
def ser_sin(x,t,param):
    return x[0] + x[1] * t + x[2] * t * t

#x0 = [ 3.71041163,  3.66129372, -0.00799794]
# def ser_sin(x,t,param):
#     return x[0] + x[2]*(t-x[1])**2 

def ser_fun_min(x,t,y,param):
    return ser_sin(x,t,param) - y

t = np.array(range(X.shape[1])).astype('float') + 6.
convL = []
x1 = x0
for i in range(X.shape[0]):
    x1,n = least_squares(ser_fun_min,x1,args=(t,X[i],x0))
    convL.append(x1)

convL = pd.DataFrame(convL,columns=['t_inter','t_slope','t_convex']).replace(np.nan,0)
sact = sact.reset_index()
sact = pd.concat([sact,convL],axis=1)
    
print(sact.head(1))
hL = sact.columns[[bool(re.search(':',x)) for x in sact.columns]]
sact = sact.replace(np.nan,0)
msact = np.array(sact.loc[:,hL])#.transpose()
sact.loc[:,'t_max'] = np.max(msact,axis=1)
sact.loc[:,'t_std'] = np.std(msact,axis=1)
sact.loc[:,'t_sum'] = np.nansum(msact,axis=1)
sact.loc[:,'t_median'] = msact.argmax(axis=1) + 6
#sact.loc[:,'t_median'] = -sact['t_slope']*2./sact['t_convex']

psum = pd.DataFrame(index=range(nBin+2))
sact.loc[:,'z_reg'], psum['y_reg'] = binning_outlier(sact['y_reg'],nBin)
sact.loc[:,'c_max'], psum['c_max'] = binning_outlier(sact['t_max'],nBin)
sact.loc[:,'c_std'], psum['c_std'] = binning_outlier(sact['t_std'],nBin)
sact.loc[:,'c_sum'], psum['c_sum'] = binning_outlier(sact['t_sum'],nBin)
sact.loc[:,'c_dist'], psum['c_dist'] = binning_outlier(sact['t_dist'],nBin)
sact.loc[:,'c_median'], psum['c_median'] = binning_outlier(sact['t_median'],nBin)
sact.loc[:,'c_inter'], psum['c_inter'] = binning_outlier(sact['t_inter'],nBin)
sact.loc[:,'c_slope'], psum['c_slope'] = binning_outlier(sact['t_slope'],nBin)
sact.loc[:,'c_convex'], psum['c_convex'] = binning_outlier(sact['t_convex'],nBin)
sact.loc[:,'c_tech'] = pd.factorize(sact.BROADCAST_)[0]
sact.loc[:,'c_type'] = pd.factorize(sact['t_type'])[0]
factL = np.unique(sact.BROADCAST_)
psum.loc[range(len(factL)),'c_tech'] = factL
factL = np.unique(vist["t_type"])
psum.loc[range(len(factL)),'c_type'] = factL
psum.to_csv(baseDir + "out/activity_range.csv")
sact.to_csv(baseDir + outFile)

if False: ##pca
    X = scale(msact)
    y = pline
    pca = PCA().fit(X)
    y = np.std(pca.transform(X), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()
    foo = pca.transform(msact)
    bar = pd.DataFrame({"PC1":foo[:, 0],"PC2":foo[:, 1],"Class":y})
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    plt.show()
    lda = LinearDiscriminantAnalysis().fit(X, y)
    scaling = lda.scalings_[:, 0]
    xmin = np.trunc(np.min(X)) - 1
    xmax = np.trunc(np.max(X)) + 1
    ncol = len(set(y))
    binwidth = 0.5

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

##53 locations
##1046,1342 no cell data


