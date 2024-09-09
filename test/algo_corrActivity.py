#%pylab inline
#http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html#reading-multivariate-analysis-data-into-python
import os, sys, gzip, random, csv, json, datetime
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.signal
from scipy.spatial.distance import pdist, wminkowski, squareform

def plog(text):
    print(text)

plog('----------------------------load----------------------------')
if False:
    fs = pd.read_csv(os.environ['LAV_DIR']+'raw/jll_cilac_selection.csv')
    fs1 = pd.read_csv(os.environ['LAV_DIR']+'raw/JLL_activities_test.csv')
    act = pd.merge(fs2[["ci-lac","x","y"]],fs1,left_on=['ci-lac'],right_on=["cilac"])
    act.index = np.asarray(act['cilac'],dtype=np.str)

poi = pd.read_csv(baseDir + "raw/poi_tank.csv")
poiA = pd.read_csv(baseDir + "raw/poi_sani_visits.csv")
poiA = pd.merge(poi,poiA,how='right',sort=False)
hL = ['6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21', '22', '23']
mact = np.array(poiA[hL])
mact[np.isnan(mact)] = 0
sact = pd.read_csv(baseDir + "raw/activity_clust.csv")
sact['poi'] = sact['poi'].apply(lambda x: int(x))
act = poiA
hsum = mact.sum(axis=0).astype('float')
if False:
    plt.plot(hsum)
    plt.show()

    
msact[np.isnan(msact)] = 0
print(msact)
print(mact)
if False:
    pd.plotting.scatter_matrix(msact,alpha=0.2,figsize=(6, 6),diagonal="kde")
    plt.tight_layout()
    plt.show()

# Rowwise mean of input arrays & subtract from input arrays themeselves
A_mA = mact - mact.mean(1)[:,None]
B_mB = msact - msact.mean(1)[:,None]
ssA = (A_mA**2).sum(1);
ssB = (B_mB**2).sum(1);
print(np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None])))

plog('----------------------------clean----------------------------')
rsum = mact.sum(axis=1).astype('float')
psum = np.percentile(rsum,[x*100./4. for x in range(5)])
rquant = pd.qcut(rsum,5,range(5)) > 1
rquant = rsum > psum[2]
act = act[rquant]
mact = mact[rquant]
rsum = mact.sum(axis=1).astype('float')
mact = mact / rsum[:,np.newaxis]
plog('------------------------------distance------------------------------')
if False:
    mDist1 = pdist(act[['x','y']],metric="euclidean")
    print(mDist1)
    from scipy.cluster.hierarchy import cophenet
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(act[['x','y']], 'ward')
    from scipy.cluster.hierarchy import fcluster
    clustD = 0.1
    act['zone'] = fcluster(Z, clustD, criterion='distance')#k, criterion='maxclust')
plog('-----------------------------stat-prop------------------------------')
def ser_sin(x,t,param):
    return x[0] + x[1] * t + x[2] * t * t

def ser_fun_min(x,t,y,param):
    return ser_sin(x,t,param) - y

x0 = [-0.016734,0.019848,-0.000972]
x1 = x0
from scipy.optimize import leastsq as least_squares
t = np.array(range(mact.shape[1])).astype('float')
convL = []
for i in range(mact.shape[0]):
    x1,n = least_squares(ser_fun_min,x1,args=(t,mact[i],x0))
    convL.append(x1)

pact = pd.DataFrame(np.median(mact,axis=1),columns=["median"])
pact['max'] = np.max(mact,axis=1)
pact['std'] = np.std(mact,axis=1)
pact['sum'] = rsum
convL = pd.DataFrame(convL)
pact = pd.concat([pact,convL],axis=1)
plog('--------------------------------affinity-matrices----------------------')
cpact1 = np.corrcoef(mact)
cpact2 = np.corrcoef(pact)
cpact3 = np.cov(mact)
cpact4 = pdist(mact,'correlation')
plog('----------------------------------clustering---------------------------')
from sklearn.cluster import AffinityPropagation, MeanShift, SpectralClustering
# af = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
#           damping=0.5, max_iter=200, preference=-100, verbose=False).fit(cpact1)
af = SpectralClustering(n_clusters=6, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol=0.0, assign_labels="kmeans", degree=3, coef0=1, kernel_params=None).fit(cpact1)
plog(set(af.labels_))
act.loc[:,'lab1'] = af.labels_
af = SpectralClustering(n_clusters=6, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol=0.0, assign_labels="kmeans", degree=3, coef0=1, kernel_params=None).fit(cpact2)
plog(set(af.labels_))
act.loc[:,'lab2'] = af.labels_
#af = MeanShift(bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1).fit(cpact3)
af = SpectralClustering(n_clusters=6, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol=0.0, assign_labels="kmeans", degree=3, coef0=1, kernel_params=None).fit(cpact3)
plog(set(af.labels_))
act.loc[:,'lab3'] = af.labels_
act.loc[:,'lab4'] = pd.cut(pact.iloc[:,5],bins=5,labels=range(5))

plog('-------------------------------write---------------------------------')
act.to_csv(baseDir+'out/poi_clustering.csv')

plog('------------------------sort-affinity-matrix-------------------------')
## order clusters
from scipy.cluster import hierarchy as hc
d = pd.DataFrame(cpact2)
link = hc.linkage(d.values, method='centroid')
o1 = hc.leaves_list(link)
mat = d.iloc[o1,:]
mat = mat.iloc[:, o1[::-1]]

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(cpact1)
axarr[1,0].imshow(cpact2)
axarr[0,1].imshow(mat)
#axarr[0,1].imshow(cpact3)
plt.show()

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')


