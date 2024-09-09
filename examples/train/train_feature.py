"""
train_feature:
utils to describe feature importance, handle outliers 
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def variance(X):
    """reduce dimensionality based on low variance"""
    p_M = sp.stats.describe(X)
    varL = np.sqrt(p_M.variance)/p_M.mean
    cSel = [x for x in range(X.shape[1])]
    cNeg = [i for i,x in enumerate(varL) if np.isnan(x) or np.abs(x) < 1.]
    T = np.delete(X,cNeg,axis=1)
    return T, cNeg

def std(X,n_tail=2):
    """exclude the low standard deviation variables"""
    p_M = X.describe()
    varL = p_M.iloc[2,:]/p_M.iloc[1,:]
    varL = np.abs(varL)
    varL = varL.sort_values(ascending=True)
    return varL.head(varL.shape[0]-n_tail).index, varL
    
def chi2(X,y):
    """chisquare score"""
    X_new = SelectKBest(chi2,k=2).fit_transform(X,y)
    return X_new
    
def kmeans(X,n_clust=5):
    """cluster features by kmeans"""
    pca = PCA().fit(X)
    A_q = pca.components_.T
    if False:
        plt.imshow(A_q)
        plt.show()
        
    kmeans = KMeans(n_clusters=n_clust).fit(A_q)
    clusters = kmeans.predict(A_q)
    cluster_centers = kmeans.cluster_centers_
    dists = defaultdict(list)
    for i, c in enumerate(clusters):
        dist = euclidean_distances(A_q[i, :], cluster_centers[c, :])[0][0]
        dists[c].append((i, dist))

    return [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]

def treeClas(X,y):
    """feature importance by tree classifier"""
    clf = sk.ensemble.ExtraTreesClassifier()
    clf = clf.fit(X, y)
    clf.feature_importances_  
    model = sk.feature_selection.SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    X_new.shape
    return X_new
        
def regression(X,y):
    """evaluate feature importance with regression coefficients"""
    clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    fit = clf.fit(X,y)
    return fit.coef_

def regQuadratic(y):
    """regression with a 4th grade polynomial"""
    def ser_fun(x,t):
        return x[0] + x[1]*t + x[2]*t*t + x[3]*t*t*t + x[4]*t*t*t*t
    def ser_fun_min(x,t,y):
        return ser_fun(x,t) - y
    x0 = [2.29344311e-02,4.91473902e-03,-1.56686399e+01,-2.52343576e-04,2.91360123e-06,-9.72000000e-04]
    t = np.array(range(y.shape[0]))
    x1,n = least_squares(ser_fun_min,x0,args=(t,y))
    return ser_fun(x1,t)

class clustLib():
    def __init__(self):
        return

    def group2Nei(self,X1,X2,max_d=0.1,max_nei=None):
        cellL = pd.DataFrame()
        for i in range(X1.shape[0]):
            x_c, y_c = X1[['x','y']].iloc[i]
            disk = ((X2['X']-x_c)**2 + (X2['Y']-y_c)**2)
            disk = disk.loc[disk <= max_d**2]
            if max_nei:
                if disk.shape[0] > max_nei:
                    disk = disk.sort_values()
                    disk = disk.head(max_nei)
            tmp = cells.loc[disk.index]
            tmp.loc[:,"cluster"] = cvist['cluster'].iloc[i]
            cellL = pd.concat([cellL,tmp],axis=0)
        cellL = cellL.groupby('cilac').head(1)

    def groupNei(self,X1,max_d=0.1,max_nei=None):
        Z = linkage(X1, 'ward')
        return fcluster(Z,max_d,criterion='distance')#k, criterion='maxclust')

    # tree = cKDTree(np.c_[tvist['x'],tvist['y']])
    # point_neighbors_list = []
    # for point in points:
    #     distances, indices = tree.query(point, len(points), p=2, distance_upper_bound=max_distance)
    #     point_neighbors = []
    #     for index, distance in zip(indices, distances):
    #         if distance == inf:
    #             break
    #         point_neighbors.append(points[index])
    #     point_neighbors_list.append(point_neighbors)
    
    def cluster(self,mact):
        """spectral clustering and area sorting"""
        A_mA = mact - mact.mean(1)[:,None]
        ssA = (A_mA**2).sum(1);
        rsum = mact.sum(axis=1).astype('float')
        psum = np.percentile(rsum,[x*100./4. for x in range(5)])
        rquant = pd.qcut(rsum,5,range(5)) > 1
        rquant = rsum > psum[2]
        act = act[rquant]
        mact = mact[rquant]
        rsum = mact.sum(axis=1).astype('float')
        mact = mact / rsum[:,np.newaxis]
        cpact1 = np.corrcoef(mact)
        #pact4 = pdist(mact,'correlation')
        # af = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,damping=0.5, max_iter=200, preference=-100, verbose=False).fit(cpact1)
        af = SpectralClustering(n_clusters=6, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity="rbf", n_neighbors=10, eigen_tol=0.0, assign_labels="kmeans", degree=3, coef0=1, kernel_params=None).fit(cpact1)
        plog(set(af.labels_))
        print(af.labels_)
        d = pd.DataFrame(cpact2)
        link = hc.linkage(d.values,method='centroid')
        o1 = hc.leaves_list(link)
        mat = d.iloc[o1,:]
        mat = mat.iloc[:, o1[::-1]]
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(cpact1)
        axarr[0,1].imshow(mat)
        plt.show()

def featureImportance(X,y,tL,method=0):
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from xgboost import XGBClassifier
    modelN = 'gradBoost'

    if method == 0:
        modelN = 'Extra tree classifier'
        mod = sk.ensemble.ExtraTreesClassifier(n_estimators=250,random_state=0)
        mod.fit(X, y)
        importances = mod.feature_importances_
        std = np.std([tree.feature_importances_ for tree in mod.estimators_],axis=0)

    if method == 1:
        modelN = 'select k best'
        mod = SelectKBest(score_func=chi2, k=X.shape[1])
        fit = mod.fit(X, y)
        importances = fit.scores_
        std = np.apply_along_axis(np.std,0,fit.transform(X))

    if method == 2: #recursive feature elimination
        modelN = 'logistic regression'
        mod = LogisticRegression()
        rfe = RFE(mod, 3)
        fit = rfe.fit(X, y)
        importances = fit.ranking_
        std = .1

    if method == 3:
        modelN = 'X gradBoost'
        mod = XGBClassifier()
        fit = mod.fit(X, y)
        importances = mod.feature_importances_
        std = .01#np.apply_along_axis(np.std,0,fit.transform(X))
        
    else:
        mod = sk.ensemble.GradientBoostingRegressor()
        mod.fit(X, y)
        importances = mod.feature_importances_
        std = np.std([tree[0].feature_importances_ for tree in mod.estimators_],axis=0)

    print(modelN)
    indices = np.argsort(importances)[::-1]
    impD = pd.DataFrame({"importance":importances,"std":std,"idx":np.argsort(importances)[::-1],"label":np.array(tL)[indices]})
    impD.sort_values("importance",inplace=True,ascending=False)
    impD = impD.reset_index()
    return impD, modelN

