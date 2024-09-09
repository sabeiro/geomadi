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

def plog(text):
    print(text)

plog('-------------------load/def------------------------')
mist = pd.read_csv(baseDir + "raw/tank/visit_bon.csv")
#sact = pd.read_csv(baseDir + "raw/tank/tank_activity_"+fSux+".csv.tar.gz",compression="gzip")
sact = pd.read_csv(baseDir + "raw/tank/act_test.csv.tar.gz",compression="gzip")
hL = sact.columns[[bool(re.search('T??:',x)) for x in sact.columns]]
hL1 = mist.columns[[bool(re.search('T??:',x)) for x in mist.columns]]
hL = sorted(list(set(hL) & set(hL1)))

from sklearn.ensemble import ExtraTreesClassifier

for i,g in sact.groupby('id_clust'):
    X = g[hL].values.T
    if not any(mist['id_clust'] == i):
        continue
    y = mist[mist['id_clust'] == i][hL].values[0]
    
    # clf = ExtraTreesClassifier()
    # clf.fit(X, y)
    # print(clf.feature_importances_)
    # plt.plot(clf.feature_importances_)
    # plt.show()

    pca = PCA(n_components=3)
    fit = pca.fit(X)
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    shA_X = shared(X_train)
    linear_model = pm.Model()
    with linear_model: 
        alpha = pm.Normal("alpha", mu=y_train.mean(),sd=10)
        betas = pm.Normal("betas", mu=0, sd=1000, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sd=100) 
        #mu = alpha + np.array([betas[j]*shA_X[:,j] for j in range(X.shape[1])]).sum()
        mu = alpha + np.array([betas[j]*X_train[:,j] for j in range(X.shape[1])]).sum()
        likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y_train)
        #map_estimate = pm.find_MAP(model=linear_model, fmin=optimize.fmin_powell)
        step = pm.NUTS()
        trace = pm.sample(1000, step)


    basic_model = pymc3.Model()
    with basic_model:
        n,z,alpha,beta,alpha_post,beta_post,iterations = 50, 10, 12, 12, 22, 52, 100000
        iterations = 100000
        theta = pymc3.Beta("theta",alpha=alpha,beta=beta)
        y = pymc3.Binomial("y",n=n,p=theta,observed=z)
        start = pymc3.find_MAP() 
        step = pymc3.Metropolis()
        trace = pymc3.sample(iterations,step,start,random_seed=1,progressbar=True)
        

    pm.traceplot(trace);
    plt.show()
    ppc = pm.sample_ppc(trace, model=linear_model, samples=1000)
    list(ppc.items())[0][1].shape 
    for idx in range(10):
        predicted_yi = list(ppc.items())[0][1].T[idx].mean()
        actual_yi = y_te[idx]
        print(predicted_yi, actual_yi)
    
    sns.kdeplot(y_train, alpha=0.5, lw=4, c='b')
    for i in range(100):
        sns.kdeplot(ppc['likelihood'][i], alpha=0.1, c='g')
    plt.show()

print(sklearn.metrics.mutual_info_score(X[0],y))
cline = np.apply_along_axis(lambda x: sklearn.metrics.mutual_info_score(x,y),axis=1,arr=X)
##ordinary least square regression
beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X, X.T)), X), y)



import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (9, 9)
import seaborn as sns
from IPython.core.pylabtools import figsize
from scipy.stats import percentileofscore
from scipy import stats
# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import scipy
import pymc3 as pm

df = pd.read_csv('../../raw/tmp/student-mat.csv')
df = df[~df['G3'].isin([0, 1])]
df = df.rename(columns={'G3': 'Grade'})
df['percentile'] = df['Grade'].apply(lambda x: percentileofscore(df['Grade'], x))

def format_data(df):
    labels = df['Grade']
    df = df.drop(columns=['school', 'G1', 'G2', 'percentile'])
    df = pd.get_dummies(df)
    most_correlated = df.corr().abs()['Grade'].sort_values(ascending=False)
    most_correlated = most_correlated[:8]
    df = df.ix[:, most_correlated.index]
    df = df.drop(columns = 'higher_no')
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size = 0.25, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = format_data(df)

formula = 'Grade ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])
with pm.Model() as normal_model:
    family = pm.glm.families.Normal()
    pm.GLM.from_formula(formula,data=X_train,family=family)
    normal_trace = pm.sample(draws=2000,chains=2,tune=500,njobs=-1)

import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (9, 9)
import seaborn as sns
from IPython.core.pylabtools import figsize
from scipy.stats import percentileofscore
from scipy import stats
# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import scipy
import pymc3 as pm

df = pd.read_csv('../../raw/tmp/student-mat.csv')
df = df[~df['G3'].isin([0, 1])]
df = df.rename(columns={'G3': 'Grade'})
df['percentile'] = df['Grade'].apply(lambda x: percentileofscore(df['Grade'], x))

def format_data(df):
    labels = df['Grade']
    df = df.drop(columns=['school', 'G1', 'G2', 'percentile'])
    df = pd.get_dummies(df)
    most_correlated = df.corr().abs()['Grade'].sort_values(ascending=False)
    most_correlated = most_correlated[:8]
    df = df.ix[:, most_correlated.index]
    df = df.drop(columns = 'higher_no')
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size = 0.25, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = format_data(df)

formula = 'Grade ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])
with pm.Model() as normal_model:
    family = pm.glm.families.Normal()
    pm.GLM.from_formula(formula,data=X_train,family=family)
    normal_trace = pm.sample(draws=2000,chains=2,tune=500,njobs=-1)

def plot_trace(trace):
    ax = pm.traceplot(trace, figsize=(14, len(trace.varnames)*1.8),
                      lines={k: v['mean'] for k, v in pm.summary(trace).iterrows()})
    matplotlib.rcParams['font.size'] = 16
    for i, mn in enumerate(pm.summary(trace)['mean']):
        ax[i, 0].annotate('{:0.2f}'.format(mn), xy = (mn, 0), xycoords = 'data', size = 8,
                          xytext = (-18, 18), textcoords = 'offset points', rotation = 90,
                          va = 'bottom', fontsize = 'large', color = 'red')

plot_trace(normal_trace);
plt.show()

model_formula = 'Grade = '
for variable in normal_trace.varnames:
    model_formula += ' %0.2f * %s +' % (np.mean(normal_trace[variable]), variable)

print(' '.join(model_formula.split(' ')[:-1]))
