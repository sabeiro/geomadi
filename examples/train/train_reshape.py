"""
train_reshape:
reshape dataframes and time series in preparation for training.
"""

import random, csv, datetime, re, os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
import geomadi.series_stat as s_s

def day2time(hL,date_format="%Y-%m-%dT"):
    """transform day format into datetime (for plotting)"""
    return [datetime.datetime.strptime(x,date_format) for x in hL]

def hour2time(hL,date_format="%Y-%m-%dT%H:%M:%S"):
    """transform time format into datetime (for plotting)"""
    return day2time(hL,date_format=date_format)

def timeCol(g):
    """returns all the columns with time format"""
    return sorted([x for x in g.columns if bool(re.search("T",x))])

def overlap(hL,hL1):
    """return only the overlapping values"""
    return sorted(list(set(hL) & set(hL1)))

def date2isocal(hL,date_format="%Y-%m-%dT"):
    """transform a date into a isocalendar"""
    isocal = [datetime.datetime.strptime(x,date_format).isocalendar() for x in hL]
    if len(date_format) < 11:
        iL = ["%02d-%02dT" % (x[1],x[2]) for x in isocal]
    else:
        hour = [datetime.datetime.strptime(x,date_format).hour for x in hL]
        iL = ["%02d-%02dT%02d" % (x[1],x[2],y) for (x,y) in zip(isocal,hour)]
    return iL

def nonnull(y):
    """return only non null numbers"""
    return np.array([x for x in y if x == x])

def mergeDataframe(dfL,idField="id_poi"):
    """merge a list of dataframes into a single dataframe"""
    if len(dfL) > 1:
        tact = pd.merge(dfL[0],dfL[1],on=idField,how="outer",suffixes=["_x",""])
    else:
        tact = dfL[0]
    for i in range(2,len(dfL)):
        tact = tact.merge(dfL[i],on=idField,how="outer",suffixes=["_x",""])
    for i in [x for x in tact.columns if bool(re.search("_x",x))]:
        del tact[i]
    return tact

def mergeFiles(projD,fL,idField="id_poi"):
    """merge files into a dataframe"""
    dfL = []
    for f in fL:
      tact = pd.read_csv(projD+f,compression="gzip",dtype={idField:str})
      dfL.append(tact)
    return mergeDataframe(dfL,idField)

def mergeDir(projDir,idField="id_poi"):
    """merge all files in a directory"""
    fL = os.listdir(projDir)
    return mergeFiles(projDir,fL,idField)

def binOutlier(y,nBin=6,threshold=3.5,isLabel=False):
    """bin with special treatment for the outliers"""
    n = nBin
    ybin = [threshold] + [x*100./float(n-1) for x in range(1,n-1)] + [100.-threshold]
    pbin = np.unique(np.nanpercentile(y,ybin))
    n = min(n,pbin.shape[0])
    delta = (pbin[n-1]-pbin[0])/float(n-1)
    pbin = [np.nanmin(y).min()] + [x*delta + pbin[0] for x in range(n)] + [np.nanmax(y).max()]
    if False:
        plt.hist(y,fill=False,color="red")
        plt.hist(y,fill=False,bins=pbin,color="blue")
        plt.show()
        sigma = np.std(y) - np.mean(y)
    t = np.array(pd.cut(y,bins=np.unique(pbin),labels=range(len(np.unique(pbin))-1),right=True,include_lowest=True))
    t[np.isnan(t)] = -1
    t = np.asarray(t,dtype=int)
    if isLabel:
        return [pbin[x] for x in t], pbin
    return t, pbin

def binMatrix(X,nBin=6,threshold=2.5):
    """bin a continuum parametric matrix"""
    c_M = pd.DataFrame()
    psum = pd.DataFrame(index=range(nBin+2))
    for i in range(X.shape[1]):
        xcol = X[:,i]
        c_M.loc[:,i], binN = binOutlier(xcol,threshold=2.5)
        psum.loc[range(len(binN)),i] = binN
    return c_M, psum
    
def binVector(y,nBin=6,threshold=2.5):
    """bin a continuum parametric vector"""
    y, psum = t_f.binOutlier(y,nBin=nBin,threshold=threshold)
    y = np.array(y).astype(int)
    return y, psum

def binMask(typeV):
    """turn a comma separated list into a mask integer"""
    typeV = typeV.astype(str)
    typeV.loc[typeV=="nan"] = ""
    typeV = [re.sub("\]","",x) for x in typeV]
    typeV = [re.sub("\[","",x) for x in typeV]
    stypeL = [str(x).split(", ") for x in list(set(typeV))]
    stypeL = np.unique(list(chain(*stypeL)))
    maskV = []
    for i,p in enumerate(typeV):
        binMask = 0
        for j,t in enumerate(stypeL):
            binMask += 2**(j*bool(re.search(t,str(p))))
        maskV.append(binMask)
    return maskV

def id2dim(tist,sL,idField="id_poi"):
    """partition a dataset in learning matrices and reference data"""
    XL = []
    yL = []
    for i,g in tist.groupby(idField):
        X, y = g[sL].values, g['ref'].values
        XL.append(X)
        yL.append(y)
    XL = np.array(XL)
    yL = np.array(yL)
    return XL, yL

def id2group(sact,mist,hL,idField="id_poi"):
    """from two datasets create a learning matrix and reference data"""
    XL = []
    yL = []
    idL = []
    for i,g in sact.groupby(idField):
        setL = mist[idField] == i
        if sum(setL) == 0:
            continue
        y = mist.loc[setL,hL].values
        X = g[hL].values
        if y.sum() <= 0.:
            continue
        if X.sum() <= 0.:
            continue
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        XL.append(X.T)
        yL.append(y[0])
        idL.append(i)
    #XL = np.array(XL)
    #yL = np.array(yL)
    return XL, yL, idL

def factorize(X):
    """factorize all string fields"""
    if type(X).__module__ == np.__name__:
        return X
    for i in X.select_dtypes(object):
        X.loc[:,i], _ = pd.factorize(X[i])
    X = X.replace(float("Nan"),0)
    return X

def applyBackfold(X):
    """add the margin to the matrix mirroring the boundary"""
    # X = np.vstack((X[-2,],X[-1,],X,X[0,],X[1,],X[2,]))
    # X = np.hstack((X[:,-2].reshape(-1,1),X[:,-1].reshape(-1,1),X,X[:,0].reshape(-1,1),X[:,1].reshape(-1,1)))
    X = np.vstack((X[-1,],X,X[0,]))
    X = np.hstack((X[:,-1].reshape(-1,1),X,X[:,0].reshape(-1,1)))
    return X

def removeBackfold(X):
    """remove the margins from the image"""
    #return X[2:-3,2:-2]
    return X[1:-1,1:-1]

def applyInterp(X,step=3):
    """interpolate betwen the steps"""
    Y = np.zeros((X.shape[0]*step,X.shape[1]*step))
    Y[:] = np.nan
    Y[::step,::step] = X
    ix = np.arange(X.shape[1])/X.shape[1]
    iy = np.arange(X.shape[0])/X.shape[0]
    interp = sp.interpolate.interp2d(ix,iy,X,kind='cubic')
    ix = np.arange(Y.shape[1])/Y.shape[1]
    iy = np.arange(Y.shape[0])/Y.shape[0]
    return interp(ix,iy)
    
def removeInterp(X,step=3):
    """interpolate betwen the steps"""
    return X[::step,::step]
    
def applyRandom(X,delta=.5):
    """apply random number to the matrix"""
    G = np.random.rand(X.shape[0],X.shape[1])*delta - delta*.5
    return  X*(1. - G)

def weekMatrix(g,isBackfold=True,roll=0,col='value'):
    """define a week as a 7x24 matrix"""

def isocalInWeek(df,idField="id_poi",isBackfold=True,roll=0,col='value'):
    """transform a dataframe with isocalendar into a series of images with 7(d)x24(h) pixels"""
    hL = timeCol(df)
    dm = df.melt(id_vars=idField,value_vars=hL)
    dm.columns = [idField,"variable",col]
    dm.loc[:,"week"] = dm['variable'].apply(lambda x: x[0:2])
    dm.loc[:,"wday"] = dm['variable'].apply(lambda x: x[3:5])
    dm.loc[:,"hour"] = dm['variable'].apply(lambda x: x[6:8])
    dl = []
    for i,g in dm.groupby([idField,'week']):
        X = weekMatrix(g,isBackfold=isBackfold,roll=roll,col=col)
        if X == 0 : continue
        dl.append({idField:i[0],"week":i[1],"norm":norm,"values":X})
    return dl

def splitInWeek(df,idField="id_poi",isBackfold=True,roll=0,col='value',isInterp=False):
    """morph a time series into an image representing a week (7x24) for convolutional neural network training"""
    time = [datetime.datetime.fromtimestamp(x) for x in df['time']]
    df.loc[:,"day"]  = [x.strftime("%Y-%m-%dT") for x in time]
    df.loc[:,"hour"] = [x.hour for x in time]
    ical = [x.isocalendar() for x in time]
    df.loc[:,"wday"] = [x[2] for x in ical]
    df.loc[:,"week"] = ["%02d-%02dT" % (x[0],x[1]) for x in ical]
    dl = []
    for i,g in df.groupby([idField,'week']):
        X = g.pivot_table(columns="hour",index="wday",values=col,aggfunc=np.sum)
        X = X.values
        if X.shape != (7,24): continue
        if roll > 0:   X = np.roll(X,roll,axis=1)
        if isBackfold: X = applyBackfold(X)
        if isInterp:   X = s_s.interpMissing2d(X)
        norm = X.sum().sum()
        if norm < 1e-10: continue
        if norm != norm: continue
        X = X/norm
        dl.append({idField:i[0],"week":i[1],"norm":norm,"values":X})
    return dl

def dayInWeek(df,idField="id_poi"):
    """morph a time series into an image representing a week (7xN) for convolutional neural network training"""
    hL = [x for x in df.columns if bool(re.search("T",x))]
    wd = day2time(hL)
    X, den, idL, normL = [], [], [], []
    for i,g in df.groupby(idField):
        dw = pd.DataFrame({idField:i,"year":[x[0] for x in wd],"week":[x[1] for x in wd],"day":[x[2] for x in wd],"count":g.values[0][1:]})        
        dwP = dw.pivot_table(index=[idField,"year","week"],columns="day",values="count",aggfunc=np.sum)
        setL = ~dwP.isnull().any(axis=1)
        dwP = dwP.loc[setL,:]
        norm = dwP.values.max()
        X.append(dwP.values/norm)
        normL.append(norm)
        den.append(dwP.reset_index())
        idL.append(i)
    den = pd.concat(den)
    return X, idL, den, norm

def loadMnist():
    """download mnist digit dataset for testing"""
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    X = np.concatenate((x_train,x_test))
    X = np.reshape(X,(X.shape[0],28,28))
    return X

def splitLearningSet(X,y,f_train=0.80,f_valid=0):
    """split a learning set"""
    seed = 128
    rng = np.random.RandomState(seed)
    f_test = 1 - f_valid
    N = X.shape[0]
    shuffleL = random.sample(range(N),N)
    partS = [0,int(N*f_train),int(N*(f_test)),N]
    y_train = np.asarray(y[shuffleL[partS[0]:partS[1]]])
    y_test  = np.asarray(y[shuffleL[partS[1]:partS[2]]])
    y_valid = np.asarray(y[shuffleL[partS[2]:partS[3]]],dtype=np.int32)
    x_train = np.asarray(X[shuffleL[partS[0]:partS[1]]])
    x_test  = np.asarray(X[shuffleL[partS[1]:partS[2]]])
    x_valid = np.asarray(X[shuffleL[partS[2]:partS[3]]],dtype=np.int32)
    return y_train, y_test, y_valid, x_train, x_test, x_valid

def day2isocal(df,idField="id_poi",isDay=True):
    """convert a series of time over multiple years into a isocalendar time"""
    hL = df.columns[[bool(re.search('-??T',x)) for x in df.columns]]
    nhL = df.columns[[not bool(re.search('-??T',x)) for x in df.columns]]
    if isDay: ical = date2isocal(hL,date_format="%Y-%m-%dT")
    else: ical = date2isocal(hL,date_format="%Y-%m-%dT%H:%M:%S")
    df = df[list(nhL) + list(hL)]
    df.columns = list(nhL) + list(ical)
    mist = df.groupby(df.columns,axis=1).agg(np.mean)
    sist = df.groupby(df.columns,axis=1).agg(np.std)
    ucal = sist.columns[[bool(re.search('-??T',x)) for x in sist.columns]]
    sist.loc[:,ucal] = sist.loc[:,ucal]/mist.loc[:,ucal]
    mist.loc[:,nhL] = df.loc[:,nhL]
    sist.loc[:,nhL] = df.loc[:,nhL]
    return mist, sist

def hour2day(g):
    """remove time and add up to days"""
    hL = g.columns[[bool(re.search('-??T',x)) for x in g.columns]]
    nhL = g.columns[[not bool(re.search('-??T',x)) for x in g.columns]]
    dL = [x[:11] for x in hL]
    g = g[list(nhL) + list(hL)]
    g.columns = list(nhL) + list(dL)
    c = g.groupby(g.columns,axis=1).agg(sum)
    c.loc[:,nhL] = c.loc[:,nhL]
    return c
    
def isocal2day(g,dateL,idField="id_poi"):
    """express isocalendar columns into days from a reference list"""
    hL = [x for x in g.columns.values if bool(re.match(".*-.*",str(x)))]
    dL = pd.DataFrame({"isocal":hL})
    dL = pd.merge(dL,dateL[['day','isocal']],how="left",on="isocal")
    dL = dL[dL['day'] == dL['day']]
    g = g[[idField] + list(dL['isocal'])]
    cL = [{x['isocal']:x['day']} for i,x in dL.iterrows()]
    g.columns = [idField] + list(dL['day'])
    return g

def ical2date(hL,year=None):
    """return a look up list with date and isocalendar"""
    if year == None:
        year = datetime.datetime.today().year
    orig = datetime.datetime.strptime(str(year) + "-01-01","%Y-%m-%d")
    lL = [orig + datetime.timedelta(days=x) for x in range(365)]
    dL = [x.strftime("%Y-%m-%dT") for x in lL]
    ical = ["%02d-%02dT" % (x.isocalendar()[1],x.isocalendar()[2]) for x in lL]
    lookUp = pd.DataFrame({"date":dL
                           ,"ical":ical
                           })
    lookUp = lookUp.groupby("date").first().reset_index()
    lookUp = lookUp.groupby("ical").first().reset_index()
    lookUp = lookUp[lookUp['ical'].isin(ical)]
    return lookUp


def dateFromIso(iso_year,iso_week,iso_day):
    """date from isocalendar, 4th of January avoids calendar week 53 like in Gregorian calendar"""
    jan4 = datetime.date(iso_year, 1, 4)
    _, jan4_week, jan4_day = fourth_jan.isocalendar()
    return jan4 + datetime.timedelta(days=iso_day-jan4_day, weeks=iso_week-jan4_week)
