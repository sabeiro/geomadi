import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import urllib3, requests

def serDecompose(y,period=7):
    """decompose signal"""
    serD = pd.DataFrame({'y':y})
    serD.loc[:,"run_av"] = serRunAv(y,steps=period)
    serD.loc[:,"smooth"] = serSmooth(y,width=3,steps=period)
    serD['diff'] = serD['smooth'] - serD['smooth'].shift()
    serD['grad'] = serD['diff'] - serD['diff'].shift()
    serD['e_av'] = serD['y'].ewm(halflife=period).mean()
    serD['stat'] = serD['run_av'] - serD['diff']
    return serD

def ser3to1(x,y,z):
    return x**2 + y**2 + z**2

def getSeries(sData,baseUrl="http://example.com"):
    """time series download"""
    resq = requests.get(baseUrl+urllib3.urlencode(sData))##,'callback':'?'}))
    ser = np.array(resq.json()['data']).astype(np.float)
    timeL = pd.DataFrame({'t':(np.array(resq.json()['data'])[:,0])/1000-3600})
    timeL = timeL.t.apply(lambda x: datetime.datetime.fromtimestamp(x))
    trainD = pd.DataFrame({'y':ser[:,1]/1000000},index=[timeL[0]+datetime.timedelta(x) for x in range(timeL.size)])
    trainD.index = [pd.datetime.strptime(str(x)[0:10],'%Y-%m-%d') for x in trainD.index]
    trainD['t'] = ser[:,0]/1000000000
    trainD.dropna(inplace=True)
    tempD = trainD.copy(deep=True)
    tempD['week'] = [x.isocalendar()[1] for x in tempD.index]
    sWeek = tempD.groupby(["week"]).sum()
    sWeek['count'] = tempD.groupby(["week"]).size()
    sWeek['y'] = sWeek['y']/sWeek['count']
    normY = sWeek['y'].max()
    sWeek['y'] = sWeek['y']/normY
    base = datetime.datetime.strptime("2017-01-01","%Y-%m-%d")
    nWeek = sWeek.shape[0]*7
    sWeek.index = [base + datetime.timedelta(days=x) for x in range(0,nWeek,7)]
    sWeek.t = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in sWeek.index]
    sWeek['month'] = sWeek.index.month
    sMonth = sWeek.groupby(['month']).sum()
    sMonth['count'] = sWeek.groupby(['month']).size()
    sMonth['y'] = sMonth['y']/sMonth['count']
    normY = sMonth['y'].max()
    sMonth['y'] = sMonth['y']/normY
    sMonth.index = ['2017-%02d-01' % x for x in range(1,len(sMonth)+1)]
    sMonth.index = [datetime.datetime.strptime(x,"%Y-%m-%d") for x in sMonth.index]
    sMonth.t = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in sMonth.index]
    trainD['roll'] = serSmooth(trainD['y'],4,11)
    trainD['e_av'] = pd.ewma(trainD['y'],halflife=14)
    #trainD['e_av'] = pd.Series.ewm(trainD['y'],halflife=14)
    trainD['dev'] = trainD['y'] - trainD['y'].shift()
    trainD['diff'] = trainD['y'] - trainD['e_av']
    trainD['stat'] = trainD['dev']
    return trainD, sWeek, sMonth

def getStartParam(sData,baseUrl="http://example.com"):
    """retriev standard parameters"""
    resq = requests.get(baseUrl+urllib3.urlencode(sData))##,'callback':'?'}))
    x0 = resq.json()['data']
    return x0
##    return [x[1] for x in x0]

def plotSer(sDay,predS,nr):
    """plot series properties"""
    plt.plot(sDay.t,sDay.y,'-k',label='series')
    plt.plot(sDay.t,sDay.stat,label='stat')
    plt.plot(predS.t,predS.pred,label='prediction')
    plt.plot(predS.t,predS['hist'],label='history')
    plt.plot(predS.t,predS['trend'],label='trend')
    plt.plot(predS.t,predS['lsq'],label='lsq')
    plt.plot(sDay.t,sDay['resid'],label='residual')
    plt.xlabel('$t$')
    plt.ylabel('$y$')
    plt.title("time series interpolation " + nr)
    plt.legend()
    plt.show()


def getHistory(sDay,nAhead,x0,hWeek):
    from kapteyn import kmpfit
    nLin = sDay.shape[0] + nAhead
    nFit = sDay.shape[0] if int(x0['obs_time']) <= 14 else int(x0['obs_time'])
    sDay['hist'] = sp.interpolate.interp1d(hWeek.t,hWeek.y,kind="cubic")(sDay['t'])
    histNorm = sDay['hist'].mean()
    sDay['hist'] = ( (sDay['hist']-sDay['hist'].min())/(sDay['hist'].max()-sDay['hist'].min()) + .5)*x0['hist_adj']
    fitD = np.array([sDay.t.tail(nFit),sDay.y.tail(nFit)])
    fitobj = kmpfit.Fitter(residuals=ser_residuals,data=fitD)
    fitobj.fit(params0=x0['poly'])
    x0['poly'] = fitobj.params
    sDay['stat'] = (sDay['y']-ser_poly(x0['poly'],sDay.t))
    sDay['stat'].ix[0:(sDay.shape[0]-nFit)] = sDay['stat'][sDay.shape[0]-nFit]
    t_test = np.linspace(sDay['t'][0],sDay['t'][sDay.shape[0]-1]+sDay.t[nAhead]-sDay.t[0],nLin)
    predS = pd.DataFrame({'t':t_test},index=[sDay.index[0]+datetime.timedelta(days=x) for x in range(nLin)])
    predS['y'] = sDay.y
    predS['hist'] = sp.interpolate.interp1d(hWeek.t,hWeek.y,kind="cubic")(predS['t'])
    predS['hist'] = ( (predS['hist']-predS['hist'].min())/(predS['hist'].max()-predS['hist'].min())*x0['hist_adj'] + .5)
    predS['trend'] = ser_poly(x0['poly'],predS['t'])
    predS['trend'].ix[0:(sDay.shape[0]-nFit)] = predS['trend'][sDay.shape[0]-nFit]
    predS['pred'] = 0
    predS['lsq'] = 0
    # plt.plot(sDay.t,sDay.y,'-',label='series')
    # plt.plot(sDay.t,sDay['hist'],'-',label='hist')
    # plt.plot(hWeek.t,hWeek.y,label='week')
    # plt.plot(sDay.t,sDay['stat'],'-',label='stat')
    # plt.plot(predS.t,predS['trend'],'-',label='fit')
    # plt.legend()
    # plt.show()
    return predS, x0

