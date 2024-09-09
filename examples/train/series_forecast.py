import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import urllib3, requests
import statsmodels.api as sm
import statsmodels.tsa as tsa
import scipy.optimize as sco
from scipy.optimize import curve_fit
import geomadi.algo_holtwinters as ht

##----------------------------optimization-functions------------------------------
def ser_poly(p,x):
    """4th order poynom"""
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 #+ p[5]*x**5

def ser_residuals(p,data):
    """residual function"""
    x,y = data
    return (y-ser_poly(p,x))

def ser_sin(x,t,param):#print 2.*np.pi/(sDay.t[7]-sDay.t[0])
    """custom sinus function"""
    return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
##confInt = stats.t.interval(0.95,len(y)-1,loc=np.mean(y),scale=stats.sem(y))

def ser_fun_min(x,t,y,param):
    """residual on sinus function"""
    return ser_sin(x,t,param) - y

def ser_exp(x,decay):
    """exponential decay"""
    return np.exp(-decay*x)

##---------------------------forecast-methods----------------------------

def serLsq(sDay,nAhead,x0,hWeek):
    nFit = sDay.shape[0] #if int(x0['obs_time']) <= 14 else int(x0['obs_time'])
    predS, x0 = getHistory(sDay,nAhead,x0,hWeek)
    predS = predS.tail(nFit+nAhead)
    freqP = x0['freq']
    res_lsq = least_squares(ser_fun_min,x0['lsq'],args=(sDay.t,sDay.stat,x0['freq']))#loss='soft_l1',f_scale=0.1,
    x0['lsq'] = [x for x in res_lsq[0]]
    predS['lsq'] = ser_sin(res_lsq[0],predS.t,x0['freq']) #fun(res_robust.x,t_test)
    sDay['resid'] = sDay['y'] - ser_sin(res_lsq[0],sDay.t,x0['freq'])/sDay['hist'] - ser_poly(x0['poly'],sDay['t'])# lm.predict(sDay)
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean())**2
    x0['res'][1] = rSquare.sum()
    x0['res'][2] = rSquare.sum()/sDay['y'].tail(x0['res'][0]).sum()
    #predS['pred'] = (predS['lsq'] + lm.predict(predS))*predS['hist']*x0['hist_adj']
    predS['pred'] = ser_sin(res_lsq[0],predS.t,x0['freq'])*sDay['hist'] + ser_poly(x0['poly'],predS['t'])
    predS = predS.drop(predS.index[0])
    return predS, x0

def bestArima(sDay,nAhead,x0,hWeek):
    dta = sDay['y']
    dta.index = [pd.datetime.strptime(str(x)[0:10],'%Y-%m-%d') for x in dta.index]
    t_line = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in dta.index]
    t_line1 = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in hWeek.index]
    sExog = pd.DataFrame({'y':sp.interpolate.interp1d(t_line1,hWeek.y,kind="cubic")(t_line)})
    grid = (slice(1, 4, 1),slice(1,2,1),slice(1, 3, 1))
    def objfunc(order,endog,exog):
        fit = sm.tsa.ARIMA(endog,order).fit(trend="c",method='css-mle',exog=exog)
        return fit.aic
    par = sco.brute(objfunc,grid,args=(dta,sExog), finish=None)
    return par

def serArma(sDay,nAhead,x0,hWeek):
    predS, x0 = getHistory(sDay,nAhead,x0,hWeek)
    dta = sDay['y']
    dta.index = [pd.datetime.strptime(str(x)[0:10],'%Y-%m-%d') for x in dta.index]
    sDay.index = dta.index
    t_line = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in dta.index]
    t_line1 = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in hWeek.index]
    sExog = pd.DataFrame({'y':sp.interpolate.interp1d(t_line1,hWeek.y,kind="cubic")(t_line)})
    #par = bestArima(dta,sExog)
    sExog.index = dta.index
    result = sm.tsa.ARIMA(dta,(x0['arma'][0],x0['arma'][1],x0['arma'][2])).fit(trend="c",method='css-mle',exog=sExog)
    predT = [str(dta.index[0])[0:10],str(dta.index[len(dta)-1])[0:10]]
    histS = pd.DataFrame({'pred':result.predict(start=predT[0],end=predT[1])})
    #predT = [str(dta.index[0])[0:10],str(dta.index[len(dta)-1]+datetime.timedelta(days=nAhead))[0:10]]
    predS = predS[0:(len(dta)-1)]
    predS['pred'] = result.predict(start=predT[0],end=predT[1])
    # predS = pd.DataFrame({'pred':result.predict(start=predT[0],end=predT[1])})
    # predS.index = [dta.index[0]+datetime.timedelta(days=x) for x in range(0,len(dta)+nAhead)]
    # predS['t'] = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in predS.index]
    # predS['hist'] = sp.interpolate.interp1d(t_line1,hWeek.y,kind="cubic")(predS['t'])
    # predS['hist'] = predS['hist']/predS['hist'].mean()
    # predS['pred'] = predS['pred']*predS['hist']
    predS['trend'] = ser_poly(x0['poly'],predS.t)
    predS['y'] = sDay['y']
    predS['pred'] = (predS['pred']*predS['hist']+predS['trend'])
    x0['response'] = [x for x in result.params]
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean())**2
    x0['res'][1:2] = [rSquare.sum(),rSquare.sum()/sDay['y'].tail(x0['res'][0]).sum()]
    return predS, x0
    # plt.plot(dta,'-k',label="series")
    # plt.plot(sExog,label="exo")
    # plt.plot(hWeek.y,label="hist")
    # plt.plot(predS,'-b',label="pred")
    # plt.legend()
    # plt.show()
    # steps = 1
    # tsa.arima_model._arma_predict_out_of_sample(res.params,steps,res.resid,res.k_ar,res.k_ma,res.k_trend,res.k_exog,endog=dta, exog=None, start=len(dta))

def SerBayes(sDay,nAhead,x0,hWeek):
    import pydlm    
    dta = sDay['y']
    dta.index = [pd.datetime.strptime(str(x)[0:10],'%Y-%m-%d') for x in dta.index]
    t_line = [float(calendar.timegm(x.utctimetuple()))/1000000 for x in dta.index]
    dta.index = t_line
    model = pydlm.dlm(dta)
    model = model + pydlm.trend(degree=1,discount=0.98,name='a',w=10.0)
    model = model + pydlm.dynamic(features=[[v] for v in t_line],discount=1,name='b',w=10.0)
    model = model + pydlm.autoReg(degree=3,data=dta.values,name='ar3',w=1.0)
    allStates = model.getLatentState(filterType='forwardFilter')
    model.evolveMode('independent')
    model.noisePrior(2.0)
    model.fit()
    model.plot()
    model.turnOff('predict')
    model.plotCoef(name='a')
    model.plotCoef(name='b')
    model.plotCoef(name='ar3')

def serHolt(sDay,nAhead,x0,hWeek):
    predS, x0 = getHistory(sDay,nAhead,x0,hWeek)
    Y = [x for x in sDay.y]
    ##Yht, alpha, beta, gamma, rmse = ht.additive([x for x in sDay.y],int(x0[0]),nAhead,x0[1],x0[2],x0[3])
    nAv = int(x0['holt'][0]) if int(x0['holt'][0]) > 1 else 5
    Yht, alpha, beta, gamma, rmse = ht.additive([x for x in sDay.y],x0['holt'][0],nAhead,x0['holt'][1],x0['holt'][2],x0['holt'][3])
    sDay['resid'] = sDay['y'] - Yht[0:sDay.shape[0]]
    x0['holt'] = [x0['holt'][0],alpha,beta,gamma,rmse]
    nLin = sDay.shape[0] + nAhead
    t_test = np.linspace(sDay['t'][0],sDay['t'][sDay.shape[0]-1]+sDay.t[nAhead]-sDay.t[0],nLin)
    #predS = pd.DataFrame({'t':t_test},index=[sDay.index[0]+datetime.timedelta(days=x) for x in range(nLin)])
    predS['pred'] = Yht
    # predS['hist'] = sp.interpolate.interp1d(hWeek.t,hWeek.y,kind="cubic")(predS['t'])
    # predS['hist'] = predS['hist']/predS['hist'].mean()
    # predS['pred'] = predS['pred']*predS['hist']*x0['hist_adj']
    # predS['trend'] = ser_poly(x0['poly'],predS.t)
    # predS['trend'].ix[0:(sDay.shape[0]-nFit)] = predS['trend'][sDay.shape[0]-nFit]
    predS['lsq'] = 0
    predS['y'] = sDay['y']
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean())**2
    x0['res'][1] = rSquare.sum()
    x0['res'][2] = rSquare.sum()/sDay['y'].tail(x0['res'][0]).sum()
    return predS, x0
    
def serAuto(sDay,nAhead,x0,hWeek):
    predS, x0 = getHistory(sDay,nAhead,x0,hWeek)
    todayD = datetime.datetime.today()
    todayD = todayD.replace(hour=0,minute=0,second=0,microsecond=0)
    dta = pd.DataFrame({'y':sDay.y})
    dta['day'] = sDay.index.weekday
    phase = dta.head(int(x0['obs_time'])).groupby(['day']).mean()
    phase['std'] = dta.groupby(['day']).std()['y']
    phase = phase.sort_values(['y'],ascending=False)
    phase['csum'] = phase['y'].cumsum()/phase['y'].sum()
    phaseN = phase.index[0] - todayD.weekday()
    r,q,p = sm.tsa.acf(sDay['y'].tail(phaseN+int(x0['obs_time'])).squeeze(),qstat=True)
    popt, pcov = curve_fit(ser_exp,np.array(range(0,6)),r[0:6]-min(r),p0=(x0['decay'][0]))
    X = np.array(range(0,r.size,7))
    popt1, pcov1 = curve_fit(ser_exp,X,r[X],p0=(x0['decay'][0]))
    autD = pd.DataFrame({'r':r,'exp':ser_exp(range(0,r.size),popt),'exp1':ser_exp(range(0,r.size),popt1)})    
    x0['decay'] = [x for x in popt]
    wN = 0
    sY = np.random.normal(phase['y'].head(1),dta.y.std())
    for i in predS.index:
        wN = 6 - np.abs(phase.index[0] - i.weekday())
        wN = wN + 1 if wN < 6 else 0
        if(wN == 0):
            sY = np.random.normal(phase['y'].head(1),dta.y.std()/2)
        sY = sY*(1+predS['hist'][i]*x0['hist_adj'])
        predS.loc[i,'pred'] = sY*ser_exp(float(wN),popt)

    predS['pred'] = serSmooth(predS['pred'],16,5)
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]
    freqP = x0['freq']
    res_lsq = least_squares(ser_fun_min,x0['lsq'],args=(sDay['t'],sDay['resid'],x0['freq']))
    predS['lsq'] = ser_sin(res_lsq[0],predS['t'],x0['freq']) # fun(res_robust.x,t_test)
    x0['lsq'][0:res_lsq[0].size] = res_lsq[0]
    predS['pred2'] = predS['pred']
    predS['pred'] = predS['pred'] + predS['lsq']
    sDay['resid'] = sDay['y'] - predS['pred'][0:sDay.shape[0]]    
    rSquare = (sDay['resid'].tail(x0['res'][0]) - sDay['resid'].tail(x0['res'][0]).mean())**2
    x0['res'][1] = rSquare.sum()
    x0['res'][2] = rSquare.sum()/sDay['y'].tail(x0['res'][0]).sum()
    # sDay.to_csv('tmpAuto1.csv')
    # predS.to_csv('tmpAuto2.csv')
    # autD.to_csv('tmpAuto3.csv')
    return predS, x0
    
