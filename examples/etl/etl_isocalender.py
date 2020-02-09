import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
import geomadi.train_reshape as t_r

custD = "mc"
custD = "tank"
custD = "bast"
idField = "id_poi"

if False:
    mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_h.csv.gz",compression="gzip",index_col=0)
    hL = mist.columns.values
    dateL = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hL]
    ical = ["%d-%02d-%02dT%02d" % (x.isocalendar()[0],x.isocalendar()[1],x.isocalendar()[2],x.hour) for x in dateL]
    mist.columns = ical
    g = mist.iloc[0]
    varS = []
    icaL = []
    for i,g in mist.iterrows():
        ser = pd.DataFrame(g)
        ser.loc[:,"year"] = [x.split("-")[0] for x in ser.index]
        ser.loc[:,"week"] = [x.split("-")[1] for x in ser.index]
        ser.loc[:,"wday"] = [x.split("-")[2] for x in ser.index]
        ser.loc[:,"hour"] = [x.split("T")[1] for x in ser.wday]
        ser.loc[:,"wday"] = [x.split("T")[0] for x in ser.wday]
        for j,s in ser.groupby("week"):
            M = s.pivot_table(index="wday",columns="year",values=s.columns[0],aggfunc=np.mean)
            C = np.triu(M.corr(),k=1)
            c = np.mean(C[np.abs(C)>0])
            s_c = np.std(C[np.abs(C)>0])
            m = M.mean(axis=1)
            e = np.mean(M.std(axis=1))/np.mean(m)
            varS.append({idField:s.columns[0],"week":j,"cor":c,"err":e,"cor_var":s_c})
            ical = ["%s-%02d" % (j,int(x)) for x in M.index]
            ica = pd.DataFrame({idField:s.columns[0],"ical":ical,"mean":m,"std":e})
            icaL.append(ica)
    varS = pd.DataFrame(varS)
    icaL = pd.concat(icaL)

    varS.to_csv(baseDir+"raw/"+custD+"/ref_ical_week.csv",index=False)
    icaL.to_csv(baseDir+"raw/"+custD+"/ref_ical_wday.csv",index=False)

if False:
    idL = np.unique(varS[idField])
    plt.title("weekly correlation over years for single location")
    for i,g in varS.groupby(idField):
        plt.errorbar(g['week'],g['cor'],yerr=g['err'],label=i,alpha=.3)#,fmt='o')
        if i == idL[20]:
            break
    #plt.legend()
    plt.show()

if False:
    print('------------------build-isocalender-frame------------------------')
    import geomadi.train_reshape as t_r
    importlib.reload(t_r)
    pist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_h.csv.gz",compression="gzip")
    mist, sist = t_r.day2isocal(pist,idField,isDay=False)
    mist.to_csv(baseDir + "raw/"+custD+"/ref_iso_h.csv.gz",compression="gzip",index=False)
    pist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
    mist, sist = t_r.day2isocal(pist,idField,isDay=True)
    mist.to_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",index=False)
    if False:
        print('-------------------plot-weekly-deviations-------------------')
        hL = sist.columns[[bool(re.search('-??T',x)) for x in sist.columns]]
        nhL = sist.columns[[not bool(re.search('-??T',x)) for x in sist.columns]]
        wcal = [x.split("-")[0]+"T" for x in hL]
        tist = sist[list(nhL) + list(hL)]
        tist.columns = list(nhL) + list(wcal)
        tist = tist.groupby(tist.columns,axis=1).agg(np.mean)
        tL = tist.columns[[bool(re.search('-??T',x)) for x in tist.columns]]
        lookUp = t_r.ical2date(hL)
        y = pd.DataFrame({"y":tist[tL].mean(axis=0)})
        y.loc[:,"ical"] = [x[0:2] + "-01T" for x in y.index]
        y = pd.merge(y,lookUp,on="ical",how="left")
        y.dropna(inplace=True)
        y.loc[:,"week"] = y['ical'].apply(lambda x: int(x[:2]))
        y1 = y.copy()
        t1 = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in y1['date']]
        #t2 = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in y2['date']]
        plt.title("deviation on isocalendar")
        plt.bar(y1['week'],y1['y'],width=1,label="ref",alpha=1.)
        #plt.bar(t2,y2['y'],width=5,label="bast",alpha=1.)
        plt.xticks(rotation=15)
        # plt.legend()
        plt.show()

if False:
    plog('--------------------check-isocalender---------------------')
    mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_h.csv.gz",compression="gzip")
    X1 = mist[[x for x in mist.columns if bool(re.search("2016-",x))]].replace(float('nan'),0)
    X2 = mist[[x for x in mist.columns if bool(re.search("2017-",x))]].replace(float('nan'),0)
    isocal = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S").isocalendar() for x in X1.columns]
    colL = ["%02d-%02d-%02d" % (x.isocalendar()[1],x.weekday(),x.hour) for x in isocal]
    cL = []
    for i in range(X1.shape[0]):
        
        cL.append(sp.stats.pearsonr(X1[i],X2[i])[0])
    x1 = X1.sum(axis=1)
    x2 = X2.sum(axis=1)
    plt.plot(X1.sum(axis=0))
    plt.plot(X2.sum(axis=0))
    plt.show()

if False:
    print('-----------------------------build-daily-values--------------------')
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv",dtype={idField:str})
    dirc = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz")
    dirc.loc[:,idField] = dirc[idField].apply(lambda x: str(int(x)))
    hL = t_r.timeCol(dirc)
    dirg = pd.DataFrame({idField:dirc[idField],"daily_visit":dirc[hL].T.mean()})
    poi.loc[:,"daily_visit"] = poi.merge(dirg,on=idField,how="left")['daily_visit_y'].values
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
