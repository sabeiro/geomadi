#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph 
import geomadi.train_execute as t_e
import geomadi.train_reshape as t_r
from dateutil import tz
from datetime import timezone
import geomadi.train_viz as t_v
from_zone = tz.gettz('UTC')
to_zone = tz.gettz('Europe/Berlin')
cred = json.load(open(baseDir + "credenza/geomadi.json","r"))

if False:
    dirT = pd.read_csv(baseDir + 'raw/bast/bast_dir-isocal.csv.gz',compression="gzip")
    dirT.loc[:,"isoD"] = dirT['isocal'].apply(lambda x:x[:5])
    dirD = dirT.groupby(["id_tile","isoD"]).agg(np.sum).reset_index()
    dirD = dirD[dirD['isoD'] < '39-07']
    def clampF(x):
        return pd.Series({"r":sp.stats.pearsonr(x['dirc'],x['bast'])[0]
                          ,"d":(sum(x['dirc'])-sum(x['bast']))/(sum(x['dirc'])+sum(x['bast']))})
    
    dirG = dirT.groupby(['id_tile']).apply(clampF).reset_index()

if False:
    t_v.kpiDis(dirG,tLab="working (d)",col_cor="r",col_dif="d",col_sum="d")

if False:
    print('-------------------------------model-comparison-----------------------------')
    bast = pd.read_csv(baseDir + "raw/bast/trainRes.csv")
    bast.rename(columns={"bast":"ref","footfall":"act"},inplace=True)
    scorB = t_e.scorPerf(bast,step="simple",idField=idField)
    bast = pd.read_csv(baseDir + "raw/bast/trainRes.csv")
    bast.rename(columns={"bast":"ref","pred_simpleEnc":"act"},inplace=True)
    scorD = t_e.scorPerf(bast,step="simple",idField=idField)
    bast = pd.read_csv(baseDir + "raw/bast/trainRes.csv")
    bast.rename(columns={"bast":"ref","pred_convNet_9x24":"act"},inplace=True)
    scorE = t_e.scorPerf(bast,step="simple",idField=idField)
    #bast = bast[bast["id_poi"].isin(poi['id_bast'])]
    for t,scor in zip(['raw','simple','convNet'],[scorB,scorD,scorE]):
        fig, ax = plt.subplots(1,2)
        t_v.kpiDis(scor,tLab="model %s" % t,col_cor="r_simple",col_dif="v_simple",col_sum="s_simple",isRel=False,ax=ax[0])
        t_v.kpiDis(scor,tLab="model %s" % t,col_cor="r_simple",col_dif="d_simple",col_sum="s_simple",isRel=True,ax=ax[1])
        plt.show()

if False:
    print("----------------------bast-agreement-on-selected-locations--------------------")
    dirc = pd.read_csv(baseDir + "raw/bast/dirCount_d.csv.gz")
    hL = [x for x in dirc.columns if bool(re.search("T",x))]
    hL = [x for x in hL if x[5:7] == "02"]
    dirc = dirc[['id_poi'] + hL]
    cL = [datetime.datetime.strptime(x,"%Y-%m-%dT").isocalendar() for x in hL]
    cL = ["%02d-%02dT" % (x[1],x[2]) for x in cL]
    dirc.columns = ['id_poi'] + cL
    
    bast = pd.read_csv(baseDir + "raw/bast/bast17_d.csv.gz")
    hL = [x for x in bast.columns if bool(re.search("T",x))]
    hL = [x for x in hL if x[5:7] == "02"]
    bast = bast[['id_poi'] + hL]
    cL1 = [datetime.datetime.strptime(x,"%Y-%m-%dT").isocalendar() for x in hL]
    cL1 = ["%02d-%02dT" % (x[1],x[2]) for x in cL1]
    bast.columns = ['id_poi'] + cL1

    cL = sorted(list(set(cL) & set(cL1)))
    dirc = dirc[['id_poi'] + cL]
    bast = bast[['id_poi'] + cL]

    tist = t_e.joinSource(dirc,bast,how="inner",idField="id_poi")
    tist = tist.dropna()
    tist.loc['id_poi'] = tist['id_poi'].apply(lambda x: str(int(x)))
    poiT = pd.read_csv(baseDir + "raw/tank/poi.csv")
    poiT = poiT[poiT['use'] == 3]
    norm = tist['ref'].sum()/tist['act'].sum()
    tist.loc[:,'act'] = tist['act']*norm

    scorM1 = t_e.scorPerf(tist,step="all",idField=idField)
    scorM3 = t_e.scorPerf(tist,step="feb",idField=idField)
    scorM2 = t_e.scorPerf(tist,step="sep",idField=idField)

    import importlib
    importlib.reload(shl)
    fig, ax = plt.subplots(1,2)
    t_v.kpiDis(scorM3,tLab="february",col_cor="r_feb",col_dif="d_feb",col_sum="s_feb",isRel=False,ax=ax[0])
    t_v.kpiDis(scorM2,tLab="september",col_cor="r_sep",col_dif="d_sep",col_sum="s_sep",isRel=False,ax=ax[1])
    # shl.plotHistogram(scorM3['r_feb'],label="correlation",ax=ax[1])
    plt.show()

    scorM3 = pd.merge(scorM3,poiT[['id_poi','id_bast']],left_on="id_poi",right_on="id_bast",suffixes=["_x",""],how="left")
    del scorM3['id_bast']
    scorM3.columns = ['id_bast', 'r_feb', 'd_feb', 'v_feb', 's_feb', 'id_poi']
    scorM3.to_csv(baseDir + "raw/bast/scor_Feb.csv")
    
if False:
    print("----------------------join-years------------------------")
    fL = ["bast14_d.csv.gz","bast15_d.csv.gz","bast16_d.csv.gz","bast17_d.csv.gz"]
    vistL = []
    for f in fL:
        vistL.append(pd.read_csv(baseDir + "raw/bast/" + f,compression="gzip"))
    vist = pd.merge(vistL[0],vistL[1],on="id_poi",how="outer")
    vist = pd.merge(vist,vistL[2],on="id_poi",how="outer")
    vist = pd.merge(vist,vistL[3],on="id_poi",how="outer")
    vist.to_csv(baseDir + "raw/bast/" + "ref_visit_d.csv.gz",compression="gzip",index=False)

    print("----------------------join-years------------------------")
    fL = ["bast14_h.csv.gz","bast15_h.csv.gz","bast16_h.csv.gz","bast17_h.csv.gz"]
    vistL = []
    for f in fL:
        vistL.append(pd.read_csv(baseDir + "raw/bast/" + f,compression="gzip"))
    vist = pd.merge(vistL[0],vistL[1],on="id_poi",how="outer")
    vist = pd.merge(vist,vistL[2],on="id_poi",how="outer")
    vist = pd.merge(vist,vistL[3],on="id_poi",how="outer")
    vist.to_csv(baseDir + "raw/bast/" + "ref_visit_h.csv.gz",compression="gzip",index=False)

    
if False:
    print('------------------------join-tables-together----------------------')
    tileM = pd.read_csv(baseDir + "raw/bast/id_bast_tile.csv")
    dirT = pd.read_csv(baseDir + "raw/bast/bastDir2017.csv.gz",compression="gzip")
    basT = pd.read_csv(baseDir + "raw/bast/bast17.csv.gz",compression="gzip")
    dirT.loc[:,"isocal"] = dirT['time'].apply(lambda x: "%02d-%02d-%02d" % (datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")).isocalendar())
    basT.loc[:,"isocal"] = basT['time'].apply(lambda x: "%02d-%02d-%02d" % (datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")).isocalendar())
    basT.loc[:,"isocal"] = basT.apply(lambda x: "%s:%s" % (x['isocal'][5:],x['time'][11:13]),axis=1)
    dirT.loc[:,"isocal"] = dirT.apply(lambda x: "%s:%s" % (x['isocal'][5:],x['time'][11:13]),axis=1)
    basT = pd.merge(basT,tileM,on="id_bast",how="left")
    dirT = pd.merge(dirT,basT,on=["id_tile","isocal"],how="left")
    dirT = dirT[['id_tile','isocal','dirc','bast']]
    dirT.to_csv(baseDir + 'raw/bast/bast_dir-isocal.csv.gz',compression="gzip",index=False)
    
if False:
    print('---------------------------------direction-counts--------------------------')
    tileL = [int(x) for x in tileL]
    tileS = ", ".join([str(x) for x in tileL])
    queL = ""
    for i in range(1,31):
        queI = "psql -p "+cred['postgres']['port']+" -U "+cred['postgres']['user']+" -d "+cred['postgres']['db']+" -h "+cred['postgres']['host']+' -c "'
        queI += "COPY (SELECT * FROM "+cred['postgres']['table']+"201809%02d"%i+" WHERE tile_id in ("+tileS+") ) TO '/tmp/dirCount_201809%02d.csv' "%i+'(format csv,header true);"'
        queL += queI + "\n"

    queL += "cp /tmp/dirCount_* " + baseDir + workDir + "dirCount/"
    open(baseDir+"log/bast/"+"dircount_query.sh","w").write(queL)

if True:
    print('---------------------format-bast----------------------')
    for i in ["14","15","16","17"]:
        print(i)
        basT = pd.read_csv(baseDir + "log/bast/bast"+i+".csv.gz",compression="gzip")
        basT.loc[:,"date"] = basT['date'].astype(str)
        basT.loc[:,"time"] = basT.apply(lambda x: "20%s-%s-%sT%02d:00:00" % (x['date'][:2],x['date'][2:4],x['date'][4:6],x['hour']-1),axis=1)
        basT.loc[:,"bast"] = basT.apply(lambda x: x['dir1']+x['dir2'],axis=1)
        basT = basT[['id_bast','time','bast']]
        basT.columns = ['id_poi','time','ref']
        #basT = basT[basT['id_bast'].isin(tileM['id_bast'])]
        basP = basT.pivot_table(index="id_poi",columns="time",values="ref",aggfunc=np.sum).reset_index()
        basP.to_csv(baseDir + "raw/bast/bast"+i+"_h.csv.gz",compression="gzip",index=False)
        basD = t_r.hour2day(basP)
        basD.to_csv(baseDir + "raw/bast/bast"+i+"_d.csv.gz",compression="gzip",index=False)

if False:
    ##https://www.bast.de/BASt_2017/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Stundenwerte.html?nn=1817946
    bast = pd.read_csv(baseDir + "log/bast_2016.csv",sep=";")
    bast = bast[['Zst','Datum','Stunde','KFZ_R1','KFZ_R2','Lkw_R1','Lkw_R2']]
    bast.columns = ["id","day","hour","vehicle_r1","vehicle_r2","tir_r1","tir_r2"]
    bast.loc[:,"day"] = bast.day.astype(str)
    bast.loc[:,"day"] = bast['day'].apply(lambda x: "20%s-%s-%s" % (x[0:2],x[2:4],x[4:6]))
    bast.to_csv(baseDir + "log/bast_2016.csv.tar.gz",compression="gzip",index=False)
        
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
