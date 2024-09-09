import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import datetime as dt
import pymongo
import geomadi.proc_lib as plib
from multiprocessing.dummy import Pool as ThreadPool 

cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))

custD = "mc"
idField = "id_poi"

if False:
    plog('----------------rename-time-columns----------------')
    gist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip",index_col=0)
    gist = gist.replace(float('nan'),0.)
    hLg = gist.columns[[bool(re.search('-',x)) for x in gist.columns]]
    for i in hLg:
        gist.rename(columns={i:i+"T"},inplace=True)
    gist.to_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")

if False:
    plog('-----------------join-sources------------------')
    poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    tact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip")

    dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount.csv.gz",compression="gzip")
    dirc = dirc.replace(float('nan'),0.)
    dirM = pd.melt(dirc,id_vars="id_poi")
    dirM.loc[:,"day"] = dirM['variable'].apply(lambda x: x[:11])
    dirc = dirM.pivot_table(index="id_poi",columns="day",values="value",aggfunc=np.sum)
    hLd = dirc.columns[[bool(re.search('-??T',x)) for x in dirc.columns]]
    hL = sorted(list(set(hL) & set(hLd)))

    dirc.loc[:,idField] = dirc.index
    tirc = pd.melt(dirc,id_vars=idField)
    tirc.columns = [idField,'time','foot']
    tact = pd.merge(tact,tirc,on=[idField,"time"],how="left")
    tact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip",index=False)

if False:
    plog('----------------parse-log-------------------------')
    act, fL = plib.parsePath(baseDir + "log/mc/act_cilac")
    act.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').save(baseDir+"raw/mc/act")
    act = act.toPandas()
    act.to_csv(baseDir+"raw/mc/act.csv.gz",compression="gzip",index=False)

if False:
    plog('-------------------format-customer-input------------------')
    metric = "rev"
    #metric = "visit"
    i = "19"
    vist = pd.read_csv(baseDir+"raw/mc/ref/ref_"+metric+"_"+i+".csv.gz",compression="gzip")
    vist.loc[:,idField] = vist['Store'].apply(lambda x: x.split(" ")[0])
    vist.loc[:,"hour"] = vist['Hour'].apply(lambda x: x[:2])
    hL = [x for x in vist.columns if bool(re.search("\.",x))]
    lL = [{x:"%s-%s-%sT" % (x[6:10],x[3:5],x[0:2])} for x in hL]
    for l in lL: vist = vist.rename(columns=l)
    hL = [x for x in vist.columns if bool(re.search("-",x))]
    vist = vist[[idField,"hour"]+hL]
    vist = vist.replace("€","")
    vist.to_csv(baseDir+"raw/mc/ref/ref_"+metric+"_"+i+".csv.gz",compression="gzip",index=False)
    
if False:
    plog('-------------------join-files------------------')
    metric = "rev"
    #metric = "visit"
    vistL = []
    for i in ["18","19"]:
        vist = pd.read_csv(baseDir + "raw/mc/ref/ref_"+metric+"_"+i+".csv.gz",compression="gzip")
        vistL.append(vist)
    vist = pd.merge(vistL[0],vistL[1],on=[idField,"hour"],how="outer")
    vist.to_csv(baseDir + "raw/mc/ref_"+metric+"_h.csv.gz",compression="gzip",index=False)

if False:
    plog('------------------------join-activity-footfall-----------------')
    minH = 6
    bast = pd.read_csv(baseDir + "raw/mc/corr_fact_bast_h_wd.csv")
    act = pd.read_csv(baseDir + "raw/mc/act_tile.csv.tar.gz",compression="gzip")
    act = act[~np.isnan(act['HOD'])]
    act.loc[:,"DATE_"] = act["DATE_"].apply(lambda x: str(x))
    act.loc[:,"HOD"] = act['HOD'].apply(lambda x: int(x)) + 1 #from utc
    act = act[act["HOD"] > minH]
    act = act[act["HOD"] < 24]
    act.loc[:,"TileID_int"] = act['TileID_int'].apply(lambda x: int(x))
    tL = [dt.datetime.strptime(x,"%d.%m.%Y %H:%M:%S") + dt.timedelta(hours=int(y)) for x,y in zip(act['DATE_'],act['HOD'])]
    act.loc[:,'time'] = [x.strftime("%Y-%m-%dT%H:%M:%S") for x in tL]
    act = act[act['time'] > "2017-04-24"]
    act = act[act['time'] < "2017-05-15"]
    act = act.pivot_table(index="TileID_int",columns="time",values="ext_cnt_korr",aggfunc=np.sum).replace(np.nan,0).reset_index()
    hL = act.columns[[bool(re.search(':',x)) for x in act.columns]].sort_values()
    act = act[['TileID_int'] + list(hL)]
    poi = pd.read_csv(baseDir + "raw/mc/poi_mc.csv")
    matchL = poi[['Sitenummer','TileID','corrFactor_ms']]
    matchL = matchL.groupby('TileID').head(1)
    act = pd.merge(act,matchL,left_on="TileID_int",right_on="TileID",how="left")
    
if False:
    plog('-----------------------------bast-correction-factor----------------------')
    hL = act.columns[[bool(re.search(':',x)) for x in act.columns]].sort_values()
    hLt = [dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hL]
    wdSet = [t.weekday() for t in hLt]
    hSet = [t.hour for t in hLt]
    for i,b in bast.iterrows():
        cSet = (b['Weekday'] == wdSet ) & (b['Hour'] == hSet)
        cSet = act.columns[[j+1 for j,x in enumerate(cSet) if x]]
        act.loc[:,cSet] = act[cSet].multiply(b['calib_factor'])
        
if False:
    plog('----------------------------substitute-missing-/-uncomplete-with-average------------------')
    hLa = list(act.columns[[bool(re.search(':',x)) for x in act.columns]])
    hTa = np.array([dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hLa])
    ta = np.arange(dt.datetime.strptime(hLa[0],"%Y-%m-%dT%H:%M:%S"),dt.datetime.strptime(hLa[-1],"%Y-%m-%dT%H:%M:%S"), datetime.timedelta(hours=1)).astype(dt.datetime)
    hourAv = pd.DataFrame()
    for i in range(minH,24): #average pro hour
        colL = [j+1 for j,x in enumerate(hTa) if x.hour == i]
        hourAv.loc[:,str(i)] = act.iloc[:,colL].apply(lambda x:np.nanmean(x),axis=1)
    hbar = [x for x in ta if x not in hTa]
    hTa  = [x for x in hTa if x.hour > minH]
    hbar = [x for x in hbar if x.hour > minH]
    for i in hbar: #substitute missing with average
        t = i.strftime("%Y-%m-%dT%H:%M:%S")
        h = str(i.hour)
        act.loc[:,t] = hourAv.loc[:,h]
    
if False:
    plog('-----------------remove spikes with average----------------------')
    for i in hTa: 
        t = i.strftime("%Y-%m-%dT%H:%M:%S")
        h = str(i.hour)
        colSel = act.loc[:,t] < 3.*hourAv.loc[:,h]
        act.loc[colSel,t] = hourAv.loc[colSel,h]

    hLa = list(act.columns[[bool(re.search(':',x)) for x in act.columns]])
    hTa = np.array([dt.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hLa])
    hTa = [x for x in hTa if x.hour > minH]
    hLa = [dt.datetime.strftime(x,"%Y-%m-%dT%H:%M:%S") for x in hTa]
    act.loc[np.isnan(act["corrFactor_ms"]),"corrFactor_ms"] = 1
    act.loc[:,hLa] = act[hLa].multiply(act['corrFactor_ms'].values,axis=0)
    act = act[["Sitenummer"] + hLa]
    cL = act.columns.values
    cL[0] = 'id_poi'
    act.columns = cL
    hL = act.columns[[bool(re.search(':',x)) for x in act.columns]].sort_values()

    vist = pd.read_csv(baseDir + "raw/mc/poi_mc_visit.csv.tar.gz",compression="gzip")
    vist = vist[:-1]
    vist.loc[:,"Date"] = vist["Date"].apply(lambda x: str(x))
    vist.loc[:,"Hour"] = vist['Hour'].apply(lambda x: int(x))
    vist.loc[:,"Sitenummer"] = vist['Sitenummer'].apply(lambda x: int(x))
    tL = [dt.datetime.strptime(x,"%d.%m.%Y %H:%M:%S") + dt.timedelta(hours=int(y)) for x,y in zip(vist['Date'],vist['Hour'])]
    vist.loc[:,'time'] = [x.strftime("%Y-%m-%dT%H:%M:%S") for x in tL]
    vist = vist.pivot_table(index="Sitenummer",columns="time",values="Guestcount",aggfunc=np.sum).replace(np.nan,0).reset_index()
    vist = vist[['Sitenummer'] + [x for x in hL]]
    cL = vist.columns.values
    cL[0] = 'id_poi'
    vist.columns = cL
    act.to_csv(baseDir  + "raw/mc/act_mc.csv.tar.gz",compression="gzip",index=False)
    vist.to_csv(baseDir + "raw/mc/act_mc_visit.csv.tar.gz",compression="gzip",index=False)
    
if False:
    plog('-------------------------plot-footfall-activities-------------------------')
    y1 = act[hL].sum(axis=0).values/act[hL].sum().sum()
    y2 = vist[hL].sum(axis=0).values/vist[hL].sum().sum()
    plt.plot(y1,label="footfall")
    plt.plot(y2,label="visit")
    plt.legend()
    plt.xlabel("hours")
    plt.ylabel("norm counts")
    plt.show()
    xcorr1 = sp.signal.correlate(y1,y2)
    # for i in range(1,5):
    #     xcorr2 = sp.signal.correlate(y1[i:],y2[:-i])
    #     plt.plot(xcorr2)
    plt.plot(xcorr1,label="cross correlation")
    plt.xlabel("hours")
    plt.ylabel("norm counts")
    plt.show()

if False:
    plog('------------------------------filter-cells-----------------------------')
    sact = pd.read_csv(baseDir + "log/"+custD+"/act_cilac.csv.gz",compression="gzip")
    hL = mist.columns[[bool(re.search('-??-',x)) for x in mist.columns]]
    for i in hL:
        mist.rename(columns={i:i+"T"},inplace=True)
    hL1 = mist.columns[[bool(re.search('-??T',x)) for x in mist.columns]]
    sact = sact.replace(float('nan'),0.)
    sact = sact[sact['cilac'].isin(mapL['cilac'])]
    sact.to_csv(baseDir + "raw/"+custD+"/act_cilac.csv.gz",compression="gzip",index=False)

if False:
    plog('----------------------------pivot-reference--------------------------')
    for i in ["ref_visit","ref_rev"]:
        mist = pd.read_csv(baseDir + "raw/"+custD+"/"+i+".csv.gz",compression="gzip")
        mist = mist.replace(float('nan'),0.)
        gist = mist.groupby(idField).agg(sum)
        gist.to_csv(baseDir + "raw/"+custD+"/"+i+"_d.csv.gz",compression="gzip")    
        hL1 = mist.columns[[bool(re.search('.-.',x)) for x in mist.columns]]
        mist = pd.melt(mist,id_vars=[idField,"hour"],value_vars=hL1)
        mist.loc[:,"time"] = mist.apply(lambda x: "%sT%02d:00:00" % (x['variable'],x['hour']),axis=1)
        mist = mist.pivot_table(index=idField,columns="time",values="value",aggfunc=np.sum)
        mist.to_csv(baseDir + "raw/"+custD+"/"+i+"_h.csv.gz",compression="gzip")

if False:
    plog('-----------------------join-customer-data--------------------')
    site1 = pd.read_csv(baseDir + "raw/mc/input/site1.csv",sep="\t")
    site2 = pd.read_csv(baseDir + "raw/mc/input/site2.csv")
    site3 = pd.read_csv(baseDir + "raw/mc/input/site3.csv")
    site3 = site3[site3['state'] == 'Site eröffnet']
    site4 = pd.read_csv(baseDir + "raw/mc/input/site4.csv",sep="\t")
    def removeDuplicates(poi):
        tL = [x for x in poi.columns if bool(re.search("_y",x))]
        for i in tL:
            j = i.split("_")[0]
            setL = poi[j] != poi[j]
            poi.loc[setL,j] = poi.loc[setL,i]
            del poi[i]
        return poi
    poi = pd.merge(site1,site2,on="id_poi",how="outer",suffixes=["","_y"])
    poi = removeDuplicates(poi)
    poi = pd.merge(poi,site3,on="id_poi",how="outer",suffixes=["","_y"])
    poi = removeDuplicates(poi)
    poi = pd.merge(poi,site4,on="id_poi",how="outer",suffixes=["","_y"])
    poi = removeDuplicates(poi)
    poi = poi.sort_values("id_poi")
    poi = poi.groupby("id_poi").first().reset_index()
    poi = poi[poi['x'] == poi['x']]
    poi.loc[:,"x"] = poi['x'].apply(lambda x: re.sub(",",".",str(x)))
    poi.loc[:,"y"] = poi['y'].apply(lambda x: re.sub(",",".",str(x)))
    poi.loc[:,"x"] = poi['x'].astype(float)
    poi.loc[:,"y"] = poi['y'].astype(float)
    poi.to_csv(baseDir + "raw/mc/poi_raw.csv",index=False)
    print(poi.columns)

if False:
    cred = json.load(open(baseDir + "credenza/geomadi.json"))
    poi = pd.read_csv(baseDir + "raw/mc/whitelist/poi.csv")
    client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])
    coll = client["tdg_infra_internal"]["grid_250"]
    def find_coord(id_poi):
        print(id_poi)
        neiN = coll.find({'tile_id':id_poi})
        for n in neiN:
            loc = geometry.Polygon(n['geom']['coordinates'][0]).centroid.xy
            return {'id_poi':id_poi,"x":loc[0][0],"y":loc[1][0]}
    coll = client["tdg_infra_internal"]["node_tile"]
    def node_tile(id_poi):
        print(id_poi)
        neiN = coll.find({'tile_id':int(id_poi)})
        for n in neiN:
            return {'id_poi':id_poi,"id_node":n['node_id']}

    print(node_tile(poi[idField].values[0]))
    tileL = []
    for i,g in poi.iterrows():
        print("processing %.2f" % (i/g.shape[0]))
        tileL.append(find_coord(i))
    tileL = pd.DataFrame(tileL)
    tileL.to_csv(baseDir + "raw/mc/whitelist/tile_pos.csv.gz",compression="gzip",index=False)
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
