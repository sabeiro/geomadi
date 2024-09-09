import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
import geomadi.train_viz as t_v
import geomadi.series_stat as s_l
import geomadi.train_execute as t_e
import geomadi.train_score as t_s
import geomadi.train_model as tlib
import geomadi.train_shape as shl
import geomadi.train_reshape as t_r
import geomadi.train_viz as t_v
import custom.lib_custom as l_c
import joypy
gra.style()

print('--------------------------------define------------------------')
custD = "tank"
idField = "id_poi"
version = "11u"
version = "prod"
modDir = baseDir + "raw/"+custD+"/model/"
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: str(x)+"T")
dateL.loc[:,"time"] = dateL["day"]
dateL = dateL[dateL['day'] > '2018-12-31T']
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
poi = poi[poi['competitor'] == 0]
poi = poi[poi['use'] == 3]
corrF = 0.3181321540397981
corrO = 0.4226623389831198#0.464928
corP = poi[[idField]]
corP.loc[:,"cor_foot"] = 1.
corP.loc[corP[idField] == '1351',"cor_foot"] = [5.559189]

print('---------------------------------load-join------------------------------')
mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip",dtype={idField:str})
mist = mist[mist[idField].isin(poi[idField])]
mist.loc[:,"ref_day"] = (mist.shape[1] - mist.isnull().sum(axis = 1)) - mist.shape[1]
poi.loc[:,"ref_day"] = pd.merge(poi,mist,on=idField,how="left")["ref_day"].values
poi = poi.sort_values("daily_visit")
pact = pd.read_csv(baseDir + "raw/"+custD+"/act_predict/act_predict_"+version+".csv.gz",compression="gzip",dtype={idField:str})
wact = pd.read_csv(baseDir + "raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip",dtype={idField:str})
ical = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",dtype={idField:str})
ical = t_r.isocal2day(ical,dateL)
dirc = t_r.mergeDir(baseDir+"raw/"+custD+"/viaCount/")
# dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount/dirCount_d.csv.gz",compression="gzip",dtype={idField:str})
deli = pd.read_csv(baseDir + "raw/"+custD+"/delivery/act_ref_foot_19_03.csv",dtype={idField:str})
deli.rename(columns={"direction_count":"foot"},inplace=True)
vist = t_e.joinSource(pact,ical,how="outer",idField=idField,isSameTime=False)
vist = t_e.concatSource(vist,wact,how="left",idField=idField,varName="weighted")
vist = t_e.concatSource(vist,dirc,how="left",idField=idField,varName="foot")
vist.loc[:,"foot"] = vist['foot']*corrF
vist = vist[vist['time'] == vist['time']]
vist = vist[vist[idField].isin(poi[idField])]
vist = vist.sort_values([idField,'time'])
vist.loc[:,"day"] = vist['time'].apply(lambda x:x[:11])
vist.loc[:,"deli_old"] = pd.merge(vist,deli,on=[idField,"day"],how="left",suffixes=["_x",""])["act"].values
vist.loc[:,"foot_old"] = pd.merge(vist,deli,on=[idField,"day"],how="left",suffixes=["_x",""])["foot"].values
vist = vist.merge(corP,on=idField,how="left")
vist.loc[:,"foot"] = vist['foot']*vist['cor_foot']

vist.loc[:,"deli"] = vist['act']*.5 + vist['weighted']*.5
vist.loc[:,"capt"] = vist['deli']/vist['foot']*100.
vist.loc[:,"capt_old"] = vist['deli_old']/vist['foot_old']*100.

vist.loc[:,"month"] = vist['day'].apply(lambda x: x[5:7])
febr = vist[vist['month']=='02']
marc = vist[vist['month']=='03']
comp = pd.concat([febr[[idField,'deli','foot','month']],marc[[idField,'deli','foot','month']]],axis=0)
comp2 = pd.merge(febr[[idField,'day','deli','foot','month']],marc[[idField,'day','deli','foot','month']],on=[idField,'day'],how='outer',suffixes=["_feb","_march"])

vist.loc[:,"deli_tot"] = vist.loc[:,"deli"]
setL = vist['deli'] != vist['deli']
vist.loc[setL,"deli_tot"] = vist.loc[setL,"deli_old"]
vist.to_csv(baseDir + "tmp/ciccia.csv")

scorR = t_s.scorPerf(vist,step="regressor",idField=idField)

print('-------------------calc-averages------------------------')
cist = vist#[vist['time'] < '2019-03-01T']
tL = ['id_poi','act','ref','weighted','foot','foot_old','deli_old','deli','capt','capt_old']
dist = cist[tL].groupby(idField).agg(np.nanmean).reset_index()
cist = cist[tL+['month']].groupby([idField,'month']).agg(np.nanmean).reset_index()
dist = dist.sort_values('capt')
dist.loc[:,"rank"] = list(range(dist.shape[0]))
dist = dist.sort_values('capt_old')
dist.loc[:,"rank_old"] = list(range(dist.shape[0]))
fist = vist[vist['deli_tot'] == vist['deli_tot']]

if False:
    print('-----------correlation-against-isocal-----------------')
    def clampF(x):
        return pd.Series({"r":sp.stats.pearsonr(x['deli_tot'],x['ref'])[0]})
    rist = fist.groupby([idField,'month']).apply(clampF).reset_index()
    rist.boxplot(column="r",by="month")
    plt.show()

if False:
    print('-----------------------sankey-for-ranking--------------------------')
    dist.loc[:,"diff"] = dist['rank'] - dist['rank_old']
    dist.loc[:,"diff_capt"] = dist['capt'] - dist['capt_old']
    
    from pySankey import sankey
    dist.loc[:,"bin"] = ["%.2f"%x for x in t_f.binOutlier(dist['capt'],nBin=5,isLabel=True)[0]]
    dist.loc[:,"bin_old"] = ["%.2f"%x for x in t_f.binOutlier(dist['capt_old'],nBin=5,isLabel=True)[0]]
    sankey.sankey(dist['bin'],dist['bin_old'],aspect=20,fontsize=12)
    plt.title("capture rate shift - binned")
    plt.show()

    dist = dist.sort_values('diff')
    t_v.plotParallel(dist[[idField,'rank_old','rank']],idField)#,colormap=plt.get_cmap("Set2"))
    plt.title("ranking shift")
    plt.show()

    t_v.plotParallel(dist[[idField,'capt_old','capt']],idField)#,colormap=plt.get_cmap("Set2"))
    plt.title("capt shift")
    plt.show()
    
    dist[['rank','rank_old']]
    print(sp.stats.spearmanr(dist['rank'],dist['rank_old'])[0])
    
if False:
    print('------------------------compare-with-previous-month------------------------')
    fist = vist[vist['deli_tot'] == vist['deli_tot']]
    fist.loc[:,"capt"] = fist['deli_tot']/fist['foot']*100.
    import matplotlib.patches as mpatches

    vist.boxplot(column=["capt","capt_old"])
    plt.show()
    
    fig, ax = plt.subplots(1,1)
    bx1 = vist.boxplot(column=["capt"],by=idField,ax=ax,return_type="dict")
    [[item.set_color('blue') for item in bx1[key]['boxes']] for key in bx1.keys()]
    [[item.set_color('blue') for item in bx1[key]['medians']] for key in bx1.keys()]
    bx2 = vist.boxplot(column=['capt_old'],by=idField,ax=ax,return_type="dict")
    [[item.set_color('orange') for item in bx2[key]['boxes']] for key in bx2.keys()]
    [[item.set_color('orange') for item in bx2[key]['medians']] for key in bx2.keys()]
    blue_patch = mpatches.Patch(color='blue',label='via')
    red_patch = mpatches.Patch(color='orange',label='tile')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(rotation=15)
    plt.show()
    
    fig, ax = plt.subplots(1,1)
    bx1 = fist.boxplot(column=["deli_tot"],by="month",ax=ax,return_type="dict")
    [[item.set_color('blue') for item in bx1[key]['boxes']] for key in bx1.keys()]
    [[item.set_color('blue') for item in bx1[key]['medians']] for key in bx1.keys()]
    bx2 = fist.boxplot(column=['ref'],by="month",ax=ax,return_type="dict")
    [[item.set_color('orange') for item in bx2[key]['boxes']] for key in bx2.keys()]
    [[item.set_color('orange') for item in bx2[key]['medians']] for key in bx2.keys()]
    red_patch = mpatches.Patch(color='orange', label='isocalendar')
    blue_patch = mpatches.Patch(color='blue', label='activities')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xticks(rotation=15)
    plt.show()

if True:
    print('---------------------------plot-delivery-curves---------------------')
    i = '1218'
    g = vist[vist[idField] == i]
    for i,g in vist.groupby(idField):
        tL = t_r.day2time(g['time'])
        plt.figure(figsize=(10,6))
        plt.title("poi: %s" % i)
        plt.plot(tL,g['deli'],label="delivery",linewidth=2)
        plt.plot(tL,g['ref'],label="isocalendar",linewidth=2)
        plt.plot(tL,g['act'],label="predicted",alpha=.5)
        plt.plot(tL,g['weighted'],label="weighted",alpha=.5)
        plt.plot(tL,g['deli_old'],label="past delivery",linewidth=2,alpha=.7)
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()
        break

if False:
    print('---------------show-bias---------------------')
    l = np.array(range(cist.shape[0]))
    cist = cist.sort_values('bias',ascending=False)
    fig, ax = plt.subplots(1,1)
    ax.bar(l,cist['bias'],width=.4,label="version 10",alpha=.5)
    ax.bar(l+.50,cist['version 11'],width=.4,label="version 11",alpha=.5)
    ax.legend()
    ax.set_xticklabels(cist[idField].values)
    plt.ylabel('deviation')
    plt.xlabel('id_poi')
    plt.show()

if False:
    print('---------------------------month-compatibility------------------------')
    comp = []
    fist = vist[vist['deli_tot'] == vist['deli_tot']]
    fist = fist[fist['deli_tot'] == fist['deli_tot']]
    for i,g in fist.groupby('month'):
        comp.append(g)
    scorD = []
    for j in range(len(comp)-1):
        scorL = []
        for i,g1 in comp[-j].groupby(idField):
            g2 = comp[-(j-1)]
            c = g2[g2[idField] == i]
            y1 = t_r.nonnull(g['deli_tot'])
            y2 = t_r.nonnull(c['deli_tot'])
            y3 = t_r.nonnull(g['foot'])
            y4 = t_r.nonnull(c['foot'])
            scorL.append({
                idField:i
                ,"p_value_act":sp.stats.ttest_ind(y1, y2)[1]
                ,"p_value_foot":sp.stats.ttest_ind(y3, y4)[1]
                ,"p_value_capt":sp.stats.ttest_ind(y1/y3, y2/y4)[1]
                #,"chi_2":sp.stats.chisquare(y1)
            })
        scorD.append(pd.DataFrame(scorL))
    scorD[0].boxplot(column=['p_value_act','p_value_foot','p_value_capt'])
    plt.show()

    scorM = pd.merge(scorD[1],scorD[0],on=idField,suffixes=["_old","_new"])
    
    from pySankey import sankey
    scorM.loc[:,"bin_new"] = ["%.2f"%x for x in t_f.binOutlier(scorM['p_value_capt_new'],nBin=10,isLabel=True)[0]]
    scorM.loc[:,"bin_old"] = ["%.2f"%x for x in t_f.binOutlier(scorM['p_value_capt_old'],nBin=10,isLabel=True)[0]]
    sankey.sankey(scorM['bin_new'],scorM['bin_old'],aspect=20,fontsize=12)
    plt.title("month compatibility - capt - p_value binned")
    plt.show()


    
    
    l = np.array(range(scorL.shape[0]))
    scorL.sort_values('p_value_capt',ascending=False,inplace=True)
    fig, ax = plt.subplots(1,1)
    ax.bar(l,scorL['p_value_act'],width=.4,label="activities",alpha=.5)
    ax.bar(l+.25,scorL['p_value_foot'],width=.4,label="footfall",alpha=.5)
    ax.bar(l+.75,scorL['p_value_capt'],width=.4,label="capt",alpha=.5)
    ax.legend()
    ax.set_xticklabels(scorL[idField].values)
    plt.ylabel('p_values')
    plt.xlabel('id_poi')
    plt.show()
    
    
if False:
    print('---------------------------joyplot-------------------------------')
    fig, axes = joypy.joyplot(comp2,column=['deli_feb','deli_march'],by=idField,ylim='own',figsize=(12,6),alpha=.5)#,colormap=cm.summer_r)
    plt.legend()
    plt.title('density distribution of activity (blue:feb, orange: march)')
    plt.show()

    fig, axes = joypy.joyplot(vist,column=['capt','capt_old'],by=idField,ylim='own',figsize=(12,6),alpha=.5)#,colormap=cm.summer_r)
    plt.legend()
    plt.title('density distribution of activity (blue:via, orange: tile)')
    plt.show()

    
    fig, axes = joypy.joyplot(comp2,column=['foot_feb','foot_march'],by=idField,ylim='own',figsize=(12,6),alpha=.5)#,colormap=cm.summer_r)
    plt.legend()
    plt.title('density distribution of footfall (blue:feb, orange: march)')
    plt.show()
    
    fig, axes = joypy.joyplot(vist,column=['capt'],by=idField,ylim='own',figsize=(12,6),alpha=.5,colormap=cm.summer_r)
    plt.title('density distribution of capture rate')
    plt.show()
    
if False:
    plog('-----------------enrich-capt-----------------')
    sist = vist.copy()
    sist = sist[sist['day'] > '2019-01-31T']
    norm = 0.4226623389831198
    sist.loc[:,"foot"] = sist["foot"]*norm*1.1
    sist.loc[:,"capt"] = sist["ref"]/sist["foot"]*100
    idL = np.unique(sist[idField])
    i = idL[0]
    i = '1218'
    sist.to_csv(baseDir + "tmp/capt.csv",index=False)
    for j,i in enumerate(idL):
        y = sist.loc[sist[idField] == i,"capt"].values
        y = sist.loc[sist[idField] == i,"ref"].values
        y1 = sist.loc[sist[idField] == i,"foot"].values
        y2 = sist.loc[sist[idField] == i,"act"].values
        y1 = y1*y.sum()/y1.sum()
        y2 = y2*y.sum()/y2.sum()
        setL = ~np.isnan(y)
        t = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in sist.loc[setL,"day"]]
        y = y[setL]
        plt.title(i)
        plt.plot(t,y1,label="foot")#+5*j)
        plt.plot(t,y,label="ref")#+5*j)
        plt.plot(t,y2,label="act")#+5*j)
        plt.legend()
        plt.show()

    shl.plotConfidenceInterval(sist['capt'],label="correlation",nInt=28,color="green")
    plt.show()
    
    sist.boxplot(column="capt",by=idField)
    plt.xticks(rotation=15)
    plt.show()
    
    gist = sist.groupby(idField).agg(np.mean).reset_index()
    gist.loc[:,"capt"] = gist["act"]/gist["foot"]*100
    gist.loc[:,"people_slip"] = gist["act"]/gist["ref"]
    print(gist.describe())
    gist.to_csv(baseDir + "raw/"+custD+"/poi_capt.csv",index=False)
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    poi.loc[:,"capt"] = pd.merge(poi,gist,on=idField,how="left",suffixes=["_x",""])["capt"]
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)

if False:
    print('-------------------------------kpi-distribution-----------------------')
    fig, ax = plt.subplots(1,2)
    shl.kpiDis(scorX,tLab="cross valid",col_cor="r_x_reg",col_dif="v_x_reg",col_sum="s_x_reg",isRel=True,ax=ax[0])
    #shl.kpiDis(kpiL,tLab="cross valid",col_cor="cor",col_dif="dif",col_sum="dif",isRel=False,ax=ax[0])
    shl.kpiDis(scorX,tLab="cross valid",col_cor="r_x_reg",col_dif="d_x_reg",col_sum="s_x_reg",isRel=False,ax=ax[1])
    #shl.kpiDis(scorR,tLab="regressor",col_cor="r_regressor",col_dif="d_regressor",col_sum="s_regressor",isRel=False,ax=ax[1])
    plt.show()

    idL = set(vist[idField])
    i = '1351'#np.unique(sact[idField])[6]
    g = vist[vist[idField] == i]
    t = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in g['day']]
    cact = gact.iloc[:,1:]
    cact.index = gact[idField]
    cact = cact.loc[:,cact.columns > '2019-01-31T']
    for i,g in vist.groupby(idField):
        y1 = cact[cact.index == i].values[0]
        y = g['act']
        y = y*y1.sum()/y.sum()
        pName = poi.loc[poi[idField] == i,"name"].values[0]
        plt.title("id: %s - %s - cor %.2f " % (i,pName,sp.stats.pearsonr(g['act'],g['ref'])[0]))
        plt.plot(t,y,label="activities")
        plt.plot(t,g['ref'],label="reference")
        #plt.plot(t,y1,label="mapping")
        plt.legend() 
        plt.xlabel("day")
        plt.ylabel("count")
        plt.xticks(rotation=15)
        plt.show()
    
    scorX.to_csv(baseDir + "raw/"+custD+"/scor_flexValid.csv",index=False)
    
    t_e.plotSum(vist,isLoc=False)
    plt.show()

    shl.plotHistogram(tist['precipProbability'],label="correlation")
    plt.show()

    importlib.reload(shl)    
    fig, ax = plt.subplots(1,2)
    shl.plotConfidenceInterval(kpiL['cor'],ax=ax[0],label="correlation",nInt=10,color="green")
    shl.plotConfidenceInterval(kpiL['dif'],ax=ax[1],label="difference",nInt=10)
    plt.show()

    importlib.reload(shl)
    fig, ax = plt.subplots(1,2)
    shl.plotHistogram(scorX['r_x_reg'],label="correlation",ax=ax[0])
    shl.plotHistogram(scorX['v_x_reg'],label="relative error",ax=ax[1])
    plt.show()
        

if False:
    print('-----------------delivery-check-------------------')
    dirc = pd.read_csv(baseDir + "raw/tank/delivery/act_ref_foot_19_03.csv",dtype={idField:str})
    dirG = dirc.groupby('day').agg(np.mean).reset_index()

    via1 = pd.read_csv(baseDir + "raw/tank/dirc_via_feb.csv",dtype={'location':str})
    del via1['time_origin']
    via2 = pd.read_csv(baseDir + "raw/tank/dirc_via_mar.csv",dtype={'location':str})
    via = pd.concat([via1,via2])
    del via['dir']
    via.loc[:,"count"] = via['count']*0.464928
    via.loc[:,"day"] = via['day'].apply(lambda x: x + "T")
    via.rename(columns={"location":idField,"count":"via_count"},inplace=True)
    via = via[via[idField].isin(np.unique(dirc[idField]))]
    viaG = via.groupby('day').agg(np.mean).reset_index()
    sect = pd.merge(dirc,via,on=["id_poi","day"],how="left",suffixes=["_via","_tile"])
    secG = sect.groupby('day').agg(np.mean).reset_index()

    if False:
        plt.plot(t_r.day2time(secG['day']),secG['foot'],label="tile")
        plt.plot(t_r.day2time(secG['day']),secG['via_count'],label="via")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()

    if False:
        i = random.choice(np.unique(df[idField]))
        d = sect[sect[idField] == i]
        plt.title("location %s" % (i))
        plt.plot(t_r.day2time(d['day']),d['foot'],label="direction count")
        plt.plot(t_r.day2time(d['day']),d['via_count'],label="via count")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()



if False:
    print('---------------------traffic-germany-------------------')
    dirT = pd.read_csv(baseDir + "raw/"+"bast"+"/dirCount_d.csv.gz",compression="gzip",dtype={idField:str})
    dirT = dirT[[x for x in dirT.columns if x > '2019-01-00T']]
    hT = t_r.timeCol(dirT)
    tT = t_r.day2time(hT)
    yT = dirT[hT].mean()
    yT = yT[yT > yT.mean()*.2]
    tT = t_r.day2time(yT.index)

    dirB = pd.read_csv(baseDir + "raw/"+"bast"+"/ref_visit_d.csv.gz",compression="gzip",dtype={idField:str})
    dirB = dirB[[x for x in dirB.columns if x > '2017-01-00T']]
    dirB = dirB[[x for x in dirB.columns if x < '2017-04-00T']]
    hL = t_r.timeCol(dirB)
    dirB.columns = [re.sub('2017-','2019-',x) for x in hL]
    hB = t_r.timeCol(dirB)
    tB = t_r.day2time(hB)
    yB = dirB[hB].mean()

    from sklearn import datasets, linear_model
    regr = linear_model.LinearRegression()
    XT = np.reshape(np.array((range(len(tT)))),(-1,1))
    regr.fit(XT,yT.values)
    yTi = regr.predict(XT)

    XB = np.reshape(np.array((range(len(tB)))),(-1,1))
    regr.fit(XB,yB.values)
    yBi = regr.predict(XB)
    
    plt.plot(tT,yT/yT.max(),label="tile",color="b")
    plt.plot(tT,yTi/yT.max(),label="tile interp",color="b")
    plt.plot(tB,yB/yB.max(),label="bast",color="orange")
    plt.plot(tB,yBi/yB.max(),label="bast interp",color="orange")
    plt.legend()
    plt.show()
    

    data = [dirc[hL1].mean(),dirc[hL2].mean()]
    plt.boxplot(data)
    plt.xticks([1,2],["feb","mar"])
    plt.show()




if False:
    print("-----------------radar-plot-of-most-relevant-kpi--------------------")
    tL = [ 'daily_visit',  'y_var', 'score', 'var_longShort', 'forecastability',"ref_day"]
    t_v.plotRadar(poi,tL)







print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
