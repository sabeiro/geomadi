import os, sys, gzip, random, csv, json, re, datetime
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import geopandas as gpd
import numpy as np
import geomadi.proc_lib as plib
import matplotlib.pyplot as plt
import tarfile
import shutil

def plog(text): print(text)
cred = json.load(open(baseDir + '/credenza/geomadi.json'))

import pymongo
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])

import importlib
importlib.reload(plib)

crmF = json.load(open(baseDir + "raw/basics/crmFact.json"))
ageC = pd.DataFrame(crmF['age'])
genC = pd.DataFrame(crmF['gender'])

projDir = baseDir + "log/statWeek/act/"
cityL = [d for d in os.listdir(projDir) if os.path.isdir(os.path.join(projDir, d))]
fL = []
for path,dirs,files in os.walk(projDir):
    for f in files:
        if re.search("csv.gz",f):
            fL.append(path+"/"+f)
            
fL = sorted(fL)
ageL = []
genL = []
mccL = []
homL = []
domL = []
for f in fL:
    zipD = pd.read_csv(f,compression="gzip")
    pivL = ["dominant_zone","time","day"]
    metr = re.sub("d_","",f)
    metr = metr.split("_")[0].split("/")[-1]
    day  = f.split("_")[1].split(".")[0]
    city = f.split("/")[-2]
    zipD.loc[:,"day"] = day
    zipD.loc[:,"city"] = city
    zipD.loc[zipD['time'].isnull(),"time"] = "0-24"
    if bool(re.search("age",metr)):
        zipD = zipD[zipD['age'] > -1]
        zipD = pd.merge(zipD,ageC,on="age",how="left")
        zipD.loc[:,"count"] = zipD['count']*zipD['corr']
        zipD1 = zipD.groupby(pivL).agg(sum).reset_index()
        zipD = pd.merge(zipD,zipD1,on=pivL,how="left",suffixes=["","_y"])
        zipD.loc[:,"percent"] = zipD["count"]/zipD["count_y"]*100.
        del zipD['corr'], zipD['age'], zipD['count_y'], zipD['corr_y'], zipD['age_y'], zipD['count']
        ageL.append(zipD)
    elif bool(re.search("gender",metr)):
        domL.append(zipD[['dominant_zone','time','day','city','count']])
        zipD = zipD[zipD['gender'] > -1]
        zipD = pd.merge(zipD,genC,on="gender",how="left")
        zipD.loc[:,"count"] = zipD['count']*zipD['corr']
        zipD1 = zipD.groupby(pivL).agg(sum).reset_index()
        zipD = pd.merge(zipD,zipD1,on=pivL,how="left",suffixes=["","_y"])
        zipD.loc[:,"percent"] = zipD["count"]/zipD["count_y"]*100.
        del zipD['corr'], zipD['gender'], zipD['count_y'], zipD['corr_y'], zipD['gender_y'], zipD['count']
        genL.append(zipD)
    elif bool(re.search("home",metr)):
        zipD = zipD[zipD['home_zone'] > -1]
        zipD1 = zipD.groupby(pivL).agg(sum).reset_index()
        zipD = pd.merge(zipD,zipD1,on=pivL,how="left",suffixes=["","_y"])
        zipD.loc[:,"percent"] = zipD["count"]/zipD["count_y"]*100.
        del zipD['count_y'], zipD['count'], zipD['home_zone_y']
        homL.append(zipD)
    elif bool(re.search("mcc",metr)):
        zipD = zipD[zipD['mcc'] > -1]
        zipD1 = zipD.groupby(pivL).agg(sum).reset_index()
        zipD = pd.merge(zipD,zipD1,on=pivL,how="left",suffixes=["","_y"])
        zipD.loc[:,"percent"] = zipD["count"]/zipD["count_y"]*100.
        del zipD['count_y'], zipD['count'], zipD["mcc_y"]
        mccL.append(zipD)
        
#mccL = pd.concat(mccL)
#mccL.to_csv(baseDir + "raw/statWeek/act/mcc.csv.gz",compression="gzip",index=False)
ageL = pd.concat(ageL)
genL = pd.concat(genL)
homL = pd.concat(homL)
domL = pd.concat(domL)

ageL.to_csv(baseDir + "raw/statWeek/act/age.csv.gz",compression="gzip",index=False)
genL.to_csv(baseDir + "raw/statWeek/act/gen.csv.gz",compression="gzip",index=False)
homL.to_csv(baseDir + "raw/statWeek/act/hom.csv.gz",compression="gzip",index=False)
domL.to_csv(baseDir + "raw/statWeek/act/dom.csv.gz",compression="gzip",index=False)

if False:
    plog('-------------filter-out-city--------------')
    ageL = pd.read_csv(baseDir + "raw/statWeek/act/age.csv.gz",compression="gzip")
    genL = pd.read_csv(baseDir + "raw/statWeek/act/gen.csv.gz",compression="gzip")
    homL = pd.read_csv(baseDir + "raw/statWeek/act/hom.csv.gz",compression="gzip")
    domL = pd.read_csv(baseDir + "raw/statWeek/act/dom.csv.gz",compression="gzip")

    cityL = ['aldi','kaufland']
    ageL = ageL[ageL['city'].isin(cityL)]
    genL = genL[genL['city'].isin(cityL)]
    homL = homL[homL['city'].isin(cityL)]
    domL = domL[domL['city'].isin(cityL)]

    dayL = ['tu','we','th']
    ageL.loc[ageL['day'].isin(dayL),"day"] = "wd"
    genL.loc[genL['day'].isin(dayL),"day"] = "wd"
    homL.loc[homL['day'].isin(dayL),"day"] = "wd"
    domL.loc[domL['day'].isin(dayL),"day"] = "wd"
    
    ageL.to_csv(baseDir + "raw/statWeek/act/kpmg/age.csv.gz",compression="gzip",index=False)
    genL.to_csv(baseDir + "raw/statWeek/act/kpmg/gen.csv.gz",compression="gzip",index=False)
    homL.to_csv(baseDir + "raw/statWeek/act/kpmg/hom.csv.gz",compression="gzip",index=False)
    domL.to_csv(baseDir + "raw/statWeek/act/kpmg/dom.csv.gz",compression="gzip",index=False)

if False:
    print('-----------------------delivery-specific-filters-------------------------')
    c = "essen"
    c = "darmstadt"
    t = "hour"
    t = "day"
    zipG = pd.read_csv(baseDir + "raw/statWeek/winter/odm_"+c+"_mtc_"+t+".csv.gz",compression="gzip")
    #zipG.loc[zipG['wday'].isin(['friday','monday','thursday','tuesday','wednsday',]),"wday"] = "workday_mean"
    #zipG.loc[zipG['wday'] == "workday_mean","count"] = zipG.loc[zipG['wday'] == "workday_mean","count"]/5.
    #zipG.loc[zipG['wday'].isin(['monday','tuesday','sunday',"workday_mean"]),"wday"]
    zipG = zipG.loc[zipG['wday'].isin(['saturday','sunday',"workday"]),:]
    zipG.to_csv(baseDir + "raw/others/odm_"+c+"_mtc_"+t+".csv.gz",compression="gzip",index=False)
    zipD = zipG[['wday','count']].groupby('wday').agg(sum).reset_index()
    plt.title("weekday sum city %s - %s resolution" % (c,t))
    plt.bar(zipD['wday'],zipD['count'])
    plt.show()
    idlist = np.unique(zipG.loc[zipG['origin'] == zipG['destination'],"origin"])
    zipT = zipD.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean)
    pd.DataFrame({"id_list":idlist}).to_csv(baseDir+"raw/others/idlist_"+c+".csv",index=False)
    print(np.unique(zipG['wday']))

if False:
    plog('-------------------aggregate-days-----------------------')
    odmD = pd.read_csv(baseDir + "raw/statWeek/odm_berlin_mtc_day.csv.gz",compression="gzip")
    odmD = odmD[odmD['wday'].isin(['sunday','workday','saturday'])]
    if False:
        odmD.to_csv(baseDir + "raw/others/odm_berlin_mtc_hour.csv.gz",compression="gzip",index=False)

    odmG = odmD.groupby("origin").agg(sum).sort_values("count")
    pd.DataFrame(custL['berlin']).to_csv(baseDir + "raw/others/idlist_berlin.csv",index=False)
    dayA = pd.DataFrame({"day":['sunday', 'workday', 'friday', 'tuesday', 'monday', 'wednsday', 'thursday'],"wday":['sunday', 'workday', 'friday', 'tuesday', 'monday', 'wednsday', 'thursday']})
    
if False:
    dateL = os.listdir(baseDir + "raw/others/boston/tar/")
    dateL = [x for x in dateL if bool(re.search("csv",x))]
    aact = pd.DataFrame()
    oact = pd.DataFrame()
    gact = pd.DataFrame()
    zact = pd.DataFrame()
    pact = pd.DataFrame()    
    for i in dateL:
        day = i.split("_")[0]
        act = pd.read_csv(baseDir + "raw/others/boston/tar/" + i)
        act.loc[:,"dominant_zone"] = act['dominant_zone'].astype(int)
        act.loc[:,"day"] = day
        act.loc[:,"count"] = act['count']/4.
        if bool(re.search("age",i)):
            aact = pd.concat([aact,act],axis=0)
        elif bool(re.search("overnight",i)):
            oact = pd.concat([oact,act],axis=0)
        elif bool(re.search("gender",i)):
            gact = pd.concat([gact,act],axis=0)
        elif bool(re.search("home_zone",i)):
            zact = pd.concat([zact,act],axis=0)
        else :
            print(i)
            pact = pd.concat([pact,act],axis=0)

    pivL = ["dominant_zone","time","day"]
            
    aact = aact[aact['age'] > -1]
    aact = pd.merge(aact,ageC,on="age",how="left")
    aact.loc[:,"count"] = aact['count']*aact['corr']
    aact1 = aact.groupby(pivL).agg(sum).reset_index()
    aact = pd.merge(aact,aact1,on=pivL,how="left",suffixes=["","_y"])
    aact.loc[:,"count"] = aact["count"]/aact["count_y"]
    del aact['corr'], aact['age'], aact['count_y'], aact['corr_y'], aact['age_y']
    aact.to_csv(baseDir + "raw/others/boston/" + "act_duesseldorf_age.csv",index=False)
    
    oact = oact[oact['overnight_zip'] > -1]
    oact1 = oact.groupby(pivL).agg(sum).reset_index()
    oact = pd.merge(oact,oact1,on=pivL,how="left",suffixes=["","_y"])
    oact.loc[:,"count"] = oact["count"]/oact["count_y"]
    del oact['count_y'], oact['overnight_zip_y']
    oact.to_csv(baseDir + "raw/others/boston/" + "act_duesseldorf_overnight.csv",index=False)
    
    gact = gact[gact['gender'] > -1]
    gact = pd.merge(gact,genC,on="gender",how="left")
    gact.loc[:,"count"] = gact['count']*gact['corr']
    gact1 = gact.groupby(pivL).agg(sum).reset_index()
    gact = pd.merge(gact,gact1,on=pivL,how="left",suffixes=["","_y"])
    gact.loc[:,"count"] = gact["count"]/gact["count_y"]
    del gact['count_y'], gact['gender'], gact['corr'], gact['gender_y'], gact['corr_y']
    gact.to_csv(baseDir + "raw/others/boston/" + "act_duesseldorf_gender.csv",index=False)

    zact = zact[zact['home_zone'] > -1]
    zact1 = zact.groupby(pivL).agg(sum).reset_index()
    zact = pd.merge(zact,zact1,on=pivL,how="left",suffixes=["","_y"])
    zact.loc[:,"count"] = zact["count"]/zact["count_y"]
    del zact['count_y'], zact['home_zone_y']
    zact.to_csv(baseDir + "raw/others/boston/" + "act_duesseldorf_homezone.csv",index=False)

    pact.to_csv(baseDir + "raw/others/boston/" + "act_duesseldorf_sum.csv",index=False)
    print(oact[oact['overnight_zip']>0.]['count'].sum()/pact['count'].sum())

if False:
    plog('-------------------------boston-consulting---------------------')
    idL = custL['duesseldorf']
    projDir = "log/samba/Data_Science/Customer_Projects_DE/Statistical_Week/tar/"
    dateL = os.listdir(baseDir + projDir)
    for i in dateL:
        print(i)
        act = pd.read_csv(baseDir + projDir + i,compression="gzip")
        colL = act.columns.values
        colL[0] = 'dominant_zone'
        act.columns = colL
        act = act[act['count'] > 0.]
        if act['time'].dtype == object:
            act.loc[:,"time"] = act['time'].apply(lambda x:x[:2])
            #act.to_csv(baseDir + projDir + i,compression="gzip",index=False)
        act = act[act['dominant_zone'].isin(idL)]
        act.to_csv(baseDir+"raw/others/boston/tar/"+i,compression="gzip",index=False)

if False:
    print('------------------------------check-local-time------------------------')
    winT = pd.read_csv(baseDir + "raw/statWeek/winter/odm_berlin_mtc_hour.csv.gz")
    sumT = pd.read_csv(baseDir + "raw/statWeek/summer/odm_berlin_mtc_hour.csv.gz")
    w = winT.groupby(['wday','time_origin']).agg(sum).reset_index()
    s = sumT.groupby(['wday','time_origin']).agg(sum).reset_index()
    w = w[w['wday'] == 'saturday']
    s = s[s['wday'] == 'sat']
    plt.plot(w['time_origin'],w['count'],label="winter")
    plt.plot(s['time_origin'],s['count'],label="summer")
    plt.legend()
    plt.show()
    
if False:
    print('----------------------shift-night-hours----------------------')
    c = "essen"
    t = "hour"
    zipG = pd.read_csv(baseDir+"raw/statWeek/winter/city/odm_"+c+"_mtc_"+t+".csv.gz",compression="gzip")
    hourS = pd.DataFrame({"hour":range(24),"hour_shift":[(x+2)%24 for x in range(24)]})
    hL = list(range(24))
    dayL = ['monday','tuesday','wednsday','thursday','friday','saturday','sunday']
    shiftL = np.roll(dayL,shift=-1)
    weekD = {'monday':'0','tuesday':'1','wednsday':'2','thursday':'3','friday':'4','saturday':'5','sunday':'6','workday':'7','workday_mean':'8'}
    weekS = pd.DataFrame({'monday':hL,'tuesday':hL,'wednsday':hL,'thursday':hL,'friday':hL,'saturday':hL,'sunday':hL,'workday':hL,'workday_mean':hL})
    weekS = weekS.melt()
    weekS.columns = ['wday','hour']
    weekS.loc[:,"wday_shift"] = weekS['wday']
    for i,d in enumerate(dayL):
        setL = (weekS['wday'] == d) & (weekS['hour'].isin([0,1,2]))
        weekS.loc[setL,'wday_shift'] = shiftL[i]
    
    weekS = weekS.rename(columns={"hour":"time_origin"})
    zipD = zipG.merge(weekS,on=["wday","time_origin"])
    zipD = zipD.sort_values(['time_origin','origin'])
    print(zipD.head())
    del zipD['wday']
    zipD = zipD.rename(columns={"wday_shift":"wday"})
    zipG = zipG.sort_values(['time_origin','origin'])
    
    zipH = zipG.groupby(['wday','time_origin']).agg(np.sum).reset_index()
    zipH.loc[:,"t"] = zipH.apply(lambda x: "%s-%02d" % (weekD[x['wday']],x['time_origin']),axis=1)
    zipH = zipH.sort_values(['t'])
    plt.plot(zipH['t'],zipH['count'],label="original")
    zipH = zipD.groupby(['wday','time_origin']).agg(np.sum).reset_index()
    zipH.loc[:,"t"] = zipH.apply(lambda x: "%s-%02d" % (weekD[x['wday']],x['time_origin']),axis=1)
    zipH = zipH.sort_values(['t'])
    plt.plot(zipH['t'],zipH['count'],label="shifted")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    zipH1 = zipG.groupby(['wday','time_origin']).agg(np.sum).reset_index()
    zipH2 = zipD.groupby(['wday','time_origin']).agg(np.sum).reset_index()
    print( (zipH1['count']-zipH2['count']).count()/zipH1['count'].sum()*100.)

    zipD.loc[zipD['wday'].isin(dayL[:5]),"wday"] = "workday_mean"
    zipT = zipD.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean)
    zipT.to_csv(baseDir+"raw/others/odm_"+c+"_mtc_"+t+"_shift.csv.gz",compression="gzip")
    #zipD.to_csv(baseDir+"raw/others/odm_"+c+"_mtc_"+t+"_shift.csv.gz",compression="gzip",index=False)
    zipT1 = pd.read_csv(baseDir+"raw/others/odm_"+c+"_mtc_"+t+".csv.gz",compression="gzip")
    print(zipT['count'].sum()/zipT1['count'].sum())
    zipG1 = zipT.groupby('wday').agg(sum)
    zipG2 = zipT1.groupby('wday').agg(sum)

if False:
    od1 = pd.read_csv(baseDir + "/raw/statWeek/summer/city/odm_dresden_mtc_hour.csv.gz",compression="gzip")
    od2 = pd.read_csv(baseDir + "tmp/dresden/statWeek_odm_dresden_h.csv")
    od3 = pd.read_csv(baseDir + "/raw/statWeek/winter/city/odm_dresden_mtc_hour.csv.gz",compression="gzip")
    dayL = ['mon','tue','wed','thu']
    od1.loc[od1['wday'].isin(dayL),"wday"] = "workday"
    dayL = ['sun']
    od1.loc[od1['wday'].isin(dayL),"wday"] = "sunday"
    dayL = ['sat']
    od1.loc[od1['wday'].isin(dayL),"wday"] = "saturday"
    dayL = ['fri']
    od1.loc[od1['wday'].isin(dayL),"wday"] = "friday"
    dayL = ['monday','tuesday','wednsday','thursday']
    od3.loc[od3['wday'].isin(dayL),"wday"] = "workday"
    print(set(od1['wday']))
    print(set(od2['wday']))
    print(set(od3['wday']))
    od1 = od1.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean).reset_index()
    od2 = od2.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean).reset_index()
    od3 = od3.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean).reset_index()

    od1.to_csv(baseDir + "tmp/dresden/odm_dresden_h_p11_summer.csv.gz",compression="gzip",index=False)
    od2.to_csv(baseDir + "tmp/dresden/odm_dresden_h_p10_winter.csv.gz",compression="gzip",index=False)
    od3.to_csv(baseDir + "tmp/dresden/odm_dresden_h_p11_winter.csv.gz",compression="gzip",index=False)
    
    print(od1['count'].sum()/od2['count'].sum())
    print(od3['count'].sum()/od2['count'].sum())

    od11 = od1[od1['origin'].isin(od1['destination'])]
    od12 = od2[od2['origin'].isin(od2['destination'])]
    od13 = od3[od3['origin'].isin(od3['destination'])]
    print(len(np.unique(od1['origin']))/len(np.unique(od2['origin'])))
    tL1 = np.unique(od11['origin'])
    tL2 = set(od21['origin'])
    print( set(od21['origin']).difference(od11['origin']) )
    
    odm = od3.merge(od2,on=['wday','origin','destination','time_origin','time_destination'],how="inner")
    odm.loc[:,"asym"] = .5*(odm['count_x'] - odm['count_y'])/(odm['count_x'] + odm['count_y'])
    mtc = gpd.read_file(baseDir + "gis/geo/mtc.shp")
    mtc.loc[:,"x"] = mtc['geometry'].apply(lambda x: x.centroid.xy[0][0])
    mtc.loc[:,"y"] = mtc['geometry'].apply(lambda x: x.centroid.xy[1][0])
    odm = odm.merge(mtc[['id','x','y']],left_on="origin",right_on="id",how="left")
    odm = odm.merge(mtc[['id','x','y']],left_on="destination",right_on="id",how="left")
    odm.loc[:,"r"] = np.sqrt((odm['x_x'] - odm['x_y'])**2 + (odm['y_x'] - odm['y_y'])**2)
    print(odm['count_x'].sum()/odm['count_y'].sum())
    plt.scatter(odm['r'],odm['asym'])
    plt.xlabel("distance")
    plt.ylabel("asymmetry")
    plt.show()

    odm.boxplot(column=['asym'])
    plt.show()
    
    odm.to_csv(baseDir + "tmp/dresden/odm_dresden_h_asymmetry.csv.gz",compression="gzip",index=False)
