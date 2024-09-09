import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
#sambaDir = os.environ['SAMBA_PATH']
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import geomadi.proc_lib as plib
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_utc_timestamp,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
import tarfile
import shutil
sc = pyspark.SparkContext.getOrCreate()
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
import pymongo
cred = json.load(open(baseDir + '/credenza/geomadi.json'))

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])
import importlib
importlib.reload(plib)

crmF = json.load(open(baseDir + "raw/basics/crmFact.json"))
ageC = pd.DataFrame(crmF['age'])
genC = pd.DataFrame(crmF['gender'])
cityN = "berlin"

ags8 = gpd.read_file(baseDir + "gis/geo/ags8.shp")
ags8L = {}
for i in ["Oberhaching","Stuttgart","Duesseldorf","Berlin","Hannover","Darmstadt","Dresden","Essen"]:
    ags8L[i.lower()] = list(ags8[ags8['GEN'] == i]['AGS'].values)
ags8L['visitBerlin'] = []
#ags8L['all'] = list(ags8['AGS']) 

custL = {"oberhaching":[12737,12735,8274,8275,8284,8285,12772,12773,12736]
         ,"stuttgart":[15,16,161,163,237,239,395,396,14878,14879,14880,14881,14882,14883,14884,14894,14895,14896,14897,14898,14899,14909,14910,14911,14912,14913,14914,14923,14924,14925,14926,14927,14939,14940,14941,14942,4952,14953,14954,14871]
         ,"duesseldorf":[385,383,144,146]}

if True:
    ags5 = gpd.read_file(baseDir + "gis/destatis/mtc_ags8.shp")
    custL['berlin'] = ags5[ags5['GEN']=='Berlin']['id'].values
    custL['hannover'] = ags5[ags5['GEN']=='Hannover']['id'].values
    custL['darmstadt'] = ags5[ags5['GEN']=='Darmstadt']['id'].values
    custL['dresden'] = ags5[ags5['GEN']=='Dresden']['id'].values
    tL = [13512, 13530, 13533, 13534, 13583, 13596, 13604, 8758, 8783, 8784, 8790, 20615, 8794, 8795, 8801, 8808, 8813, 8814, 8841, 8842, 8843, 8844, 8845, 20667, 8846, 8848, 13492, 13510, 13511]
    custL['dresden'] = list(custL['dresden']) + tL
    print(list(filter (lambda x : x not in custL['dresden'], tL)))
    custL['visitBerlin'] = [1834,1835,1810,1811,571,4279]
    custL["essen"] = ags5[ags5['GEN']=='Essen']['id'].values
    tL = ags5.loc[[bool(re.search("an der Ruhr",x)) for x in ags5['GEN']],"id"]
    #custL['all'] = list(ags5['id'].values)
    custL["essen"] = list(custL['essen']) + list(tL.values)

if False:
    print('----------------------parsing-odm-all-------------------')
    s = 'winter'
    timeL = ["mtc_day","mtc_hour","ags8_day","ags8_hour","zip_hour","zip_day"]
    # s = 'summer'
    # timeL = ["mtc_day","mtc_hour","ags8_day","zip_day"]
    timeRes = timeL[0]
    c = 'oberhaching'
    for timeRes in timeL:
        projDir = "file:///" + baseDir + "log/statWeek/"+s+"/" + timeRes
        projDir = baseDir + "log/statWeek/"+s+"/" + timeRes
        wL = os.listdir(projDir)
        for d in wL:
            df, fL = plib.parseParquet(projDir+"/"+d,is_lit=False,patterN="part-r-00")
            dL = [x.split("/")[1] for x in fL]
            jL = pd.DataFrame(plib.dateList(projDir,[d]))
            jL.loc[:,"wday"] = jL['dir'].apply(lambda x: re.sub("_h","",x))
            jL.loc[:,"wday"] = jL['wday'].apply(lambda x: "_".join(x.split("_")[-2:]) )
            jL.loc[:,"wday"] = jL['wday'].apply(lambda x: re.sub("statWeek_","",x))
            jL.loc[:,"dir"]  = jL.index
            df = df.withColumn("count",df['count']/float(jL['nDate'][0]))
            try : zipD = df.toPandas()
            except: continue
            if zipD.shape[0] == 0: continue
            if len(set(zipD["time_origin"])) == 1:
                del zipD["time_origin"], zipD["time_destination"]
            else:
                zipD.loc[:,"time_origin"] = zipD['time_origin'].apply(lambda x:x[:2])
                zipD.loc[:,"time_destination"] = zipD['time_destination'].apply(lambda x:x[:2])
            zipD.to_csv(baseDir + "raw/statWeek/"+s+"/all/odm_"+timeRes+"_"+jL['wday'][0]+".csv.gz",compression="gzip",index=False)

if False:
    print('----------------------parsing-odm--------------------')
    mtc = gpd.read_file(baseDir + "gis/destatis/mtc_popDens.shp")
    mtc.loc[:,"pop_density"] = mtc['Einwohner']/mtc['Flaeche_DE']
    mtp = mtc.groupby('id').agg(sum).reset_index()
    import importlib
    importlib.reload(plib)
    s = 'winter'
    timeL = ["mtc_day","mtc_hour","ags8_day","ags8_hour","zip_hour","zip_day"]
    s = 'summer'
    timeL = ["mtc_day","mtc_hour"]
    schema = ArrayType(StructType([StructField("dir",IntegerType(),False),StructField("wday",StringType(),False),StructField("nDate",IntegerType(),False)]))
    #timeL = ["mtc_day","mtc_hour"]
    timeRes = timeL[0]
    c = 'oberhaching'
    for timeRes in timeL:
        projDir = baseDir + "log/statWeek/"+s+"/" + timeRes
        df, fL = plib.parseParquet(projDir,is_lit=True,patterN="part-r-00")
        dL = [x.split("/")[1] for x in fL]
        jL = pd.DataFrame(plib.dateList(projDir,dL))
        jL.loc[:,"wday"] = jL['dir'].apply(lambda x: re.sub("_h","",x))
        jL.loc[:,"wday"] = jL['wday'].apply(lambda x: x.split("_")[-1])
        jL.loc[:,"dir"]  = jL.index
        rdd = sc.parallelize([tuple(x) for x in jL[["dir","wday","nDate"]].values])
        dL = sqlContext.createDataFrame(rdd,["dir","wday","nDate"])#,schema)
        df = df.join(dL,df['dir'] == dL['dir'],"cross")
        sqlContext.registerDataFrameAsTable(df,"table1")    
        for c in custL:
        #for c in ['all']:
            print("city: " + c)
            if bool(re.search("ags8",timeRes)): idlist = [int(x) for x in ags8L[c]]
            else : idlist = custL[c]
            if len(idlist) == 0: continue
            idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
            if c == 'all': df1 = df
            else: df1 = sqlContext.sql("SELECT * FROM table1 WHERE origin IN ("+idLs+") OR destination in ("+idLs+")")
            try : zipD = df1.toPandas()
            except: continue
            if zipD.shape[0] == 0: continue
            del zipD['dir']
            zipD.loc[:,"count"] = zipD['count']/zipD['nDate']
            ##zipD.loc[zipD['wday'].isin(['monday','tuesday','thursday','wednsday','friday']),"wday"] = "workday"
            ##zipD.loc[zipD['wday'].isin(['monday','tuesday','thursday','wednsday']),"wday"] = "workday"
            del zipD['nDate']
            zipG = zipD.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean).reset_index()
            zipG.loc[:,"time_origin"] = zipG['time_origin'].apply(lambda x:x[:2])
            zipG.loc[:,"time_destination"] = zipG['time_destination'].apply(lambda x:x[:2])
            if len(set(zipG["time_origin"])) == 1:
                del zipG["time_origin"], zipG["time_destination"]
            zipG.loc[:,"count"] = zipG['count'].apply(lambda x:int(x))
            zipG.to_csv(baseDir + "raw/statWeek/"+s+"/city/odm_"+c+"_"+timeRes+".csv.gz",compression="gzip",index=False)
            if bool(re.search("ags8",timeRes)):
                continue
            zipT = zipG[['origin','count']].groupby("origin").agg(sum).reset_index()
            zipT.loc[:,"origin"] = zipT['origin'].astype(int)
            zipT = pd.merge(zipT,mtp[['id','Einwohner']],left_on="origin",right_on="id",how="left",suffixes=["","_o"])
            zipT.loc[:,"o_dens"] = zipT['count']/zipT['Einwohner']
            zipT = zipT[zipT['id'].isin(idlist)]
            zipT.to_csv(baseDir + "gis/statWeek/statWeek_tripDens_"+c+".csv",index=False)

if False:
    plog('--------------------------print-sum-------------------------')
    df.groupBy().sum().collect()
    df.rdd.map(lambda x: (1,x[1])).reduceByKey(lambda x,y: x + y).collect()[0][1]

if False:
    plog('------------------------mtc-pop-dens--------------------')
    mtc = gpd.read_file(baseDir + "gis/destatis/mtc_popDens.shp")
    mtc.loc[:,"pop_density"] = mtc['Einwohner']/mtc['Flaeche_DE']
    tL = ['id'] + list(mtc.columns[mtc.dtypes == float])
    mtg = mtc.groupby('id').agg(np.mean).reset_index()
    mtg.to_csv(baseDir + "raw/destatis/mtc_popDens.csv",index=False)

if False:
    plog('-------------------old-dom-------------------')
    zipD = pd.read_csv(baseDir + "log/statWeek/actRep/zip_single.csv")
    zipD.to_csv(baseDir + "log/statWeek/actRep/zip_single.csv.gz",index=False,compression="gzip")
    
if False:
    plog('---------------------parse-odm-via---------------------')
    import importlib
    importlib.reload(plib)
    projDir = "log/odm_via/"
    dateL = os.listdir(baseDir + projDir)
    collE = client["tdg_infra_internal"]["nodes"]
    for j,i in enumerate(dateL):
        nameF = i.split(".")[0]
        nameF = "_".join(nameF.split("_")[2:]) + "_" + str(j)
        df, fL, job = plib.parseTarParquet(baseDir+projDir,i,patterN="odm_result",isLit=True)
        iL = [(i,x.split("/")[0]) for i,x in enumerate(fL)]
        iL = [(x[0],x[1].split("_")[-1]) for x in iL]
        rdd = sc.parallelize(iL)
        dL = sqlContext.createDataFrame(rdd,["dir","day"])
        #df = df.join(dL,on="dir",how="left")
        df = df.drop('dir')#.collect()
        zipD = df.toPandas()
        zipD.loc[:,"count"] = zipD['count']/len(job['date_list'])
        zipD.to_csv(baseDir+"gis/ptv/"+nameF+".csv",index=False)
        shutil.rmtree(fL[0].split("/")[0])
        zipG = zipD.groupby('location').agg(sum).reset_index()
        locD = job['odm_via_conf']['input_locations']
        locT = []
        for i in locD:
            for j in i['node_list']:
                neiN = collE.find({"node_id":j})
                nei = neiN[0]
                i['x'] = nei['loc']['coordinates'][0]
                i['y'] = nei['loc']['coordinates'][1]
                locT.append(i)
        locT = pd.DataFrame(locT)
        locT = pd.merge(locT,zipG,left_on="location_id",right_on="location",how="left")
        locT.to_csv(baseDir + "gis/ptv/"+nameF+"_loc.csv",index=False)
    
if False:
    plog('------------------------parse-actRep--------------------')
    projDir = baseDir + "log/statWeek/actRep/hourly"
    df, fL = plib.parsePath(projDir,is_lit=True)
    sqlContext.registerDataFrameAsTable(df,"table1")
    for c in custL:
        print(c)
        idlist = custL[c]
        idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
        df1 = sqlContext.sql("SELECT * FROM table1 WHERE dominant_zone IN ("+idLs+")")
        zipD = df1.toPandas()
        del zipD['dir']
        zipD.loc[:,"wday"] = zipD['name']
        zipD.loc[:,"count"] = zipD['count']/4.
        zipD.loc[zipD['name'].isin(['monday','tuesday','thursday','wednsday','friday']),"wday"] = "workday"
        del zipD['name']
        zipG = zipD.groupby(['wday','origin','destination','time_origin','time_destination']).agg(np.mean).reset_index()
        zipG.loc[:,"time_origin"] = zipG['time_origin'].apply(lambda x:x[:2])
        zipG.loc[:,"time_destination"] = zipG['time_destination'].apply(lambda x:x[:2])
        zipG.loc[:,"count"] = zipG['count'].apply(lambda x:int(x))
        zipG.to_csv(baseDir + "raw/others/statWeek_odm_h_"+c+".csv",index=False)

if False:
    plog('---------------parse-tared-activities---------------')
    import importlib
    import shutil
    importlib.reload(plib)
    projDir = "log/touristen/visitBerlin/"
    dateL = os.listdir(baseDir + projDir)
    for i in dateL:
        nameF = i.split(".")[0]
        nameF = "_".join(nameF.split("_")[2:])
        df, fL = plib.parseTar(baseDir+projDir,i,idlist=custL['visitBerlin'],patterN="part-00000",isLit=True)
        iL = [(i,x.split("/")[-3]) for i,x in enumerate(fL)]
        rdd = sc.parallelize(iL)
        dL = sqlContext.createDataFrame(rdd,["dir","day"])
        df = df.join(dL,on="dir",how="left")
        df = df.drop('dir')#.collect()
        zipD = df.toPandas()
        zipD = zipD[zipD['day'] != "output"]
        del zipD['day']
        if bool(re.search("age",nameF)):
            zipD = zipD[zipD['age'] > -1]
            zipD = pd.merge(zipD,ageC,on="age",how="left")
            zipD.loc[:,"count"] = zipD['count']*zipD['corr']
            del zipD['corr']
        if bool(re.search("gen",nameF)):
            zipD = zipD[zipD['gender'] > -1]
            zipD = pd.merge(zipD,genC,on="gender",how="left")
            zipD.loc[:,"count"] = zipD['count']*zipD['corr']
            del zipD['corr']
        zipD.to_csv(baseDir+"gis/visitBerlin/"+nameF+".csv",index=False)
        shutil.rmtree(fL[0].split("/")[0])

    dateL = os.listdir(baseDir+"gis/visitBerlin/")
    dateL = [x for x in dateL if bool(re.search("_h_",x))]
    zipL = []
    for i in dateL:
        act = pd.read_csv(baseDir+"gis/visitBerlin/" + i)
        day = i.split("_")[-1]
        day = day.split(".")[0]
        act.loc[:,"day"] = day        
        zipL.append(act)
    zipD = pd.concat(zipL)
    zipD.to_csv(baseDir+"gis/visitBerlin/touristen_day_h.csv",index=False)

if False:
    plog('---------------parse-tared-activities---------------')
    import importlib
    import shutil
    importlib.reload(plib)
    projDir = "log/statWeek/cw41/"
    dateL = os.listdir(baseDir + projDir)
    for i in dateL:
        nameF = i.split(".")[0]
        nameF = "_".join(nameF.split("_")[2:])
        df, fL = plib.parseTarParquet(baseDir+projDir,i,idlist=custL['oberhaching'],patterN="odm_result",isLit=True)
        iL = [(i,x.split("/")[3]) for i,x in enumerate(fL)]
        iL = [(x[0],re.sub("statWeek_","",x[1])) for x in iL]
        iL = [(x[0],re.sub("_h","",x[1])) for x in iL]
        iL = [(x[0],"2018-10-"+x[1]) for x in iL]
        rdd = sc.parallelize(iL)
        dL = sqlContext.createDataFrame(rdd,["dir","day"])
        df = df.join(dL,on="dir",how="left")
        df = df.drop('dir')#.collect()
        zipD = df.toPandas()
        zipD.to_csv(baseDir+"raw/others/"+nameF+".csv.gz",index=False,compression="gzip")
        shutil.rmtree(fL[0].split("/")[0])
        
if False:
    odm1 = pd.read_csv(baseDir + "raw/others/"+"oberhaching_cw41"+".csv.gz",compression="gzip")
    odm2 = pd.read_csv(baseDir + "raw/others/"+"oberhaching_cw41_p11_5"+".csv.gz",compression="gzip")
    print("missing odm %.2f" % ( 2.*(odm1.shape[0] - odm2.shape[0])/(odm1.shape[0] + odm2.shape[0])) )
    print("missing counts %.2f" % ( 2.*(odm1['count'].sum()-odm2['count'].sum())/(odm1['count'].sum()+odm2['count'].sum())) )
    print("missing odm %.2f" % ( 1. - (odm2.shape[0])/(odm1.shape[0])) )
    print("missing counts %.2f" % ( 1. - (odm2['count'].sum())/(odm1['count'].sum())) )
    
# df.coalesce(1).write.mode('overwrite').format("com.databricks.spark.csv").save(baseDir + "log/demo/statWeek_mtc.csv")
# zipD = pd.read_csv(baseDir + "log/demo/statWeek_mtc")

if True:
    plog('-------------------------statWeek-remote--------------------')
    weekL = ["mo","tu","we","th","fr","sa","su"]
    metrL = ["age","gender","zip","mcc","home"]
    fL = plib.browseFS("/tdg/stat_week/")
    idlist = list(idlist)
    idLs = ','.join(['"'+str(x)+'"' for x in np.unique(custL[cityN])])
    dirL = dict()
    for i in metrL:
        dirL[i] = {}
    for i in fL:
        met = [x for x in metrL if bool(re.search(x,i))][0]
        wee = [x for x in weekL if bool(re.search("_"+x+"$",i))][0]
        dirL[met][wee] = i
    for m in dirL.keys():
        for d in dirL[m].keys():
            projDir = "/tdg/stat_week/"+dirL[m][d]+"/2018/"
            for j in range(6):
                gL = plib.browseFS(projDir)
                projDir = projDir + "/" + gL[0]
            dirL[m][d] = projDir

    for m in dirL.keys():
        fL = [dirL[m][d] for d in dirL[m].keys()]
        for d in dirL[m].keys():
            f = dirL[m][d] 
            df = plib.readRemoteCsv(f)
            sqlContext.registerDataFrameAsTable(df,"table1")
            df = sqlContext.sql("SELECT * FROM table1 WHERE dominant_zone IN ("+idLs+")")
            df = df.withColumn("day",func.lit(d))
            if f==fL[0] :
                ddf = df
            else :
                ddf = ddf.unionAll(df)
        zipD = ddf.toPandas()
        zipD.to_csv(baseDir + "log/statWeek/city/" + "actRep_" + cityN + "_" + m + ".csv.gz",index=False,compression="gzip")

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

