import os, sys, gzip, random, csv, json
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import geomadi.geo_octree as g_o
import lernia.train_reshape as t_r

ret = gpd.read_file(baseDir + "gis/retail/retail.shp")
retD = pd.read_csv(baseDir + "raw/gps/retail.csv")
retD = retD.sort_values("n")
retD.loc[:,"rank"] = list(range(retD.shape[0]))
shop = pd.read_csv(baseDir + "raw/gps/retail_act.csv")
shop = shop.rename(columns={"Center_ID":"id"})
shoD = shop.groupby('id').agg(np.sum).reset_index()
shoD = shoD.sort_values('unique_visitors')
shoD.loc[:,"rank"] = list(range(shoD.shape[0]))
retD = retD.merge(shoD,on="id",how="inner")
retD.loc[:,"dens_gps"] = retD['n']/retD['area']
retD.loc[:,"dens_act"] = retD['unique_visitors']/retD['area']
print(sp.stats.spearmanr(retD['n'],retD['unique_visitors']))
print(sp.stats.spearmanr(retD['dens_gps'],retD['dens_act']))

retD.loc[:,"rank_gps"], _ = t_r.binOutlier(retD['rank_x'],nBin=10)
retD.loc[:,"rank_act"], _ = t_r.binOutlier(retD['rank_y'],nBin=10)
t_v.plotSankey(retD,col_1="rank_gps",col_2="rank_act")
retD.to_csv(baseDir + "raw/gps/retail_comp.csv",index=False)

dens = pd.read_csv(baseDir + "raw/gps/dens_month.csv.gz",compression="gzip",dtype={'octree':str})

import importlib
importlib.reload(g_o)

for j,k in ret.iterrows():
    gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
    poly = k['geometry']
    g3 = gO.boundingBox(poly.exterior.bounds)
    setL = dens['octree'].apply(lambda x: x[:len(g3)] == g3)
    densP = dens[setL]
    count = 0.
    for i,g in densP.iterrows():
        p = sh.geometry.Polygon(gO.decodePoly(g['octree']))
        a = p.intersection(poly)
        share = a.area/p.area
        count += g['n']*share
    ret.loc[j,"n"] = [count]
    
ret.to_file(baseDir + "gis/retail/retail.shp")
ret.loc[:,"area"] = ret['geometry'].apply(lambda x: x.area)
ret[['id','Name','n','area']].to_csv(baseDir + "raw/gps/retail.csv",index=False)

if False:
    polyL = [sh.geometry.Polygon(gO.decodePoly(g3))]
    polyF = gpd.GeoDataFrame({"geometry":polyL,'n':ret['name'],'octree':[g3]})
    polyF.to_file(baseDir + "gis/gps/bbox_octree.shp")

if False:
    print('------------------unique-retail-----------------------')
    retD = pd.read_csv(baseDir + "raw/gps/retail/unique.csv.gz",compression="gzip")
    retD.loc[:,"poi"] = retD['poi'] + 1
    retE = pd.read_csv(baseDir + "raw/gps/retail.csv")
    retC = pd.read_csv(baseDir + "raw/gps/retail_comp.csv")
    retD.loc[:,"dt"] = retD['dt']/60.
    retD = retD[retD['poi'] < 18]
    retE = retE[retE['id'] < 18]
    print(retE['n'].sum(),retD['event'].sum())
    
    plt.title("returning visitors")
    retU = retD.groupby("id").agg(len)
    retU.sort_values("dt",inplace=True)
    retUu = retU.groupby("dt").agg(len)
    plt.bar(retUu.index,retUu['event'])
    plt.ylabel("counts")
    plt.xlabel("times")
    plt.show()

    retU = retD.groupby(["poi","id"]).agg(len).reset_index()
    retUu = retU.groupby(["poi","event"]).agg(sum).reset_index()
    retU = retUu.pivot_table(index="poi",columns="event",values="day",aggfunc=np.sum)
    retU = retU.sort_values(1,ascending=False)
    retL = retC[retC['id'] < 18]
    retL.index = retL['id']
    retL = retL.loc[retU.index]
    retU.index = retL['Name']
    plt.imshow(retU)
    plt.xlabel("days")
    plt.ylabel("poi")
    plt.yticks(range(retL.shape[0]),retL['Name'])
    plt.show()
    retU.to_csv(baseDir + "raw/gps/retail_returning.csv")

    plt.title("event frequency")
    retU = retD.groupby("event").agg(len)
    retU.sort_values("dt",inplace=True)
    plt.bar(retU.index,retU['poi'])
    plt.ylabel("counts")
    plt.xlabel("frequency")
    plt.show()

    plt.title("dwelling time distribution min")
    retD[retD['dt']>0].boxplot(column="dt")
    plt.show()

    plt.title("dwelling time distribution min")
    def clampF(x):
        return pd.Series({"dwelling":np.median(x['dt'])})
    retU = retD[retD['dt']>0].groupby("wday").apply(clampF).reset_index()
    plt.bar(retU['wday'],retU['dwelling'],label="unique",alpha=.5)
    plt.ylabel("dwelling")
    plt.xlabel("weekday")
    plt.show()

    plt.title("dwelling time distribution shopping center")
    def clampF(x):
        return pd.Series({"dwelling":np.median(x['dt'])})
    retU = retD[retD['dt']>0].groupby("poi").apply(clampF).reset_index()
    retU = retU.merge(retC,left_on="poi",right_on="id",how="left")
    retU.sort_values("dwelling",inplace=True,ascending=False)
    retU.loc[:,"rank"] = range(retU.shape[0])
    plt.bar(retU['rank'],retU['dwelling'],label="unique",alpha=.5)
    plt.ylabel("dwelling")
    plt.xlabel("weekday")
    plt.xticks(retU['rank'],retU['Name'],rotation=15)
    plt.show()
    
    plt.title("weekday order")
    def clampF(x):
        return pd.Series({"count":sum(x['event'])/len(x['event'])})
    retU = retD.groupby("wday").apply(clampF).reset_index()
    plt.bar(retU['wday'],retU['count'],label="unique",alpha=.5)
    plt.ylabel("counts")
    plt.xlabel("weekday")
    plt.show()

    plt.title("day frequency")
    retU = retD.groupby("day").agg(len)
    retU.sort_values("dt",inplace=True)
    plt.bar(retU.index,retU['poi'])
    plt.ylabel("counts")
    plt.xlabel("frequency")
    plt.show()

    plt.title("poi frequency")
    retU = retD.groupby("poi").agg(len)
    retU = retU.merge(retC,left_on="poi",right_on="id",how="left")
    retU.sort_values("event",inplace=True,ascending=False)
    retU.loc[:,"rank"] = range(retU.shape[0])
    plt.bar(retU['rank'],retU['dt'])
    plt.ylabel("counts")
    plt.xlabel("poi")
    plt.xticks(retU['rank'],retU['Name'],rotation=15)
    plt.show()

    plt.title("poi frequency events")
    retU = retC[retC['id'] < 18]
    retU.sort_values("n",inplace=True,ascending=False)
    retU.loc[:,"rank"] = range(retU.shape[0])
    plt.bar(retU['rank'],retU['n'])
    plt.ylabel("counts")
    plt.xlabel("poi")
    plt.xticks(retU['rank'],retU['Name'],rotation=15)
    plt.show()

    plt.title("poi frequency activities")
    retU = retD.groupby("poi").agg(len)
    retU = retU.merge(retC,left_on="poi",right_on="id",how="left")
    retU.sort_values("unique_visitors",inplace=True,ascending=False)
    retU.loc[:,"rank"] = range(retU.shape[0])
    plt.bar(retU['rank'],retU['unique_visitors'])
    plt.ylabel("counts")
    plt.xlabel("poi")
    plt.xticks(retU['rank'],retU['Name'],rotation=15)
    plt.show()

    plt.title("poi order")
    retU = retD.groupby("poi").agg(len).reset_index()
    retU.sort_values("event",inplace=True,ascending=False)
    retU.loc[:,"rank"] = range(retU.shape[0])
    retE1 = retE.merge(retU,left_on="id",right_on="poi")
    plt.bar(retU['rank'],retU['dt']/max(retU['dt']),label="unique",alpha=.5)
    plt.bar(retE1['rank'],retE1['n']/max(retE1['n']),label="event",alpha=.5)
    plt.ylabel("counts")
    plt.xlabel("poi")
    plt.xticks(retE1['rank'],retE1['Name'],rotation=15)
    plt.legend()
    plt.show()

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
