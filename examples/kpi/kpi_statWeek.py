import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.train_reshape as t_r

od10h = pd.read_csv(baseDir + "doc/darmstadt/statWeek_odm_darmstadt_h.csv.gz",compression="gzip")
od10d = pd.read_csv(baseDir + "doc/darmstadt/statWeek_odm_darmstadt_d.csv.gz",compression="gzip")

od11h = pd.read_csv(baseDir + "raw/statWeek/winter/odm_darmstadt_mtc_hour.csv.gz",compression="gzip")
od11d = pd.read_csv(baseDir + "raw/statWeek/winter/odm_darmstadt_mtc_day.csv.gz",compression="gzip")

odch = pd.read_csv(baseDir + "raw/statWeek/winter/odm_darmstadt_ags8_hour.csv.gz",compression="gzip")
odcd = pd.read_csv(baseDir + "raw/statWeek/winter/odm_darmstadt_ags8_day.csv.gz",compression="gzip")

comm = pd.read_csv(baseDir + "raw/basics/commuter.csv")
comm.loc[:,"commuter"] = comm['pendouttot'] + comm['pendintota']
comm.loc[:,"tripPerson"] = comm['commuter']/comm['poptotal']
if False:
    from sklearn import datasets, linear_model
    regr = linear_model.LinearRegression()
    XT = np.reshape(np.log(comm['poptotal'].values),(-1,1))
    regr.fit(XT,np.log(comm['tripPerson']))
    yT = regr.predict(XT)

    fig, ax = plt.subplots(1,1)
    plt.scatter(comm['poptotal'],comm['tripPerson'])
    plt.plot(np.exp(XT[:,0]),np.exp(yT),color="red",linewidth=2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("population")
    ax.set_ylabel("trip per person")    
    plt.show()
comm = comm[comm['region']=='Darmstadt-Dieburg']

od11h = od11h[od11h['wday'].isin(list(set(od10d['wday'])))]
od11d = od11d[od11d['wday'].isin(list(set(od10d['wday'])))]

odd = pd.merge(od10d,od11d,on=["wday","origin","destination"],how="outer",suffixes=["_10","_11"])
odh = pd.merge(od10h,od11h,on=["wday","origin","destination","time_origin","time_destination"],how="outer",suffixes=["_10","_11"])

wd = odd[odd['wday'] == 'workday']
ags = odcd[odcd['wday'] == 'workday']
wd.to_csv(baseDir + "tmp/odm_10-11_darmstadt.csv.gz",compression="gzip",index=False)
odh.to_csv(baseDir + "tmp/odm_h_10-11_darmstadt.csv.gz",compression="gzip",index=False)
agsEx = ags[ags['origin'] != ags['destination']]
odEx = wd[wd['origin'] != wd['destination']]

print("from/to Wiesbaden")
print(ags[(ags['origin'] == 6414000) | (ags['destination'] == 6414000)])
c = agsEx[(agsEx['origin'] == 6414000)]['count'] .values + agsEx[agsEx['destination'] == 6414000]['count'].values
print("ratio %f" % (c/agsEx['count'].sum()) )


print(odd['count_10'].sum()/odd['count_11'].sum())
print(odd['count_10'].mean()/odd['count_11'].mean())
trip = pd.DataFrame({"p10":wd['count_10'].sum()/158254.
                     ,"p11":wd['count_11'].sum()/158254.
                     ,"p11_ex":odEx['count_11'].sum()/158254.
                     ,"p11_ags":ags['count'].sum()/158254.
                     ,"p11_ags_ex":agsEx['count'].sum()/158254.
                     ,"atlas":comm['tripPerson'].values[0]*2.},index=[0])

plt.bar(trip.columns,trip.iloc[0])
plt.xlabel('version')
plt.ylabel('trip per inhabitant')
plt.show()

print(odd.describe())
odd.boxplot(column=["count_10","count_11"],by="wday")
plt.show()

if False:
    print('-----------------evalutate-night-hours---------------')
    c, t = 'essen', 'hour'
    zipG = pd.read_csv(baseDir + "raw/others/odm_"+c+"_mtc_"+t+".csv.gz",compression="gzip")
    zipH = zipG.groupby(['wday','time_origin']).agg(np.sum).reset_index()
    zipH.loc[:,"t"] = zipH.apply(lambda x: x['wday'] + "-" + str(x['time_origin']),axis=1)
    plt.plot(zipH['t'],zipH['count'])
    plt.xticks(rotation=45)
    plt.show()

if False:
    print('----------------essen-------------------')
    essen = gpd.read_file(baseDir + "gis/others/essen.geojson")
    mtc = gpd.read_file(baseDir + "gis/geo/mtc.shp")
    mtcE = mtc[mtc['geometry'].intersects(essen['geometry'][0])]
    c, t = 'essen', 'hour'
    zipG = pd.read_csv(baseDir + "raw/others/odm_"+c+"_mtc_"+t+".csv.gz",compression="gzip")
    zipG = pd.read_csv(baseDir+"raw/others/odm_"+c+"_mtc_"+t+"_shift.csv.gz",compression="gzip")
    zipG1 = zipG[zipG['origin'].isin(mtcE['id'])]
    print(zipG1["count"].sum()/zipG['count'].sum())

    zipW = zipG1.groupby(['wday','time_origin']).agg(sum).reset_index()
    zipW.loc[:,"day_n"] = zipW.apply(lambda x: weekD[x['wday']],axis=1)
    zipW = zipW.sort_values(["day_n",'time_origin'])
    del zipW['origin'], zipW['destination'], zipW['time_destination']
    zipW.to_csv(baseDir + "raw/others/odm_"+c+"_mtc_"+t+"_gebiet.csv",index=False)

if False:
    print('-------------------symmetry--------------------------')
    c = "berlin"
    o,d = 16173, 13494
    c = "essen"
    o,d = 13975, 5719
    cityL = ["oberhaching","stuttgart","duesseldorf","berlin","hannover","darmstadt","dresden","essen"]
    c = cityL[-1]
    weekL = []
    for c in cityL:
        odm = pd.read_csv(baseDir + "gis/statWeek/odm_"+c+"_mtc_day.csv")
        odmD = odm.merge(odm,left_on=['wday','origin','destination'],right_on=['wday','destination','origin'])
        odmD = odmD[odmD['origin_x'] != odmD['destination_x']]
        odmD = odmD[odmD['origin_x'] != odmD['origin_y']]
        odmD.loc[:,"dif"] = 2.*(odmD['count_x']-odmD['count_y'])/(odmD['count_x']+odmD['count_y'])
        odmD.loc[:,"tot"] = odmD['count_x'] + odmD['count_y']
        odmG = odmD.groupby(['wday','origin_x']).agg(np.mean).reset_index()
        for i,g in odmG.groupby('wday'):
            x = np.average(g['dif'])
            s = np.std(g['dif'])
            xw = np.average(g['dif'],weights=g['tot'])
            weekL.append({"city":c,"wday":i,"mean":x,"mean_w":xw,"std":s})
            
    weekL = pd.DataFrame(weekL)
    weekL.loc[:,"weight"] = weekL["mean_w"]*weekL["std"]/8.
    
    weekL.boxplot(column="mean_w",by="city")
    plt.xticks(rotation=15)
    plt.show()
    
    weekL.boxplot(column="mean_w",by="wday")
    plt.xticks(rotation=15)
    plt.show()
    
    fig, ax = plt.subplots(1,1)
    hist, bins = np.histogram(odmG['dif'],bins=21)
    bins = [ (x1+x2)*.5 for x1, x2 in zip(bins[1:],bins[:-1])]
    x = np.average(odmG['dif'],weights=odmG['count_x'])
    ax.bar(bins,hist)
    ax.axvline(x,color="r")
    plt.show()

    odm = pd.read_csv(baseDir + "gis/statWeek/odm_"+c+"_mtc_day.csv")
    odmD = odm.merge(odm,left_on=['wday','origin','destination'],right_on=['wday','destination','origin'])
    odmD = odmD[odmD['origin_x'] != odmD['destination_x']]
    odmD = odmD[odmD['origin_x'] != odmD['origin_y']]
    odmD.loc[:,"dif"] = 2.*(odmD['count_x']-odmD['count_y'])/(odmD['count_x']+odmD['count_y'])
    odmD.loc[:,"tot"] = odmD['count_x'] + odmD['count_y']
    odmE  = odmD[odmD['origin_x'] == o ]# | (odmD['destination_x'] == o )]
    odmE1 = odmD[odmD['origin_x'] == d ]# | (odmD['destination_x'] == d )]
    setL  = (odmD['origin_x'] == o ) & (odmD['origin_y'] == d )
    odmE2 = odmD[setL]

    fig, ax = plt.subplots(1,4)
    for i,g,j in zip(range(4),[odmD,odmE,odmE1,odmE2],["region","city","suburb","city-suburb"]):
        hist, bins = np.histogram(g['dif'],bins=21,weights=g['tot'])
        bins = [ (x1+x2)*.5 for x1, x2 in zip(bins[1:],bins[:-1])]
        x = np.average(g['dif'],weights=g['tot'])
        ax[i].bar(bins,hist)
        ax[i].axvline(x,color="r")
        ax[i].set_title("%s asym %.2f # %.0f" % (j,x,x*g['tot'].sum()) )
    plt.show()

    mtcL = sorted(list(set(odm['origin']) & set(odm['destination'])))
    asyL = []
    for i in mtcL:
        x1 = odm.loc[odm['origin'] == i,'count'].sum()
        x2 = odm.loc[odm['destination'] == i,'count'].sum()
        asyL.append({"mtc":i,"origin":x1,"destination":x2})
    asyL = pd.DataFrame(asyL)
    asyL.loc[:,"asym"] = 2.*(asyL['origin']-asyL['destination'])/(asyL['origin']+asyL['destination'])
    asyL.loc[:,"tot"] = asyL['origin']+asyL['destination']
    asyL = asyL.sort_values("asym")

    nl = np.percentile(asyL['tot'],[25,50,75,100])
    fig, ax = plt.subplots(1,4)
    for i,c in enumerate(nl):
        g = asyL[asyL['tot'] < c]
        hist, bins = np.histogram(g['asym'],bins=21,weights=g['tot'])
        bins = [ (x1+x2)*.5 for x1, x2 in zip(bins[1:],bins[:-1])]
        x = np.average(g['asym'],weights=g['tot'])
        ax[i].bar(bins,hist)
        ax[i].axvline(x,color="r")
        ax[i].set_title("max %d lost %.0f" % (int(c),x*g['tot'].sum()))
    plt.show()

    
    asyD = {}
    asyD['all'] = np.average(odmD['dif'],weights=odmD['tot'])
    asyD['essen_center'] = np.average(odmE['dif'],weights=odmE['tot'])
    asyD['alterburg'] = np.average(odmE2['dif'],weights=odmE2['tot'])
    asyD['alterburg-essen'] = np.average(odmE1['dif'],weights=odmE1['tot'])
    asyD = pd.DataFrame(asyD,index=[0])
    plt.bar(asyD.columns,asyD.values[0])
    plt.show()

    t_v.plotHist(odmE1['dif'],nBin=21)
    plt.show()

    t_v.plotHist(odmD['dif'],nBin=21)
    plt.show()

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
