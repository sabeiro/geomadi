import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as sh
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
from shapely.geometry.polygon import Polygon
import geomadi.geo_octree as g_o
import allink.dis_energy as d_e

gO = g_o.h3tree()
custD = "dep"
dep = pd.read_csv(baseDir + "raw/"+custD+"/ride_dep.csv.gz")
weatL = pd.read_csv(baseDir+"raw/"+custD+"/weather_h.csv.gz",compression="gzip")

#ride_rev
tL = dep['ended_at'].apply(lambda x: datetime.datetime.strptime(x[:19],"%Y-%m-%d %H:%M:%S"))
dep.loc[:,"time"] = tL.apply(lambda x: x.timestamp())
dep.loc[:,"week"] = tL.apply(lambda x: x.isocalendar()[1])
dep.loc[:,"weekday"] = tL.apply(lambda x: x.weekday())
dep.loc[:,"hour"] = tL.apply(lambda x: x.hour)
dep.loc[:,"day"] = ["%04d-%02d-%02d" % (x.year,x.month,x.day) for x in tL]
dep = dep.merge(weatL[['day','hour','apparentTemperature']],on=["day","hour"],how="left")
dep.rename(columns={"apparentTemperature":"temp"},inplace=True)
dep.loc[:,"temp"] = 1. - dep["temp"]/40.
dep.loc[:,"n"] = 1
dep = dep[dep['cost'] > - 10.] ## refund
dep.loc[:,"cost"] = dep["cost"].abs()
dep.loc[:,"downstream_cost"] = dep["downstream_cost"].abs()
dep.loc[:,"rev"] = dep['cost'] #+ dep['downstream_cost']
shift = pd.DataFrame({"hour":list(range(24)),"shift":[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]})
dep = dep.merge(shift,on="hour",how="left")

if False:
    dep.boxplot(column=["cost","downstream_cost","rev"])
    plt.show()

for d,k in zip([8,9,10],[980,140,20]):
    print(d)
    precDigit = d
    dep.loc[:,"geohash"] = dep.apply(lambda x: gO.encode(x['end_longitude'],x['end_latitude'],precision=precDigit),axis=1)
    """historical series per area, week, weekday, shift"""
    dens = dep.groupby(["geohash","week","weekday","shift","event_type"]).agg(sum).reset_index()
    dens.loc[:,"temp"] = dens['temp']/dens['n']
    setL = dens['event_type'] == "ride"
    dens.loc[:,"n_ride"] = 0
    dens.loc[setL,"n_ride"] = dens.loc[setL,"n"]
    dens.loc[:,"n_dep"] = 0
    dens.loc[~setL,"n_dep"] = dens.loc[~setL,"n"]
    dens = dens.sort_values("n_ride",ascending=False)
    def clampF(x):
        return pd.Series({"rev":sum(x['rev'])
                          ,"n":sum(x['n']),"n_ride":sum(x['n_ride']),"n_dep":sum(x['n_dep'])
                          ,"activation":d_e.chemPot(x['n_ride'],x['n_dep'],max_occ=k,thermal_noise=x['temp'])
                          #,"prob":d_e.sumProb(x['n_ride'],x['n_dep'])
        })
    """group per area, weekday and shift"""
    densP = dens.groupby(['geohash','weekday','shift']).apply(clampF).reset_index()
    densP.dropna(inplace=True)
    densP.loc[:,"err"]   = densP['activation'].apply(lambda x: x[2])
    densP.loc[:,"noise"] = densP['activation'].apply(lambda x: x[1])
    densP.loc[:,"activation"]  = densP['activation'].apply(lambda x: x[0])
    densP = densP.sort_values("n_ride",ascending=False)
    # densP.boxplot(column=["activation","noise"])
    # plt.show()
    densP.loc[:,"prob"] = densP.apply(lambda x: d_e.numericProb(x['activation'],thermal_noise=x['noise']),axis=1)
    densP.loc[:,"urev"] = densP['rev']/densP['n']
    densP.loc[:,"pot"] = densP['prob']*densP['urev']
    """prepare-plot"""
    densS = densP.groupby('geohash').agg(np.mean).reset_index()
    densS.to_csv(baseDir + "raw/dep/chem_pot_"+str(precDigit)+".csv.gz",compression="gzip",index=False)
    densS.sort_values('activation',ascending=False,inplace=True)
    polyL = densS.apply(lambda x: sh.geometry.Polygon(gO.decodePoly(x['geohash'])),axis=1)
    poly  = gpd.GeoDataFrame(densS,geometry=polyL)
    poly.loc[:,"area"] = poly['geometry'].apply(lambda x: x.area)
    poly.loc[:,"dens"] = poly.apply(lambda x: x['n']/x['area'],axis=1)
    poly.loc[:,"urev"] = poly['rev']/poly['n']
    poly.to_file(baseDir + "gis/dep/chem_pot_"+str(precDigit)+".shp")

print(densP.describe())
    
if False:
    print('------------display-fit-------------')
    intL = []
    for i in range(1):
        y = densP.iloc[i]["prob"]
        t, y1, x0, err = s_i.fitFermi(y)
        intL.append({"i":i,"activation":x0[0],"noise":x0[1],"err":err})
        plt.plot(y1,label=str(i))
        plt.plot(y,label=str(i))
    plt.show()
    intL = pd.DataFrame(intL)
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("fitting on noise")
    intL.boxplot(column=["activation","noise","err"],ax=ax[0])
    ax[1].set_title("fix noise")
    intL.boxplot(column=["activation","noise","err"],ax=ax[1])
    # ax[0].set_ylim([-1,4])
    # ax[1].set_ylim([-1,4])
    plt.show()

if False:
    print('----------------------------area-average-values----------------------')
    for d in [8,9,10]:
        print(d)
        precDigit = d
        dep.loc[:,"geohash"] = dep.apply(lambda x: gO.encode(x['end_longitude'],x['end_latitude'],precision=precDigit),axis=1)
        densS = dep.groupby(["geohash"]).agg(sum).reset_index()
        polyL = densS.apply(lambda x: sh.geometry.Polygon(h3.h3_to_geo_boundary(x['geohash'])),axis=1)
        poly  = gpd.GeoDataFrame(densS,geometry=polyL)
        poly.loc[:,"area"] = poly['geometry'].apply(lambda x: x.area)
        poly.loc[:,"dens"] = poly.apply(lambda x: x['n']/x['area'],axis=1)
        poly.loc[:,"urev"] = poly['rev']/poly['n']
        poly.to_file(baseDir + "gis/dep/rev_hash"+str(precDigit)+".shp")

if False:
    print('------------fit-activation-distribution---------------')
    import series_interp as s_i
    importlib.reload(s_i)
    projDir = baseDir + "raw/ride/"
    dL = [x for x in os.listdir(projDir) if bool(re.search("_prob",x))]
    dis = pd.read_csv(projDir + dL[2],index_col=0)
    X = dis.values
    intL = []
    for i in range(1):#X.shape[0]):
        i = 169
        t, y1, x0, err = s_i.fitFermi(X[i])
        intL.append({"i":i,"activation":x0[0],"noise":x0[1],"err":err})
        plt.plot(y1,label="fit",linewidth=3)
        plt.plot(X[i],label="measure",linewidth=2)
    plt.legend()
    plt.show()
    intL = pd.DataFrame(intL)
    intL = intL.sort_values("activation",ascending=False)

    intL1 = intL

    fig, ax = plt.subplots(1,2)
    ax[0].set_title("fitting on noise")
    intL.boxplot(column=["activation","noise","err"],ax=ax[0])
    ax[1].set_title("fix noise")
    intL1.boxplot(column=["chem","noise","err"],ax=ax[1])
    ax[0].set_ylim([-1,4])
    ax[1].set_ylim([-1,4])
    plt.show()


    plt.title("Fermi fitting" % (x0[0]))
    i = 11
    y = X[i]
    t, y1, x0, err = s_i.fitFermi(y)
    plt.plot(X[i],label="empirical")
    plt.plot(y1,label="distribution")
    i = 169
    y = X[i]
    t, y1, x0, err = s_i.fitFermi(y)
    plt.plot(X[i],label="empirical")
    plt.plot(y1,label="distribution")
    i = 40
    y = X[i]
    t, y1, x0, err = s_i.fitFermi(y)
    plt.plot(X[i],label="empirical")
    plt.plot(y1,label="distribution")
    plt.legend()
    plt.show()


