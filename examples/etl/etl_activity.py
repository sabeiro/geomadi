#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import seaborn as sns

def plog(text):
    print(text)

projDir = "log/striezelmarkt/" + "single_dom/2017/"
def parseFolder(projDir,repDir):
    dateL1 = os.listdir(baseDir + projDir)
    dateL = []
    act = pd.DataFrame()
    for i,l in enumerate(dateL1):
        dateL = dateL + [dateL1[i] + "/" + x + "/" for x in os.listdir(baseDir + projDir + str(dateL1[i]))]
    for i,l in enumerate(dateL):
        tmp = pd.read_csv(baseDir + projDir + l + repDir)
        act = pd.concat([act,tmp],axis=0)
    act = act[np.logical_not(act['dominant_zone'] == '-1')]
    return act


act = parseFolder("log/striezelmarkt/" + "single_dom/2017/","activity_report/output/international.dominant_zone/part-00000")
act = act.pivot_table(index="time",columns="international",values="count",aggfunc=np.sum).fillna(0).reset_index()
act.loc[:,'time'] = act['time'].apply(lambda x: re.sub("T","",str(x)))
act.columns = ['day','national','international']
act = act[['day','national','international']]
act.loc[:,'national'] = act['national'].apply(lambda x: round(x))
act.loc[:,'international'] = act['international'].apply(lambda x: round(x))
act.to_clipboard(index=False)

act = parseFolder("log/striezelmarkt/" + "hours_dom/2017/","activity_report/output/international.dominant_zone/part-00000")
#act.loc[:,"count"] = act["count"] + act['international']
def timeSplit(x):
    try:
        t = datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S+00:00") + datetime.timedelta(hours=1)
    except:
        t = datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S+00:00+1") + datetime.timedelta(hours=1)
    return t.strftime("%Y-%m-%d"), t.hour, t
        
if False:
    fig, ax = plt.subplots(4,4)
    gact = act.groupby("dominant_zone")
    for g in gact:
        g = g[1].groupby('time').sum().reset_index()
        h = g['time'].apply(lambda x: timeSplit(str(x)))
        t = [x[2] for x in h]
        g.loc[:,'h'] = ["%02d-%02d" % (x[1],x[1]+1) for x in h]
        g = g.groupby('h').sum().reset_index()
        plt.xticks(rotation=45)
        plt.plot(g['h'],g['count'])
    plt.show()
    act.to_csv(baseDir + "striezelmarkt_cilac.csv")

act = act.pivot_table(index="time",values="count",aggfunc=np.sum).fillna(0).reset_index()
h = act['time'].apply(lambda x: timeSplit(x))
act.loc[:,'day'] = [x[0] for x in h]
act.loc[:,'hour'] = ["%02d-%02d" % (x[1],x[1]+1) for x in h]
act.loc[:,'count'] = act['count'].apply(lambda x: round(x))
act = act[['day','hour','count']]
act.to_clipboard(index=False)

act = parseFolder("log/striezelmarkt/" + "single_dom_overnight/2017/","activity_report/output/international.dominant_zone.overnight_zip/part-00000")    
act = act[np.logical_not(act['overnight_zip'] == -1)]
act.loc[:,'time'] = act['time'].apply(lambda x: re.sub("T","",str(x)))
act = act.pivot_table(index=["time","overnight_zip"],values="count",aggfunc=np.sum).fillna(0).reset_index()
act.loc[:,'count'] = act['count'].apply(lambda x: round(x))
act = act.loc[act['count']>0]
act.to_clipboard(index=False)

act = act.loc[act['time'] > '2017-11-28']
act = act.loc[act['time'] < '2017-12-25']
act = act.pivot_table(index="overnight_zip",values="count",aggfunc=np.sum).fillna(0).reset_index()
act.loc[:,'count'] = act['count'].apply(lambda x: round(x))
act.to_clipboard(index=False)

act = parseFolder("log/striezelmarkt/" + "single_dom_mcc/2017/","activity_report/output/international.dominant_zone.mcc/part-00000")
act = act.pivot_table(index="mcc",values="count",aggfunc=np.sum).fillna(0).reset_index()
act.loc[:,'count'] = act['count'].apply(lambda x: round(x))
mcc = pd.read_csv(baseDir + "raw/MCC.csv")
mcc = mcc.groupby("MCC").head(1)
act = pd.merge(act,mcc[["MCC","Country"]],left_on="mcc",right_on="MCC",how="left")
act = act.loc[act['count']>0]
act.to_clipboard(index=False)

#
act = pd.read_csv(baseDir+"raw/act_mtc_weekday.csv.tar.gz",compression='gzip')
act.columns = ['zone','count','hour']
act = act[act['zone'].isin([15792,15776,8540,16033,8542,13249,20236])]
hact = act.pivot_table(index=["zone","hour"],values="count",aggfunc=np.sum).reset_index()
act = act.pivot_table(index="zone",values="count",aggfunc=np.sum).reset_index()
act.loc[:,'edge'] = [2,2,4,1,1,1,8]
act.to_clipboard(index=False)
hact['hour'] = hact['hour'].replace("0+1",24)
hact['hour'] = hact['hour'].replace("1+1",25)
hact['hour'] = hact['hour'].apply(lambda x:int(x))
hact = pd.merge(hact,act,left_on="zone",right_on="zone",how="left",suffixes=["","_y"])
hact.loc[:,"density"] = hact['count']/hact['edge']/hact['edge']
hact.loc[:,'zone'] = hact['zone'].apply(lambda x:int(x))
hact.to_clipboard()

groups = hact.groupby('zone')
fig, ax = plt.subplots(2,1)
for name, group in groups:
    ax[0].plot(group['hour'], group['count'], marker='o', linestyle='', ms=12, label=name)
ax[0].legend()
for name, group in groups:
    ax[1].plot(group['hour'], group['density'], marker='o', linestyle='', ms=12, label=name)
ax[1].legend()
plt.show()

sns.pointplot('hour', 'count', data=hact[hact['zone']==8542], hue='zone', fit_reg=False)
plt.show()

hact = hact.sort_values("hour")
groups = hact.groupby('zone')
colL = [plt.cm.viridis(i/10,1) for i in range(10)]
cI = 0
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['hour'], group['count'], marker='o', linestyle='-', ms=12, label=name,color=colL[cI])
    plt.ylim((-45.450000000000045, 6828.4499999999998))
    ax.legend()
    cI += 1
plt.show()

plog('-------------------------------week-by-week--------------------------')
act = pd.read_csv(baseDir + "raw/act_markt.csv")
def apL(x):
    return str(x[0]) + "T%02d" % x[1]
act.loc[:,'t'] = [datetime.datetime.strptime(x,"%Y-%m-%dT%H") for x in act[['day','hour']].apply(lambda x: apL(x),axis=1).values]
act.loc[:,'wday'] = act['t'].apply(lambda x: x.weekday())
act.loc[:,'week'] = act['t'].apply(lambda x: x.week)
wact = act.groupby('week')
weekD = []
def apL(x):
    return "%02d-%02d" % (x[0],x[1])
for n,w in wact:
    w.loc[:,'d-h'] = ["%02d-%02d" % (x,y) for (x,y) in zip(w['wday'],w['hour'])]
    weekD.append(w[['d-h','count']])

for n in range(len(weekD)-1):
    w1 = weekD[n]
    w2 = weekD[n+1]
    w = pd.merge(w1,w2,left_on="d-h",right_on="d-h",how="inner")
    w.index = w['d-h']
    colN = str(n+1) + "-" + str(n)
    w.loc[:,colN] = w['count_y'] -  w['count_x']
    if n==0:
        diffD = w[colN]
    else:
        #diffD = diffD.reindex(pd.Index(np.unique([x for x in diffD.index] + [y for y in w.index])))
        #diffD.loc[w.index,str(n)] = w.loc[w.index,'count']
        diffD = pd.concat([diffD,w[colN]],axis=1)

diffD.loc[:,'day'] = [str(x)[0:2] for x in diffD.index]
diffP = diffD.groupby("day").agg(np.average).reset_index()

diffD.plot()
plt.show()
diffP.plot()
plt.show()

diffD.to_csv(baseDir + "deltaWeek_striezelmarkt.csv")
act.to_csv(baseDir + "act_striezelmarkt.csv")


print(diffP.head())
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
