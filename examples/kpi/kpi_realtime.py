import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import lernia.train_viz as t_v
import lernia.train_reshape as t_r
import albio.series_stat as s_s
import lernia.train_score as t_s
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import seaborn as sns
from scipy import signal

idField = "id_poi"
custD = "realtime"

print('------------------------load-reshape------------------------')

poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
agg = pd.read_csv(baseDir + "raw/"+custD+"/poi_realtime_agg.csv.gz",compression="gzip")
agg.loc[:,"crm"] = agg.apply(lambda x: "age_%d_gender_%d" % (x['age'],x['gender']),axis=1)
agg.rename(columns={"location_id":idField},inplace=True)
cross = pd.read_csv(baseDir + "raw/"+custD+"/poi_realtime_cross.csv.gz",compression="gzip")
crm = pd.read_csv(baseDir + "raw/"+custD+"/poi_realtime_crm.csv.gz",compression="gzip")
aggL = agg.groupby([idField,'age','gender']).agg(np.mean).reset_index()
aggA = agg.groupby([idField,'age']).agg(sum).reset_index()[[idField,'age','share']]
aggG = agg.groupby([idField,'gender']).agg(sum).reset_index()[[idField,'gender','share']]
aggA.loc[:,"share"] = aggA["share"]/7.
aggG.loc[:,"share"] = aggG["share"]/7.
cL = list(np.unique(agg['crm']))
print(cL)
# aggA = aggA.pivot_table(index=idField,columns="age",values = agg.pivot_table(index=[idField,"date","hour"],columns="age",values="share",aggfunc=np.sum))
houA = aggA.pivot_table(index=idField,columns="age",values="share",aggfunc=np.sum)
houA.columns = ["age_1","age_2","age_3","age_4","age_5"]
houA = houA.replace(float('nan'),0)
houA.loc[:,houA.columns] = np.multiply(houA.values,1./houA.sum(axis=1)[:, np.newaxis])

aggAh = agg.groupby([idField,'date','hour','age']).agg(np.mean).reset_index()
aggAh = aggAh.pivot_table(index=[idField,'date','hour'],columns='age',values='share',aggfunc=np.sum)
norm = 1./aggAh.sum(axis=1)
aggAh.loc[:,aggAh.columns] = np.multiply(aggAh.values,norm.values[:,np.newaxis])
aggAh.columns = ["age_"+str(x) for x in aggAh.columns]

aggGh = agg.groupby([idField,'date','hour','gender']).agg(np.mean).reset_index()
aggGh = aggGh.pivot_table(index=[idField,'date','hour'],columns='gender',values='share',aggfunc=np.sum)
norm = 1./aggGh.sum(axis=1)
aggGh.loc[:,aggGh.columns] = np.multiply(aggGh.values,norm.values[:,np.newaxis])
aggGh.columns = ["gender_"+str(x) for x in aggGh.columns]

agg.loc[:,'crm'] = agg.apply(lambda x: "age_"+str(x['age'])+'_gender_'+str(x['gender']),axis=1)
hou = agg.pivot_table(index=[idField,'date','hour'],columns='crm',values='share',aggfunc=np.sum)
hou = hou.replace(float('nan'),0)
norm = 1./hou[cL].sum(axis=1)
hou.loc[:,cL] = np.multiply(hou[cL].values,norm.values[:,np.newaxis])
cL = t_r.overlap(cross.columns,hou.columns)
hou = hou.reset_index()
aggX = hou[ [idField]+cL ].groupby(idField).agg(np.mean)
crossX = cross[ [idField]+cL ].groupby(idField).agg(np.mean)
norm = 1./aggX.sum(axis=1)
aggX.loc[:,aggX.columns] = np.multiply(aggX.values,norm.values[:,np.newaxis])
norm = 1./crossX.sum(axis=1)
crossX.loc[:,crossX.columns] = np.multiply(crossX.values,norm.values[:,np.newaxis])
print("missing", [x for x in aggX.index if not x in crossX.index])
aggX = aggX[aggX.index.isin(crossX.index)]

ts = [datetime.datetime.fromtimestamp(x) for x in crm['date']]
crm.loc[:,"min"] = [x.minute for x in ts]
crm.loc[:,"hour"] = [x.hour for x in ts]
xtab = pd.crosstab(crm['hour'],crm['min'],margins=False)

ts = [datetime.datetime.fromtimestamp(x) for x in cross['date']]
cross.loc[:,"min"] = [x.minute for x in ts]
cross.loc[:,"hour"] = [x.hour + x.minute/60. for x in ts]
ts = [datetime.datetime.fromtimestamp(x) for x in crm['date']]
crm.loc[:,"hour"] = [x.hour + x.minute/60. for x in ts]
crost = cross.groupby('hour').agg(np.mean)
hout = hou.groupby('hour').agg(np.mean)
crmt = crm.groupby('hour').agg(np.mean)
norm = 1./hout.sum(axis=1)
hout.loc[:,hout.columns] = np.multiply(hout.values,norm.values[:,np.newaxis])
norm = 1./crost.sum(axis=1)
crost.loc[:,crost.columns] = np.multiply(crost.values,norm.values[:,np.newaxis])
norm = 1./crmt.sum(axis=1)
crmt.loc[:,crmt.columns] = np.multiply(crmt.values,norm.values[:,np.newaxis])

ts = [datetime.datetime.fromtimestamp(x) for x in cross['date']]
cross.loc[:,"hour"] = [x.hour for x in ts]
cross.loc[:,"day"] = [x.strftime("%Y-%m-%d") for x in ts]
crosH = cross.groupby([idField,'day','hour']).agg(np.mean)
norm = 1./crosH[cL].sum(axis=1)
crosH.loc[:,cL] = np.multiply(crosH[cL].values,norm.values[:,np.newaxis])
cross.loc[:,"hour"] = [x.hour + x.minute/60. for x in ts]

tL = sorted([x for x in crm.columns if any([re.search("age",x),re.search("gender",x)])])
gL = sorted([x for x in crm.columns if any([re.search("gender",x)])])
aL = sorted([x for x in crm.columns if any([re.search("age",x)])])
aL = aL[:-1]
gL = gL[:-1]

import importlib
importlib.reload(t_v)

if False:
    print('-------------time-distribution--------------')
    fig, ax = plt.subplots(1,4)
    grm = crm.groupby("hour").agg(len).reset_index()
    gagg = agg.groupby("hour").agg(len).reset_index()    
    ax[0].set_title("hour distribution")
    ax[0].bar(grm['hour'],grm[idField])
    grm = crm.groupby("min").agg(len).reset_index()
    ax[1].set_title("minute distribution")
    ax[1].bar(grm['min'],grm[idField])
    ax[2].set_title("cross tab")
    cmap = plt.get_cmap("RdBu")
    ax[2] = sns.heatmap(xtab,cmap=cmap,square=True,ax=ax[2])
    ax[3].set_title("hour distribution/past delivery")
    ax[3].bar(gagg['hour'],gagg['date'])
    plt.show()

    aggH = agg.groupby('hour').agg(np.mean)
    plt.bar(aggH.index,aggH['share'])
    plt.xlabel('hour')
    plt.ylabel('share')
    plt.show()

if False:
    print('--------------------typical-day------------------------')
    fig, ax = plt.subplots(1,2)
    for c in cL:
        # ax[0].plot(crost.index,crost[c],label=c)
        # ax[1].plot(hout.index,hout[c],label=c)
        ax[0].plot(crost.index,crost[c],label=c)
        ax[1].plot(hout.index,hout[c],label=c)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title("10 min")
    ax[1].set_title("hourly")
    ax[0].set_ylim([0.,0.25])
    ax[1].set_ylim([0.,0.25])
    ax[0].legend(fontsize='x-small')
    ax[1].legend(fontsize='x-small')
    plt.show()
    fig, ax = plt.subplots(1,2)
    for c in aL:
        ax[0].plot(crmt.index,crmt[c],label=c)
    for c in gL:
        ax[1].plot(crmt.index,crmt[c],label=c)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title("10 min age")
    ax[1].set_title("10 min gender")
    plt.show()
    
    importlib.reload(t_v)
    colL = ['blue','red','green','yellow','purple','red','brown','olive','cyan','#ffaa44','#441188']
    colL = colL + colL
    fig, ax = plt.subplots(1,3)
    for i,c in enumerate(cL):
        t_v.plotBinned(cross,col_y=c,col_bin="hour",ax=ax[0],label=c,color=colL[i],alpha=.05)
        t_v.plotBinned(crosH,col_y=c,col_bin="hour",ax=ax[1],label=c,color=colL[i],alpha=.05)
        t_v.plotBinned(hou,col_y=c,col_bin="hour",ax=ax[2],label=c,color=colL[i],alpha=.05)
    ax[2].legend(fontsize='x-small')
    ax[0].set_title("10 min")
    ax[1].set_title("10 min in hour")
    ax[2].set_title("hourly")
    ax[0].set_ylim([-0.05,0.4])
    ax[1].set_ylim([-0.05,0.4])
    ax[2].set_ylim([-0.05,0.4])
    plt.show()

    fig, ax = plt.subplots(1,3)
    cross.boxplot(column=cL,ax=ax[0])
    crosH.boxplot(column=cL,ax=ax[1])
    hou.boxplot(column=cL,ax=ax[2])
    ax[0].set_title("10 min")
    ax[1].set_title("10 min in hour")
    ax[2].set_title("hourly")
    for tick in ax[0].get_xticklabels(): tick.set_rotation(15)
    for tick in ax[1].get_xticklabels(): tick.set_rotation(15)
    for tick in ax[2].get_xticklabels(): tick.set_rotation(15)
    plt.show()

    plt.title("standard deviation")
    y = cross[cL].std(axis=0)
    plt.bar(y.index,y,label="10 min")
    y = crosH[cL].std(axis=0)
    plt.bar(y.index,y,label="10 min in hour")
    y = hou[cL].std(axis=0)
    plt.bar(y.index,y,label="hour")
    plt.xticks(rotation=15)
    plt.legend()
    plt.show()

    importlib.reload(t_v)
    colL = ['blue','red','green','yellow','purple','red','brown','olive','cyan','#ffaa44','#441188']
    colL = colL + colL
    fig, ax = plt.subplots(1,2)
    for i,c in enumerate(aL):
        t_v.plotBinned(crm,col_y=c,col_bin="hour",ax=ax[0],label=c,color=colL[i],alpha=.05)
    for i,c in enumerate(gL):
        t_v.plotBinned(crm,col_y=c,col_bin="hour",ax=ax[1],label=c,color=colL[i],alpha=.05)
    ax[0].legend(fontsize='x-small')
    ax[1].legend(fontsize='x-small')
    plt.show()

    fig, ax = plt.subplots(1,2)
    for i,c in enumerate(aL):
        t_v.plotBinned(crm,col_y=c,col_bin="hour",ax=ax[0],label=c,color=colL[i],alpha=.05)
    for i,c in enumerate(agL):
        t_v.plotBinned(aggAh,col_y=c,col_bin="hour",ax=ax[1],label=c,color=colL[i],alpha=.05)
    ax[0].set_ylim([-0.05,0.4])
    ax[1].set_ylim([-0.05,0.4])
    ax[0].set_title("10 min")
    ax[1].set_title("hourly")
    ax[0].legend(fontsize='x-small')
    ax[1].legend(fontsize='x-small')
    plt.show()

    fig, ax = plt.subplots(1,2)
    for i,c in enumerate(gL):
        t_v.plotBinned(crm,col_y=c,col_bin="hour",ax=ax[0],label=c,color=colL[i],alpha=.05)
    for i,c in enumerate([1,2]):
        t_v.plotBinned(aggGh,col_y=c,col_bin="hour",ax=ax[1],label=c,color=colL[i],alpha=.05)
    ax[0].set_ylim([0.1,0.9])
    ax[1].set_ylim([0.1,0.9])
    ax[0].set_title("10 min")
    ax[1].set_title("hourly")
    ax[0].legend(fontsize='x-small')
    ax[1].legend(fontsize='x-small')
    plt.show()

if False:
    print('------------------------standard-deviation-per-location--------------------------')
    stdL = []
    for i,g in cross.groupby(idField):
        s = g[cL].std(axis=0)
        stdL.append({idField:i,"s":np.mean(s)})
    stdL = pd.DataFrame(stdL)
    stdL.boxplot(column='s')
    plt.show()

    t_v.plotHist(stdL['s'],nBin=7,lab="std")
    plt.title('distribution of standar deviation per location')
    plt.xlabel('mean standard deviation')
    plt.ylabel('count')
    plt.show()

if False:
    print('--------------------------basic-statistics---------------------')
    cenA = pd.read_csv(baseDir + "raw/"+custD+"/age_census.csv")
    cenA.loc[:,:] = cenA.values/cenA.values.sum()
    cenAG = pd.concat([cenA,cenA],axis=1)
    cenAG.columns = [x+"_gender_1" for x in aL] + [x+"_gender_2" for x in aL]
    cenAG.loc[:,:] = cenAG.values/cenAG.values.sum()
    cenAG = cenAG[cL]
    aggAG = agg[['crm','share']].groupby('crm').agg(sum).T
    aggAG.loc[:,cL] = aggAG.values/aggAG.values.sum()
    aggAA = aggA[['age','share']].groupby('age').agg(sum).T
    aggGG = aggG[['gender','share']].groupby('gender').agg(sum).T
    aggAA.loc[:,:] = aggAA.values/aggAA.values.sum()
    aggGG.loc[:,:] = aggGG.values/aggGG.values.sum()
    print(.5/aggGG)
    print(cenA.values/aggAA.values)
    corrF = aggAG.copy()
    corrF.loc[:,cL] = cenAG.values/aggAG.values
    plt.bar(corrF.columns,corrF.values[0]-1.)
    plt.xticks(rotation=15)
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    t_v.plotPie(cenAG,cL,isValue=False,ax=ax[0])
    ax[0].set_title("age split census")
    t_v.plotPie(aggAG,cL,isValue=False,ax=ax[1])
    ax[1].set_title("age split hour")
    plt.show()
    
    fig, ax = plt.subplots(1,3)
    ax[0].set_title("age split short")
    t_v.plotPie(crm,aL,isValue=False,ax=ax[0])
    ax[1].set_title("age split agg")
    t_v.plotPie(aggAA,list(aggAA.columns),isValue=False,ax=ax[1])
    ax[2].set_title("age split census")
    t_v.plotPie(cenA,list(cenA.columns),isValue=False,ax=ax[2])
    plt.show()

    fig, ax = plt.subplots(1,2)
    ax[0].set_title("gender split short")
    t_v.plotPie(crm,gL,isValue=False,ax=ax[0])
    ax[1].set_title("gender split agg")
    t_v.plotPie(aggGG,list(aggGG.columns),isValue=False,ax=ax[1])
    plt.show()

    crm[gL].plot(kind='density')
    plt.show()


    import joypy
    crosm = cross.melt(value_vars=cL)
    fig, axes = joypy.joyplot(crosm,column='value',by='variable',ylim='own',figsize=(12,6),alpha=.5)#,colormap=cm.summer_r)
    plt.legend()
    plt.title('age_gender densities')
    plt.show()
    
    cross[cL].plot(kind='density')
    plt.show()
    
    crm[gL].plot(kind='density')
    plt.show()

    crm[aL].plot(kind='density')
    plt.show()

    fig, ax = plt.subplots(1,2)
    ax[0].set_title("gender values histogram")
    t_v.plotHistogram(crm[gL].values.ravel(),ax=ax[0])
    ax[1].set_title("age values histogram")
    t_v.plotHistogram(crm[aL].values.ravel(),ax=ax[1])
    plt.show()
    
    nbins = 30
    crmN = crm[gL].dropna()
    x, y = crmN[gL[0]].values,crmN[gL[1]].values
    # x = np.random.normal(size=500)
    # y = x * 3 + np.random.normal(size=500)
    k = sp.stats.kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)
    plt.show()

    sns.jointplot(x=crm[gL[0]], y=crm[gL[1]], kind='kde')
    plt.show()

if False:
    print('----------------------correlation-compared-to-aggregate---------------')
    corA = agg.pivot_table(index=idField,columns="age",values="share",aggfunc=np.sum).corr()
    crmA = crm[aL].corr()

    fig, ax = plt.subplots(1,2)
    cmap = plt.get_cmap("RdBu")
    ax[0].set_title("age correlation - aggregate")
    ax[0] = sns.heatmap(corA,cmap=cmap,linewidths=.0,square=True,ax=ax[0])
    ax[1].set_title("age correlation - short time")
    ax[1] = sns.heatmap(crmA,cmap=cmap,linewidths=.0,square=True,ax=ax[1])
    plt.show()

    fig, ax = plt.subplots(1,2)
    cmap = plt.get_cmap("RdBu")
    ax[0].set_title("age correlation - aggregate")
    ax[0] = sns.heatmap(aggX.corr(),cmap=cmap,linewidths=.0,square=True,ax=ax[0])
    ax[1].set_title("age correlation - short time")
    ax[1] = sns.heatmap(crossX.corr(),cmap=cmap,linewidths=.0,square=True,ax=ax[1])
    for tick in ax[0].get_xticklabels(): tick.set_rotation(15)
    for tick in ax[1].get_xticklabels(): tick.set_rotation(15)
    plt.show()


if False:
    print('--------------------multivariate-t-distribution----------------')
    idL = np.unique(agg[idField])
    testL = []
    for i in idL:
        set1 = hou[idField] == i
        set2 = cross[idField] == i
        cV = {}
        for c in cL:
            cV[c] = sp.stats.ttest_ind(hou.loc[set1,c],cross.loc[set2,c])[1]
        d = {idField:i}
        d.update(cV)
        testL.append(d)
    testL = pd.DataFrame(testL)
    X = testL[cL].dropna().values
    y = np.log(X.ravel())
    ax = t_v.plotHistogram(abs(y),isLog=True)
    ax.set_title("t-test exponents distribution")
    plt.show()
    
    idL = np.unique(crossX.index)
    corL = []
    for i in idL:
        x1 = aggX.loc[i].values
        x2 = crossX.loc[i].values
        r = t_s.calcMetrics(x1/x1.sum(),x2/x2.sum())
        r.update({idField:i})
        corL.append(r)
    corL = pd.DataFrame(corL)

    t_v.kpiDis(corL,col_cor="cor",col_dif="rel_err",col_sum="cor")
    plt.show()

    # aggA1 = agg.pivot_table(index=idField,columns="age",values="share",aggfunc=np.mean)
    # norm = 1./aggA1.sum(axis=1)
    # aggA1 = np.multiply(aggA1.values,norm.values[:,np.newaxis])
    # crmA1 = crm[ [idField] + aL].groupby(idField).agg(np.mean)
    # C = signal.correlate2d(aggA1,crmA1.T,mode="full")

    fig, ax = plt.subplots(1,2)
    xcorr = np.corrcoef(aggX,crossX)
    ax[0].set_title("cross correlation")
    ax[0].imshow(xcorr)
    ax[1].set_title("value distribution")
    ax[1].hist(xcorr.ravel(),bins=20)
    plt.show()
    
    X = crm[aL[:-2]].values
    importlib.reload(s_s)
    y1, S1 = s_s.fromMultivariate(crm[aL[:-2]].values,mode='normal')
    y2, S2 = s_s.fromMultivariate(aggA[aggA.columns[:-1]].values,mode='normal')
    ttest = []
    for i in aL:
        ttest.append(sp.stats.ttest_ind(houA[i],crm[i])[1])

if False:
    print('---------------clustering--------------------')
    from scipy.cluster.hierarchy import cophenet
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import fcluster
    cluL = pd.DataFrame(index=aggX.index)
    clustD = 0.4
    clu = fcluster(linkage(aggX,'ward'), clustD, criterion='distance')
    print(set(clu))
    cluL.loc[:,"hour"] = clu
    clu = fcluster(linkage(crossX,'ward'), clustD, criterion='distance')
    print(set(clu))
    cluL.loc[:,"10min"] = clu
    t_v.plotSankey(cluL,col_1="hour",col_2="10min",title="reclustering: hour->10min")

if False:
    print('---------------clustering--------------------')
    from sklearn.cluster import KMeans
    X = aggX[cL].values
    clusterer = KMeans(copy_x=True,init='k-means++',max_iter=300,n_clusters=6,n_init=10,n_jobs=1,precompute_distances='auto',random_state=None,tol=0.0001,verbose=2)
    clusterer.fit(X)
    centroids = clusterer.cluster_centers_
    corr = pd.DataFrame(np.corrcoef(np.array(centroids)))
    ax = sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.show()
    cluL = pd.DataFrame(index=aggX.index)
    cluL.loc[:,"hour"] = clusterer.predict(aggX[cL].values)
    cluL.loc[:,"10min"] = clusterer.predict(crossX[cL].values)
    t_v.plotSankey(cluL,col_1="hour",col_2="10min",title="reclustering: hour->10min")

if False:
    print('-------------location-relations---------------')
    import networkx as nx
    cor = aggX
    G = nx.Graph()
    for i in cluL.index: G.add_node(i,size=1)
    for i in set(cluL['hour']): G.add_node(i,size=4)
    for i,g in cluL.iterrows():
        G.add_edge(i,g['hour'],color='b')
        G.add_edge(i,g['10min'],color='r')

    nodS = aggX.sum(axis=1)*30
    labels = {}
    for i in cluL.index: labels[i] = i
    colors = [G[u][v]['color'] for u,v in G.edges()]
    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G,pos,width=1,alpha=0.3,edge_color=colors)
    nx.draw_networkx_nodes(G,pos,node_color='g',node_size=nodS,alpha=0.3)
    #nx.draw_networkx_labels(G,pos,labels,font_size=18)
    plt.axis('off')
    plt.show()
    

        
if False:
    print('-------------------------boxplot---------------------')

    t_v.boxplotOverlap(hou,cross,cL,lab1='hour',lab2='10 min')
    t_v.boxplotOverlap(hou,crosH,cL,lab1='hour',lab2='10 min')
    
    t_v.boxplotOverlap(houA,crm[aL],aL,lab1='hour',lab2='10 min')

    t_v.boxplotOverlap(aggAh,crm[aL],aL,lab1='hour',lab2='10 min')
    
    
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("hourly")
    #houA.boxplot(ax=ax[0])
    aggAh.boxplot(ax=ax[0])
    ax[1].set_title("10 min")
    crm[aL].boxplot(ax=ax[1])
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("hourly")
    houA.boxplot(ax=ax[0])
    ax[1].set_title("10 min")
    crm[aL].boxplot(ax=ax[1])
    plt.show()

if False:
    print('------------------correction-factors-------------------')
    
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
