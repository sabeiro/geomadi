#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pymongo
import geomadi.kernel_lib as k_l
from scipy import signal as sg
import seaborn as sns
import matplotlib

#percentage over 150km, 2/3 is too much

ops = {"isClamp":False,"150km":False,"weighting":True,"low_counts":False}

nodeL = pd.read_csv(baseDir + "raw/nissan/junct_loc.csv")
zipM = pd.read_csv(baseDir + "raw/nissan/enter2exit_dist.csv")

via4 = pd.read_csv(baseDir + "raw/nissan/enter2exit_v4.csv",index_col=0)
via5 = pd.read_csv(baseDir + "raw/nissan/enter2exit_v5.csv",index_col=0)
via6 = pd.read_csv(baseDir + "raw/nissan/enter2exit_v6.csv",index_col=0)
print(via4.sum().sum())
print(via5.sum().sum())
print(via6.sum().sum())

fL = ["raw/nissan/enter2exit_short_sym.csv","raw/nissan/enter2exit_v1.csv","raw/nissan/enter2exit_v2.csv","raw/nissan/enter2exit_v3.csv","raw/nissan/enter2exit_v4.csv"]
stepL = ["odm","noisy","via_nodes","AB_1d","AB_+d"]
fL = ["raw/nissan/enter2exit_v6.csv"]
stepL = ["thueringen"]
kpiD = []
Tl = []
Pl = []
Xl = []
for i,f in enumerate(fL):
    viaP = pd.read_csv(baseDir + f,index_col=0)
    # viaP.columns = [int(x) for x in viaP.columns]
    # viaP.index = [int(x) for x in viaP.index]
    # if i == 0:
    #     viaP.loc[33,:] = 0.
    viaP = viaP[viaP.columns[viaP.columns.isin(viaP.index)]]
    viaP = viaP.loc[viaP.index[viaP.index.isin(viaP.columns)]]
    if ops['isClamp']: #clamp odd
        viaP.index   = [(x-(x%2) ) for x in viaP.index]
        viaP.columns = [(x-(x%2) ) for x in viaP.columns]
    viaT = pd.melt(viaP.reset_index(),value_vars=viaP.columns,id_vars="enter")
    viaT.columns = ["enter","exit","count"]
    # viaT.loc[:,"enter"] = viaT['enter'].astype(int)
    # viaT.loc[:,"exit"] = viaT['exit'].astype(int)
    if ops['150km']: #996 to junctions over 150km
        nodeD = pd.read_csv(baseDir + "raw/nissan/en2ex_dist_int.csv")
        nodeD = nodeD[['dist','j_en','j_ex']].groupby(["j_en","j_ex"]).agg(np.mean).reset_index()
        nodeD1 = nodeD[['j_ex','j_en','dist']]
        nodeD1.columns = ['j_en','j_ex','dist']
        nodeD = pd.concat([nodeD,nodeD1],axis=0)
        nodeD = nodeD.groupby(["j_en","j_ex"]).agg(np.mean).reset_index()
        viaT = pd.merge(viaT,nodeD,left_on=["enter","exit"],right_on=["j_en","j_ex"],how="left")
        viaT.loc[viaT['dist'] > 150,"exit"] = 996

    viaP = viaT.pivot_table(index="enter",columns="exit",values="count",aggfunc=np.sum)
    viaP = viaP[viaP.columns[viaP.columns.isin(viaP.index)]]
    viaP = viaP.loc[viaP.index[viaP.index.isin(viaP.columns)]]
    # viaS = viaT[viaT['enter']<60]
    # viaS = viaS[viaS['exit'] <60]
    # # viaS = viaS[viaS['enter']!=45]
    # # viaS = viaS[viaS['exit'] !=45]
    enS = viaT[["enter","count"]].groupby("enter").agg(sum).reset_index()
    exS = viaT[["exit","count"]].groupby("exit").agg(sum).reset_index()
    viaD = pd.merge(enS,exS,left_on="enter",right_on="exit",how="outer",suffixes=["_en","_ex"])
    del viaD['exit']
    viaD.columns = ["junction","enter","exit"]
    viaD.loc[:,"diff"] = (viaD['enter']-viaD['exit'])/(viaD['enter']+viaD['exit'])*2.
    viaD.loc[:,"max"]  = viaD[['enter','exit']].apply(lambda x: max(x),axis=1)
    if ops['weighting']:
        normF = 1./np.percentile(viaD['max'],80)
        normF = 1./viaD['max'].max()
        def nDecay(x,normF):
            interc = 0.
            return x*(1.-interc)*normF + interc
        
        viaD.loc[:,"diff"] = viaD['diff']*[nDecay(x,normF) for x in viaD['max']]
    if i == 0:
        difD = pd.DataFrame({"junction":viaD['junction'],"dif_" + str(i):viaD['diff'],"max_"+str(i):viaD['max']})
    else:
        difD = pd.merge(difD,pd.DataFrame({"junction":viaD['junction'],"dif_" + str(i):viaD['diff'],"max_"+str(i):viaD['max']}),on="junction",how="outer")

    X = viaP.values
    u1, u2 = np.triu_indices(X.shape[0],k=1)
    idL = pd.DataFrame([{"pair": "%s-%s" % (viaP.index[i],viaP.columns[j]),"diff":viaP.iloc[i,j]-viaP.iloc[j,i],"r":i,"c":j,"mean":np.mean([viaP.iloc[i,j],viaP.iloc[j,i]])} for i,j in zip(u1,u2)])
    idL.loc[:,"abs"] = idL['diff'].apply(lambda x:abs(x))
    idL.sort_values('abs',inplace=True)
    idL = pd.merge(idL,zipM,on="pair",how="left")
    if ops['low_counts']:
        selT = (X<100) + (X.T<100)
        X[selT] = 0
    U = np.triu(X,1)
    L = np.tril(X,-1).T
    Ndif = sum([abs(x) for x in (U.ravel()-L.ravel()) ])
    T = abs((U-L)/(U+L))*2.
    if ops['weighting']:
        normF = 1./np.percentile(X.ravel(),80)
        x = X.ravel()
        normF = 1./max(x[~np.isnan(x)])
        def nDecay(x,normF):
            interc = 0.3
            return x*(1.-interc)*normF + interc

        nDecay = np.vectorize(nDecay)
        W = nDecay(X,normF)
        T = T*W
    P = abs(T).copy()
    P[abs(T) <  .1] = .0
    P[abs(T) >= .1] = .1
    P[abs(T) >= .2] = .2
    P[abs(T) >= .3] = .3
    T = np.floor(T*100)
    asyM = pd.DataFrame(T,columns=viaP.columns,index=viaP.index)
    asyM.to_csv(baseDir + "raw/nissan/asym_v" + str(i) + ".csv")
    Xl.append(viaP)
    Tl.append(asyM)
    Pl.append(pd.DataFrame(P,columns=viaP.columns,index=viaP.index))
    N  = X.sum()
    N1 = len(P[P == 0.0])
    N2 = len(P[P == 0.1])
    N3 = len(P[P >  0.1])
    kpiD.append({"asym":Ndif/N,"diag":X[np.diag_indices_from(X)].sum()/N,"10%":N1/(N1+N2+N3),"20%":N2/(N1+N2+N3),"30%":N3/(N1+N2+N3)})

kpiD = pd.DataFrame(kpiD,index=stepL)
difD.sort_values("junction",inplace=True)
difD.to_csv(baseDir + "raw/nissan/diff_history.csv",index=False)

print(kpiD)

if False:
    plog('--------------------asymmetry heatmap-------------------')
    T1 = Tl[0]
    T2 = Tl[-1]
    cmap = matplotlib.colors.ListedColormap(['#00F00090','#F0F00090','#FF880090','#BB000090'],name='from_list',N=None)
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(1,1)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.set(font_scale=0.6)
#    ax[0] = sns.heatmap(Pl[0],cmap=cmap,linewidths=.0,annot=Tl[0],square=True,ax=ax[0],cbar=None)
#    ax[1] = sns.heatmap(Pl[0],cmap=cmap,linewidths=.0,annot=Tl[0],square=True,ax=ax[1],cbar=cbar_ax)
    ax = sns.heatmap(Pl[-1],cmap=cmap,linewidths=.0,square=True,ax=ax,cbar_ax=cbar_ax)
    ax.set_title(stepL[0])
    #    ax[1].set_title("AB+d")
    ax.set_ylabel('FROM')
    ax.set_xlabel('TO')
    colorbar = ax.collections[0].colorbar
    # for a in ax:
    #     a.set_ylabel('FROM')
    #     a.set_xlabel('TO')
    # _, labels = plt.yticks()
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.setp(labels, rotation=0)
    plt.show()

if False:
    plog('--------weighting plot--------')
    x = np.linspace(0,3000.,100)
    normF = 1./max(x)
    def nDecay(x,normF):
        interc = 0.
        return x*(1.-interc)*normF + interc
    nDecay = np.vectorize(nDecay)
    y = nDecay(x,normF)
    def nDecay(x,normF):
        interc = 0.3
        return x*(1.-interc)*normF + interc
    nDecay = np.vectorize(nDecay)
    y1 = nDecay(x,normF)
    plt.title("weigthing function definition")
    plt.plot(x,y,label="intercept = 0")
    plt.plot(x,y1,label="intercept = 0.3")
    plt.ylabel("weight")
    plt.xlabel("counts")
    plt.legend()
    plt.show()
    
if False:
    plog('-----------count heatmap--------------')
    T1 = Xl[0].replace(float('nan'),0).astype(int)
    T2 = Xl[-1].replace(float('nan'),0).astype(int)
   
    cmap = plt.get_cmap("PiYG") #BrBG
    cmap = cmap.reversed()
    fig, ax = plt.subplots(1,2)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    pL = [0,1,2,4]
    sns.set(font_scale=0.8)
    ax[0] = sns.heatmap(T1,cmap=cmap,linewidths=.0,annot=None,square=True,ax=ax[0],cbar=None,vmax=2000.)
    ax[1] = sns.heatmap(T2,cmap=cmap,linewidths=.0,annot=None,square=True,ax=ax[1],cbar_ax=cbar_ax,vmax=2000.)
    ax[0].set_title("zip2zip")
    ax[1].set_title("AB+d")
    colorbar = ax[1].collections[0].colorbar
    for a in ax:
        a.set_ylabel('FROM')
        a.set_xlabel('TO')
    _, labels = plt.yticks()
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.setp(labels, rotation=0)
    plt.show()

if False:
    plog('----------------column history----------------')
    cmap = matplotlib.colors.ListedColormap(['#00F00090','#F0F00090','#FF000090','#BB000090'],name='from_list',N=None)
    dL = [x for x in difD.columns if bool(re.search("dif_",x))]
    for i,g in enumerate(dL):
        d = difD[g].abs()
        l = difD[g].abs()
        y = difD[g].replace(float("nan"),0)
        l[d <  .1] = .0
        l[d >= .1] = .1
        l[d >= .2] = .2
        l[d >= .3] = .3
        difD.loc[:,"step_"+ str(i)] = l
        difD.loc[:,"lab_" + str(i)] = y.apply(lambda x: int((x*100.)))
    sL = [x for x in difD.columns if bool(re.search("step_",x))]
    lL = [x for x in difD.columns if bool(re.search("lab_",x))]
    stepD = difD[sL]
    labD = difD[dL]*100.
    stepD.index = difD['junction']
    stepD.columns = [stepL[x] for x in range(stepD.shape[1])]
    sns.set(font_scale=0.8)
    
    fig, axn = plt.subplots(1,6)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for i, ax in enumerate(axn.flat):
        j = [int(i*stepD.shape[0]/6),int((i+1)*stepD.shape[0]/6)]
        sns.heatmap(stepD.iloc[j[0]:j[1]],cmap=cmap,linewidths=.0
                    ,ax=ax,annot=difD[lL].iloc[j[0]:j[1]],square=True
                    ,cbar=i==5,cbar_ax=None if i==5 else cbar_ax)
        for tick in ax.get_xticklabels(): tick.set_rotation(45)
        for tick in ax.get_yticklabels(): tick.set_rotation(45)
    cbar = axn[5].collections[0].colorbar
    for j, lab in enumerate(['10%','20%','30%','40%']):
        cbar.ax.text(.5,(2*j+1)/8.0,lab,ha='center',va='center')
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('asymmetry value', rotation=270)
    plt.show()

if False:
    plog('-----------------------quad heatmap-----------------------')
    cmap = matplotlib.colors.ListedColormap(['#00F00090','#F0F00090','#FF000090','#BB000090'],name='from_list',N=None)
    pL = [0,1,2,4]
    fig, ax = plt.subplots(2,2)#,sharex=True,sharey=True)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    sns.set(font_scale=0.8)
    ax[0][0] = sns.heatmap(Pl[pL[0]],cmap=cmap,linewidths=.0,annot=Tl[pL[0]],square=True,ax=ax[0][0],cbar=None)
    ax[0][1] = sns.heatmap(Pl[pL[1]],cmap=cmap,linewidths=.0,annot=Tl[pL[1]],square=True,ax=ax[0][1],cbar_ax=cbar_ax)
    ax[1][0] = sns.heatmap(Pl[pL[2]],cmap=cmap,linewidths=.0,annot=Tl[pL[2]],square=True,ax=ax[1][0],cbar=None)
    ax[1][1] = sns.heatmap(Pl[pL[3]],cmap=cmap,linewidths=.0,annot=Tl[pL[3]],square=True,ax=ax[1][1],cbar=None)
    ax[0][0].set_title(stepL[pL[0]])
    ax[0][1].set_title(stepL[pL[1]])
    ax[1][0].set_title(stepL[pL[2]])
    ax[1][1].set_title(stepL[pL[3]])
    colorbar = ax[0][1].collections[0].colorbar
    colorbar.set_ticks([0.1, 0.2, 0.3])
    colorbar.set_ticklabels(["10%","20%","30%",">30%"])
    for a in ax:
        for b in a:
            b.set_ylabel('FROM')
            b.set_xlabel('TO')
    _, labels = plt.yticks()
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.setp(labels, rotation=0)
    plt.show()

if False:
    plog('-----------------correlation matrix-------------------')
    import importlib
    import cv2
    import scipy.ndimage as ndimage
    cmap = plt.get_cmap("PiYG") #BrBG
#    cmap = plt.get_cmap("hsv") #BrBG
    importlib.reload(k_l)
    sigma = 1
    X[X == 0] = 1
    Xl = np.log(X)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(ndimage.gaussian_filter(Xl,sigma=(sigma),order=0),cmap=cmap)
    ax[0].grid(visible=False)
    ax[0].set_title("raw matrix: 100%")
    X1 = Xl.copy()
    X1[np.diag_indices_from(X1)] = 0
    ax[1].imshow(ndimage.gaussian_filter(X1,sigma=(sigma),order=0),cmap=cmap)
    ax[1].grid(visible=False)
    N1 = (np.nansum(X[np.diag_indices_from(X)])/np.nansum(X))
    ax[1].set_title("no diagonal: %.2f%%" % (1.- N1) )
    Xk = k_l.gkern(nlen=5,nsig=3)
    X2 = cv2.filter2D(src=X1,kernel=Xk,ddepth=-1)
    X2 = X1.copy()
    x2 = X2.ravel()
    x2 = x2[~np.isnan(x2)]
    X2[X2 > X2.max()*.5] = np.mean(X2[X2>0])
    x2 = X2.ravel()
    x2 = x2[~np.isnan(x2)]
    ax[2].imshow(ndimage.gaussian_filter(X2,sigma=(sigma),order=0),cmap=cmap)
    N2 = (np.nansum(X2[X2 > X2.max()*.98])/np.nansum(X))
    ax[2].set_title("no outlier: %.2f%%" % (1.- N1 - N2) )
    ax[2].grid(visible=False)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    corM = ndimage.gaussian_filter(viaP.corr(),sigma=(1),order=0)
    im = ax.imshow(corM,aspect='auto',cmap=cmap)
    ax.set_xticklabels(viaP.index)
    ax.set_yticklabels(viaP.columns)
    plt.grid(False)
    plt.show()

if False:
    x = idL[idL['angle_dif'] == idL['angle_dif']]['angle_dif']
    y = idL[idL['angle_dif'] == idL['angle_dif']]['mean']
    print(sp.stats.pearsonr(x,y))
    plt.plot(x,y)
    plt.show()

if False:
    x = np.log10(viaT['dist'][setL])
    y = np.log10(viaT['count'][setL])
    for i in [float('nan'),float('inf'),-float('inf')]:
        y = y.replace(i,0)
    fig, ax = plt.subplots(1)
    ax.scatter(viaT['dist'],viaT['count'])
    ax.set_title("distance decay correlation %.2f" % (sp.pearsonr(x,y)[0]) )
    plt.xlabel("distance")
    plt.ylabel("counts")
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()
    setL = viaT['dist'] == viaT['dist']




print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
