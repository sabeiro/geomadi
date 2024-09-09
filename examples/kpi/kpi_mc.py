#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph
import geomadi.train_shapeLib as shl
import seaborn as sns
import geomadi.train_execute as t_e

def plog(text):
    print(text)

custD = "tank"
custD = "mc"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
mapL = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")

gist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip",index_col=0)
hLg = gist.columns[[bool(re.search('-??T',x)) for x in gist.columns]]

dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount.csv.gz",compression="gzip")
dirc = dirc.replace(float('nan'),0.)
dirM = pd.melt(dirc,id_vars="id_poi")
dirM.loc[:,"day"] = dirM['variable'].apply(lambda x: x[:11])
dirc = dirM.pivot_table(index="id_poi",columns="day",values="value",aggfunc=np.sum)
hLd = dirc.columns[[bool(re.search('-??T',x)) for x in dirc.columns]]

if False:
    plog('---------------display-score----------------------')
    act = pd.read_csv(baseDir + "raw/"+custD+"/act_vist_play.csv.gz",compression="gzip")
    scor = t_e.scorKpi(act,step="paly",idField=idField)
    shl.kpiDis(scor,tLab="visits vs footfall - daily",col_cor="r_play",col_dif="d_play",col_sum="s_play")
    act = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip")
    scor = t_e.scorKpi(act,step="cilac",idField=idField)
    shl.kpiDis(scor,tLab="cilac mapping weighted",col_cor="r_cilac",col_dif="d_cilac",col_sum="s_cilac",saveF=baseDir + "www/f_food/kpi_cilac_weight.png")

    act = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_ref.csv.gz",compression="gzip")
    scor = t_e.scorKpi(act,step="cilac",idField=idField)
    shl.kpiDis(scor,tLab="cilac mapping",col_cor="r_cilac",col_dif="d_cilac",col_sum="s_cilac",saveF=baseDir + "www/f_food/kpi_cilac_weight.png")

    shl.kpiDis(scor,tLab="visits vs footfall - daily",col_cor="r_traj",col_dif="d_traj",col_sum="s_traj")
    shl.kpiDis(scor,tLab="visits vs activities - daily",col_cor="r_act",col_dif="d_act",col_sum="s_act")
    shl.kpiDis(scor,tLab="footfall vs activities - daily",col_cor="r",col_dif="d",col_sum="s")

poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
poi = pd.merge(poi,scor,on="id_poi",how="left")
poi.loc[poi['type'] != poi['type'],"type"] = ''
poi.loc[:,"drive"] = poi['type'].apply(lambda x: bool(re.search("without",x)))

if False:
    fig, ax = plt.subplots(2,2)
    poi[poi['r_traj']==poi['r_traj']].boxplot(column="r_traj",by="type",ax=ax[0][0])
    poi[poi['r_traj']==poi['r_traj']].boxplot(column="r_traj",by="drive",ax=ax[0][1])
    poi[poi['r_act']==poi['r_act']].boxplot(column="r_act",by="drive",ax=ax[1][0])
    poi[poi['r']==poi['r']].boxplot(column="r",by="drive",ax=ax[1][1])
    for i,a in enumerate(ax.flat):
        for tick in a.get_xticklabels():
            tick.set_rotation(10)
    plt.show()

if False:
    plog('---------------------scoring-evolution-heatmap--------------------')
    scorM = pd.read_csv(baseDir + "raw/"+custD+"/scorMap.csv")
    scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnType.csv")
    scorL = pd.merge(scorM,scorL,on="id_poi",how="outer")
    tL = [x for x in scorL.columns if bool(re.search("r_",x))]
    scorL.sort_values(tL[-1],inplace=True)
    sns.set(font_scale=1.2)
    def clampF(x):
        return pd.Series({"perf":len(x[x>0.6])/len(x)})
    scorV = scorL[tL].apply(clampF)
    labL = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    cmap = plt.get_cmap("PiYG") #BrBG
    scorL.index = scorL[idField]
    yL = scorL[scorL[tL[-1]]>0.6].index[0]

    fig, ax = plt.subplots(1,1)#,sharex=True,sharey=True)
    ax.set_title("learn")
    ax = sns.heatmap(scorL[tL],cmap=cmap,linewidths=.0,cbar=None,ax=ax)
    ax.hlines(y=yL,xmin=tL[0],xmax=tL[-1],color="r",linestyle="dashed")
    ax.set_xticklabels(labL)
    for tick in ax.get_xticklabels():
        tick.set_rotation(10)
    plt.show()


if False:
    scorP = pd.read_csv(baseDir + "raw/"+custD+"/scor_playDay.csv")
    scorL = pd.read_csv(baseDir + "raw/"+custD+"/scor_learnDay.csv")
    tL = [x for x in scorL.columns if bool(re.search("r_",x))]
    tP = [x for x in scorP.columns if bool(re.search("r_",x))]    
    scorL.sort_values(tL[-1],inplace=True)
    scorP.sort_values(tP[-1],inplace=True)
    sns.set(font_scale=1.2)
    
    def clampF(x):
        return pd.Series({"perf":len(x[x>0.6])/len(x)})
    scorV = scorL[tL].apply(clampF)
    labL = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    scorV = scorP[tP].apply(clampF)
    labP = ["%s-%.0f" % (x,y*100.) for (x,y) in zip(scorV.columns.values,scorV.values[0])]
    cmap = plt.get_cmap("PiYG") #BrBG
    scorL.index = scorL[idField]
    scorP.index = scorP[idField]
    yL = scorL[scorL[tL[-1]]>0.6].index[0]
    yP = scorP[scorP[tP[-1]]>0.6].index[0]

    fig, ax = plt.subplots(1,2)#,sharex=True,sharey=True)
    ax[0].set_title("learn")
    ax[0] = sns.heatmap(scorL[tL],cmap=cmap,linewidths=.0,cbar=None,ax=ax[0])
    ax[0].hlines(y=yL,xmin=tL[0],xmax=tL[-1],color="r",linestyle="dashed")
    ax[0].set_xticklabels(labL)
    ax[1].set_title("play")
    ax[1] = sns.heatmap(scorP[tP],cmap=cmap,linewidths=.0,cbar=None,ax=ax[1])
    ax[1].hlines(y=yP,xmin=tP[0],xmax=tP[-1],color="r",linestyle="dashed")
    ax[1].set_xticklabels(labP)
    for i in range(2):
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(10)
    plt.show()



print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
