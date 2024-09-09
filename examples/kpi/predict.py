#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
#import modin.pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import etl.etl_mapping as e_m
import geomadi.series_lib as s_l
import geomadi.train_execute as t_e
import geomadi.train_lib as tlib

def plog(text):
    print(text)

custD = "tank"
custD = "mc"
idField = "id_poi"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
scorM = pd.read_csv(baseDir + "raw/"+custD+"/scorMap.csv")
    
if False:
    plog('------------------predict-performance---------------------')
    importlib.reload(tlib)
    y, _ = t_f.binOutlier(scorP['r_reg'],nBin=7,threshold=0.1)
    p_M = shl.shapeLib(X).calcPCA()[:,:5]
    tMod = tlib.trainMod(p_M,y)
    #tMod.tune(paramF=baseDir+"train/tank_kpi.json",tuneF=baseDir+"train/tank_tune_poi.json")
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/tank_kpi.json",test_size=.2)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/tank_kpi.json",test_size=.3)
    mod, trainR = tMod.loopMod(paramF=baseDir+"train/tank_kpi.json",test_size=.4)
    tMod.save(mod,baseDir + "train/tank_kpi.pkl")
    if False:
        tMod.plotRoc()
        cm = tMod.plotConfMat()

