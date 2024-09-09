#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)

lte = pd.read_csv(baseDir + "log/tracebox/table_call_lte.csv.tar.gz",compression="gzip",sep="\t")
lte_c = pd.read_csv(baseDir + "log/tracebox/lte_columns.csv")
lte.columns = lte_c['column_name']

lte = lte[lte['START_TIME'] != '\\N']
lte = lte[lte['END_TIME'] != '\\N']
lte = lte[lte["START_TIME_MS"] == lte['START_TIME_MS']]
lte = lte[lte["END_TIME_MS"] == lte['END_TIME_MS']]

lte.loc[:,"START_TIME"] = lte['START_TIME'].astype(str)
lte.loc[:,"START_TIME_MS"] = lte['START_TIME_MS'].astype(int)
lte.loc[:,"END_TIME"] = lte['END_TIME'].astype(str)
lte.loc[:,"END_TIME_MS"] = lte['END_TIME_MS'].astype(int)
lte.loc[:,"START_CELL_ID"] = lte['START_CELL_ID'].astype(int)
lte.loc[:,"END_CELL_ID"] = lte['END_CELL_ID'].astype(int)

lte = lte[['START_TIME','START_TIME_MS','END_TIME','END_TIME_MS','M_TMSI', 'S_TMSI', 'TAC', 'START_CELL_ID',  'END_CELL_ID','POS_FIRST_LOC', 'POS_FIRST_LON', 'POS_FIRST_LAT','POS_LAST_LON', 'POS_LAST_LAT']]

lte.loc[:,'ts1'] = lte[['START_TIME',"START_TIME_MS"]].apply(lambda x: datetime.datetime.strptime(x[0],"%Y-%m-%d %H:%M:%S").timestamp()*1000 + float(x[1]),axis=1)
lte.loc[:,'ts2'] = lte[['END_TIME',"END_TIME_MS"]].apply(lambda x: datetime.datetime.strptime(x[0],"%Y-%m-%d %H:%M:%S").timestamp()*1000 + float(x[1]),axis=1)

lte = lte[['S_TMSI','TAC','ts1','POS_FIRST_LON','POS_FIRST_LAT','ts2','POS_LAST_LON','POS_LAST_LAT']]
lte.columns = ['s_tmsi','tac','ts1','x1','y1','ts2','x2','y2']
lte.to_csv(baseDir + "log/tracebox/tracebox_move.csv.gz",index=False,compression="gzip")

ho = pd.read_csv(baseDir + "log/tracebox/table_call_ho.csv.tar.gz",compression="gzip",sep="\t")
ho_c = pd.read_csv(baseDir + "log/tracebox/ho_columns.csv")
ho.columns = ho_c['column_name']

ho = ho[ho["DATE_TIME_MS"] == ho['DATE_TIME_MS']]
ho.loc[:,"DATE_TIME"] = ho['DATE_TIME'].astype(str)
ho.loc[:,"DATE_TIME_MS"] = ho['DATE_TIME_MS'].astype(int)
ho.loc[:,"CELL_ID"] = ho['CELL_ID'].astype(int)
ho.loc[:,"MOVING"] = ho['MOVING'].astype(int)

#ho = ho[['DATE_TIME','DATE_TIME_MS','LON','LAT','M_TMSI','CELL_ID']]
ho.loc[:,'ts'] = ho[['DATE_TIME',"DATE_TIME_MS"]].apply(lambda x: datetime.datetime.strptime(x[0],"%Y-%m-%d %H:%M:%S").timestamp()*1000 + float(x[1]),axis=1)
ho = ho[['M_TMSI','ts','LON','LAT','CELL_ID','TARGET_CELL']]
ho.columns = ['m_tmsi','ts','x','y','cilac','cilac2']
ho.to_csv(baseDir + "log/tracebox/tracebox_pos.csv.gz",index=False,compression="gzip")

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')



