import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.series_stat as s_s

ser = pd.read_csv(baseDir + "../erp/erp_motion/hvb_2019.csv",dtype={"amount":float})
ser.loc[:,"time"] = ser['date'].apply(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y"))
ser.loc[:,"rev"] = np.cumsum(ser['amount'])
ser.sort_values('time',inplace=True)
ger = ser.groupby("group").agg(np.sum).reset_index()

plt.plot(ser['time'],ser['rev'])
plt.show()




print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
