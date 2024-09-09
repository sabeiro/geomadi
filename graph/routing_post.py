#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)

zipN = pd.read_csv(baseDir + "log/nissan/zip_node.csv")
zipL = [{int(x['nearest_graph_node']):x['PLZ']} for i,x in zipN.iterrows()]
convT = pd.read_csv(baseDir + "log/nissan/node2node_motor.csv.tar.gz",compression="gzip")
convT.loc[:,"origin_id"] = convT['origin_id'].apply(lambda x: int(x))
c = convT.iloc[0]

convT = pd.read_csv(baseDir + "log/nissan/zip2zip_motor.csv.tar.gz",compression="gzip")
convM = pd.melt(convT,id_vars="PLZ")
convM = convM[~np.isnan(convM['value'])]
convM = convM[convM['value'] > 0.]
convM.loc[:,"variable"] = convM['variable'].apply(lambda x: int(x))
convM.to_csv(baseDir + "log/nissan/zip2zip_melt.csv",index=False)


zipN = zipN.groupby("nearest_graph_node").head(1)
convT.index = pd.merge(convT,zipN,left_on="origin_id",right_on="nearest_graph_node",how="left")['PLZ']
del convT['origin_id']
conTmp = pd.DataFrame({"node":[int(x) for x in convT.columns]})
conTmp.loc[:,"ciccia"] = 1
zipN.loc[:,"nearest_graph_node"] = zipN["nearest_graph_node"].astype(int)
convT.columns = pd.merge(conTmp,zipN,left_on="node",right_on="nearest_graph_node",how="left")['PLZ']
convT.to_csv(baseDir + "log/nissan/zip2zip_motor.csv.tar.gz",compression="gzip")

odmS = pd.read_csv(baseDir + "log/nissan/odm_shortbreak.csv.tar.gz",compression="gzip")
odmL = pd.read_csv(baseDir + "log/nissan/odm_longbreak.csv.tar.gz",compression="gzip")




print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
