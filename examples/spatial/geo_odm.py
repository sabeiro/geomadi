#%pylab inline
import os, sys, gzip, random, csv, json
sys.path.append(os.environ['LAV_DIR']+'/src/')
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from datetime import datetime

def plog(text):
    print(text)

import pymongo

key_file = os.environ['LAV_DIR'] + '/credenza/geomadi.json'
cred = []
with open(key_file) as f:
    cred = json.load(f)

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
infra_conn_dev = client["tdg_17d08"]["infrastructure"]
infra_conn_subway = client["subway_graph"]["munich_cilac_nodes"]
client.database_names()

gridSe = client["telia_se_grid"]['grid_250']
cur = gridSe.find()  
odm = pd.read_csv(os.environ['LAV_DIR']+"/log/ODM20170901.csv.tar.gz",compression='gzip',sep=',',quotechar='"',names=["count","orig","dest","h_orig","h_dest"],header=0)
odm = odm.replace(np.nan,0)
odm['orig'] = odm['orig'].apply(lambda x: int(x))
odm['dest'] = odm['dest'].apply(lambda x: int(x))
odm['count'] = odm['count'].apply(lambda x: int(x))
tileL = odm['orig'].unique()


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')


