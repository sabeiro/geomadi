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

#cc = gridSe.find({ "tile_id": 1782  })
c = gridSe.find().next()
mat = np.array(c['geom']['coordinates'][0])
dx = (mat[1][0] - mat[0][0])/2.
dy = (mat[1][1] - mat[0][1])/2.

if False:
    plt.scatter([x[0] for x in mat],[x[1] for x in mat])
    plt.show()

convL = []
for c in gridSe.find():
    convL.append([c['tile_id'],c['geom']['coordinates'][0][0][0]+dx,c['geom']['coordinates'][0][0][1]+dy])
convL = pd.DataFrame(convL)
convL.to_csv(os.eviron("LAV_DIR") + "gis/telia/gridMap.csv")
print('done')

# def find(dict_list, key, value_list):
#     return [dict for dict in dict_list if dict[key] in value_list]



# from pandas.tools.plotting import scatter_matrix
# df = DataFrame(randn(1000, 4), columns=['a', 'b', 'c', 'd'])
# scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
