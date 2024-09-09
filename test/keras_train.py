#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
from sklearn.model_selection import train_test_split

sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def plog(text):
    print(text)

mist = pd.read_csv(baseDir + "raw/tank/act_vist_year.csv.tar.gz",compression="gzip")
poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
mist.loc[:,"day"]  = mist['time'].apply(lambda x:x[0:10])
mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
mist.loc[:,"month"] = mist['time'].apply(lambda x:x[5:7])
mist.loc[:,"wday"] = mist['day'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
dist = mist.pivot_table(index=["id_clust","hour"],columns="month",values="ref",aggfunc=np.mean)
dist.replace(float("Nan"),0,inplace=True)
idL = dist.index.values




w, h = 8, 5;
Matrix = [[0 for x in range(w)] for y in range(h)] 


dist  = mist[ ['id_clust','wday','ref','act']].groupby(['id_clust','wday']).agg(sum).reset_index()
dist.loc[:,"y_dif"] = dist['ref']/dist['act']
dist.replace(float("Inf"),0,inplace=True)
dist.loc[:,"id_cat"], _ = dist['id_clust'].factorize()

X = dist[['wday','id_clust']].values
y = np.array((dist['y_dif'] > np.mean(dist['y_dif']))*1)
#y, y_bin = tlib.binVector(y,nBin=5,threshold=10)
#y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


X = xr.Dataset({k: xr.concat(dist[k].to_xarray(),dim='id_clust') for k in dist.keys()})
X = dist.to_xarray()


def index_hash(*codes):
    m = len(codes[0])
    n = len(codes)
    a = np.empty((m, n), dtype=np.int64)
    for i in range(n):
        a[:,i] = codes[i]
    a = np.apply_along_axis(lambda x: hash(x.data.tobytes()), 1, a)
    return a


input_shape = (img_x, img_y, 1)

model = Sequential()
model.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test),callbacks=[history])



print('-----------------te-se-qe-te-ve-be-te-ne------------------------')


