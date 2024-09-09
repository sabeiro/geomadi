"""
train_keras:
parent class for keras learning regarding 
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM, RepeatVector, Dropout
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from keras import optimizers
import geomadi.train_score as t_s
session_conf = tensorflow.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tensorflow.set_random_seed(1)
sess = tensorflow.Session(graph=tensorflow.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

class trainKeras:
    """train_keras basics utilities"""
    def __init__(self,X):
        """save data set matrix"""
        self.X = X
        self.setBoundary(X)
        self.history = []

    def setBoundary(self,X):
        """Save the normalization boundaries"""
        self.minX = [np.min(X[:,x]) for x in range(X.shape[1])]
        self.maxX = [np.max(X[:,x]) for x in range(X.shape[1])]

    def setLimit(self,y):
        """Save the normalization boundaries"""
        self.yMin = np.min(y)
        self.yMax = np.max(y)

    def setOptimizer(self,name="adam",**args):
        """set optimizer for the model"""
        if name == "sgd":
            opt = optimizers.SGD(**args)
        elif name == "rmsdrop":
            opt = optimizers.RMSprop(**args)
        elif name == "adagrad":
            opt = optimizers.Adagrad(**args)
        elif name == "adadelta":
            opt = optimizers.Adadelta(**args)
        elif name == "adam":
            opt = optimizers.Adam(**args)
        elif name == "adamax":
            opt = optimizers.Adamax(**args)
        else: 
            opt = optimizers.Adadelta(**args)
        self.opt = opt
        
    def setX(self,X):
        """reset data set"""
        self.setBoundary(X)
        self.X = X

    def getX(self):
        """return data set"""
        return self.X

    def getEpochs(self):
        """return the total number of epochs"""
        history = self.history
        loss = np.concatenate([x for x in history.history['loss']])
        return len(loss)
    
    def plotHistory(self):
        """plot history of training performace"""
        if not self.history:
            raise Exception("train the model first")
        history = self.history
        loss = np.concatenate([x for x in history.history['loss']])
        val_loss = np.concatenate([x for x in history.history['val_loss']])
        plt.plot(loss,label='train')
        plt.plot(val_loss,label='test')
        plt.legend()
        plt.show()

    def saveModel(self,fName):
        """save last trained model"""
        if not self.model:
            raise Exception("train the model first")
        fName = fName.split(".")[0]
        json.dump(self.model.to_json(),open(fName+".json","w"))
        # with open(fName+".json","w") as json_file:
        #     json_file.write(self.model.to_json())
        self.model.save_weights(fName+".h5")

    def loadModel(self,fName):
        """load trained model"""
        fName = fName.split(".")[0]
        json_file = open(fName+'.json','r')
        model_json = json_file.read()
        json_file.close()
        model = keras.model.model_from_json(model_json)
        model.load_weights(fName+".h5")
        self.model = model

    def getModel(self):
        """return the encoder"""
        if not self.model:
            raise Exception("train the model first")
        return self.model

    def delModel(self):
        """delete current model"""
        try:
            del self.model
        except:
            pass
        
    def scale(self,X):
        """scaler function"""
        X = X.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(X)
        return scaled, scaler

    def invert_scale(self,scaler, X, value):
        new_row = [x for x in X] + [value]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]
    
    def defConv2D(self):
        """define a convolutional neural network for small images"""
        k_size = (3,2) #(5,5)
        convS = [8,16] #[32,64]
        model = Sequential()
        model.add(Conv2D(convS[0],kernel_size=k_size,strides=(1,1),activation='relu',input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(convS[1],k_size,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(1000,activation='relu'))
        model.add(Dense(num_class,activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        return model

    def newModel(self):
        """pointer to the model"""
        self.model = self.defConv2D()
        
    def defModel(self):
        """return a new model if not previously defined"""
        try: return self.model
        except: pass
        self.model = self.newModel()
        return self.model

    def splitSet(self,X,y,portion=.75):
        """split data set"""
        N = X.shape[0]
        N_tr = int(N*portion)
        shuffleL = random.sample(range(N),N)
        X_train = X[shuffleL][:N_tr]
        X_test = X[shuffleL][N_tr:]
        y_train = y[shuffleL][:N_tr]
        y_test = y[shuffleL][N_tr:]
        return X_train, X_test, y_train, y_test

    def predict(self,X_test):
        return self.model.predict(X_test)
    
    def train(self,y,**args):
        """train the current model, calculate scores"""
        X_train, X_test, y_train, y_test = self.splitSet(self.X,y)
        self.defModel()
        history = self.autoencoder.fit(X_train,X_train,validation_data=(X_test,X_test),**args)#,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        self.history.append(history)
        y_pred = self.predict(X_test)
        kpi = t_s.calcMetrics(y_pred, y_test)
        return self.model, kpi
    
    def featKnockOut(self,g,y):
        """feature knock out, recursively remove one feature at time and calculate performances"""
        tL = g.columns
        perfL = []
        self.setX(g.values)
        model, kpi = self.train(y,epoch=200)
        for j in range(9):
            model, kpi = self.train(y,epoch=50)
            kpi['feature'] = "all"
            perfL.append(kpi)
        for i in tL:
            print(i)
            c = g.drop(columns=[i])
            self.setX(c.values)
            self.delModel()
            model, kpi = self.train(y,epoch=200)
            for j in range(9):
                model, kpi = self.train(y,epoch=50)
                kpi['feature'] = "- " + i
                perfL.append(kpi)
        perfL = pd.DataFrame(perfL)
        self.perfL = perfL
        return perfL
    
    def plotFeatImportance(self,perfL):
        """boxplot of scores per feature"""
        fig, ax = plt.subplots(1,2)
        perfL.boxplot(by="feature",column="rel_err",ax=ax[0])
        perfL.boxplot(by="feature",column="cor",ax=ax[1])
        for a in ax:
            for tick in a.get_xticklabels():
                tick.set_rotation(45)
        plt.show()

