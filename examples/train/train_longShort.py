"""
train_longShort:
implementation of the long short term memory to forecast time series
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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import geomadi.train_score as t_s
from geomadi.train_keras import trainKeras

class timeSeries(trainKeras):
    """keras on time series"""
    def __init__(self,X):
        trainKeras.__init__(self,X)

    def difference(self,dataset,interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    def inverse_difference(self,history, yhat, interval=1):
        return yhat + history[-interval]

    def longShort(self):
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim)(inputs)
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)
        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        return encoder

    def modelLongShort(self,input_shape=(1,1)):
        model = Sequential()
        model.add(LSTM(50,input_shape=input_shape))
	#model.add(LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]),stateful=True))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
	#model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model
        return model

    def modelLongShort1(self,input_shape=(1,1)):
        regressor = Sequential()
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=input_shape))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        self.model = regressor
        return regressor

    def newModel(self):
        print("new long short term memory model")
        print(self.input_shape)
        self.model = self.modelLongShort(input_shape=self.input_shape)
        return self.model
        
    def defModel(self):
        try:
            return self.model
        except:
            pass
        self.model = self.newModel()
        return self.model
    
    def trainReset(self,train, batch_size, nb_epoch, neurons):
        X, y = train[:, 0:-1], train[:, -1]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        model = Sequential()
        model.add(LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]),stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            model.reset_states()
        return model

    def forecastLongShort(self, batch_size, X):
        X = X.reshape(1, 1, len(X))
        yhat = self.model.predict(X, batch_size=batch_size)
        return yhat[0,0]

    def toSupervised(self,dataset,look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def prepareSupervised(self,data,lag=1):
        df = DataFrame(data)
        columns = [df.shift(i) for i in range(1, lag+1)]
        columns.append(df)
        df = concat(columns, axis=1)
        df.fillna(0, inplace=True)
        return df
    
    def prepareLongShort(self,X,y,n_in=1,n_out=1,dropnan=True):
        """perform a n_in shift for each of training feature"""
        X = np.append(y[:,np.newaxis],X,axis=1)
        scaled, scaler = self.scale(X)
        n_vars = 1 if type(scaled) is list else scaled.shape[1]
        df = pd.DataFrame(scaled)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(1, n_out):
            cols.append(df.shift(-i))
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        reframed = pd.concat(cols, axis=1)
        reframed.columns = names
        reframed.replace(float('nan'),0,inplace=True)
        X1 = reframed.values
        self.setLimit(y)
        y = (y - self.yMin)/(self.yMax+self.yMin)
        return X1, y, scaler

    def splitSet(self,X,y,portion=.5,n_in=2,n_out=1):
        X1, y1, scaler = self.prepareLongShort(X,y,n_in=n_in,n_out=n_out,dropnan=False)
        ahead = int(X.shape[0]*portion)
        y_train, y_test = y1[:ahead], y1[ahead:]
        X_train, X_test = X1[:ahead,:], X1[ahead:,:]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        self.input_shape = (X_train.shape[1],X_train.shape[2])
        return X_train, X_test, y_train, y_test

    def predict(self,X,y):
        X_train, X_test, y_train, y_test = self.splitSet(X,y,portion=.0)
        y_scaled = self.model.predict(X_test)
        y_pred = y_scaled[:,0]*(self.yMax+self.yMin) + self.yMin
        y_test1 = y_test*(self.yMax+self.yMin) + self.yMin
        return y_pred, y_test1

    def forecastLongShort(self,y,ahead=28,epoch=50,n_in=2,n_out=1):
        """forecast on ahead time points"""
        portion = 1. - ahead/len(y)
        self.setLimit(y)
        X_train, X_test, y_train, y_test = self.splitSet(self.X,y,portion=portion)
        self.defModel()
        self.history = self.model.fit(X_train,y_train,epochs=epoch,batch_size=72,validation_data=(X_test,y_test),verbose=0,shuffle=False)#,callbacks=[TensorBoard(log_dir='/tmp/lstm')])
        y_scaled = self.model.predict(X_test)
        y_pred = y_scaled[:,0]*(self.yMax+self.yMin) + self.yMin
        y_test1 = y_test*(self.yMax+self.yMin) + self.yMin
        if False:
            plt.plot(y[ahead:],label="test")
            plt.plot(y_pred,label="pred")
            plt.legend()
            plt.show()
        
        kpi = t_s.calcMetrics(y_pred, y_test1)
        return self.model, kpi

    def plotPredVsRef(self,X,y,portion=.5):
        X_train, X_test, y_train, y_test = self.splitSet(X,y,portion=.5)
        y_scaled = self.model.predict(X_test)
        y_pred = y_scaled[:,0]*(self.yMax+self.yMin) + self.yMin
        y_test1 = y_test*(self.yMax+self.yMin) + self.yMin
        y_pred1 = self.model.predict(X_train)
        y_pred1 = np.reshape(scaler.inverse_transform(y_pred1),len(y_pred1))[1:]
        y_train1 = np.reshape(scaler.inverse_transform([y_train]),len(y_train))[:-1]
        kpi1 = t_s.calcMetrics(y_train1,y_pred1)
        fig, ax = plt.subplots(1,2)
        ax[0].plot(y_train1)
        ax[0].plot(y_pred1)
        ax[0].set_title("train cor %.2f rel_err %.2f" % (kpi1['cor'],kpi1['rel_err']))
        ax[1].plot(y_test1)
        ax[1].plot(y_pred)
        ax[1].set_title("test cor %.2f rel_err %.2f" % (kpi['cor'],kpi['rel_err']))
        plt.show()
        
    def forecastSingle(self,y,look_back=1,ahead=28,epochs=100,isPlot=False):
        #np.random.seed(7)
        X = y.copy()
        if len(y.shape) == 1:
            X = np.reshape(y,(len(y),1))
        X = X.astype('float32')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        X = self.scaler.fit_transform(X)
        train_size = int(len(y) - ahead)
        test_size = ahead
        train, test = X[0:train_size,:], X[train_size:,:]
        X_train, y_train = self.toSupervised(train, look_back)
        X_test, y_test = self.toSupervised(test, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        self.input_shape = (1,look_back)
        self.defModel()
        self.model.fit(X_train,y_train,epochs=epochs,batch_size=1,verbose=0)
        y_pred = self.model.predict(X_test)
        # X_test1 = X_train[-ahead:]
        # y_test1 = y_train[-ahead:]
        # y_pred = []
        # for i in range(ahead):
        #     yhat = model.predict(X_test1)
        #     yhat = list(yhat[-1])
        #     y_pred.append(yhat[0])
        #     for j in range(X_test1.shape[2]-1):
        #         yhat.append(X_test1[-1][0][j+1])
        #     np.vstack((X_test1,[[yhat]]))
        # y_pred = np.array(y_pred)
        y_pred = np.reshape(scaler.inverse_transform(y_pred),len(y_pred))[1:]
        y_test1 = np.reshape(scaler.inverse_transform([y_test]),len(y_test))[:-1]
        kpi = t_s.calcMetrics(y_test1,y_pred)
        if isPlot:
            y_pred1 = self.model.predict(X_train)
            y_pred1 = np.reshape(self.scaler.inverse_transform(y_pred1),len(y_pred1))[1:]
            y_train1 = np.reshape(self.scaler.inverse_transform([y_train]),len(y_train))[:-1]
            kpi1 = t_s.calcMetrics(y_train1,y_pred1)
            fig, ax = plt.subplots(1,2)
            ax[0].plot(y_train1)
            ax[0].plot(y_pred1)
            ax[0].set_title("train cor %.2f rel_err %.2f" % (kpi1['cor'],kpi1['rel_err']))
            ax[1].plot(y_test1)
            ax[1].plot(y_pred)
            ax[1].set_title("test cor %.2f rel_err %.2f" % (kpi['cor'],kpi['rel_err']))
            plt.show()
        return self.model, kpi
    
    def predictLongshort(self,y,nStep=60):
        """prediction on time series"""
        X = self.X
        sc = MinMaxScaler(feature_range = (0, 1))
        X = sc.fit_transform(X)
        X_train = []
        y_train = []
        for i in range(nStep,X.shape):
            X_train.append(X[i-nStep:i, 0])
            y_train.append(X[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        regressor = self.modelLongShort1()
        regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        if False:
            dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
            inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60, 76):
                X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted_stock_price = regressor.predict(X_test)
                predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        return regressor


