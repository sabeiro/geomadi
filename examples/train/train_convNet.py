"""
train_weekImg:
implementation of different models with keras to predict time series represented like images.
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM, RepeatVector, Dropout, ZeroPadding2D, Cropping2D
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
import geomadi.train_score as t_s
import geomadi.train_reshape as t_r
from geomadi.train_keras import trainKeras

class weekImg(trainKeras):
    """train on image representation of weeks per hourly values"""
    def __init__(self,X,model_type="convNet",isBackfold=True):
        trainKeras.__init__(self,X)
        self.isBackfold = isBackfold
        self.model_type = model_type

    def getX(self):
        return self.X
    
    def getEncoder(self):
        """return the encoder"""
        if not self.encoder:
            raise Exception("train the model first")
        return self.encoder
        
    def getDecoder(self):
        """return the decoder"""
        if not self.decoder:
            raise Exception("train the model first")
        return self.decoder

    def getAutoencoder(self):
        """return the autoencoder"""
        if not self.autodecoder:
            raise Exception("train the model first")
        return self.autodecoder

    def defSimpleEncoder(self,encoding_dim=32):
        """define simple encoder"""
        pix_dim = self.X.shape[1]*self.X.shape[2]#784
        input_img = Input(shape=(pix_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(pix_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
        return encoder, decoder, autoencoder

    def defDeepEncoder(self,layDim=[64,32,16],encoding_dim=32):
        """define deep encoder"""
        pix_dim = self.X.shape[1]*self.X.shape[2]
        input_img = Input(shape=(pix_dim,))
        encoded = Dense(layDim[0], activation='relu')(input_img)
        encoded = Dense(layDim[1], activation='relu')(encoded)
        encoded = Dense(layDim[2], activation='relu')(encoded)
        decoded = Dense(layDim[1], activation='relu')(encoded)
        decoded = Dense(layDim[0], activation='relu')(decoded)
        decoded = Dense(pix_dim, activation='sigmoid')(decoded)
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img,encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = 1#Model(encoded_input,decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])
        print(autoencoder.summary())
        self.model = autoencoder
        return encoder, decoder, autoencoder

    def defShortConv(self,convS=[8,16,1],kernel_size=(3,3),max_pooling=(3,3)):
        """define a short convolutional network"""
        self.setOptimizer("adadelta",lr=0.1)
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        encoded = ZeroPadding2D(((0,0),(0,1)))(input_img)
        encoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        
        decoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(1,kernel_size,activation='relu',padding='same')(decoded)
        decoded = Cropping2D(cropping=((0, 0), (0, 1)), data_format=None)(decoded)
        
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=encoder.layers[-1].output_shape[1:])
        decoder_layer = autoencoder.layers[-1]
        decoder = 1#Model(encoded_input, decoder_layer(encoded_input))##to fix
        autoencoder.compile(optimizer=self.opt,loss='binary_crossentropy',metrics=['accuracy'])
        print(autoencoder.summary())
        return encoder, decoder, autoencoder
    
    def defNoBackfold(self,convS=[8,16,8],kernel_size=(3,3),max_pooling=(3,3)):
        """define a short convolutional network"""
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        encoded = ZeroPadding2D(((1,1),(1,2)))(input_img)
        encoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        
        decoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(convS[2],kernel_size,activation='relu',padding='same')(decoded)
        decoded = Cropping2D(cropping=((1, 1), (1, 2)), data_format=None)(decoded)
        
        autoencoder = Model(input_img,decoded,name="autoencoder")
        encoder = Model(input_img,encoded)
        encoded_input = Input(shape=encoder.layers[-1].output_shape[1:])
        decoder_layer = autoencoder.layers[-1]
        decoder = 1#Model(encoded_input, decoder_layer(encoded_input))##to fix
        autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
        print(autoencoder.summary())
        return encoder, decoder, autoencoder
    
    def defInterp(self,convS=[32,16,8],kernel_size=(3,3),max_pooling=(3,3)):
        """define convolution neural net"""
        if not hasattr(self,"opt"):
            self.setOptimizer("adadelta",lr=0.1)
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        encoded = ZeroPadding2D(((0,0),(1,2)))(input_img)
        encoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        encoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        
        decoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(decoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(1,kernel_size,activation='sigmoid',padding='same')(decoded)
        decoded = Cropping2D(cropping=((0, 0), (1, 2)), data_format=None)(decoded)
        
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=encoder.layers[-1].output_shape[1:])
        decoder_layer = autoencoder.layers[-1]
        decoder = 1#Model(encoded_input, decoder_layer(encoded_input))##to fix
        autoencoder.compile(optimizer=self.opt,loss='binary_crossentropy',metrics=['accuracy'])
        print(autoencoder.summary())
        return encoder, decoder, autoencoder

    def defConvNet(self,convS=[8,16,8],kernel_size=(3,3),max_pooling=(2,2)):
        """define convolution neural net"""
        if not hasattr(self,"opt"):
            self.setOptimizer("adadelta",lr=0.1)
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        encoded = ZeroPadding2D(((1,2),(1,1)))(input_img)
        encoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        encoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        
        decoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(decoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(1,kernel_size,activation='sigmoid',padding='same')(decoded)
        decoded = Cropping2D(cropping=((1, 2), (1, 1)), data_format=None)(decoded)
        
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=encoder.layers[-1].output_shape[1:])
        decoder_layer = autoencoder.layers[-1]
        decoder = 1#Model(encoded_input, decoder_layer(encoded_input))##to fix
        autoencoder.compile(optimizer=self.opt,loss='binary_crossentropy',metrics=['accuracy'])
        print(autoencoder.summary())
        return encoder, decoder, autoencoder

    def defConvFlat(self,convS=[8,16,8],kernel_size=(3,3),max_pooling=(2,2)):
        """define convolution neural net"""
        if not hasattr(self,"opt"):
            self.setOptimizer("adadelta",lr=0.1)
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        encoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(input_img)
        encoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        
        decoded = Conv2D(convS[2],kernel_size,activation='relu',padding='same')(encoded)
        decoded = Conv2D(1,kernel_size,activation='sigmoid',padding='same')(decoded)
        
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        autoencoder.compile(optimizer=self.opt,loss='binary_crossentropy',metrics=['accuracy'])
        print(autoencoder.summary())
        return encoder, 1, autoencoder

    def defConvNetOld(self,convS=[8,16,8],kernel_size=(3,3),max_pooling=(2,2)):
        """define convolution neural net"""
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        encoded = ZeroPadding2D(((1,2),(1,1)))(input_img)
        encoded = Conv2D(convS[0],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        encoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling,padding='same')(encoded)
        encoded = Conv2D(convS[2],kernel_size,activation='sigmoid',padding='same')(encoded)
        encoded = MaxPooling2D(max_pooling, padding='same')(encoded)
        
        decoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(encoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(convS[1],kernel_size,activation='relu',padding='same')(decoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(convS[0],kernel_size,activation='relu')(decoded)
        decoded = UpSampling2D(max_pooling)(decoded)
        decoded = Conv2D(1,kernel_size,activation='sigmoid',padding='same')(decoded)
        decoded = Cropping2D(cropping=((1, 2), (1, 1)), data_format=None)(decoded)
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=encoder.layers[-1].output_shape[1:])
        decoder_layer = autoencoder.layers[-1]
        decoder = 1#Model(encoded_input, decoder_layer(encoded_input))##to fix
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        print(autoencoder.summary())
        return encoder, decoder, autoencoder

    def varAutoenc(self,convS=[8,16,8],kernel_size=(3,3),max_pooling=(2,2)):
        x = Input(batch_shape=(batch_size, original_dim))
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim),mean=0., std=epsilon_std)
            return z_mean + K.exp(z_log_sigma) * epsilon
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        vae = Model(x, x_decoded_mean)
        encoder = Model(x, z_mean)
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)
        def vae_loss(x, x_decoded_mean):
            xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

        vae.compile(optimizer='rmsprop', loss=vae_loss)
        
    def seq2seq(self):
        input_img = Input(shape=(self.X.shape[1],self.X.shape[2],1))
        inputs = Input(shape=(self.X.shape[1]*self.X.shape[2],self.X.shape[0]))
        encoded = LSTM(latent_dim)(inputs)
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)
        sequence_autoencoder = Model(inputs, decoded)
        encoder = Model(inputs, encoded)
        
    def reshape(self,X):
        if self.model_type in ["simple","variational"]:
            return np.reshape(X,(len(X),np.prod(X.shape[1:])))
        elif self.model_type == "deep":
            return np.reshape(X,(len(X),np.prod(X.shape[1:])))
        else :
            return np.reshape(X, (len(X),X.shape[1],X.shape[2], 1))
        
    def defEncoder(self):
        if self.model_type == "simple":
            return self.defSimpleEncoder(encoding_dim=32)
        elif self.model_type == "deep":
            return self.defDeepEncoder()
        elif self.model_type == "convNet":     
            return self.defConvNet()
        elif self.model_type == "shortConv":     
            return self.defShortConv()
        elif self.model_type == "convFlat":     
            return self.defConvFlat() 
        elif self.model_type == "convPad":
            return self.defConvPad()
        elif self.model_type == "variational":
            return self.varAutoenc()
        elif self.model_type == "noBackfold":
            return self.defNoBackfold()
        elif self.model_type == "interp":
            return self.defInterp()
        else:
            return self.defSimpleEncoder(encoding_dim=32)
        
    def runAutoencoder(self,split_set=.75,**args):
        """perform a simple autoencoder"""
        y = np.array(list(range(self.X.shape[0])))
        X_train, X_test, y_train, y_test = self.splitSet(self.X,y,portion=split_set)
        if not hasattr(self,"autoencoder"):
            self.encoder, self.decoder, self.autoencoder = self.defEncoder()
        X_train = self.reshape(X_train)
        X_test = self.reshape(X_test)
        history = self.autoencoder.fit(X_train,X_train,validation_data=(X_test,X_test),**args)#,callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        self.history.append(history)
        self.model = self.autoencoder
        
    def runEncoder(self,Y,split_set=.75,**args):
        """perform a simple autoencoder"""
        X_train, X_test, Y_train, Y_test = self.splitSet(self.X,Y,portion=split_set)
        if not hasattr(self,"autoencoder"):
            self.encoder, self.decoder, self.autoencoder = self.defEncoder()
        X_train = self.reshape(X_train)
        X_test = self.reshape(X_test)
        Y_train = self.reshape(Y_train)
        Y_test = self.reshape(Y_test)
        history = self.autoencoder.fit(X_train,Y_train,validation_data=(X_test,Y_test),**args)
        self.history.append(history)
        self.model = self.autoencoder
        
    def plotImg(self,nline=12):
        """plot a sample of weekly representations"""
        n = min(nline*nline,self.X.shape[0])
        N = self.X.shape[0]
        shuffleL = random.sample(range(N),N)
        plt.figure()
        cmap = plt.get_cmap("viridis")
        for i in range(n):
            ax = plt.subplot(nline,nline,i+1)
            plt.imshow(self.X[shuffleL[i]],cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.tight_layout()
        plt.subplots_adjust(left=0.1,bottom=0.5,right=None,top=None,wspace=None,hspace=None)
        plt.show()

    def plotTimeSeries(self,nline=6):
        """plot a time series along side with a the image representation"""
        glL = []
        N = self.X.shape[0]
        shuffleL = random.sample(range(N),N)
        cmap = plt.get_cmap("viridis")
        plt.figure()
        plt.title("image representation of time series")
        for i in shuffleL[:nline]:
            glL.append(self.X[i])
            j = [1,5,9,3,7,11]
            ncol = int(nline*2/4)
            ncol = ncol + (ncol %2)
        for i in range(nline):
            ax = plt.subplot(4,ncol,i*2+1)
            plt.plot(glL[i].ravel())
            ax = plt.subplot(4,ncol,i*2+2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(glL[i],cmap=cmap)

        plt.xlabel("hours")
        plt.ylabel("frequency")
        plt.show()

    def plotMorph(self,nline=4,n=10):
        """plot the morphing between original images and autoencoded"""
        rawIm = self.reshape(self.X)
        decIm = self.autoencoder.predict(rawIm)
        N = self.X.shape[0]
        shuffleL = random.sample(range(N),N)
        cmap = plt.get_cmap("viridis")
        plt.figure()
        for i in range(int(nline*n/2)):
            c = (i % n)
            r = int( (i - c)/n )
            ax = plt.subplot(nline,n, c + (r*2)*n + 1)
            plt.imshow(decIm[shuffleL[i]].reshape(self.X.shape[1], self.X.shape[2]),cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.subplots_adjust(bottom=.4)
            ax = plt.subplot(nline,n,c + (r*2+1)*n + 1)
            plt.imshow(rawIm[shuffleL[i]].reshape(self.X.shape[1], self.X.shape[2]),cmap=cmap)
            ax.set_title("^")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.subplots_adjust(bottom=.1)
            plt.tight_layout(pad=1.08, h_pad=0.1, w_pad=None, rect=None)
        plt.subplots_adjust(left=None,bottom=0.4,right=None, top=None, wspace=None, hspace=None)
        plt.show()

        
