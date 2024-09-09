#http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA

mist = pd.read_csv(baseDir + "raw/tank/visit_bon.csv")
sact = pd.read_csv(baseDir + "raw/tank/act_test.csv.tar.gz",compression="gzip")
hL = sact.columns[[bool(re.search('T??:',x)) for x in sact.columns]]
hL1 = mist.columns[[bool(re.search('T??:',x)) for x in mist.columns]]
hL = sorted(list(set(hL) & set(hL1)))

for i,g in sact.groupby('id_clust'):
    X = g[hL].values.T
    if not any(mist['id_clust'] == i):
        continue
    y = mist[mist['id_clust'] == i][hL].values[0]

    plt.plot(X.sum(1))
    plt.show()
    
ica = FastICA(n_components=int(X.shape[1]/2))
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
pca = PCA(n_components=int(X.shape[1]/2))
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
plt.figure()
models = [X,S_, H]
names = ['Observations (mixed signal)','ICA recovered signals','PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()



#!/usr/bin/env python
from __future__ import division
import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# loading wav files
fs_1, voice_1 = wavfile.read(baseDir + "raw/demixed.wav")
fs_2, voice_2 = wavfile.read(baseDir + "raw/mixed.wav")
# reshaping the files to have same size
m = voice_1.shape[0]
voice_2 = voice_2[:m]

# plotting time domain representation of signal
if False:
    figure_1 = plt.figure("Original Signal")
    plt.subplot(2, 1, 1)
    plt.title("Time Domain Representation of voice_1")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.plot(np.arange(m)/fs_1, voice_1)
    plt.subplot(2, 1, 2)
    plt.title("Time Domain Representation of voice_2")
    plt.plot(np.arange(m)/fs_2, voice_2)
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()

# mix data
voice = np.c_[voice_1, voice_2]
A = np.array([[1, 0.5], [0.5, 1]])
A = np.array([1, 0.5,0.5, 1])
X = np.dot(voice, A)
X = voice

# plotting time domain representation of mixed signal
if False:
    figure_2 = plt.figure("Mixed Signal")
    plt.subplot(2, 1, 1)
    plt.title("Time Domain Representation of mixed voice_1")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.plot(np.arange(m)/fs_1, X[:, 0])
    plt.subplot(2, 1, 2)
    plt.title("Time Domain Representation of mixed voice_2")
    plt.plot(np.arange(m)/fs_2, X[:, 1])
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()

# blind source separation using ICA
ica = FastICA()
print("Training the ICA decomposer .....")
t_start = time.time()
ica.fit(X)
t_stop = time.time() - t_start
print("Training Complete; took %f seconds" % (t_stop))
# get the estimated sources
S_ = ica.transform(X)
# get the estimated mixing matrix
A_ = ica.mixing_
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# plotting time domain representation of estimated signal
figure_3 = plt.figure("Estimated Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of estimated voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, S_[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of estimated voice_2")
plt.plot(np.arange(m)/fs_2, S_[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()



sig1, fs1, enc1 = wavread('file1.wav')
sig2, fs2, enc2 = wavread('file2.wav')
mixed1 = sig1 + 0.5 * sig2
mixed2 = sig2 + 0.6 * sig1
from mdp import fastica
from scikits.audiolab import flacread, flacwrite
from numpy import abs, max
 
# Load in the stereo file
recording, fs, enc = flacread('mix.flac')
 
# Perform FastICA algorithm on the two channels
sources = fastica(recording)
 
# The output levels of this algorithm are arbitrary, so normalize them to 1.0.
sources /= max(abs(sources), axis = 0)
 
# Write back to a file
flacwrite(sources, 'sources.flac', fs, enc)
rec1, fs, enc = flacread('Mixdown (1).flac') # Mono file
rec2, fs, enc = flacread('Mixdown (2).flac')
rec3, fs, enc = flacread('Mixdown (3).flac')
 
sources = fastica(array([rec1,rec2,rec3]).transpose())




 #!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np



class AMUSE:



    def __init__(self, x, n_sources, tau):

        self.x = x
        self.n_sources = n_sources
        self.tau = tau
        self.__calc()
        
    def __calc(self):

       #BSS using eigenvalue value decomposition

       #Program written by A. Cichocki and R. Szupiluk at MATLAB
      
        R, N = self.x.shape
        Rxx = np.cov(self.x)
        U, S, U = np.linalg.svd(Rxx)
        
        if R > self.n_sources:
            noise_var = np.sum(self.x[self.n_sources+1:R+1])/(R - (self.n_sources + 1) + 1)
        else:
            noise_var = 0
        
        h = U[:,0:self.n_sources]
        T = np.zeros((R, self.n_sources))
        
        for m in range(0, self.n_sources):
            T[:, m] = np.dot((S[m] - noise_var)**(-0.5) ,  h[:,m])
        
        T = T.T
        y = np.dot(T, self.x)
        R1, N1 = y.shape
        Ryy = np.dot(y ,  np.hstack((np.zeros((R1, self.tau)), y[:,0:N1 - self.tau])).T) / N1
        Ryy = (Ryy + Ryy.T)/2
        D, B  = np.linalg.eig(Ryy)
        
        self.W = np.dot(B.T, T)
        self.sources = np.dot(self.W, self.x)



Example of use of the class AMUSE.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/bruno/PESQUISAS/Python/') #directory of the AMUSE class


import numpy as np

import pylab as pl
from AMUSE import AMUSE
from scipy.io import wavfile
    
#read sources
fs, s1 = wavfile.read('/home/bruno/MESTRADO/Pesquisa/ArquivosAudio/drums.wav')
fs, s2 = wavfile.read('/home/bruno/MESTRADO/Pesquisa/ArquivosAudio/bass.wav')


#5 sec of sources

s1 = s1[0:220500]
s2 = s2[0:220500]
s = np.c_[s1, s2].T


A = np.array([[10.2,12.5],[5.7,2.4]])  #Matrix mixture. Merges the sources

x = np.dot(A, s) #observed signal

amuse = AMUSE(x, 2, 1)
s_hat = amuse.sources #estimate sources. 1/8 size of the original signal
W = amuse.W #separation matrix


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/bruno/PESQUISAS/Python/') #directory of the AMUSE class


import numpy as np

import pylab as pl
from AMUSE import AMUSE
import pywt
from scipy.io import wavfile
    
#read sources
fs, s1 = wavfile.read('/home/bruno/MESTRADO/Pesquisa/ArquivosAudio/drums.wav')
fs, s2 = wavfile.read('/home/bruno/MESTRADO/Pesquisa/ArquivosAudio/bass.wav')


#5 sec of sources

s1 = s1[0:220500]
s2 = s2[0:220500]
s = np.c_[s1, s2].T


A = np.array([[10.2,12.5],[5.7,2.4]])  #Matrix mixture. Merges the sources

x = np.dot(A, s) #observed signal


Wavelet = pywt.Wavelet('db2') #set wavelet

level_max = 3 #set level decomposition


#compute the coefficients

#the aproximation signal is given by coeffs1[0] and coeffs2[0]
coeffs1 = pywt.wavedec(x[0], Wavelet, level=level_max)
coeffs2 = pywt.wavedec(x[1], Wavelet, level=level_max)


x_ = np.c_[coeffs1[0], coeffs2[0]].T



amuse = AMUSE(x_, 2, 1)

s_hat_dwt = amuse.sources #estimate sources. 1/8 size of the original signal
W = amuse.W #separation matrix


s_hat = np.dot(W, x) #estimate sources. Same size of the original signal
 t = np.arange(0, 5, 1./fs)


fig0 = pl.figure()

pl.subplot(211)
pl.title('Original Sources')
pl.plot(t, s1)
pl.subplot(212)
pl.plot(t, s2)
pl.show()


fig1 = pl.figure()

pl.subplot(211)
pl.title('Observed Signal')
pl.plot(t, x[0])
pl.subplot(212)
pl.plot(t, x[1])
pl.show()


fig2 = pl.figure()

pl.subplot(211)
pl.title('Estimated Sources')
pl.plot(t, s_hat[0])
pl.subplot(212)
pl.plot(t, s_hat[1])


for fig in [fig0, fig1, fig2]:   

    for ax in fig.get_axes():
        ax.titlesize = 10
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Amplitude')
        ax.grid()
            
pl.show()




from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
import scipy as scipy
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.cluster.vq import whiten
import train_shapeLib as shl

fSux = "30"
sact = pd.read_csv(baseDir + "out/tank_activity_"+fSux+".csv.tar.gz",compression="gzip")
hL = sact.columns[[bool(re.search(':',x)) for x in sact.columns]]
X = np.array(sact[hL])
mist = pd.read_csv(baseDir + "out/tank_visit_max.csv")
importlib.reload(shl)
scoreL = shl.scoreLib()
y = scoreL.score(sact,mist,hL,"cluster")['y_corr']
y = np.array(y) / np.max(y)

input_img = Input(shape=(64,1))  # adapt this if using `channels_first` image data format
encoded = Dense(16, activation='relu')(input_img)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(1, activation='relu')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='SGD', loss='mean_squared_error')
autoencoder.summary()
x_train = np.transpose(emat[:,0:50000])
x_train = np.expand_dims(x_train,axis=2)
#x_train = np.reshape(x_train, (x_train.shape[1], 64, 1))
x_test = np.transpose(emat[:,50000:80000])
x_test = np.expand_dims(x_test,axis=2)
#x_test = np.reshape(x_test, (x_test.shape[1], 64, 1))

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=150,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


preds = autoencoder.predict(x_test)
plt.plot(x_test[0:500,1,0])
plt.plot(preds[0:500,1,0])
plt.show()
