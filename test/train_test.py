from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from keras.models import Sequential
from keras.layers import Dense
wineD = pd.read_csv(baseDir + "raw/winequality-red.csv")
wineD.loc[:,"type"] = 0
tmp = pd.read_csv(baseDir + "raw/winequality-white.csv")
tmp.loc[:,"type"] = 1
wineD = pd.concat([wineD,tmp])

X = dist[['wday','id_poi']]
y = (dist['y_dif'] > np.mean(dist['y_dif']))*1
#y, y_bin = tlib.binVector(y,nBin=5,threshold=10)
#y = pd.get_dummies(y)
X = np.array(X.astype(float))
y = np.array(y.astype(float))

X=wineD.ix[:,0:11]
y=np.ravel(wineD.type)
NNeu = X.shape[1]
model = Sequential()
model.add(Dense(NNeu+1, activation='relu', input_shape=(NNeu,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

y = wineD.quality
X = wineD.drop('quality', axis=1) 
NNeu = X.shape[1]
model = Sequential()
model.add(Dense(64, input_dim=NNeu, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(sum(y)/y.shape[0])
model.fit(X_train, y_train,epochs=3, batch_size=1, verbose=1)
y_pred = model.predict(X_test)
plt.hist(y,bins=20)
plt.hist(y_pred,bins=20)
plt.show()


print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test,y_pred))
print(cohen_kappa_score(y_test, y_pred))

import importlib
importlib.reload(tlib)
model = tlib.modKeras(x_train,y_train)
model.fit(x_train,y_train,epochs=5,batch_size=128,validation_data=(x_test,y_test))
y_pred = model.predict(X)




from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for train, test in kfold.split(X, Y):
    model = Sequential()
    model.add(Dense(64, input_dim=12, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.fit(X[train], Y[train], epochs=10, verbose=1)


from keras.optimizers import RMSprop
rmsprop = RMSprop(lr=0.0001)
model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
from keras.optimizers import SGD, RMSprop
sgd=SGD(lr=0.1)
model.compile(optimizer=sgd, loss='mse', metrics=['mae'])



fName = baseDir+"/train/"+'lookAlikeNeural'+str(0)
open(fName + ".json","w").write(model.to_json())
model.save_weights(fName+".h5")

##predictions = model.predict(X)





##svm
##bagging classifier
##n estim 10
##max feat 1/10
##kernel lin
##deg 3
##C 1



pred = model.predict_classes(x_valid)
print pred

net1 = NeuralNet(
    layers=[('input',layers.InputLayer),
            ('hidden',layers.DenseLayer),
            ('output',layers.DenseLayer),
    ],
    input_shape=(None,Ncat),
    hidden_num_units=3,  # number of units in 'hidden' layer
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9
    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
        max_epochs=10,
        verbose=1,
    )

net1.fit(x_train,y_train)
print("Predicted: %s" % str(net1.predict(x_test)))

imgN = 24
img = x_test[imgN]
img = np.reshape(img,(np.sqrt(img.shape[0]),np.sqrt(img.shape[0])))*255
print "Prediction is: ", pred[imgN]
pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()
