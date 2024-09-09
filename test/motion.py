#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import geomadi.lib_graph as gra
import geomadi.series_lib as s_l
import seaborn as sns
from sklearn.decomposition import PCA

def plog(text):
    print(text)

fL = os.listdir(baseDir + "../iot/raw")
ac1L = [x for x in fL if bool(re.search("ACC_100Hz",x))]
ac2L = [x for x in fL if bool(re.search("ACC_200Hz",x))]
gy2L = [x for x in fL if bool(re.search("GYR_200Hz",x))]

if True:
    plog('----------combine-series----------')
    acc1, acc2, gyr2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for f in ac1L:
        idS = f.split("_")[2]
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x_"+idS,"y_"+idS,"z_"+idS])
        acc1 = pd.concat([acc1,sen],axis=1)
    for f in ac2L:
        idS = f.split("_")[2]
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x_"+idS,"y_"+idS,"z_"+idS])
        acc2 = pd.concat([acc2,sen],axis=1)
    for f in gy2L:
        idS = f.split("_")[2]
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x_"+idS,"y_"+idS,"z_"+idS])
        gyr2 = pd.concat([gyr2,sen],axis=1)
    acc1 = acc1.replace(float('nan'),0)
    acc2 = acc2.replace(float('nan'),0)
    gyr2 = gyr2.replace(float('nan'),0)    

if True:
    plog('-------------calc-power-spectrum-------------')
    psN = 20
    psS = []
    for f in ac2L:
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x","y","z"])
        psL = {}
        psL['id'] = f.split("_")[2]
        psL['freq'] = list(range(psN))
        for i in ['x','y','z']:
            serD = s_l.serDecompose(sen[i],period=14)
            ps = np.abs(np.fft.fft(serD['smooth']))**2
            psL[i] = np.log(ps[:psN])
        pca = PCA().fit(sen.values)
        y = pca.transform(sen.values)[:,0]
        serD = s_l.serDecompose(y,period=14)
        ps = np.abs(np.fft.fft(serD['smooth']))**2
        psL['p'] = np.log(ps[:psN])
        psL['r'] = psL['x'] + psL['y'] + psL['z']
        psS.append(pd.DataFrame(psL))
    psS = pd.concat(psS)
    psPx = psS.pivot_table(index="freq",values="x",columns="id",aggfunc=np.sum)
    psPy = psS.pivot_table(index="freq",values="y",columns="id",aggfunc=np.sum)
    psPz = psS.pivot_table(index="freq",values="z",columns="id",aggfunc=np.sum)
    psPp = psS.pivot_table(index="freq",values="p",columns="id",aggfunc=np.sum)
    psPr = psS.pivot_table(index="freq",values="r",columns="id",aggfunc=np.sum)

    
if False:
    plog('-------------spectral-cross-correlation-------------')
    fig ,ax = plt.subplots(1,3)
    corM = psPp.corr()
    ax[0].set_title('PCA power spectrum')
    sns.heatmap(corM,vmax=1,square=True,annot=False,cmap='RdYlGn',ax=ax[0],cbar=False)
    corM = psPr.corr()
    ax[1].set_title('radial power spectrum')
    sns.heatmap(corM,vmax=1,square=True,annot=False,cmap='RdYlGn',ax=ax[1],cbar=False)
    corM = (psPx.corr() + psPy.corr() + psPz.corr())/3.
    ax[2].set_title('direction sum power spectrum')
    sns.heatmap(corM,vmax=1,square=True,annot=False,cmap='RdYlGn',ax=ax[2],cbar=False)
    plt.show()

    fig, ax = plt.subplots(1)
    for i in range(10):
        l = random.choice(acc2.columns)
        y = acc2[l].values
        serD = s_l.serDecompose(y,period=14)
        ps = np.abs(np.fft.fft(serD['smooth']))**2
        ax.plot(ps[:10],label=i)
    ax.set_yscale('log')
    #ax.set_xscale('log')
    plt.legend()
    plt.show()

if False:
    plog('-------------spike-detection----------------')
    import importlib
    importlib.reload(s_l)
    spL = []
    for f in ac2L:
        idS = f.split("_")[2]
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x","y","z"])
        spiD = {}
        for l in ['x','y','z']:
            y = sen[l].values
            serD = s_l.serDecompose(y,period=16)
            serT = s_l.serDecompose(serD['diff'].abs(),period=16)
            spik = serT['smooth'].values > np.mean(serT['smooth'])*7.
            spiL = []
            flip = spik[0]
            for i,j in enumerate(spik):
                if j != flip:
                    spiL.append(i)
                    flip = not flip
            spiD[l] = int(len(spiL)/2)
        spiD['id'] = idS
        spL.append(spiD)
    spL = pd.DataFrame(spL)
            
    if False:
        y = sen[l].values
        serD = s_l.serDecompose(y,period=16)
        serT = s_l.serDecompose(serD['diff'].abs(),period=16)
        spik = serT['smooth'].values > np.mean(serT['smooth'])*3.
        spiV = spik*np.mean(serT['smooth'])*3.
        print(sum(spik),len(spik))
        plt.title("spike percentage %.2f" % (sum(spik)/len(spik)))
        plt.plot(serD['y'],label="signal")
        plt.plot(serD['smooth'],label="smooth")
        plt.plot(serD['diff'].abs(),label="abs derivative")
        plt.plot(serT['smooth'],label="smooth derivative")
        plt.plot(spiV,linewidth=3,label="spike")
        plt.legend()
        plt.show()

if False:
    plog('------------------resample-freq----------------')
    for f in ac2L:
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x","y","z"])
        
        
if False:
    plog('-------------categorize-spikes-from-spectrum---------------')
    feat = psPx.T.sort_values('id')
    spik = spL.sort_values('id')['x'] > 0
    X = feat.values
    y = spik.values * 1
    if False:
        feat.T.plot()
        plt.show()
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(10, input_shape=(20,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=500, batch_size=10,verbose=1)
    y_pred = np.array(model.predict(X) > 0.5)[:,0]
    print(sum( abs(y*1-y_pred*1) ) )

    from keras.utils import to_categorical
    y_binary = to_categorical(y)
    model = Sequential()
    model.add(Dense(10, input_shape=(20,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y_binary, epochs=500, batch_size=10)
    y_pred = np.array(model.predict(X) > 0.5)[:,1]
    print(sum( abs(y*1-y_pred*1) ) )
    
    import geomadi.train_modelList as t_m
    import geomadi.train_lib as t_l
    from sklearn.metrics import confusion_matrix
    tM = t_l.trainMod(X,y)
    mL = t_l.modelList()
    clf = t_m.binCla[1]
    mod = clf.fit(X,y)
    y_pred = mod.predict(X)
    print(sum( abs(y*1-y_pred*1) ) )
    cm = confusion_matrix(y,y_pred)
    cm = np.array(cm)
    print(cm)
    from sklearn.externals import joblib
    joblib.dump(clf,'out/svc_model.pkl', compress=9)
    
    if False:
        print("on diagonal %.2f" % (sum(cm.diagonal())/sum(sum(cm))) )
        plt.xlabel("prediction")
        plt.ylabel("score")
        plt.imshow(cm)
        plt.show()

if False:
    plog('-------------sensors-correlation-------------')
    senS = []
    for f in ac1L:
        idS = f.split("_")[2]
        sen = pd.read_csv(baseDir + "../iot/raw/"+f,names=["x","y","z"])
        stat = sen.mean()
        tmp = sen.std()
        tmp.index = ['s_x','s_y','s_z']
        stat = pd.concat([stat,tmp],axis=0)
        pV = {}
        for i in sen.columns:
            pV["p_25_"+i] = np.percentile(sen[i],25) 
            pV["p_75_"+i] = np.percentile(sen[i],75) 
        stat = pd.concat([stat,pd.Series(pV)],axis=0)
        stat = pd.DataFrame([stat])
        stat.index = [idS]
        senS.append(stat)
    senS = pd.concat(senS)
    corS = senS.T.corr()
    ax = sns.heatmap(corS, vmax=1, square=True,annot=False,cmap='RdYlGn')
    plt.show()

if False:
    plog('-------------sensor-relations---------------')
    import networkx as nx
    cor = corR
    G = nx.Graph()
    for i in cor.index:
        G.add_node(i)
    for i in cor.index:
        for j in cor.index:
            if abs(cor.loc[i,j]) > 0.8:
                if i !=j :
                    G.add_edge(i,j)

    nodS = cor.sum(axis=1)*10
    pos = nx.circular_layout(G)
    pos = nx.spectral_layout(G)
    pos = nx.spring_layout(G)
    labels = {}
    for i in cor.index:
        labels[i] = i
    nx.draw_networkx_edges(G,pos,width=.6,alpha=0.3)
    nx.draw_networkx_nodes(G,pos,node_color='g',node_size=nodS,alpha=0.3,with_labels=True)
    nx.draw_networkx_labels(G,pos,labels,font_size=18)
    plt.show()
    
if False:
    plog('----------sensors-in-space----------')
    fig, ax = plt.subplots(1,2)
    pivS = pd.melt(senS,id_vars=["id"],value_vars=["x","y","z"])
    pivS.boxplot(column="value",by="variable",ax=ax[0])
    pivS = pd.melt(senS,id_vars=["id"],value_vars=["s_x","s_y","s_z"])
    pivS.boxplot(column="value",by="variable",ax=ax[1])
    plt.show()
    
if False:
    plog('----------total-motion----------')
    xL = [x for x in acc2.columns if bool(re.search("x_",x))]
    yL = [x for x in acc2.columns if bool(re.search("y_",x))]
    zL = [x for x in acc2.columns if bool(re.search("z_",x))]
    ac2M = pd.DataFrame({"a_x":acc2[xL].mean(axis=1),"a_y":acc2[yL].mean(axis=1),"a_z":acc2[zL].mean(axis=1)})
    xL = [x for x in gyr2.columns if bool(re.search("x_",x))]
    yL = [x for x in gyr2.columns if bool(re.search("y_",x))]
    zL = [x for x in gyr2.columns if bool(re.search("z_",x))]
    gy2M = pd.DataFrame({"a_x":gyr2[xL].mean(axis=1),"a_y":gyr2[yL].mean(axis=1),"a_z":gyr2[zL].mean(axis=1)})
    print(acc1.shape,acc2.shape,gyr2.shape)
    
    for i in ['x','y','z']: # equation of motion
        ac2M.loc[:,"v_"+i] = 0
        ac2M.loc[:,i] = 0
        gy2M.loc[:,"v_"+i] = 0
        gy2M.loc[:,i] = 0
        for j in range(ac2M.shape[0]-1):
            ac2M.loc[j+1,"v_"+i] = ac2M.loc[j,'a_'+i] + ac2M.loc[j,'v_'+i]
            ac2M.loc[j+1,i] = ac2M.loc[j,'a_'+i]*.5 + ac2M.loc[j,'v_'+i] + ac2M.loc[j,i]
            gy2M.loc[j+1,"v_"+i] = gy2M.loc[j,'a_'+i] + gy2M.loc[j,'v_'+i]
            gy2M.loc[j+1,i] = gy2M.loc[j,'a_'+i]*.5 + gy2M.loc[j,'v_'+i] + gy2M.loc[j,i]

    gyM = gy2M.rolling(10).mean().iloc[::10]
    acM = ac2M.rolling(10).mean().iloc[::10]
            
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.quiver(acM['x'],acM['y'],acM['z'],gyM['x'],gyM['y'],gyM['z'],label='displacement')
    ax.legend()
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot(acM['v_x'],acM['v_y'],acM['v_z'], label='velocity')
    ax.legend()
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot(acM['a_x'],acM['a_y'],acM['a_z'], label='acceleration')
    ax.legend()
    plt.show()
    
    for i in acc1.columns[:20]:
        plt.plot(acc1[i],label=i)
    plt.legend()
    plt.show()

if False:
    plog('---------------unsupervised-segment-detection------------')
    l = random.choice(acc1.columns)
    y = acc1[l].values
    serD = s_l.serDecompose(y,period=14)
    yt = serD['smooth'].values[::3]
    segment_len = 16
    slide_len = 2
    segments = []
    for start_pos in range(0, len(yt), slide_len):
        end_pos = start_pos + segment_len
        segment = np.copy(yt[start_pos:end_pos])
        if len(segment) != segment_len:
            continue
        segments.append(segment)
    print(len(segments))
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads)**2
    windowed_segments = []
    for segment in segments:
        windowed_segment = np.copy(segment) * window
        windowed_segments.append(windowed_segment)

    from sklearn.cluster import KMeans
    clusterer = KMeans(copy_x=True,init='k-means++',max_iter=300,n_clusters=18,n_init=10,n_jobs=1,precompute_distances='auto',random_state=None,tol=0.0001,verbose=2)
    #clusterer.fit(windowed_segments)
    clusterer.fit(segments)
    centroids = clusterer.cluster_centers_
    nearest_centroid_idx = clusterer.predict(segments)[0]
    nearest_centroid = np.copy(centroids[nearest_centroid_idx])
    corr = pd.DataFrame(np.corrcoef(np.array(centroids)))
    ax = sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.show()

    y1 = segments[0]
    y2 = nearest_centroid
    plt.figure()
    plt.title("corr %.2f rmse %.2f" % (sp.stats.pearsonr(y1,y2)[0],np.sqrt(np.mean((y1-y2)**2))) )
    plt.plot(y1, label="Original segment")
    plt.plot(y2, label="Nearest centroid")
    plt.legend()
    plt.show()
    
    if False:
        disp = segments
        disp = windowed_segments
        disp = clusterer.cluster_centers_
        plt.figure()
        lL = random.sample(range(len(disp)),len(disp))
        for i in range(3):
            for j in range(6):
                axes = plt.subplot(3,6,6*i+j+1)
                l = lL[min(len(disp)-1,6*i+j)]
                plt.plot(disp[l],label="segment " + str(l))
                plt.legend()
        plt.tight_layout()
        plt.show()

if False:
    plog('---------plot-signal-decomposition-----------')
    import importlib
    importlib.reload(s_l)
    l = random.choice(acc1.columns)
    y = acc1[l]
    serD = s_l.serDecompose(y,period=14)

    fig, ax = plt.subplots(1)
    for i in serD.columns:
        ax.plot(serD[i],label=i)
    plt.legend()
    plt.show()
    
if False:
    plog('------------------permuter---------------')
    import string
    def isPermuted(w1,w2):
        cL = [c for c in string.ascii_lowercase]
        d1, d2 = {}, {}
        for c in cL:
            d1[c], d2[c] = 0, 0
        for w in w1.lower() :
            d1[w] = d1[w] + 1
        for w in w2.lower() :
            d2[w] = d2[w] + 1
        return d1 == d2

    print(isPermuted("qwerty", "wqeyrt") )
    print(isPermuted("aab","bba"))

if False:
    plog('------------------dictionary-learning----------------')
    from sklearn import datasets
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.image import extract_patches_2d
    faces = datasets.fetch_olivetti_faces()
    print('Learning the dictionary... ')
    rng = np.random.RandomState(0)
    kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True)
    patch_size = (20, 20)
    buffer = []
    t0 = time.time()
    index = 0
    for _ in range(6):
        for img in faces.images:
            data = extract_patches_2d(img, patch_size,max_patches=50,random_state=rng)
            data = np.reshape(data, (len(data), -1))
            buffer.append(data)
            index += 1
            if index % 10 == 0:
                data = np.concatenate(buffer, axis=0)
                data -= np.mean(data, axis=0)
                data /= np.std(data, axis=0)
                kmeans.partial_fit(data)
                buffer = []
            if index % 100 == 0:
                print('Partial fit of %4i out of %i' % (index, 6 * len(faces.images)))

    dt = time.time() - t0
    print('done in %.2fs.' % dt)
    plt.figure(figsize=(4.2, 4))
    for i, patch in enumerate(kmeans.cluster_centers_):
        plt.subplot(9, 9, i + 1)
        plt.imshow(patch.reshape(patch_size),cmap=plt.cm.gray,interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' % (dt, 8 * len(faces.images)), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()


    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
