#%pylab inline
##http://scikit-learn.org/stable/modules/ensemble.html
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
outFile = "raw/activity_train.csv"
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize

from sklearn import preprocessing
import train_modelList as tml
import sklearn as sk

plog('---------------------------decision-tree----------------------')

from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
import pydotplus
from sklearn import tree
import collections

sact = pd.read_csv(baseDir + 'out/activity_class'+outSuf+'.csv')
cL = [x for x in sact.columns if re.search("c_",x)]
tL = [x for x in sact.columns if re.search("t_",x)]

fig, ax = plt.subplots(1,1,figsize=(10,20))
X = sact[tL]
y = sact['z_corr']
clf = XGBClassifier()
clf = clf.fit(X,y.ravel())
ax = plot_tree(clf)#,rankdir='LR')
#plot_importance(clf,ax=ax)#,rankdir='LR')
fig = ax.figure
fig.set_size_inches(10, 20)
plt.show()

X = sact[tL]
y = sact['z_corr']
clf = sk.tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
dot_data = tree.export_graphviz(clf,feature_names=cL,filled=True,rounded=True,max_depth=2)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png(baseDir + 'fig/tree.png')

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
 

#mod = model[2].fit(X_train,y_train)
from xgboost import XGBClassifier
from xgboost import plot_tree
clf = XGBClassifier()
clf.fit(X, y)
# plot single tree
plot_tree(clf,num_trees=0, rankdir='LR')
plt.show()

import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X,y)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=False, rounded=False, special_characters=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())




dot_data = sk.tree.export_graphviz(clf,out_file=None,feature_names=tL,class_names=tL,filled=True,rounded=True,special_characters=True)
import pydotplus
import graphviz
from sklearn import tree

graph = graphviz.Source(dot_data, filename="test.gv", format="png")  
graph.view()



#joblib.dump(model,os.environ['LAV_DIR']+"/train/"+'lookAlike'+str(index)+'.pkl',compress=1)



if False:   
    X = (sact[tL]).transpose()
    X = (sact[cL]).transpose()
    y = (sact['y_reg'])
    #y = (sact['y_corr'])
    cX = np.corrcoef(X)
    pX = pdist(X)
    link = hc.linkage(X, method='centroid')
    o1 = hc.leaves_list(link)
    # mat = sact.iloc[o1,:]
    # mat = mat.iloc[:, sact[::-1]]
    c, coph_dists = cophenet(link, pdist(X))
    hc.dendrogram(link,color_threshold=1,show_leaf_counts=True,labels=[re.sub('c_','',x) for x in cL])
    plt.title("dendrogram continous values")
    plt.xlabel("classifiers")
    plt.ylabel("distance")
    #plt.imshow(mat)
    plt.show()
    
    cL1 = ['t_convex','t_cell_dist','t_median','t_type','t_slope']
    cL1 = ['t_sum','t_median','t_max','t_inter','t_slope']
    df = pd.DataFrame(sact[cL1])
    df.columns=[re.sub('t_','',x) for x in cL1]
    pd.plotting.scatter_matrix(df, diagonal="kde")
    plt.tight_layout()
    plt.show()
    
    sns.heatmap(cX, vmax=1., square=False).xaxis.tick_top()
    plt.show()
    
    cm = confusion_matrix(sact['z_reg'], sact['z_corr'])
    np.set_printoptions(precision=2)
    plt.figure
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar
    plt.tight_layout
    plt.ylabel('correlation')
    plt.xlabel('regression')
    plt.show()


