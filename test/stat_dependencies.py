import numpy as np
from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression
from sklearn.linear_model           import LogisticRegression
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB
from sklearn.svm                    import SVC
import seaborn as sns
from scipy.stats.stats import pearsonr

sact = pd.read_csv(baseDir + "raw/activity_clust.csv")
hL = [x for x in sact.columns if re.search("c_",x)]
yL = [x for x in sact.columns if re.search("y_",x)]
tL = ['cell_dist','type','max', 'std', 'median', 'sum','f1', 'f2', 'f3']

X = np.array(sact[tL]).transpose()
X = np.array(sact[hL]).transpose()
y = np.array(sact['y_corr'])

cX = np.corrcoef(X)
pX = pdist(X)
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


o1 = hc.leaves_list(link)
mat = d.iloc[o1,:]
mat = mat.iloc[:, o1[::-1]]

link = hc.linkage(X, method='centroid')
c, coph_dists = cophenet(link, pdist(X))
hc.dendrogram(link,color_threshold=1,show_leaf_counts=True,labels=tL)
plt.title("dendrogram continous values")
plt.xlabel("classifiers")
plt.ylabel("distance")
#plt.imshow(mat)
plt.show()




pd.tools.plotting.scatter_matrix(sact[hL], diagonal="kde")
plt.tight_layout()
plt.show()
sns.heatmap(cX, vmax=1., square=False).xaxis.tick_top()
plt.show()

trainingData    = np.array([ [2.3, 4.3, 2.5],  [1.3, 5.2, 5.2],  [3.3, 2.9, 0.8],  [3.1, 4.3, 4.0]  ])
trainingScores  = np.array( [3.4, 7.5, 4.5, 1.6] )
predictionData  = np.array([ [2.5, 2.4, 2.7],  [2.7, 3.2, 1.2] ])

cflL = [LinearRegression(),svm.SVR(),LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(),LinearDiscriminantAnalysis(),GaussianNB(),SVC()]

cfl = cflL[0]
clf.fit(trainingData, trainingScores)
print("SVC")
print(clf.predict(predictionData))
