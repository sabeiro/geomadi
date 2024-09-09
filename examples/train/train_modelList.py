"""
train_modelList:
collection of models for performance comparison
"""
import json
import sklearn as sk
import sklearn.ensemble
import sklearn.tree
import sklearn.neural_network
import sklearn.svm
import sklearn.gaussian_process
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.discriminant_analysis
import sklearn.dummy
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeRegressor
import sklearn as sk
import sklearn.discriminant_analysis
import sklearn.dummy
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.metrics as skm
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.svm
import sklearn.tree
import statsmodels.formula.api as sm

class modelList():
    """list of sklearn models to iterate"""
    catCla = [
        {"active":True,"name":"random_forest"  ,"type":"class","score":"stack","mod":sk.ensemble.RandomForestClassifier()}
        ,{"active":True,"name":"decision_tree" ,"type":"class","score":"stack","mod":sk.tree.DecisionTreeClassifier()}
        ,{"active":True,"name":"extra_tree"    ,"type":"class","score":"stack","mod":sk.ensemble.ExtraTreesClassifier()}
        ,{"active":True,"name":"perceptron"    ,"type":"class","score":"ravel","mod":sk.neural_network.MLPClassifier()}
        ,{"active":True,"name":"k_neighbors"   ,"type":"class","score":"stack","mod":sk.neighbors.KNeighborsClassifier()}
        ,{"active":True,"name":"grad_boost"    ,"type":"logit","score":"ravel","mod":sk.ensemble.GradientBoostingClassifier()}
        ,{"active":False,"name":"support_vector","type":"logit","score":"ravel","mod":sk.svm.SVC()}
        ,{"active":True,"name":"discriminant"  ,"type":"logit","score":"ravel","mod":sk.discriminant_analysis.LinearDiscriminantAnalysis()}
        ,{"active":True,"name":"logit_reg"     ,"type":"logit","score":"ravel","mod":sk.linear_model.LogisticRegression()}
        #,sk.dummy.DummyClassifier(strategy='stratified',random_state=10)
#        ,{"active":True,"name":"keras"         ,"type":"logit","score":"ravel","mod":modKeras()}
    ]

    regL = {
        "decTree":{"active":True,"name":"decision tree reg","type":"class","score":"stack","mod":DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=None, splitter='best')}
        ,"bagReg":{"active":True,"name":"bagging regressor","type":"class","score":"stack","mod":BaggingRegressor(base_estimator=DecisionTreeRegressor(),bootstrap=True, bootstrap_features=False, max_features=1.0,max_samples=1.0, n_estimators=10, n_jobs=None, oob_score=False,random_state=None, verbose=0, warm_start=False)}
        ,"lasso":{"active":True,"name":"lasso","type":"class","score":"stack","mod":linear_model.Lasso(alpha=0.05,max_iter=1000,normalize=False,positive=True,precompute=True,random_state=None,selection='cyclic',tol=0.0001,warm_start=True)}
        ,"linear":{"active":True,"name":"linear","type":"class","score":"stack","mod":linear_model.LinearRegression(copy_X=True, fit_intercept=False,n_jobs=None,normalize=False)}
        ,"elastic":{"active":True,"name":"linear","type":"class","score":"stack","mod":linear_model.ElasticNet(alpha=1.0, copy_X=True, fit_intercept=False, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, precompute=True,random_state=None, selection='cyclic', tol=1e-4, warm_start=False)}
        ,"elastic_cv":{"active":True,"name":"linear","type":"class","score":"stack","mod":linear_model.ElasticNetCV(alphas=None, copy_X=True, cv=5, eps=0.001,fit_intercept=False, l1_ratio=0.3, max_iter=2000, n_alphas=200,n_jobs=None, normalize=False, positive=False, precompute='auto',random_state=None,selection='cyclic',tol=0.0001,verbose=0)}
        ,"perceptron_time":{"active":True,"name":"linear","type":"class","score":"stack","mod":sklearn.neural_network.MLPRegressor(activation='relu', alpha=1e+6, batch_size='auto', beta_1=0.85,beta_2=0.999, early_stopping=False, epsilon=1e-09,hidden_layer_sizes=(7,7,), learning_rate='constant',learning_rate_init=0.09, max_iter=250, momentum=0.9,n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,random_state=None, shuffle=True, solver='lbfgs', tol=1e-4,validation_fraction=0.15, verbose=False, warm_start=False)}
        ,"perceptron":{"active":True,"name":"linear","type":"class","score":"stack","mod":sklearn.neural_network.MLPRegressor(activation='relu',learning_rate='constant',solver='lbfgs',epsilon=1e-09)}
    }
    gridL = {
        "lasso":{
            'alpha':[0.1,0.5,1.,1.5,2.]
            #,"fit_intercept":[False,True]
            ,"selection":['random','cyclic']
            ,"tol":[0.1,0.001,0.0001,0.00001,0.000001]
            #,"warm_start":[False,True]
        }
        ,"elastic":{
            "alpha":[0.001,0.01,0.1,1.]
            ,"l1_ratio":[0.0,0.5,1.0]
            #"fit_intercept":[False,True]
            #,"selection":['random','cyclic']
            #,"positive":[True,False]
            ,"tol":[1e-3,1e-4,1e-5,1e-6]
            #,"warm_start":[False,True]
        }
        ,"decTree":{'criterion':["mse","friedman_mse","mae"],
                    'max_depth': [10,20,30,40,None],
                    'max_features': ['auto','sqrt'],
                    'min_samples_leaf': [1,2,3],
                    'min_samples_split': [2,3,4,5,10]
        }
        ,"bagging":{'n_estimators':[10,20,30,40,50]}
        ,"perceptron":{
            #"activation":["relu",'logistic','tanh']
            #,"learning_rate":['constant','invscaling','adaptive']
            #"solver":['lbfgs','sgd','adam']
            #,"alpha":[1e+6,1e+7,1e+8]
            #,"tol":[1e-4,1e-3,1e-2]
            #,"learning_rate_init":[0.09,0.1,0.11]
            #,"validation_fraction":[0.125,0.15,0.175]
            #"max_iter":[250,270,290]
            #,"epsilon":[1e-9,1e-8,1e-7] #adam
            "beta_1":[0.75,0.8,0.85,0.9] # adam
            #,"beta_2":[0.99,0.95,0.96,0.97] # adam
            #,"momentum":[0.6,0.7,0.8,0.9] #sgd
            ,"hidden_layer_sizes":[(8,),(8,1,),(8,16,8,)]
        }
    }

    othCla = [
        sk.svm.SVR()
        ,sk.gaussian_process.GaussianProcessClassifier()
        ,sk.naive_bayes.GaussianNB()
        ,sk.discriminant_analysis.QuadraticDiscriminantAnalysis()
        ,sk.linear_model.LinearRegression()
        ,sk.ensemble.AdaBoostClassifier()
    ]

    def __init__(self,paramF="train.json"):
        self.paramF = paramF
        return

    def nCat(self):
        return len(modelList.catCla)

    def retCat(self,n):
        return modelList.catCla[n]

    def get_params(self):
        params = []
        for mod in modelList.catCla:
            params.append(mod['mod'].get_params())
        with open(self.paramF,'w') as f:
            f.write(json.dumps(params))
        return params

    def set_params(self):
        with open(self.paramF) as f:
            params = json.load(f)
        for i,clf in enumerate(modelList.catCla):
            mod = clf['mod']
            mod.set_params(**params[i])
        return params

    def get_model_name(self):
        return [x['name'] for x in modelList.catCla]


tuneL = {'loss': 'deviance', 'max_features': 'sqrt', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100}
tuneL = {'loss': 'deviance', 'max_features': 'sqrt', 'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 50}
tuneL = {'loss': 'deviance', 'max_features': 'auto', 'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 200}
tuneL = {'loss': 'deviance', 'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': 3, 'learning_rate': 0.1}
catCla = [
    ##random forest
    sk.ensemble.RandomForestClassifier(n_estimators=tuneL['n_estimators'],criterion='entropy',max_features=tuneL['max_features'],max_depth=5,bootstrap=True,oob_score=True,n_jobs=12,random_state=33)
    ##random forest 2
    ,sk.ensemble.RandomForestClassifier(n_estimators=tuneL['n_estimators'],criterion='gini',n_jobs=12,max_depth=15,max_features=tuneL['max_features'],min_samples_split=2,random_state=None)
    ##decision tree
    ,sk.tree.DecisionTreeClassifier(criterion="gini",random_state=tuneL['n_estimators'],max_depth=10,min_samples_leaf=5)
    ##extra tree
    ,sk.ensemble.ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',max_depth=None, max_features='auto', max_leaf_nodes=None,min_impurity_decrease=1e-07, min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0,n_estimators=250,n_jobs=1,oob_score=False,random_state=0,verbose=0,warm_start=False)
    ##neural network
    ,sk.neural_network.MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto', beta_1=0.9,beta_2=0.999, early_stopping=False, epsilon=1e-08,hidden_layer_sizes=(tuneL['n_estimators'],), learning_rate='constant',learning_rate_init=0.001, max_iter=200, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=None,shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,verbose=False, warm_start=False)
    ##k-neighbors
    ,sk.neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=3, p=2,weights='uniform')
]

binCla = [
    ##dummy variables
    #sk.dummy.DummyClassifier(strategy='stratified',random_state=10)
    ##gradient boost
    sk.ensemble.GradientBoostingClassifier(criterion='friedman_mse',init=None,learning_rate=tuneL['learning_rate'], loss=tuneL['loss'], max_depth=tuneL['max_depth'],max_features=tuneL['max_features'], max_leaf_nodes=None,min_impurity_decrease=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=tuneL['n_estimators'], presort='auto', random_state=10,subsample=1.0, verbose=0, warm_start=False)
    ##support vector
    ,sk.svm.SVC(C=1.0,cache_size=200,class_weight=None,coef0=0.0,decision_function_shape=None,degree=3,gamma='auto',kernel='rbf',max_iter=-1,probability=True,random_state=0,shrinking=True,tol=0.001,verbose=False)
    ,sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)
    ##logistic regression
    ,sk.linear_model.LogisticRegression(C=100.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,max_iter=tuneL['n_estimators'], multi_class='ovr',n_jobs=12,penalty='l2',random_state=None,solver='liblinear',tol=0.0001,verbose=0,warm_start=False)
    ]

othCla = [
    ##support vector
    sk.svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    ##gaussian process
    ,sk.gaussian_process.GaussianProcessClassifier(copy_X_train=True,kernel=1**2 * RBF(length_scale=1), max_iter_predict=tuneL['n_estimators'],multi_class='one_vs_rest', n_jobs=1, n_restarts_optimizer=0,optimizer='fmin_l_bfgs_b', random_state=None, warm_start=True)
    ##naive bayesias
    ,sk.naive_bayes.GaussianNB(priors=None)
    ##quadratic discriminant
    ,sk.discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,store_covariances=False, tol=0.0001)
    ,sk.linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    ##ada boost
    ,sk.ensemble.AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,learning_rate=1.0, n_estimators=50, random_state=None)
    ]


def nCat():
    return len(catCla)

def retCat(n):
    return catCla[n]

def nBin():
    return len(binCla)

def retBin(n):
    return binCla[n]
