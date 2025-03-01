‎!pip install pycox‎
‎!pip install scikit-survival‎
‎!pip install lifelines‎

‎#Imports‎
‎from google.colab import files‎


‎import pandas as pd‎
‎import numpy as np‎
‎pd.set_option('display.max_columns'‎, ‎None)‎
‎from google.colab import drive‎
‎import matplotlib.image as mpimg‎
‎import matplotlib.pyplot as plt‎
‎from sklearn import preprocessing‎
‎from datetime import datetime‎
‎from sklearn.preprocessing import Normalizer‎
‎from sklearn.preprocessing import StandardScaler‎
‎from sklearn.preprocessing import MinMaxScaler‎
‎from sklearn.preprocessing import RobustScaler‎
‎from sklearn.preprocessing import MaxAbsScaler‎
‎from sklearn.model_selection import train_test_split‎
‎from pycox.evaluation import EvalSurv‎
‎import torchtuples as tt‎
‎from sklearn.metrics import roc_curve‎, ‎auc‎
‎from sksurv.metrics import concordance_index_censored‎
‎from sksurv.kernels import clinical_kernel‎
‎from sksurv.svm import FastSurvivalSVM‎
‎from sklearn.model_selection import ShuffleSplit‎, ‎GridSearchCV‎
‎from sksurv.util import Surv‎
‎from sksurv.ensemble import RandomSurvivalForest‎
‎from sksurv.svm import FastKernelSurvivalSVM‎
‎from pandas.core.algorithms import diff‎
‎from sksurv.linear_model import IPCRidge‎
‎from sksurv.ensemble import GradientBoostingSurvivalAnalysis‎
‎import pandas as pd‎
‎%matplotlib inline‎
‎from sklearn import set_config‎
‎from sklearn.model_selection import GridSearchCV‎, ‎KFold‎
‎from sklearn.pipeline import make_pipeline‎
‎set_config(display="text")  # displays text representation of estimators‎

‎#FastSurvivalSVM‎
‎random_state = 20‎


‎# Separate input features and target‎
‎y = df1[['status','time']]‎
‎X = df1.drop(['status','time']‎, ‎axis=1)‎


‎# setting up testing and training sets‎
‎train_x‎, ‎test_x‎, ‎train_y‎, ‎test_y = train_test_split(X‎, ‎y‎, ‎test_size=0.25‎, ‎random_state=27)‎


‎estimator = FastSurvivalSVM(max_iter=10‎, ‎tol=1e-5‎, ‎random_state=0)‎


‎x = test_x.to_numpy()‎
‎y=Surv.from_arrays(np.array(test_y['status'])‎, ‎np.array(test_y['time'])‎,
                   ‎name_event="status",name_time ="time")‎


‎estimator = FastSurvivalSVM(max_iter=10‎, ‎tol=1e-2‎, ‎random_state=0)‎


‎def score_survival_model(model‎, ‎X‎, ‎y)‎:
    ‎prediction = model.predict(X)‎
    ‎result = concordance_index_censored(y['status']‎, ‎y['time']‎, ‎prediction)‎
    ‎return result[0]‎


‎p = train_x.shape[1]‎
‎param_grid = {'alpha'‎: ‎2‎. ‎** np.arange(0‎, ‎1‎, .‎2)}‎
‎cv = ShuffleSplit(n_splits=p‎, ‎test_size=0.2‎, ‎random_state=11)‎
‎gcv = GridSearchCV(estimator‎, ‎param_grid‎, ‎scoring=score_survival_model‎,
                   ‎n_jobs=4‎, ‎refit=False‎,
                   ‎cv=cv)‎


‎import warnings‎
‎warnings.filterwarnings("ignore"‎, ‎category=FutureWarning)‎
‎gcv = gcv.fit(x,y)‎
‎round(gcv.best_score_‎, ‎7)‎, ‎gcv.best_params_‎

‎#RandomSurvivalForest‎
‎x=df1.drop(['status','time']‎, ‎axis=1)‎
‎X = x.to_numpy()‎
‎y=Surv.from_arrays(np.array(df1['status'])‎, ‎np.array(df1['time'])‎,
                   ‎name_event="status",name_time ="time")‎
‎random_state = 55‎


‎X_train‎, ‎X_test‎, ‎y_train‎, ‎y_test = train_test_split(‎
    ‎X‎, ‎y‎, ‎test_size=0.2‎, ‎random_state=random_state)‎


‎rsf = RandomSurvivalForest(n_estimators=10‎,
                           ‎min_samples_split=5‎,
                           ‎min_samples_leaf=5‎,
                           ‎max_features=None‎,
                           ‎n_jobs=-1‎,
                           ‎random_state=random_state)‎
‎rsf.fit(X_train‎, ‎y_train)‎
‎rsf.score(X_test‎, ‎y_test)‎

‎#FastKernelSurvivalSVM‎
‎def score_survival_model(model‎, ‎X‎, ‎y)‎:
    ‎prediction = model.predict(X)‎
    ‎result = concordance_index_censored(y['status']‎, ‎y['time']‎, ‎prediction)‎
    ‎return result[0]‎
‎x=df1.drop(['status','time']‎, ‎axis=1)‎
‎X = x.to_numpy()‎
‎y=Surv.from_arrays(np.array(df1['status'])‎, ‎np.array(df1['time'])‎,
                    ‎name_event="status",name_time ="time")‎


‎kernel_matrix = clinical_kernel(x)‎
‎cv = ShuffleSplit(n_splits=100‎, ‎test_size=0.3‎, ‎random_state=0)‎
‎param_grid = {"alpha"‎: ‎1.0 ** np.arange(-12‎, ‎13‎, ‎2)}‎
‎kssvm = FastKernelSurvivalSVM(optimizer="rbtree"‎, ‎kernel="linear"‎, ‎random_state=0)‎
‎kgcv = GridSearchCV(kssvm‎, ‎param_grid‎, ‎scoring=score_survival_model‎,
                     ‎n_jobs=1‎, ‎refit=False‎, ‎cv=cv)‎


‎kgcv = kgcv.fit(kernel_matrix‎, ‎y)‎
‎round(kgcv.best_score_‎, ‎3)‎, ‎kgcv.best_params_‎

‎#IPCRidge‎
‎ipc = IPCRidge(fit_intercept=True‎, ‎random_state=0)‎


‎param_grid = {'alpha'‎: ‎1‎. ‎** np.arange(0‎, ‎1‎, .‎2)}‎
‎cv = ShuffleSplit(n_splits=70‎, ‎test_size=0.5‎, ‎random_state=0)‎
‎ipcv = GridSearchCV(ipc‎, ‎param_grid‎, ‎scoring=score_survival_model‎,
                     ‎n_jobs=-1‎, ‎refit=False‎, ‎cv=cv)‎
‎ipc = IPCRidge(fit_intercept=True‎, ‎random_state=0)‎


‎ipc.fit(x,y)‎
‎prediction = ipc.predict(x)‎
‎result = concordance_index_censored(y['status']‎, ‎y['time']‎, ‎prediction)‎
‎result[0]‎


‎#GradientBoostingSurvivalAnalysis‎


‎X_train‎, ‎X_test‎, ‎y_train‎, ‎y_test = train_test_split(x‎, ‎y‎, ‎test_size=0.2‎, ‎random_state=0)‎


‎est_cph_tree = GradientBoostingSurvivalAnalysis(‎
    ‎n_estimators=100‎, ‎learning_rate=1.0‎, ‎max_depth=1‎, ‎random_state=0‎
)
‎est_cph_tree.fit(X_train‎, ‎y_train)‎

‎cindex = est_cph_tree.score(X_test‎, ‎y_test)‎
‎print(round(cindex‎, ‎7))‎


‎#ComponentwiseGradientBoostingSurvivalAnalysis‎
‎from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis‎
‎estimator = ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph")‎
                                               ‎.fit(X_train,‎ ‎y_train)‎
‎cindex = estimator.score(X_test‎, ‎y_test)‎
‎print(round(cindex‎, ‎7))‎

‎#Coxnet‎
‎alphas = 10.0 ** np.linspace(-1‎, ‎1‎, ‎50)‎
‎coefficients =‎ {}


‎cph = CoxPHSurvivalAnalysis()‎
‎for alpha in alphas‎:
    ‎cph.set_params(alpha=alpha)‎
    ‎cph.fit(x,y)‎
    ‎key = round(alpha‎, ‎5)‎
    ‎coefficients[key] = cph.coef_‎


‎coefficients = pd.DataFrame.from_dict(coefficients).rename_axis(index="feature"‎, 
                                        ‎columns="alpha").set_index(x.columns)‎
‎cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0‎, ‎alpha_min_ratio=0.1)‎
‎cox_lasso.fit(X_train‎, ‎y_train)‎
‎cox_lasso.score(X_test‎, ‎y_test)‎
