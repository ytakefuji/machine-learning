import pandas as pd
pima=pd.read_csv('pima-indians-diabetes.csv',encoding="shift-jis")
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']
from sklearn.model_selection import train_test_split
y = pima['diabetes']
X=pima.drop(['diabetes'],axis=1)

import numpy as np
np.random.seed(90)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=90,shuffle=True)
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'n_jobs':-1,
    'bagging_freq': 5,
    'verbose': 0
}

clf = lgb.train(params,
                lgb_train,
                num_boost_round=80,
                valid_sets=lgb_eval,
                verbose_eval=1,
                early_stopping_rounds=5)
y_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
pred=[]
for i in y_pred:
 if i>0.5: pred.append(1)
 else:pred.append(0)
y_pred=pred
print(y_pred)
from sklearn.metrics import *
print("roc:%.4f"%roc_auc_score(y_test,y_pred))
print("acc:%.4f"%accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print("f1:%.4f"%f1_score(y_test, y_pred))
print('recall:','%.4f'%recall_score(y_test, y_pred))
print('precision:','%.4f'%precision_score(y_test, y_pred))
