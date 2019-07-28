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

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=80,
                valid_sets=lgb_eval,
                verbose_eval=1,
                early_stopping_rounds=5)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
from sklearn.metrics import roc_auc_score
print("%.4f"%roc_auc_score(y_test,y_pred))
