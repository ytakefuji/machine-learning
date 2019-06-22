import pandas as pd
pima=pd.read_csv('pima-indians-diabetes.csv',encoding="shift-jis")
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']
from sklearn.model_selection import train_test_split
y = pima['diabetes']
X=pima.drop(['diabetes'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
'''
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
import sklearn.ensemble.GradientBoostingClassifier
import xgboost as xgb
gb=GradientBoostingClassifier(n_estimators=682)
bg=BaggingClassifier(n_estimators=682)
xg=xgb.XGBClassifier(n_estimators=682)
iso=IsolationForest(n_estimators=682)
ada=AdaBoostClassifier(xg,n_estimators=682,algorithm='SAMME.R')
'''
ext=ExtraTreesClassifier(n_estimators=682,min_samples_split=2,random_state=8)
rf=RandomForestClassifier(n_estimators=682, max_depth=None,min_samples_split=2,random_state=8)
clf=StackingClassifier(classifiers=[ext],meta_classifier=rf)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
