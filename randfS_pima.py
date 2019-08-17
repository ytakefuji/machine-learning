import pandas as pd
pima=pd.read_csv('pima-indians-diabetes.csv',encoding="shift-jis")
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']
from sklearn.model_selection import train_test_split
y = pima['diabetes']
X=pima.drop(['diabetes'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)

from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=90)
X_train,y_train= smt.fit_resample(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
clf=RandomForestClassifier(n_estimators=382, max_depth=None,min_samples_split=2,random_state=8)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("acc:",'%.4f'%clf.score(X_test,y_test))
print(confusion_matrix(y_test, y_pred))
print('f1:','%.4f'%f1_score(y_test, y_pred))
print('recall:','%.4f'%recall_score(y_test, y_pred))
print('precision:','%.4f'%precision_score(y_test, y_pred))
