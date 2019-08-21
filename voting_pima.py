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
import xgboost as xgb
xg=xgb.XGBClassifier(n_estimators=382,random_state=8)
ext=ExtraTreesClassifier(n_estimators=382,min_samples_split=2,random_state=8)
rf=RandomForestClassifier(n_estimators=382, max_depth=None,min_samples_split=2,random_state=8)
st=StackingClassifier(classifiers=[ext],meta_classifier=rf)
from sklearn.ensemble import VotingClassifier
#clf=VotingClassifier(estimators=[('rf', rf),('xg',xg),('ext',ext),('st',st)],voting='soft',weights=[4,1,4,1])
clf=VotingClassifier(estimators=[('rf', rf),('xg',xg),('ext',ext),('st',st)],voting='hard')
clf.fit(X_train,y_train)
print('%.5f'%clf.score(X_test,y_test))
