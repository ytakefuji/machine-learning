import pandas as pd
pima=pd.read_csv('pima-indians-diabetes.csv',encoding="shift-jis")
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']
from sklearn.model_selection import train_test_split
y = pima['diabetes']
X=pima.drop(['diabetes'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=784, max_depth=None,min_samples_split=2,random_state=8)
clf2=AdaBoostClassifier(clf,n_estimators=500,learning_rate=1.0,random_state=7,algorithm='SAMME')
clf2.fit(X_train,y_train)
print(clf2.score(X_test,y_test))
dic=dict(zip(X.columns,clf2.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))
