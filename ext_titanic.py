import pandas as pd
titanic=pd.read_csv('titanic.csv',encoding="shift-jis")

from sklearn.feature_extraction import DictVectorizer
def one_hot_dataframe(data, cols, replace=False):
    vec = DictVectorizer(sparse=False)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].T.to_dict().values()))
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data)

titanic=one_hot_dataframe(titanic,['pclass','embarked','sex','home.dest','room','ticket','boat'],replace=True)
mean=titanic['age'].mean()
titanic['age'].fillna(mean,inplace=True)
titanic.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split
titanic_target = titanic['survived']
titanic_data=titanic.drop(['name','row.names','survived'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(titanic_data,titanic_target,test_size=0.3,random_state=54,shuffle=True)
from sklearn.ensemble import ExtraTreesClassifier
clf=ExtraTreesClassifier(n_estimators=382, max_depth=None,min_samples_split=2,random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print (metrics.accuracy_score(y_test,y_pred))
print(clf.score(X_test,y_test))
