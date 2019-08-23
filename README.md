# open source machine learning in medicine

Open Source Machine Learning in Medicine is available from Amazon:

https://www.amazon.com/Open-Source-Machine-Learning-Medicine-ebook/dp/B07WH9H6HM/

------------------------------------------
------------------------------------------
A Titanic problem is used to demonstrate how to use three important functions in machine learning where pandas, preprocessing, and train_test_split are detailed.
The goal of machine learning is to create input-output function f: y=f(X) where y and X are output and inputs respectively. We would like to predict y values using X values by forming function f.

----------------------------
titanic.csv is a dataset with 13 parameters (row.names,pclass,survived,name,age,embarked,home.dest,room,ticket,boat,sex). "survive" is the output y to be predicted and 12 parameters are inputs X. pandas is a library to import titanic.csv data in Python:
<pre>
import pandas as pd
titanic=pd.read_csv('titanic.csv',encoding="shift-jis")
</pre>

Machine learning algoritms can take care of only numbers. Therefore, if the dataset contains non-numeric value (string), all strings must be converted into numbers by the followinig three lines:
<pre>
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
titanic=titanic.apply(le.fit_transform)
</pre>
The following two lines fills empty data in age with mean.
<pre>
mean=titanic['age'].mean()
titanic['age'].fillna(mean,inplace=True)
</pre>
---------------------------

train_test_split is a function to split the dataset X (inputs) and y (output) into X_train,X_test,y_train,y_test respectively. This example shows 0.2 for testing data, therefore 0.8 for training data. shuffle=True means shuffling data.
<pre>
titanic_target = titanic['survived']
titanic_data=titanic.drop(['survived'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(titanic_data,titanic_target,test_size=0.2,random_state=54,shuffle=True)
</pre>
--------------------------

There are two kinds of machine learning:classifier and regressor. Classifier deals with discrete numbers while regressor with continuous numbers. Two ensemble machine learning algorithms are introduced: randomforest and extratrees. To run machine learning, type the following command:
<pre>
$ python ext_titanic.py
or
$ python randf_titanic.py
</pre>

The following three lines show feature of importances in more important order.
<pre>
dic=dict(zip(titanic_data.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
 print(item[0],round(item[1],4))
</pre>

You must install pandas and scikit-learn (sklearn) in your system. To install libraries, you can use conda or pip command.

In order to install conda, download one of the following files from the site:

https://conda.io/miniconda.html

Python 2.7 will retire in 2020. I have tested Python 2.7.

----------------------------------------------------------------------
----------------------------------------------------------------------
pima-indians-diabetes.csv describes the medical records for Pima Indians.
<pre>
randf_pima.py: RandomForestClassifier
randfS_pima.py: RandomForestClassifier with SMOTE
ext_pima.py: ExtraTreesClassifier
elm_pima.py: GenELMClassifier
adaboost_pima.py: AdaBoostClassifier with RandomForestClassifier
lgbm_pima.py: LightGBM
stack_pima.py: Stacking RandomForestClassifier with ExtraTreesClassifier
randfTree_pima.py: explainable decision trees
voting_pima.py: Voting classifier
</pre>
-----------------------------------------------------------------------
Skin cancer using HAM10000

There are seven skin cancers:
<pre>
cancer	no. of images
nv          6705
mel	    1113
bkl	    1099
bcc	    514
akiec	    327
vasc	    142
df	    115
</pre>

In order to make 64S.h5, download 64S.00 and 64S.01, and 

$ cat 64S.0* >64S.h5

<pre>
skin64S_val.py: Using 64S.h5 (saved model), this can generate the same result of keras_sking64RGBs.py
keras_skin64RGB.py: Using hmnist_64_64_RGB.csv, this keras model can classify given images into one of seven skin cancers.
keras_skin64RGBs.py: Using hmnist_64_64_RGB.csv, keras model with SMOTE method can classify given images into one of seven skin cancers.
keras_skin64S.py: This can classify 28 images into one of seven skin cancers:
64S.01: cat 64S.0* >64S.h5
64S.00: 
</pre>
