# machine-learning
A Titanic problem is used to demonstrate how to use three important functions in machine learning including pandas, preprocessing, and train_test_split are detailed.
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
---------------------------


train_test_split is a function to split the dataset X (inputs) and y (output) into X_train,X_test,y_train,y_test respectively.
Two ensemble machine learning algorithms are introduced: random forest and extra trees.

--------------------------

There are two kinds of machine learning:classifier and regressor. Classifier deals with discrete numbers while regressor with continuous numbers. To run machine learning, type the following command:
<pre>
$ python ext_titanic.py
or
$ python randf_titanic.py
</pre>

You must install pandas and scikit-learn (sklearn) in your system. To install libraries, you can use conda or pip command.

In order to install conda, download one of the following files from the site:

https://conda.io/miniconda.html

Python 2.7 will retire in 2020. I have tested Python 2.7.
