# machine-learning
A Titanic problem is used to demonstrate machine learning where three important functions including pandas, one_hot_dataframe, and train_test_split are detailed.
The goal of machine learning is to create input-output function f: y=f(X) where y and X are output and inputs respectively.

titanic.csv is a dataset with 13 parameters (row.names,pclass,survived,name,age,embarked,home.dest,room,ticket,boat,sex). "survive" is the output y to be predicted and 12 parameters are inputs X.
Machine learning algoritms can take care of only numbers.
Therefore, if the dataset contains non-numeric value (string), all strings must be converted into numbers.

one_hot_dataframe is a function to convert non-numeric values (strings) to numbers (integers).
In order to train/test the dataset using machine learning, the dataset must be splited into train dataset and test dataset.

train_test_split is a function to split the dataset X (inputs) and y (output) into X_train,X_test,y_train,y_test respectively.
Two ensemble machine learning algorithms are introduced: random forest and extra trees.

To run machine learning, 

python ext_titanic.py

or

python randf_titanic.py

You must install pandas and scikit-learn (sklearn) in your system. To install libraries, you can use conda or pip command.

In order to install conda, download one of the following files from the site:

https://conda.io/miniconda.html

Python 2.7 will retire in 2020. I have tested Python 2.7.
