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
