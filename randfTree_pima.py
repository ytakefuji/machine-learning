import pandas as pd
pima=pd.read_csv('pima-indians-diabetes.csv',encoding="shift-jis")
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']
from sklearn.model_selection import train_test_split
y = pima['diabetes']
X=pima.drop(['diabetes'],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=382, max_depth=None,min_samples_split=2,random_state=8)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
tree=clf.estimators_[2]
from sklearn.tree import export_graphviz
# Export as dot file
import pydotplus
from io import StringIO
dotfile=StringIO()
export_graphviz(tree, out_file=dotfile) 
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png("tree.png")
from PIL import Image
image=Image.open('tree.png')
image.show()
