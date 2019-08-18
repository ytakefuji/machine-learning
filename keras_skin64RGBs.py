import matplotlib.pyplot as plt,itertools
import pandas as pd,numpy as np,random as rn
skin=pd.read_csv('hmnist_64_64_RGB.csv')
X=skin.drop(['label'],axis=1)
X=np.asarray(X)/255.0
X=np.subtract(X,0.5)
X=np.multiply(X,2.0)
X=X.reshape(-1,64*64*3)
y=skin['label']
y=np.asarray(y)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=90)
X,y= smt.fit_resample(X,y)
#X=X.reshape(-1,64,64,3)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=90,shuffle=True)
print(x_train.shape,y_train.shape)
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=90)
x_train,y_train= smt.fit_resample(x_train,y_train)
x_train=x_train.reshape(-1,64,64,3)
x_test=x_test.reshape(-1,64,64,3)

import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Reshape, Conv2D, MaxPool2D, concatenate, Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential, load_model
from keras import backend as K
np.random.seed(90)
rn.seed(90)
K.floatx()
tf.set_random_seed(90)
keras.initializers.Initializer()

y_test = to_categorical(y_test, num_classes=7)
y_train = to_categorical(y_train, num_classes=7)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(800, activation='relu'))
model.add(Dense(7))
model.add(Activation(tf.nn.softmax))

model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',
metrics=['acc'])

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size =32,
    validation_data=(x_test,y_test),
    verbose=1)

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

model.save('64RGBs.h5')
y_pred=model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1) 
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
print("balanced=",balanced_accuracy_score(y_true, y_pred_classes))
plot_labels = ['akiec','bcc','bkl','df','nv','vasc','mel']
report = classification_report(y_true,y_pred_classes,target_names=plot_labels)
print(report)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = range(7)) 
plt.show()
