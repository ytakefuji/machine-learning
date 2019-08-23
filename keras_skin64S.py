import matplotlib.pyplot as plt,itertools
import pandas as pd,numpy as np,random as rn
rn.seed(90)
np.random.seed(90)
n=90.0
skin=pd.read_csv('hmnist_64_64_RGB.csv')
X=skin.drop(['label'],axis=1)
X=np.asarray(X)/n
X=np.subtract(X,255.0/n/2.0+0.5)
#X=np.multiply(X,2.0)
#X=X.reshape(-1,64,64,3)
X=X.reshape(-1,64*64*3)
y=skin['label']
c0=[];c1=[];c2=[];c3=[];c4=[];c5=[];c6=[]
for i in range(10015):
 if y[i]==0:c0.append(i)
 if y[i]==1:c1.append(i)
 if y[i]==2:c2.append(i)
 if y[i]==3:c3.append(i)
 if y[i]==4:c4.append(i)
 if y[i]==5:c5.append(i)
 if y[i]==6:c6.append(i)
from random import sample
c00=sample(c0,4)
c11=sample(c1,4)
c22=sample(c2,4)
c33=sample(c3,4)
c44=sample(c4,4)
c55=sample(c5,4)
c66=sample(c6,4)
Xtest=[];ytest=[]
for i in range(4):
 ytest.append(y[c00[i]])
 Xtest.append(X[c00[i]])
for i in range(4):
 ytest.append(y[c11[i]])
 Xtest.append(X[c11[i]])
for i in range(4):
 ytest.append(y[c22[i]])
 Xtest.append(X[c22[i]])
for i in range(4):
 ytest.append(y[c33[i]])
 Xtest.append(X[c33[i]])
for i in range(4):
 ytest.append(y[c44[i]])
 Xtest.append(X[c44[i]])
for i in range(4):
 ytest.append(y[c55[i]])
 Xtest.append(X[c55[i]])
for i in range(4):
 ytest.append(y[c66[i]])
 Xtest.append(X[c66[i]])

for i in range(4):
 X=np.delete(X,c00[i],axis=0)
 X=np.delete(X,c11[i],axis=0)
 X=np.delete(X,c22[i],axis=0)
 X=np.delete(X,c33[i],axis=0)
 X=np.delete(X,c44[i],axis=0)
 X=np.delete(X,c55[i],axis=0)
 X=np.delete(X,c66[i],axis=0)
 y=y.drop(c00[i])
 y=y.drop(c11[i])
 y=y.drop(c22[i])
 y=y.drop(c33[i])
 y=y.drop(c44[i])
 y=y.drop(c55[i])
 y=y.drop(c66[i])
print(X.shape,y.shape)
Xtest=np.array(Xtest)
ytest=np.array(ytest)
print(Xtest.shape,ytest.shape)
#y=np.asarray(y)
   
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=90)
#X_train=X_train.reshape(-1,64*64*3)
X_train,y_train= smt.fit_resample(X,y)
print(np.unique(ytest,return_counts=True))
X_train=X_train.reshape(-1,64,64,3)
Xtest=Xtest.reshape(-1,64,64,3)
print(X_train.shape,Xtest.shape)

import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Reshape, Conv2D, MaxPool2D, concatenate, Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras.callbacks import EarlyStopping as ES
K.floatx()
tf.set_random_seed(90)
keras.initializers.Initializer()

ytest = to_categorical(ytest, num_classes=7)
y_train = to_categorical(y_train, num_classes=7)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(7))
model.add(Activation(tf.nn.softmax))

model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',
metrics=['acc'])

from keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
es=TerminateOnBaseline(monitor='val_acc', baseline=0.97)

history = model.fit(
    X_train, y_train,
    epochs=99,
    batch_size =32,
    validation_data=(Xtest,ytest),
    callbacks=[es],
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

model.save('64S.h5')
y_pred=model.predict(Xtest)
y_pred_classes = np.argmax(y_pred,axis = 1) 
y_true = np.argmax(ytest,axis = 1) 
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
print("balanced=",balanced_accuracy_score(y_true, y_pred_classes))
plot_labels = ['akiec','bcc','bkl','df','nv','vasc','mel']
report = classification_report(y_true,y_pred_classes,target_names=plot_labels)
print(report)
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes = plot_labels) 
#plot_confusion_matrix(confusion_mtx, classes = range(7)) 
plt.show()
