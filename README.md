# open source machine learning in medicine

Open Source Machine Learning in Medicine is available from Amazon:

https://www.amazon.com/Open-Source-Machine-Learning-Medicine-ebook/dp/B07WH9H6HM/

The first example (pima-indians-diabetes problem) using random forest ensemble algorithm is used to explain train_test_split function, and other important functions. The diabetes dataset includes data from 768 women with 9 parameters:
The second example is to diagnose skin cancer using image data. Skin cancer dataset was released by Harvard University.
Using HAM10000, this book deals with skin cancer classification problem where skin cancer images can be classified into one of seven skin cancers. Data augmentation plays a key role in machine learning. SMOTE method is introduced in this book.

------------------------------------------
# pima-indians-diabetes
pima-indians-diabetes.csv describes the medical records for Pima Indians.
<pre>
pima-indians-diabetes.csv: pima indians diabetes dataset
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
# Skin cancer using HAM10000

There are seven skin cancers in 10015 images:
<pre>
f['label'].value_counts()

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
keras_skin64S.py: This can classify 28 test images into one of seven skin cancers:
64S.01: data for 64S.h5
64S.00: data for 64S.h5
</pre>

https://www.kaggleusercontent.com/kf/5823361/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..o2RwehDj-1Qim6OODiegDg.Y3xOs4RI-eMDjAPaPW9ZVcsnyCdRdiwbeuM54AO-5ncw3CRi9g9ISUG8HHu7y6BaaWO2EdXiBGjEUwRaXN94ist_2Wa084IqGbNISaYXTNvibiA29OUz03CbAeFMa8kK-dtjh7xEVilR8eT7L4yBt65q90B0tNGG60NJncK3mBo9zFh3nK_IrSoDm5TM7UekibMG4zX6Zseb1coVFhEhCRzWq3DUMtAY5CWZoNZaQvKpWNJO1c0xs8prrEX47Wgy9XG1JlPn8THRs7-ZZrZgF0gw_hbcHzBOQbxbXZ9XPQwa4N6H4hh9H-jM_t3mepq-b9UfOim6nS93_DbzjqDAmWOVaeE3HiBjMVLmJdVdyBI1gZi5gbzsLFrWgmMvvL_27mFobitDHu_aXedJ1U0bS5KulEu5tL0I1c-2FV0VIuEZjrOzXBkcM-k2ecyK_x_xNamyM61uRr6LSAqhgeRk4FKjiigmLjWOQ2A94ifVsaaToXqf9DgopIeDgZhVXbSoycPCFvt1-IqYmNrNF53_tr58AR7pmkHo0ix8W0VchVv_ePpIuTpZlAaTNN56_L_1wyxtxDuMPy4JdV7GSK2zQ5UyEe2Mz4VE0w8FObuaYoT4kzbGL7K3FZoyowp5peAWwrCIKS_v5-rYKTAvmFKVsLZRP5YUzWR8KlT-GbGt4kc.EHDfJnqde0rthFZBHomsMw/hmnist_64_64_RGB.csv
