# 1 357
# 0 212

# 라벨 0을 112개 삭제 후 357개로 다시 SMOTE

# SMOTE 전 후 비교

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# pip install imblearn
from imblearn.over_sampling import SMOTE

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))
print(pd.Series(y).value_counts())
print(y)

'''
colist=[]
for i, ycont in enumerate(y):
    if ycont == 0:
        colist += [i]

print(colist)
y_new = np.delete(y,colist[:112],axis=0)
'''

ylist = np.where(y == 0)
ylist = ylist[0][:112]
y_new = np.delete(y,ylist,axis=0)
x_new = np.delete(x,ylist,axis=0)

print(np.unique(y, return_counts=True))

#============================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=134, stratify=y)


# 2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=55)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score: ', score)
print('acc_score: ', accuracy_score(y_pred, y_test))

from sklearn.metrics import f1_score
print('f1_macro: ', f1_score(y_pred, y_test))


print('==================== SMOTE 적용 후 ========================')
smote = SMOTE(random_state=1234)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.Series(y_train).value_counts())

# 2. 모델, 3. 훈련
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score: ', score)
print('acc_score: ', accuracy_score(y_pred, y_test))

from sklearn.metrics import f1_score
print('f1_macro: ', f1_score(y_pred, y_test))

# model.score:  0.9790209790209791
# acc_score:  0.9790209790209791
# f1_macro:  0.983050847457627
# ==================== SMOTE 적용 후 ========================
# 1    267
# 0    267
# dtype: int64
# model.score:  0.9790209790209791
# acc_score:  0.9790209790209791
# f1_macro:  0.983050847457627