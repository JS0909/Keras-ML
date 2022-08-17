import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# pip install imblearn
from imblearn.over_sampling import SMOTE
import sklearn as sk
print(sk.__version__) # 1.1.2


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)
print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    48
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
#============================================================================================

x = x[:-30]
y = y[:-30]
print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=134, stratify=y)

print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    14

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
print('f1_macro: ', f1_score(y_pred, y_test, average='macro'))
print('f1_micro: ', f1_score(y_pred, y_test, average='micro'))

# model.score:  1.0
# acc_score:  1.0
# f1_macro:  1.0
# f1_micro:  1.0

# 데이터 축소 후 (2라벨을 30개 줄임)
# model.score:  0.9459459459459459
# acc_score:  0.9459459459459459
# f1_macro:  0.8713450292397661
# f1_micro:  0.9459459459459459




print('==================== SMOTE 적용 후 ========================')
smote = SMOTE(random_state=1234)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(pd.Series(y_train).value_counts())
# 1    53
# 0    53
# 2    53

# 2. 모델, 3. 훈련
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score: ', score)
print('acc_score: ', accuracy_score(y_pred, y_test))

from sklearn.metrics import f1_score
print('f1_macro: ', f1_score(y_pred, y_test, average='macro'))
print('f1_micro: ', f1_score(y_pred, y_test, average='micro'))

# model.score:  1.0
# acc_score:  1.0
# f1_macro:  1.0
# f1_micro:  1.0