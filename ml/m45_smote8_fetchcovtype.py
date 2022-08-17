# 실습
# 저장한 증폭 데이터를 불러와서
# 완성 및 성능 비교

# 1 357
# 0 212

# 라벨 0을 112개 삭제 후 357개로 다시 SMOTE

# SMOTE 전 후 비교

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# pip install imblearn
import joblib

# 1. 데이터
path = 'D:/study_data/_save/'
x_train, x_test, y_train, y_test = joblib.load(path + 'm45_smote7_fetchcovtype.dat')

# 2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=55, verbose=2)

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

# model.score:  0.9572538949281598
# acc_score:  0.9572538949281598
# f1_macro:  0.9371510342976164
# f1_micro:  0.9572538949281598

