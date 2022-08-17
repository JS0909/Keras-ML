# 실습
# 증폭 후 해당 데이터 저장
# 증폭에서 시간 찍기 (resample에서 시간 보기)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# pip install imblearn
from imblearn.over_sampling import SMOTE
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 
print(type(x))          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
print(pd.Series(y).value_counts())
#============================================================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=134, stratify=y)

smote = SMOTE(random_state=1234, k_neighbors=5)
start = time.time()
x_train, y_train = smote.fit_resample(x_train, y_train)
end = time.time()
print(pd.Series(y_train).value_counts())

import joblib
path = 'D:/study_data/_save/'
joblib.dump([x_train, x_test, y_train, y_test], path + 'm45_smote7_fetchcovtype.dat')

print('걸린시간: ', end-start)

# 걸린시간:  19.980294466018677