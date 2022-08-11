import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape)
le = LabelEncoder()
y = le.fit_transform(y)

# PCA 반복문
# for i in range(x.shape[1]):
#     pca = PCA(n_components=i+1)
#     x2 = pca.fit_transform(x)
#     x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
#     model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
#     model.fit(x_train, y_train)
#     results = model.score(x_test, y_test)
#     print(i+1, '의 결과: ', results)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True)
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([169507, 226569,  28696,   2152,   7618,  13864,  16403],
lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(x_train, y_train) 
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# 2. 모델
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

# # 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# # 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# 모든 칼럼
# 결과:  0.8695988915948814
# 걸린 시간:  6.007132291793823

# pca
# 54 의 결과:  0.8959407244219169

# LDA n_components = 5
# 결과:  0.7694293606877619
# 걸린 시간:  3.3473618030548096

