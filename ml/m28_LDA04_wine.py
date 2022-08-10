import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape)

# PCA 반복문
# for i in range(x.shape[1]):
#     pca = PCA(n_components=i+1)
#     x2 = pca.fit_transform(x)
#     x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
#     model = XGBClassifier()
#     model.fit(x_train, y_train)
#     results = model.score(x_test, y_test)
#     print(i+1, '의 결과: ', results)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y)
print(np.unique(y_train, return_counts=True))
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_train, y_train) 
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# # 2. 모델
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)

# # 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# # 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# 모든 칼럼
# 결과:  0.9444444444444444
# 걸린 시간:  0.48240208625793457

# pca 5
# 5 의 결과:  0.9722222222222222

# pca 4
# 5 의 결과:  0.9722222222222222

# LDA n_components = 1
# 결과:  0.9722222222222222
# 걸린 시간:  0.572378396987915

# LDA n_components = 2
# 결과:  0.9722222222222222
# 걸린 시간:  0.4697086811065674
