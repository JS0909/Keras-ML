import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape)
# le = LabelEncoder()
# y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y)
print(np.unique(y_train, return_counts=True))
lda = LinearDiscriminantAnalysis(n_components=1) # y라벨 개수보다 작아야만 한다
lda.fit(x_train, y_train) 
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# 2. 모델
from xgboost import XGBClassifier
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# 모든 칼럼
# 결과:  0.7573250241545894

# pca 14
# 결과:  0.8847468760441028

# pca 4
# 결과:  0.9011631807550952

# LDA n_components = 1
# 결과:  0.956140350877193
# 걸린 시간:  0.34171247482299805