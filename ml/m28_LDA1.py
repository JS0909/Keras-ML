import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape) # (581012, 54)
le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=20)
# x = pca.fit_transform(x)
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y)
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([169472, 226640,  28603,   2198,   7594,  13894,  16408],
#       dtype=int64))
lda = LinearDiscriminantAnalysis(n_components=6) # y라벨 개수보다 작아야만 한다
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

# xgboost - gpu
# 결과:  0.8695988915948814
# 걸린 시간:  5.994714736938477

# xgboost - gpu / PCA n_component - 10
# 결과:  0.8406065247885166
# 걸린 시간:  4.38213324546814

# xgboost - gpu / PCA n_component - 20
# 결과:  0.8857946868841596
# 걸린 시간:  4.646213531494141

# PCA는 y값을 건들지 않고 x값만 축소하지만
# LDA는 y값을 x값 축소할 때 같이 연산에 포함한다