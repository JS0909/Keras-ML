import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

pca = PCA(n_components=5)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) # 새로 생성된 칼럼들의 importances
print(sum(pca_EVR)) # 0.9999999203185791

cumsum = np.cumsum(pca_EVR) # 누적합
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()


'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)

# 2. 모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
'''

# 모든 칼럼
# 결과:  0.9187290554452663

# PCA 12
# 결과:  0.8563581772863746