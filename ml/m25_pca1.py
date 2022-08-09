import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

pca = PCA(n_components=12) # 주성분 분석, 차원 축소 // n_components 개수만큼으로 x의 열을 줄임
# y 값이 없는 대표적 비 지도 학습의 하나
x = pca.fit_transform(x) 
print(x.shape) # (506, 2)

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

# 모든 칼럼
# 결과:  0.9187290554452663

# PCA 12
# 결과:  0.8563581772863746