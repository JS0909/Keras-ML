import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

for i in range(x.shape[1]):
    pca = PCA(n_components=i+1)
    x2 = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(i+1, '의 결과: ', results)



# 모든 칼럼
# 결과:  0.7573250241545894

# 14
# 결과:  0.8847468760441028

# 4
# 결과:  0.9011631807550952