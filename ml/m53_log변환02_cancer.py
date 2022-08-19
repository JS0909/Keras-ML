from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler,\
    QuantileTransformer, PowerTransformer # = 이상치에 자유로운 편
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = load_breast_cancer()
x,y = datasets.data, datasets.target
# print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LogisticRegression()
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('그냥 결과: ', round(result,4))

# 그냥 결과:  0.7063

#=================== 로그 변환 ======================
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df.value_counts())

df.plot.box()
plt.title('cancer')
plt.xlabel('Feature')
plt.ylabel('데이터값')
plt.show()

df['mean area'] = np.log1p(df['mean area']) # 0.6696
df['worst area'] = np.log1p(df['worst area']) # 0.6696

# 둘다 0.7063

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LogisticRegression()
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('로그 변환 결과: ', round(result,4))
