from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.datasets import fetch_covtype
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

# 1. 데이터
datasets = fetch_covtype()
x,y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LogisticRegression()
model = RandomForestClassifier(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = accuracy_score(y_test, y_pred)
print('그냥 결과: ', round(result,4))

# 

#=================== 로그 변환 ======================
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df.columns)

df.plot.box()
plt.title('covtype')
plt.xlabel('Feature')
plt.ylabel('데이터값')
plt.show()

# df['magnesium'] = np.log1p(df['magnesium']) #  0.8946

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LogisticRegression()
model = RandomForestClassifier(random_state=1234)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = accuracy_score(y_test, y_pred)
print('로그 변환 결과: ', round(result,4))

