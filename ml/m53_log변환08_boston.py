from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.datasets import load_boston
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
datasets = load_boston()
x,y = datasets.data, datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LinearRegression()
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('그냥 결과: ', round(result,4))

# LR 그냥 결과:  0.7665
# RF 그냥 결과:  0.9139



#=================== 로그 변환 ======================
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df.head(10))

df.plot.box()
plt.title('boston')
plt.xlabel('Feature')
plt.ylabel('데이터값')
# plt.show()

# print(df['B'].head())
# df['B'] = np.log1p(df['B']) # 0.7711
# print(df['B'].head())


# df['CRIM'] = np.log1p(df['CRIM']) # 0.7596
df['ZN'] = np.log1p(df['ZN']) # 0.7734
# df['TAX'] = np.log1p(df['TAX']) # 0.7669

# 세개 다 하면: 0.7667
# 네개 다 하면: 0.7717


x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=1234)

# scl = StandardScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# 2. 모델
# model = LinearRegression()
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
result = r2_score(y_test, y_pred)
print('로그 변환 결과: ', round(result,4))

# LR 로그 변환 결과:  0.7711
# RF 로그 변환 결과:  0.917