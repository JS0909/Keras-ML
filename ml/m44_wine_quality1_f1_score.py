import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
file = 'D:\study_data\_data\wine/winequality-white.csv'

data = pd.read_csv(file, index_col=None, header=0,sep=';')

print(data.shape) # (4898, 12)
print(data.describe()) # std: 표준편차
print(data.info())

# data = np.array(data)
data2 = data.values
# data = data.to_numpy()

print(type(data2)) # <class 'numpy.ndarray'>
print(data2.shape) # (4898, 12)

x = data2[:, :11]
y = data2[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True)) # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data['quality'].value_counts())
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5

# 분류 문제에서는 데이터의 분포때문에 스코어가 많이 갈린다. 분포를 꼭 확인하자.

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, shuffle=True, train_size=0.85, stratify=y)

# 2. 모델
model = RandomForestClassifier(random_state=666)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)

score = model.score(x_test, y_test)
print('model.score: ', score) # 0.710204081632653
print('acc_score: ', accuracy_score(y_pred, y_test)) # 0.710204081632653

from sklearn.metrics import f1_score
# 기본적으로 이진분류에서만 사용함
# 정밀도와 재현율 = 프리시전과 리콜

print('f1_macro: ', f1_score(y_pred, y_test, average='macro')) # 칼럼별로 f1을 따져서 평균을 내서 비교함
print('f1_micro: ', f1_score(y_pred, y_test, average='micro')) # micro f1 = acc score 

# 전처리 안하고
# model.score:  0.710204081632653
# acc_score:  0.710204081632653
# f1_macro:  0.4556941766626644
# f1_micro:  0.710204081632653

