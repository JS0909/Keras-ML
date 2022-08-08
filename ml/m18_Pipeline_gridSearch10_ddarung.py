# Dacon 따릉이 문제풀이
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함


# 1. 데이터
path = 'D:\study_data\_data\ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)

parameters = [ # 명시한 모델명을 앞에 언더바 두개 해주고 붙여넣어줘야 파이프+그리드서치 적용됨
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__n_jobs':[-1]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10], 'RF__n_jobs':[-1]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10], 'RF__n_jobs':[-1]},
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_split':[2,3,10]},
    ] 

# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline, Pipeline

pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestRegressor())], verbose=1)

# 3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1)

model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score: ', result)

# model.score:  0.7717306910029738

# 파이프라인 안쓰고
# best tuned acc:  0.7897815339385946
