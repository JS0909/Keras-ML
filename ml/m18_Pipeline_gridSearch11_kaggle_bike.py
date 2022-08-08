import pandas as pd 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
path = 'D:\study_data\_data\kaggle_bike/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

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

# model.score:  0.9506350426639436

# 파이프라인 안쓰고
# best tuned acc:  0.9463221004268908

