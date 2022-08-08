from tabnanny import verbose
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

parameters = [ # 명시한 모델명을 앞에 언더바 두개 해주고 붙여넣어줘야 파이프+그리드서치 적용됨
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__n_jobs':[-1]},
    {'RF__max_depth':[6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10], 'RF__n_jobs':[-1]},
    {'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10], 'RF__n_jobs':[-1]},
    {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_split':[2,3,10]},
    ] 

# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())], verbose=1)

# 3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

# model = GridSearchCV(RandomForestClassifier(), parameters, cv=5, verbose=1)
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score: ', result)

# model.score:  0.9298245614035088

# 파이프라인 안쓰고
# best tuned acc:  0.9736842105263158