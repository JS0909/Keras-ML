# [실습] feature 한개 삭제하고 성능 비교

import numpy as np
from sklearn.datasets import load_diabetes

# 1. 데이터
datasets = load_diabetes()
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
x = datasets.data
print(x.shape) # (442, 10)
x = np.delete(x, 1, axis=1)
print(x.shape) # (442, 9)
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor # pip install xgboost

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
model = GradientBoostingRegressor()
# model = XGBRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score: ', result)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score: ', r2)

print('----------------------------------------')
print(model, ': ', model.feature_importances_) # tree계열에만 있음


# model.score:  -0.06803668834514842
# r2_score:  -0.06803668834514842
# DecisionTreeRegressor() :  [0.08175505 0.01452838 0.34378855 0.08707457 0.02062155 0.10143098
#  0.06139199 0.01179111 0.15634859 0.12126924]

# model.score:  0.4076785129628565
# r2_score:  0.4076785129628565
# RandomForestRegressor() :  [0.05768957 0.01261572 0.33354815 0.09110822 0.04410219 0.06195914
#  0.06235012 0.02609681 0.22340702 0.08712305]

# sex 칼럼삭제 후
# model.score:  0.38227457076013294
# r2_score:  0.38227457076013294
# ----------------------------------------
# RandomForestRegressor() :  [0.06016791 0.32140891 0.08263489 0.05213651 0.06293746 0.05785848
#  0.0348057  0.23544637 0.09260377]

# model.score:  0.4124988763421431
# r2_score:  0.4124988763421431
# GradientBoostingRegressor() :  [0.04612784 0.01648727 0.33594916 0.0955424  0.03161077 0.06604381
#  0.03821368 0.01413885 0.27693126 0.07895497]

# model.score:  0.26078151031491137
# r2_score:  0.26078151031491137
# XGBRegressor :  [0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191 0.06551369 0.17944618 0.13779876 0.08540721]