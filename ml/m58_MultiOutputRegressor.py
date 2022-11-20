import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

x, y = load_linnerud(return_X_y=True) # 리니어 훈련용 데이터셋
print(x.shape) # (20, 3)
print(y.shape) # (20, 3)

# model = Ridge()
# model = XGBRegressor()

# cat = CatBoostRegressor()
# model = MultiOutputRegressor(estimator=cat)

lg = LGBMRegressor()
model = MultiOutputRegressor(estimator=lg)

model.fit(x, y)

print(model.predict([[2, 110, 43]]))

