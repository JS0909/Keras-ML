from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)

datasets = fetch_california_housing()
# print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x = datasets.data
y = datasets.target
# print(x.shape, y.shape)
# (20640, 8) (20640,)

# print(type(x)) # <class 'numpy.ndarray'>

x = pd.DataFrame(x, columns=datasets.feature_names)
# print(x)

######################### 다중공선성 ############################
# x에서만 쓴다 / 자기 자신이랑 다른 애들 전부랑 비교

# drop_features = ['Longitude']
# drop_features = ['Longitude', 'AveBedrms']
drop_features = ['Longitude', 'AveBedrms', 'Latitude']
x = x.drop(drop_features, axis=1)

vif = pd.DataFrame()
vif['feature'] = x.columns
vif['VIF Factor'] = [variance_inflation_factor(
    x.values, i) for i in range(x.shape[1])]

vif.sort_values('VIF Factor', ascending=False, inplace=True, ignore_index=True)

print(vif)

# 통상 5 나 10 이상이라면 다중공선성이 높아서 안좋다

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.9, random_state=123)

model = RandomForestRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('score:', score)

# score: 0.8194322552922824
# score: 0.7444603327598173
# score: 0.7416696970107874
# score: 0.6827183381840691