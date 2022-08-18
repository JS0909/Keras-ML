from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 1. 데이터
datasets = load_boston()
x,y = datasets.data, datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# 2. 모델
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

# 0.7665382927362877