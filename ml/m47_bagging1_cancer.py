import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True, stratify=y)
scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)


# 2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression # 이진 분류, 시그모이드 형태로 아웃풋 뺀다

model = BaggingClassifier(LogisticRegression(), n_estimators=100, n_jobs=-1, random_state=1234)


# 3. 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
print(model.score(x_test, y_test))

# 0.9736842105263158

# bagging(Bootstrap Aggregating) : 한 가지 모델을 여러번 돌림
# voting : 여러가지 모델을 여러번 돌림

# bagging classifier에 디시전 트리 넣으면 랜포가 됨