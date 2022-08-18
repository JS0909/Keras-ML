from tabnanny import verbose
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head(7))

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.8, random_state=704, stratify=dataset.target)

scl = MinMaxScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.fit_transform(x_test)



# 2. 모델
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)

model = VotingClassifier(estimators=[('XG', xg), ('LG', lg), ('CAT', cat)],
                         voting='soft'
                         )

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('voting result: ', round(acc, 4))



classifiers = [xg, lg, cat]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    acc2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, acc2))
    
# voting result:  0.9561
# XGBClassifier 정확도: 0.9649
# LGBMClassifier 정확도: 0.9649
# CatBoostClassifier 정확도: 0.9561