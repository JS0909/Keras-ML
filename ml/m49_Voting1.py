import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

model = VotingClassifier(estimators=[('LR', lr), ('KNN', knn)],
                         voting='soft' # hard
                         )

# 3. 훈련
model.fit(x_train,y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('voting result: ', round(acc, 4))

# voting result:  0.9825

'''
하드보팅
        a  b  c
이진    0  0  1  -> 0
0,1     1  1  0  -> 1

소프트보팅
        a       b        c
이진   0 1     0 1      0 1
0,1  0.7 0.3  0.5 0.5  0.6 0.4

0: (0.7+0.5+0.6)/3
1: (0.3+0.5+0.4)/3

-> 이건 (0.7+0.5+0.6)/3의 확률로 0이다
'''

classifiers = [lr, knn]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    acc2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, acc2))

# soft voting  
# voting result:  0.9825
# LogisticRegression 정확도: 0.9737
# KNeighborsClassifier 정확도: 0.9825

# hard voting
# voting result:  0.9737
# LogisticRegression 정확도: 0.9737
# KNeighborsClassifier 정확도: 0.9825