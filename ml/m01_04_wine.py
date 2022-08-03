from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVC


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
model = LinearSVC()


# 3. 컴파일, 훈련
model.fit(x_train, y_train )

# 4. 평가, 예측
score = model.score(x_test, y_test)
ypred = model.predict(x_test)

print('acc score: ', score)
print('y_pred: ', ypred)

# r2 score:  0.6666666666666666