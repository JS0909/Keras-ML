from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import LinearSVR


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
model = LinearSVR()


# 3. 컴파일, 훈련
model.fit(x_train, y_train )

# 4. 평가, 예측
score = model.score(x_test, y_test)
ypred = model.predict(x_test)

print('acc score: ', score)
print('y_pred: ', ypred)

# loss :  1672.607177734375  
# acc :  0.5149952410421789

# acc score:  0.2668794571758186