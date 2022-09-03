import autokeras as ak
print(ak.__version__)
import tensorflow as tf
import keras
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = \
    keras.datasets.mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# 2. 모델
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=5)
end = time.time()

# 4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print('결과: ', results)
print('시간: ', round(end-start, 4))

# 결과:  [0.03775094822049141, 0.9876000285148621]
# 시간:  4111.0163