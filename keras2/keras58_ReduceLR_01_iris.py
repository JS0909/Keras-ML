from sklearn.datasets import load_iris
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from tensorflow.python.keras.optimizer_v2 import adam



# 1. 데이터
datasets = load_iris()

print(datasets.feature_names)
x = datasets['data']
y = datasets['target']

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

# 2. 모델구성
model = Sequential()
model.add(Conv1D(80, 2, input_shape=(4,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(90))
model.add(Dropout(0.1))
model.add(Dense(70, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. compilel, fit
optimizer = adam.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # learning rate를 0.5만큼 감축시키겠다

start = time.time()
model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=128, callbacks=[es,reduce_lr])
end = time.time()-start

loss, acc = model.evaluate(x_test,y_test)

print('걸린 시간: ', end)
print('loss: ', loss)
print('acc: ', acc)

# 걸린 시간:  4.563338041305542
# loss:  0.021217865869402885
# acc:  1.0