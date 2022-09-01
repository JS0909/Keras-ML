import joblib as jb
import pandas as pd
import numpy as np
import glob
import os

from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping


path = 'D:\study_data\_data\dacon_vegi/'

train_data, label_data, vali_data, val_target, test_input, test_target = jb.load(path+'datasets.dat')

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
print(vali_data.shape) # (206, 1440, 37)

# 2. 모델
model = Sequential()
model.add(LSTM(10, 1, input_shape=(1440,37)))
model.add(Dense(10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit(train_data,label_data, epochs=100, callbacks=[Es], validation_split=0.1)


# 4. 평가, 예측
loss = model.evaluate(vali_data, val_target)
test_pred = model.predict(test_input)
print(test_pred.shape) # (195, 1)


for i in range(6):
    thislen=0
    thisfile = 'D:\study_data\_data\dacon_vegi/test_target/'+'TEST_0'+str(i+1)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])



import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submissionKeras.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()


