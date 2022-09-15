import joblib as jb
import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM, Dropout, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# 1. Data
path = 'D:\study_data\_data\dacon_vegi/'

train_data, label_data, val_data, val_target, test_input, test_target = jb.load(path+'datasets.dat')

# print(train_data[0])
# print(len(train_data), len(label_data)) # 1607 1607
# print(len(train_data[0]))   # 1440
# print(label_data)   # 1440
# print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
# print(val_data.shape) # (206, 1440, 37)
# print(val_target.shape) # (206,)

# 2. Model
model = Sequential()
model.add(Bidirectional(GRU(100, input_shape=(1440,37))))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# return_sequence=True // RNN 겹치기

# 3. Compile, Fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
model.fit(train_data,label_data, batch_size=200, epochs=50, callbacks=[Es], validation_data=(val_data, val_target))

model.save('D:\study_data\_save\_h5/vegi06.h5')
# model = load_model('D:\study_data\_save\_h5/vegi.h5')

# 4. Evaluate, Predict
loss = model.evaluate(val_data, val_target)
print(loss)

test_pred = model.predict(test_input)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(val_target, test_pred)
# print('rmse: ', rmse)


# test_pred -> TEST_ files
for i in range(6):
    thislen=0
    thisfile = 'D:\study_data\_data\dacon_vegi/test_target/'+'TEST_0'+str(i+1)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])


# TEST_ files -> zip file
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submissionKeras.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()


# vegi01
# [0.2958856523036957, 0.2879069149494171]

# vegi02
# [0.290480375289917, 0.26576611399650574]

# vegi03 현 베스트
# [0.2860572934150696, 0.25723227858543396]

# vegi04
# [0.2761073112487793, 0.24922898411750793]

# vegi05
# [0.29625755548477173, 0.2660003900527954]