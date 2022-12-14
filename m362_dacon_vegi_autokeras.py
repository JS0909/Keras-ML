import joblib as jb
import pandas as pd
import numpy as np
import os
import autokeras as ak
import keras
import time

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# 1. Data
path = 'D:\study_data\_data\dacon_vegi/'

train_data, label_data, vali_data, val_target, test_input, test_target = jb.load(path+'datasets.dat')

# print(train_data[0])
# print(len(train_data), len(label_data)) # 1607 1607
# print(len(train_data[0]))   # 1440
# print(label_data)   # 1440
# print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
# print(vali_data.shape) # (206, 1440, 37)

train_data = train_data.reshape(1607,1440*37)
vali_data = vali_data.reshape(206, 1440*37)

# 2. Model
model = ak.TimeseriesForecaster(
    overwrite=True,
    max_trials=2,
    loss='mean_absolute_error'
)

# 3. Compile, Fit
start = time.time()
model.fit(train_data, label_data, epochs=5)
end = time.time()

# model = load_model('D:\study_data\_save\_h5/auto_vegi01.h5')


# 4. Evaluate, Predict
loss = model.evaluate(vali_data, val_target)
print(loss)

test_pred = model.predict(test_input)

model = model.export_model()
model.save('D:\study_data\_save\_h5/auto_vegi01.h5')


# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(test_pred, val_target)
# print('rmse: ', rmse)


# test_pred -> TEST_ files
for i in range(6):
    thislen=0
    thisfile = 'D:\study_data\_data\dacon_vegi/test_target/'+'TEST_0'+str(i+1)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[thislen:thislen+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    thislen+=len(test['rate'])


# TEST_파일 취합, 압축파일 생성
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submissionKeras.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()