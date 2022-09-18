import joblib as jb
import pandas as pd
import numpy as np
import os

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, Flatten, LSTM, Dropout, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


# 1. Data
path = 'D:\study_data/_data\dacon_vegi/'

train_data, label_data, val_data, val_target, test_input, test_target = jb.load(path+'datasets.dat')

x_train,x_test,y_train,y_test = train_test_split(train_data,label_data,train_size=0.91,shuffle=True,random_state=123)
# print(x_train.shape)

#2. 모델 구성      
model = Sequential()
model.add(GRU(100,input_shape=(1440,37)))
# model.add(GRU(50, activation='relu'))
# model.add(GRU(50))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(80, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary()

#3. 컴파일, 훈련
es = EarlyStopping(monitor='val_loss',patience=400,mode='auto',verbose=1)
reduced_lr = ReduceLROnPlateau(monitor='val_loss',patience=300,mode='auto',verbose=1,factor=0.5)
from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.2
optimizer = adam_v2.Adam(lr=learning_rate)

model.compile(loss='mae', optimizer='adam',metrics=['acc'])
# hist = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(val_data, val_target), verbose=2,callbacks = [es,reduced_lr])
# model.save('D:\study_data/_save/_h5/minvegi01.h5')
# model = load_model('D:\study_home/_save/_h5/vegi.h5')

#4. 평가,예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)
# from sklearn.metrics import r2_score
# y_predict = model.predict(x_test)
# r2 = r2_score(y_predict,y_test)
# from sklearn.metrics import mean_squared_error
# rmse = np.sqrt(mean_squared_error(y_test,y_predict))


# y_predict = model.predict(x_test)
# print(y_test.shape) #(152,)
# print(y_predict.shape) #(152, 13, 1)

# from sklearn.metrics import accuracy_score, r2_score,accuracy_score
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)

model.fit(train_data,label_data)
test_pred = model.predict(test_input)

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

print('Done')

