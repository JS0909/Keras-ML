from re import I
import pandas as pd
import numpy as np
import glob
import os

from sklearn.ensemble import RandomForestRegressor
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, GRU, Conv1D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping


path = 'D:\study_data\_data\dacon_vegi/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))
test_input = sorted(glob.glob(path + 'test_input/*.csv'))
test_target = sorted(glob.glob(path + 'test_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)

train_data, label_data = aaa(train_input_list, train_target_list) #, False)
vali_data, val_target = aaa(val_input_list, val_target_list) #, False)

test_input, test_target = aaa(test_input, test_target)

print(train_data[0])
print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
print(vali_data.shape) # (206, 1440, 37)

# 2. 모델
model = Sequential()
model.add(Conv1D(10, 1, input_shape=(1440,37)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
model.fit(train_data,label_data, epochs=1, callbacks=[Es], validation_split=0.1)


# 4. 평가, 예측
loss = model.evaluate(vali_data, val_target)
print(loss)
test_pred = model.predict(test_input)
print(test_pred.shape) # (195, 1)


for i in range(6):
    i2=0
    a = i+1
    thisfile = 'D:\study_data\_data\dacon_vegi/test_target/'+'TEST_0'+str(a)+'.csv'
    test = pd.read_csv(thisfile, index_col=False)
    test['rate'] = test_pred[i2:i2+len(test['rate'])]
    test.to_csv(thisfile, index=False)
    i2+=len(test['rate'])


import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_vegi/test_target")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()


