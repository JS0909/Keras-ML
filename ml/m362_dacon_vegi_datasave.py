import joblib as jb
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

datalist = [train_data, label_data, vali_data, val_target, test_input, test_target]
jb.dump(datalist, path+'datasets.dat')