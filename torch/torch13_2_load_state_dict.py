import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 넘파이 형태로 스케일링하기 때문에 스케일링 데이터는 스케일링 이후 to(DEVICE)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())
# torch.Size([455, 30])

############################# DataLoader #############################
from torch.utils.data import TensorDataset, DataLoader
                            # L x, y 합친거     # L x,y + batch 합친거
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

print(train_set) # <torch.utils.data.dataset.TensorDataset object at 0x000001F0B585F130>
print('================= train_set[0] ===================')
print(train_set[0]) # 첫 행의 데이터 + 해당 행의 유니크값(라벨)
print('================= train_set[0][0] ===================')
print(train_set[0][0]) # 첫 행의 데이터들
print('================= train_set[0][1] ===================')
print(train_set[0][1]) # 첫 행의 유니크값
print('================= len(train_set) ===================')
print(len(train_set)) # 455

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=False)
# test 데이터는 배치작업 안해도 되긴 함

# 2. model
class Model(nn.Module): 
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x
        
######## 가중치 불러오기 ########
path = './_save/'
loaded_model = Model(30, 1).to(DEVICE)
loaded_model.load_state_dict(torch.load(path + 'torch13_state_dict.pt'))

pred_result = (loaded_model(x_test) >= 0.5).float()

score = (pred_result == y_test).float().mean()
acc_score = accuracy_score(y_test.cpu(), pred_result.cpu())


print(f'score:{score:.4f}')
print(f'acc_score:{acc_score:.4f}')

# score:0.9737
# acc_score:0.973