from h11 import Data
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR100
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

# 1. data
path = 'D:\study_data\_data/torch_data/'
train_dataset = CIFAR100(path, train=True, download=False)
test_dataset = CIFAR100(path, train=False, download=False)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

print(x_train.size())
# torch.Size([50000, 32, 32, 3])

x_train, x_test = x_train.reshape(-1, 3, 32, 32), x_test.reshape(-1, 3, 32, 32)
print(x_train.size(), x_test.size())
# torch.Size([50000, 3, 32, 32]) torch.Size([10000, 3, 32, 32])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

# 2. model
class CNN(nn.Module):
    def __init__(self, num_features):
        super(CNN, self).__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.3)
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.3)
        )
        
        self.hidden_layer3 = nn.Linear(32*6*6, 32)

        self.output_layer = nn.Linear(in_features=32, out_features=100)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = x.view(x.shape[0], -1) # flatten
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x
    
model = CNN(3).to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    epoch_acc = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)

EPOCHS = 20
for epoch in range(1, EPOCHS+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print(f'epoch:{epoch}, loss:{loss:.4}, acc:{acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.3f}')
    
# epoch:20, loss:3.411, acc:0.1979, val_loss:3.4811, val_acc:0.189