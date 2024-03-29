from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as tr

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

transf = tr.Compose([tr.Resize(15), tr.ToTensor()]) # 커스텀한 TensorDataset 에서 씀

path = 'D:\study_data\_data/torch_data/'

# 1. data
# train_dataset = MNIST(path, train=True, download=True, transform=transf)
# test_dataset = MNIST(path, train=False, download=True, transform=transf)
# print(train_dataset[0][0].shape) # torch.Size([1, 15, 15])

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets

print(x_train.size(), y_train.size()) # torch.Size([60000, 28, 28]) torch.Size([60000])
print(x_test.size(), y_test.size()) # torch.Size([10000, 28, 28]) torch.Size([10000])
print(np.min(x_train.numpy()), np.max(x_train.numpy())) # 0.0   1.0

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 28*28) # view() = reshape()
print(x_train.size(), x_test.size()) # torch.Size([60000, 784]) torch.Size([10000, 784])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)
print(len(train_loader)) # 60000/32 = 1875

# 2. model
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
# 평가할 때는 dropout이 자동으로 안적용된다
# train할때도 에포당 dropout되는 노드가 무엇인지는 항상 바뀐다

        self.output_layer = nn.Linear(100, 10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(28*28).to(DEVICE)

# 3. compile, fit
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4) # 0.0001

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
    model.eval() # 레이어 단계의 batch normalization, dropout 등을 쓰지 않도록 해준다
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
for epoch in range(1, EPOCHS + 1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)

    print(f'epoch:{epoch}, loss:{loss:.4}, acc:{acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.3f}')