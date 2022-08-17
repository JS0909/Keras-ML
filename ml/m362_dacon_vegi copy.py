import random
import pandas as pd
import numpy as np
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'EPOCHS':5,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':16,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

all_input_list = sorted(glob.glob('./train_input/*.csv'))
all_target_list = sorted(glob.glob('./train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

class CustomDataset(Dataset):
    def __init__(self, input_paths, target_paths, infer_mode):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.infer_mode = infer_mode
        
        self.data_list = []
        self.label_list = []
        print('Data Pre-processing..')
        for input_path, target_path in tqdm(zip(self.input_paths, self.target_paths)):
            input_df = pd.read_csv(input_path)
            target_df = pd.read_csv(target_path)
            
            input_df = input_df.drop(columns=['시간'])
            input_df = input_df.fillna(0)
            
            input_length = int(len(input_df)/1440)
            target_length = int(len(target_df))
            
            for idx in range(target_length):
                time_series = input_df[1440*idx:1440*(idx+1)].values
                self.data_list.append(torch.Tensor(time_series))
            for label in target_df["rate"]:
                self.label_list.append(label)
        print('Done.')
              
    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        if self.infer_mode == False:
            return data, label
        else:
            return data
        
    def __len__(self):
        return len(self.data_list)
    


train_dataset = CustomDataset(train_input_list, train_target_list, False)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=6)

val_dataset = CustomDataset(val_input_list, val_target_list, False)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6)

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.lstm = nn.LSTM(input_size=37, hidden_size=256, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        hidden, _ = self.lstm(x)
        output = self.classifier(hidden[:,-1,:])
        return output
    
    
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    
    best_loss = 9999
    best_model = None
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for X, Y in tqdm(iter(train_loader)):
            X = X.to(device)
            Y = Y.to(device)
            
            optimizer.zero_grad()
            
            output = model(X)
            loss = criterion(output, Y)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        val_loss = validation(model, val_loader, criterion, device)
        
        print(f'Train Loss : [{np.mean(train_loss):.5f}] Valid Loss : [{val_loss:.5f}]')
        
        if scheduler is not None:
            scheduler.step()
            
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
    return best_model

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for X, Y in tqdm(iter(val_loader)):
            X = X.float().to(device)
            Y = Y.float().to(device)
            
            model_pred = model(X)
            loss = criterion(model_pred, Y)
            
            val_loss.append(loss.item())
            
    return np.mean(val_loss)

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

best_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test_input_list = sorted(glob.glob('./test_input/*.csv'))
test_target_list = sorted(glob.glob('./test_target/*.csv'))

def inference_per_case(model, test_loader, test_path, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for X in iter(test_loader):
            X = X.float().to(device)
            
            model_pred = model(X)
            
            model_pred = model_pred.cpu().numpy().reshape(-1).tolist()
            
            pred_list += model_pred
    
    submit_df = pd.read_csv(test_path)
    submit_df['rate'] = pred_list
    submit_df.to_csv(test_path, index=False)
    
for test_input_path, test_target_path in zip(test_input_list, test_target_list):
    test_dataset = CustomDataset([test_input_path], [test_target_path], True)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    inference_per_case(best_model, test_loader, test_target_path, device)
    
import zipfile
os.chdir("./test_target/")
submission = zipfile.ZipFile("../submission.zip", 'w')
for path in test_target_list:
    path = path.split('/')[-1]
    submission.write(path)
submission.close()