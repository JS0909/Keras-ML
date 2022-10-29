import os
import cv2
import gc
import time
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A  # torchvision을 대신할만한 라이브러리. image augmentation 등의 기능이 많고 빠르다
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

dataset = "8k"

if dataset == "8k":
    df = pd.read_csv("D:\study_data\_data/team_project\Flickr8k/captions.txt")
    df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
    df.to_csv("D:\study_data\_data/team_project\Flickr8k/captions.csv", index=False)
    df = pd.read_csv("D:\study_data\_data/team_project\Flickr8k/captions.csv")
    image_path = "D:\study_data\_data/team_project\Flickr8k/Images"
    captions_path = "D:\study_data\_data/team_project\Flickr8k/"
elif dataset == "30k":
    df = pd.read_csv("D:\study_data\_data/team_project\Flickr30k/results.txt", delimiter=",") # delimiter 구분자
    df.columns = ['image', 'caption_number', 'caption']
    df['caption'] = df['caption'].str.lstrip()
    df['caption_number'] = df['caption_number'].str.lstrip()
    df.loc[19999, 'caption_number'] = "4"
    df.loc[19999, 'caption'] = "A dog runs across the grass ."
    ids = [id_ for id_ in range(len(df) // 5) for _ in range(5)]
    df['id'] = ids
    df.to_csv("D:\study_data\_data/team_project\Flickr30k/captions.csv", index=False)
    image_path = "D:\study_data\_data/team_project\Flickr30k\Images"
    captions_path = "D:\study_data\_data/team_project\Flickr30k"

print(df.head())

class CFG:
    debug = False
    image_path = image_path
    captions_path = captions_path
    batch_size = 32
    num_workers = 0     # 데이터 로더에 사용할 CPU 서브 프로세스 개수. 디폴트 0, 메인 프로세스로 처리하겠다는 뜻
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3 # 가중치가 너무 커져서 함수가 복잡해지는 경우(복잡도 과도한 증가=overfitting), loss를 최소화하기보다 가중치를 줄이는 것을 더 우선시 한다
    patience = 1
    factor = 0.8
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased" # uncased: 대소문자 구별x, MLM
    text_embedding = 768    # 버트모델 인코더 통과한 직후 차원
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0 # 템퍼쳐 값이 커질 수록 logits 값이 줄어듦. 기준이 빡세짐

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1): # val = loss
        self.count += count
        self.sum += val
        self.avg = self.sum / self.count    # 배치사이즈만큼 N빵

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(  # 토크나이징된 텍스트 input
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms
        # print(self.encoded_captions)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
        
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()    # 채널 앞으로 뺌
        item['caption'] = self.captions[idx]
        
        # print(item)
        '''{'input_ids': tensor([ 101, 2019, 2137, 4362, 1999, 1037, 2317, 1998, 6379, 6167, 2003, 2437,
        1037, 2448, 2007, 1996, 3608, 1012,  102,    0,    0,    0,    0,    0,     # input_ids : 토크나이징된 텍스트, encoded_captions로 나온 키밸류값
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,     # attention_mask : id 0인 곳에 어텐션마스크도 0, , encoded_captions로 나온 키밸류값
           0,    0,    0,    0,    0,    0]),
        'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'image': tensor([[[-0.5082, -1.0048, -0.1314,  ..., -1.0219, -1.3473, -1.0390],     # 이미지
         [ 0.5193, -0.9020, -0.4397,  ..., -1.1418, -1.0219, -0.8849],
         [ 1.8722,  0.8276, -0.4739,  ..., -1.2103, -1.2445, -1.0733],
         ...,
         [-0.6452, -0.6281, -0.7650,  ...,  0.2282, -0.5767, -0.8335],
         [-0.7650, -1.0219, -1.0562,  ...,  0.0741,  0.3994,  0.0569],
         [-0.8849, -0.9877, -1.0219,  ..., -0.4226,  0.0227,  0.2282]],

        [[-0.8978, -1.0553, -0.2150,  ..., -0.8803, -1.1253, -0.7577],
         [ 0.6429, -0.7577, -0.4601,  ..., -0.9853, -0.8277, -0.7052],
         [ 1.9909,  0.9230, -0.5476,  ..., -1.0378, -1.0028, -0.9153],
         ...,
         [-0.2850, -0.3025, -0.4951,  ...,  0.5728, -0.1450, -0.4076],
         [-0.3375, -0.5826, -0.6001,  ...,  0.3978,  0.7304,  0.4153],
         [-0.3725, -0.4251, -0.5301,  ..., -0.0749,  0.3627,  0.5728]],

        [[-0.2532, -0.1487,  0.1999,  ..., -1.0376, -1.2293, -0.8981],
         [ 1.0017, -0.0790,  0.0779,  ..., -1.1073, -1.0027, -0.8981],
         [ 2.2914,  1.5245,  0.2348,  ..., -1.1596, -1.1247, -1.0201],
         ...,
         [-0.7936, -0.7413, -0.9853,  ...,  0.5311, -0.2881, -0.6541],
         [-0.8110, -1.0201, -1.0724,  ...,  0.3742,  0.7228,  0.3568],
         [-0.8981, -0.9330, -0.9853,  ..., -0.3753,  0.3219,  0.5834]]]),
         'caption': 'An American footballer in a white and purple strip is making a run with the ball .'}   # 원래 캡션'''

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True), 
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
        
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(     # ResNet 50, 전이학습, 가중치 True로 가져옴
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
    
class TextEncoder(nn.Module):   # 텍스트 인코더는 DistilBert 모델을 사용
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):   # input_ids : 토큰화된 문장, attention_mask : 패딩부분 0
        output = self.model(input_ids=input_ids, attention_mask=attention_mask) # 모델은 DistilBert
        last_hidden_state = output.last_hidden_state
        '''last_hidden_state: tensor([[[-0.4560, -0.0427,  0.0366,  ..., -0.3436,  0.5117,  0.3217],
         [ 0.0571,  0.3289,  0.0741,  ..., -0.3507,  0.5046,  0.5981],
         [-0.3144,  0.5901,  0.1016,  ..., -0.5961,  0.4920,  0.3808],
         [-0.4737,  0.1667, -0.2677,  ..., -0.3990,  0.6218, -0.0489],
         [-0.3669,  0.1392, -0.0656,  ..., -0.1629,  0.2608, -0.2845],
         [ 0.6135,  0.3641, -0.4704,  ..., -0.1396, -0.4371, -0.2203]]],
        device='cuda:0')'''
        # last_hidden_state에서 첫줄만 가져다 씀. 첫번째 자리는 CLS 토큰 자리 (class token)
        return last_hidden_state[:, self.target_token_idx, :]
    
class ProjectionHead(nn.Module):
    # 트랜스포머의 fc_out과 비슷한 역할. text와 image 임베딩 디멘션이 서로 다른데 
    # 여기서 프로젝션 디멘션으로 디멘션 맞춰주고 어텐션 계산함
    def __init__(                       
        self,       
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)    # (32, 256)
        text_embeddings = self.text_projection(text_features)       # (32, 256)
        
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature  # 템퍼쳐 값이 커질 수록 logits 값이 줄어듦. 기준이 빡세짐.
        images_similarity = image_embeddings @ image_embeddings.T           # self 유사도, 어텐션 스코어
        texts_similarity = text_embeddings @ text_embeddings.T

        # 텍스트 피쳐와 이미지 피쳐를 행렬곱해서 transformer처럼 셀프어텐션 에너지값을 구함
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        ) # images_similarity: (32, 32)   texts_similarity: (32, 32)  tartgets: (32, 32)
        
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)   # targets.shape = (32,32)   preds.shape = (32,32)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    
'''# A simple Example
batch_size = 4
dim = 256
embeddings = torch.randn(batch_size, dim)
out = embeddings @ embeddings.T
print('sample softmax: ', F.softmax(out, dim=-1))'''

def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100    # 디버그 True로 실행하면 100개 데이터만 가져와서 해보기
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )   # 전체 데이터의 20% 만큼을 valid 데이터셋으로 만듦
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,    # input_id : 캡션 토크나이징된 것, attention_mask : 패딩부분 마스크처리
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,   
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}   # caption부분만 빼고 불러오기
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)  # batch['image'].shape = (32, 3, 224, 224), 0번째 shape 이니까 count에는 32씩 들어감
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))    # tqdm에서 중간 결과 찍어주는 용도
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

# '''
def main():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(model.image_projection.parameters(), model.text_projection.parameters()), 
        "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # reduce lr과 동일한 역할
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor     # factor 얼마나 감소시킬지
    )
    step = "epoch"
    start_time = time.time()
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
    end_time = time.time() - start_time
    return end_time
        
end_time = main()
print('took', round(end_time), 'sec.')
print(f'epochs: {CFG.epochs}    batch size: {CFG.batch_size}')
# '''

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))      # 저장된 가중치 불러와서 사용
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))     # 이미지 인코딩
            image_embeddings = model.image_projection(image_features)               # 텍스트랑 맞추기 위한 fc_out projection
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

_, valid_df = make_train_valid_dfs()
model, image_embeddings = get_image_embeddings(valid_df, "best.pt")                 # inference용 이미지는 valid 세트에서 가져옴

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])  # 입력한 문장 처리
    batch = {key: torch.tensor(values).to(CFG.device)for key, values in encoded_query.items()}
    
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)              # 이미지와 디멘션 맞춰줌 fc_out
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)         # 이미지는 get_image_embeddings에서 프로젝션했음
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)           # 이미지는 채널값에 대해, 텍스트는 토큰값에 대해 L2norm, 하는 이유는 두 임베딩값 곱할때 값의 편차를 줄이기 위해
    dot_similarity = text_embeddings_n @ image_embeddings_n.T               # (1, 256) @ (8090, 256).T = (1, 8090)
    # print(text_embeddings, text_embeddings_n)
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5) # (8090,)만든 후 argmax 인데 제일 큰거부터 (n=)9*5개 인덱스 반환함
    matches = [image_filenames[idx] for idx in indices[::5]]       # 45개 중 5개씩 건너뛰어가며 뽑아 냄 (결국 9개 이미지 반환하는 것)
    '''indices tensor([5330, 5334, 5331, 5333, 5332, 1403, 1404, 1401, 1402, 1400,  362,  361,
         364,  360,  363, 6044, 6040, 6042, 6043, 6041, 7915, 7916, 7918, 7917,
        7919, 2616, 2617, 2619, 2618, 2615, 1543, 1544, 1540, 1541, 1542, 2184,
        2182, 2183, 2181, 2180,  896,  895,  897,  899,  898], device='cuda:0')'''
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()
    
find_matches(model, 
             image_embeddings,
             query="surfers in the ocean",
             image_filenames=valid_df['image'].values,      # valid 데이터셋의 이미지들 중에서 텍스트와 매치되는 이미지를 보여줌
             n=9)

# took 2246 sec.
# epochs: 5    batch size: 32

# 구조를 쉽게 요약하면 어텐션 기법을 사용하여 이미지 피처와 텍스트 피처의 유사도를 계속 구하는 방식으로 훈련하고
# 예측의 경우 텍스트 피처를 넣으면 이미지 피처를 클래시파이어 클래스로 두고
# 그 중에서 topk - 5 의 방식으로 5장을 해당 텍스트에 관련된 이미지라고 예측