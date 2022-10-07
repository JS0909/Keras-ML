import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision

import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")
sample_dir = 'C:\study\project/team\GAN/samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
   
# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 20
batch_size = 100

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# 랜덤하게 이미지 9장 살펴보는 함수
def show_imgs(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off") # x축, y축 안보이게 설정
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show() 
       
# Image processing
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])
transform = transforms.Compose([
                transforms.ToTensor(),             # 이미지 각 픽셀값이 0 ~ 1 : centering + rescaling (정확하게 하려면 채널별로 평균을 구해서 centering 해야함)
                transforms.Normalize(mean=[0.5],   # 1 for greyscale channels
                                     std=[0.5])])  # 각각 -1 ~ 1 사이 값으로 변경됨. 이따가 G 가 생성하는 아웃풋도 tanh 를 거치기 때문에 -1 ~ 1 사이 값임

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='D:\study_data\_data/torch_data/',
                                   train=True,
                                   transform=transform,
                                   download=False)
# print(mnist[0])
# show_imgs(mnist) # transforms.Normalize 한 거 안한 거 별로 차이는 없음

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

# Discriminator :: 판별자
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),   # np.maximum(0.2*x, x) = 괄호 안의 숫자는 허용치 범위
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1), 
    nn.Sigmoid())      # discriminator 는 가짜냐 진짜냐, 0 or 1 값으로 판단하기 때문에 sigmoid 사용. 따라서 loss 지표도 binary_cross_entropy 사용

# Generator :: 생성자 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())      # -1 ~ 1 사이

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
    out = (x + 1) / 2           # data 채널은 현재 원래 -1 ~ 1 사이 값인데
    return out.clamp(0, 1)      # 0보다 작은 값은 안 씀 / 1보다 큰 값은 안 씀
                                # -1 ~ 1 을 0 ~ 1 사이 값으로 변환

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# Start training
dx_epoch = []       # epoch 별 D(x) 스코어 -> 실제 이미지로 판별한 스코어
dgx_epoch = []      # epoch 별 D(G(z)) 스코어 -> 만들어낸 이미지로 판별한 스코어
total_step = len(data_loader)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):   # data_loader 에 image, label 로 있나봄      
        # 이미지는 지금 4차원일 것. (배치개수, 채널, 가로, 세로) : (100, 1, 28, 28)  
        images = images.reshape(batch_size, -1).to(device)  # 여기서 (100, 1*28*28) 로 차원 변경
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)  # 라벨링 용 텐서 (100, 1) 짜리 1 로 채워진 데이터
        fake_labels = torch.zeros(batch_size, 1).to(device) # 라벨링 용 텐서 (100, 1) 짜리 0 으로 채워진 데이터 -> 이미지당 하나씩 0 또는 1로 매치

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1   // 여기서 second term 이란 (1-y) * log(1 - D(x)) 이부분.  y가 1이면 여기가 0이 됨
        outputs = D(images)     # 여기는 1 에 가까울 수록 적은 loss 를 만들며
        d_loss_real = criterion(outputs, real_labels)   # 진짜이미지에 대한 진짜 라벨링을 얼마나 했는지의 loss
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0  // 여기서 first term 이란 - y * log(D(x)) 이부분.  y가 0이면 여기가 0이 됨
        z = torch.randn(batch_size, latent_size).to(device) # 노이즈 이미지 하나를 생성
        # plt.imshow(z.to('cpu'))
        # plt.show()
        fake_images = G(z)
        outputs = D(fake_images)   # 여기는 0 에 가까울 수록 적은 loss 를 만든다. 지금 여기 값은 시그모이드 거쳐서 나옴 0 ~ 1 값        
        d_loss_fake = criterion(outputs, fake_labels)   # 가짜이미지에 대한 가짜 라벨링을 얼마나 했는지의 loss
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake  # D(image) 와 D(G(image)) 의 loss 합산해서 이걸 기준으로 D 에 대한 역전파
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device) # 노이즈 이미지 하나를 생성 (100, 64)
        fake_images = G(z)
        outputs = D(fake_images)    # 만들어낸 가짜 이미지를 D 에게 검사받기
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))  # 똑같은 말인데 1- 요 작업 안하고 걍 바로 D(G(z))를 키움. 이 말은 D 가 가짜이미지를 진짜 이미지라고 하는 경우의 수를 늘린다는 말과 같음
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf  // D를 여러번 돌릴 때 G를 한번 돌려서 오버피팅 방지한대, 제한적인 데이터셋에서는 그래야 컴퓨터의 제한적 계산에 따른 오버피팅 안난다.. 고 함
        # 이렇게 해야 D 가 옵티마이징 되는 동안 G 가 천천히 변할 수 있다고 함. 
        g_loss = criterion(outputs, real_labels) 
        # G가 생성한 이미지를 가지고 D가 구분해낸 (0 ~ 1 사이) 값이 real_labels (전부 1) 값과 얼마나 차이가 나는지
        # 여기서는 G 를 학습시키는 곳이기 때문에 G가 real_label 을 최대한 받아내는 것이 목표이므로 real_label 을 못받은 것을 G의 loss 로 생각해서 이것으로 역전파를 진행함
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    dx_epoch.append(real_score.mean().item())     # 각 에포당 마지막 배치들의 real_score 의 평균값을 사용함  //  real_score = D(진짜이미지) 0 ~ 1 사이          (100, 1)
    dgx_epoch.append(fake_score.mean().item())    # 각 에포당 마지막 배치들의 fake_score 의 평균값을 사용함  //  fake_score = D(생성자가만든이미지) 0 ~ 1 사이   (100, 1)
    
    # Save real images
    if (epoch+1) == 1:  # 첫번째 에포에서 
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
# torch.save(G.state_dict(), 'G.ckpt')
# torch.save(D.state_dict(), 'D.ckpt')

# plot    
plt.figure(figsize = (12, 8))
plt.xlabel('epoch')
plt.ylabel('score')
x = np.arange(num_epochs)
plt.plot(x, dx_epoch, 'g', label='D(x)')
plt.plot(x, dgx_epoch, 'b', label='D(G(z))')
plt.legend()
plt.show()
