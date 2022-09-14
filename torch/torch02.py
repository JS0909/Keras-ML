import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()                   # 쿠다 사용할 수 있는지 여부
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu') # 쿠다 사용 가능하면 'cuda' 모드, 안되면 'cpu' 모드 / device id 생략 가능
print(torch.__version__, '사용DVICE:', DEVICE)

# 1. data
x = np.array([1,2,3]) # (3, )
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1) # 1번째자리 쉐이프 늘려주기, 0이면 0번째, -1 마지막 번째
y = torch.FloatTensor(y).unsqueeze(-1) # 만약 2라면 (3,1,1)이 된다

print(x, y)
print(x.shape, y.shape) # torch.Size([3, 1]) torch.Size([3, 1])

# 2. model
# model = Sequnential()
model = nn.Linear(1, 1) # 인풋 열, 아웃풋 열 / 단층레이어

# 3. compile, fit
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss() # = loss
optimizer = optim.SGD(model.parameters(), lr=0.001) # 정의한 모델의 파라미터에 러닝레이트를 적용한 옵티마이저를 쓰겠다
# optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    # model.train() # 훈련모드 켜기(없어도 됨. 디폴트)
    optimizer.zero_grad() # 손실함수 기울기를 0으로 초기화. 두번째 역전파부터 이전 grad가 남아 있으므로 0으로 한번씩 초기화해줘야 함
    
    hypothesis = model(x)
    loss = criterion(hypothesis, y) # MSE 에 넣는 값
    
    loss.backward()  # 역전파하겠다
    optimizer.step() # 역전파한거 가지고 가중치를 갱신시키겠다

    return loss.item() # item() : 텐서형태가 아닌 알아들을 수 있는 형태로 반환한다는 뜻. sess.run() 느낌
    
epochs = 2000
for epoch in range(1, epochs+1): # 1 ~ 100 까지 돌리기
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
# 4. eval
# loss = model.evaluate(x, y)
def evaluate(model, criterion, x, y): # 평가에서는 가중치 갱신할 필요가 없음
    model.eval() # 평가모드 켜기. 가중치 갱신 안한다는 뜻
    
    with torch.no_grad(): # gradient 아예 안쓰겠다. 순전파만 사용해서 x에 대한 예상값만 사용 할 것임
        x_predict = model(x)
        results = criterion(x_predict, y)
    
    return results.item()

result_loss = evaluate(model, criterion, x, y)
print(f'최종 loss: {result_loss}')

# y_predict = model.predict([4])
results = model(torch.Tensor([[4]])) # (1, 1) 형태의 텐서로 넣어서 예측값 뽑기
print(f'4의 예측값: {results.item()}')