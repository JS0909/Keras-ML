x = 10
y = 10 # 목표결과값
w = 500 # 임의의 가중치 초기값
lr = 0.001
epochs = 600

for i in range(epochs):
    predict = x * w
    loss = (predict - y) ** 2     # mse
    
    print('Loss: ', round(loss, 4), '\tPredict: ', round(predict, 4))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) **2  # mse
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) **2  # mse
    
    if (up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
        
# w 갱신