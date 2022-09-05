import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

# 2. model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# 3. compile, fit
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

learning_rate = 0.0001

optlist = [adam.Adam, adadelta.Adadelta, adagrad.Adagrad, adamax.Adamax, rmsprop.RMSprop, nadam.Nadam]
for optimizer in optlist:
    optimizer = optimizer(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x, y, epochs=50, batch_size=1, verbose=0)

    # 4. evaluate, predict
    loss = model.evaluate(x, y)
    y_pred = model.predict([11])

    print('optimizer: ', optimizer.__class__.__name__, '   /loss: ', round(loss, 4), '/lr: ', learning_rate, '/predict result: ', y_pred)

# optimizer:  Adam    /loss:  2.3388 /lr:  0.0001 /predict result:  [[11.169116]]
# optimizer:  Adadelta    /loss:  2.4045 /lr:  0.0001 /predict result:  [[10.517084]]
# optimizer:  Adagrad    /loss:  2.3132 /lr:  0.0001 /predict result:  [[10.930161]]
# optimizer:  Adamax    /loss:  2.2448 /lr:  0.0001 /predict result:  [[10.734779]]
# optimizer:  RMSprop    /loss:  2.9819 /lr:  0.0001 /predict result:  [[9.166279]]
# optimizer:  Nadam    /loss:  2.2587 /lr:  0.0001 /predict result:  [[11.106043]]