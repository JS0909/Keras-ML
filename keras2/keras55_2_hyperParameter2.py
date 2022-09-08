# 하이퍼파라미터에 노드 추가, learning rate 추가

import numpy as np
from torch import dropout
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
import keras

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. model
def build_model(drop=0.5, activation='relu', lr=1e-1, node1=10, node2=10, node3=10):

    inputs = Input(shape=(28*28), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = adam.Adam(learning_rate=lr)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    # optimizers = ['adam', 'rmsprop', 'adadelta']
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    node1 = [10, 32, 64]
    node2 = [10, 32, 64]
    node3 = [10, 32, 64]
    
    return {'batch_size':batchs, 'lr':lr,
            # 'optimizer':optimizers, 
            'drop':dropout, 'activation':activation,
            'node1':node1, 'node2':node2, 'node3':node3
            }
    
hyperparameters = create_hyperparameter()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=10)

start = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.2)
end = time.time()-start

print('model.best_params: ', model.best_params_)
print('model.best_estimator: ', model.best_estimator_)
print('model.best_score: ', model.best_score_)
print('model.score: ', model.score)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('acc score: ', accuracy_score(y_test, y_pred))
print('걸린 시간: ', end)

# acc score:  0.9598
# 걸린 시간:  215.25915098190308

# learning rate 그리드서치 넣으면 import 오류남