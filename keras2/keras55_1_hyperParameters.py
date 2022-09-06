from tabnanny import verbose
import numpy as np
from torch import dropout
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import keras

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2. model
def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    
    return {'batch_size':batchs, 'optimizer':optimizers, 
            'drop':dropout, 'activation':activation}
    
hyperparameters = create_hyperparameter()


from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

keras_model = KerasClassifier(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=2)

start = time.time()
model.fit(x_train, y_train, epochs=1, validation_split=0.2)
end = time.time()-start

print('걸린 시간: ', end)
print('model.best_params: ', model.best_params_)
print('model.best_estimator: ', model.best_estimator_)
print('model.best_score: ', model.best_score_)
print('model.score: ', model.score)

from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(x_test)
print('acc score: ', accuracy_score(y_test, y_pred))