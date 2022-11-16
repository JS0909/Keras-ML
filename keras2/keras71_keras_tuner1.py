import tensorflow as tf
from tensorflow.keras.datasets import mnist
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizer_v2.adam import Adam # tf 2.8.2
# from keras.optimizers import Adam      # tf 2.9.1


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.

def get_model(hp):
    hp_unit1 = hp.Int('units1', min_value=16, max_value=512, step=15)
    hp_unit2 = hp.Int('units2', min_value=16, max_value=512, step=15)
    hp_unit3 = hp.Int('units3', min_value=16, max_value=512, step=15)
    hp_unit4 = hp.Int('units4', min_value=16, max_value=512, step=15)
    
    hp_drop1 = hp.Choice('dropout1', values=[0.0, 0.2, 0.3, 0.4, 0.5])
    hp_drop2 = hp.Choice('dropout2', values=[0.0, 0.2, 0.3, 0.4, 0.5])
    
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(hp_unit1, activation='relu'))   # activation도 파라미터로 잡아줄 수 있음
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit2, activation='relu'))   
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit3, activation='relu'))   
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(hp_unit4, activation='relu'))   
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(10, activation='softmax'))   
    
    model.compile(optimizer=Adam(learning_rate=hp_lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

kerastuner = kt.Hyperband(get_model,
                          directory='my_dir',
                          objective='val_accuracy',
                          max_epochs=6,
                          project_name='kerastuner-mnist'
                          )

kerastuner.search(x_train, y_train,
                  validation_data=(x_test, y_test), epochs=5)    

best_hps = kerastuner.get_best_hyperparameters(num_trials=2)[0]

print('best parameter - units1 : ', best_hps.get('units1'))
print('best parameter - units2 : ', best_hps.get('units2'))
print('best parameter - units3 : ', best_hps.get('units3'))
print('best parameter - units4 : ', best_hps.get('units4'))

print('best parameter - dropout1 : ', best_hps.get('dropout1'))
print('best parameter - dropout2 : ', best_hps.get('dropout2'))

print('best parameter - learning_rate : ', best_hps.get('learning_rate'))


# 체크포인트 땡겨와서 predict 바로 할 수 있음

