from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.python.keras.layers import concatenate, Concatenate

tf.random.set_seed(9) # 하이퍼 파라미터 튜닝 용이하게 하기 위해

breed = {0:'beagle', 1:'bichon', 2:'bulldog', 3:'chihuahua', 4:'chow_chow', 
5:'cocker_spaniel', 6:'collie', 7:'dachshund', 8:'fox_terrier', 9:'german_shepherd', 
10:'golden_retriever', 11:'greyhound', 12:'husky', 13:'jack_russell_terrier', 14:'jindo', 
15:'labrador_retriever', 16:'maltese', 17:'miniature_pinscher', 18:'papillon', 19:'pomeranian', 
20:'poodle', 21:'pug', 22:'rottweiler', 23:'samoyed', 24:'schnauzer', 25:'shiba',
26:'shihtzu', 27:'spitz', 28:'welsh_corgi', 29:'yorkshire_terrier'}

age = {0:'11year_', 1:'5month_4year', 2:'5year_10year', 3:'_4month'}
age_class = {0:'유년', 1:'청년', 2:'중장년', 3:'노년'}

# 1. 데이터
filepath = 'd:/study_data/_save/_npy/_project/'
suffix = '.npy'

x_train = np.load(filepath+'train_x'+suffix)
y1_train = np.load(filepath+'train_y1'+suffix)
y2_train = np.load(filepath+'train_y2'+suffix)

x_test = np.load(filepath+'test_x'+suffix)
y1_test = np.load(filepath+'test_y1'+suffix)
y2_test = np.load(filepath+'test_y2'+suffix)

testing_img = np.load(filepath+'testing_img'+suffix)

# print(x_train.shape) # (4000, 150, 150, 3)
# print(y1_train.shape, y2_train.shape) # (6438, 30) (6438, 4)
# print(y1_test.shape, y2_test.shape) # (1610, 30) (1610, 4)

# print(y2_train)

'''
# 2. 모델구성
# 2-1. input모델
input1 = Input(shape=(150, 150, 3))
conv1 = Conv2D(32,(2,2), padding='same', activation='swish')(input1)
mp1 = MaxPool2D()(conv1)
conv2 = Conv2D(32,(2,2), activation='swish')(mp1)
flat1 = Flatten()(conv2)
dense1 = Dense(32, activation='relu')(flat1)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(32, activation='relu')(dense2)

# 2-2. output모델1
output1 = Dense(10)(output)
output2 = Dense(10)(output1)
last_output1 = Dense(30, activation='softmax')(output2)

# 2-3. output모델2
output3 = Dense(10)(output)
output4 = Dense(10)(output3)
last_output2 = Dense(4, activation='softmax')(output4)

model = Model(inputs=input1, outputs=[last_output1, last_output2])
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
log = model.fit(x_train, [y1_train, y2_train], epochs=1000, batch_size=32, callbacks=[Es], validation_split=0.2)
model.save('D:/study_data/_save/_h5/project.h5')
'''
model = load_model('D:/study_data/_save/_h5/project.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, [y1_test, y2_test])
print('tested loss : ', loss)

y1_pred, y2_pred = model.predict(x_test)
y1_pred = tf.argmax(y1_pred, axis=1)
y1_test_arg = tf.argmax(y1_test, axis=1)
y2_pred = tf.argmax(y2_pred, axis=1)
y2_test_arg = tf.argmax(y2_test, axis=1)
acc_sc1 = accuracy_score(y1_test_arg,y1_pred)
acc_sc2 = accuracy_score(y2_test_arg,y2_pred)
print('y1_acc스코어 : ', acc_sc1)
print('y2_acc스코어 : ', acc_sc2)

y1_pred = np.array(y1_pred)
y2_pred = np.array(y2_pred)
a = range(0, 10)
for i in a:
    print(y1_pred[i])
    print(y2_pred[i], '\n')

testpred_breed, testpred_age = model.predict(testing_img)
# print('종 확률: ', testpred_breed[0][tf.argmax(testpred_breed, axis=1)])
# print('나이 확률: ', testpred_age[0][tf.argmax(testpred_age, axis=1)])


testpred_breed_arg = tf.argmax(testpred_breed, axis=1)
testpred_age_arg = tf.argmax(testpred_age, axis=1)
testpred_breed_arr = np.array(testpred_breed_arg)
testpred_age_arr = np.array(testpred_age_arg)


print('종: ', breed[testpred_breed_arr[-1]], ' // 확률:', testpred_breed[0][tuple(testpred_breed_arg)])
print('나이: ', age[testpred_age_arr[-1]], age_class[testpred_age_arr[-1]],' // 확률:', testpred_age[0][tuple(testpred_age_arg)])

dog_jung = ['chow_chow', 'bulldog', 'greyhound', 'maltese', 'miniature_pinscher',
            'papillon', 'pomeranian', 'poodle', 'shiba', 'spitz']
dog_ha = ['chihuahua', 'pug', 'shihtzu', 'yorkshire_terrier']
age_jung = '중장년'
age_ha = ['유년', '노년']

if breed[breed[testpred_breed_arr[-1]]]==dog_jung:
    num1 = 2.5
elif breed[breed[testpred_breed_arr[-1]]]==dog_ha:
    num1 = 0
else:
    num1 = 5

if age_class[testpred_age_arr[-1]]==age_jung:
    num2 = 2.5
elif age_class[testpred_age_arr[-1]]==age_ha:
    num2 = 0
else:
    num2 = 5

        

# 인덱스 튜플화해서 접근하라고 future warning 메세지 뜸
# FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; 
# use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, 
# `arr[np.array(seq)]`, which will result either in an error or a different result.  


#----------- Android studio에서 사용하기 위해 tflite 파일로 모델 변환------------------
# converter = tf.lite.TFLiteConverter.from_keras_model(model) 
# tflite_model = converter.convert() 
# open("d:/study_data/_save/converted_model.tflite", "wb").write(tflite_model)
