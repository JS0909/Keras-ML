from flask import Flask, render_template, request
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score 
 
breed = {0:'beagle', 1:'bichon', 2:'bulldog', 3:'chihuahua', 4:'chow_chow', 
5:'cocker_spaniel', 6:'collie', 7:'dachshund', 8:'fox_terrier', 9:'german_shepherd', 
10:'golden_retriever', 11:'greyhound', 12:'husky', 13:'jack_russell_terrier', 14:'jindo', 
15:'labrador_retriever', 16:'maltese', 17:'miniature_pinscher', 18:'papillon', 19:'pomeranian', 
20:'poodle', 21:'pug', 22:'rottweiler', 23:'samoyed', 24:'schnauzer', 25:'shiba',
26:'shihtzu', 27:'spitz', 28:'welsh_corgi', 29:'yorkshire_terrier'}

age = {0:'11year_', 1:'5month_4year', 2:'5year_10year', 3:'_4month'}
age_class = {0:'노년', 1:'청년', 2:'중장년', 3:'유년'}

# 1. 데이터
filepath = 'd:/study_data/_save/_npy/_project/'
suffix = '.npy'

x_train = np.load(filepath+'train_x'+suffix)
y1_train = np.load(filepath+'train_y1'+suffix)
y2_train = np.load(filepath+'train_y2'+suffix)

x_test = np.load(filepath+'test_x'+suffix)
y1_test = np.load(filepath+'test_y1'+suffix)
y2_test = np.load(filepath+'test_y2'+suffix)


# 2. 모델
from keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model

# 사전 학습된 모델 불러오기
input_tensor = Input(shape=(224,224,3))
model = VGG16(weights='imagenet', include_top=False, input_tensor = input_tensor)

# 모델 Layer 데이터화
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Layer 추가
x = layer_dict['block5_pool'].output
# Cov2D Layer +
x = Conv2D(filters = 64, kernel_size=(3, 3), activation='relu')(x)
# MaxPooling2D Layer +
x = MaxPooling2D(pool_size=(2, 2))(x)
# Flatten Layer +
x = Flatten()(x)
# FC Layer +
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x1 = Dropout(0.5)(x)
x21 = Dense(30, activation='softmax')(x1)
x22 = Dense(4, activation='softmax')(x1)

# new model 정의
new_model = Model(inputs = model.input, outputs = [x21,x22])






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

# 테스트용 이미지로 프레딕트
testing_img = np.load(filepath+'testing_img'+suffix)
testpred_breed, testpred_age = model.predict(testing_img)

testpred_breed_arg = tf.argmax(testpred_breed, axis=1)
testpred_age_arg = tf.argmax(testpred_age, axis=1)
testpred_breed_arr = np.array(testpred_breed_arg)
testpred_age_arr = np.array(testpred_age_arg)

dog_sang = ['samoyed', 'rottweiler', 'german_shepherd', 'jack_russell_terrier', 
            'husky', 'collie', 'labrador_retriever', 'golden_retriever']
dog_jung = ['jindo','chow_chow', 'bulldog', 'greyhound', 'maltese', 'miniature_pinscher',
            'papillon', 'pomeranian', 'poodle', 'shiba', 'schnauzer',
            'spitz', 'beagle', 'fox_terrier', 'bichon', 'dachshund', 'cocker_spaniel', 'welsh_corgi']
dog_ha = ['chihuahua', 'pug', 'shihtzu', 'yorkshire_terrier']
age_jung = '중장년'
age_u = '유년'
age_no = '노년'

if breed[testpred_breed_arr[-1]] in dog_sang:
    num1 = 5
elif breed[testpred_breed_arr[-1]] in dog_jung:
    num1 = 2.5
elif breed[testpred_breed_arr[-1]] in dog_ha:
    num1 = 0

if age_class[testpred_age_arr[-1]]==age_jung:
    num2 = 2.5
    age_weight = 2
elif age_class[testpred_age_arr[-1]]==age_u:
    num2 = 0
    age_weight = 3
elif age_class[testpred_age_arr[-1]]==age_no:
    num2 = 0
    age_weight = 2
else:
    num2 = 5
    age_weight = 2

exercise = num1+num2
if exercise==10:
    ex = '최상 / 산책 1시간 30분 ~ 2시간'
elif exercise==7.5:
    ex = '상 / 산책 1시간 ~ 1시간 30분'
elif exercise==5:
    ex = '중 / 산책 30분 ~ 1시간'
elif exercise==2.5:
    ex = '하 / 산책 20분 ~ 40분'
else:
    ex = '최하 / 산책 20분'
        
# food = ((weight * 30 + 70) * age_weight) / kcal

age_po = round(testpred_age[0][tuple(testpred_age_arg)]*100, 5)
breed_result = breed[testpred_breed_arr[-1]]
breed_po = round(testpred_breed[0][tuple(testpred_breed_arg)]*100, 3)
age_result = age[testpred_age_arr[-1]]
age_cl = age_class[testpred_age_arr[-1]]

# ===== 정보 출력 =====
print('종: ', breed_result, '//', breed_po,'%')
print('나이: ', age_result, age_cl, age_po, '%')
print('적정 활동량: ', ex)
# print('적정 사료양: ', round(food,3), 'g')