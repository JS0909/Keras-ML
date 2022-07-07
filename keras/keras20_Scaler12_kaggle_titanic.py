# [실습]
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# pandas의 y라벨의 종류 확인 train_set.columns.values
# numpy에서는 np.unique(y, return_counts=True)

# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

print(train_set.describe())
print(train_set.info())
print(train_set.isnull())
print(train_set.isnull().sum())
print(train_set.shape) # (10886, 12)
print(train_set.columns.values) # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape) # (712, 7)
print(y_train.shape) # (712, 1)
print(x_test.shape) # (179, 7)
print(y_test.shape) # (179, 1)

#2. 모델구성
model = Sequential()
model.add(Dense(80, input_dim=7))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
start_time = time.time()
log = model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[Es], validation_split=0.2)
end_time = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict) 
acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)
print('걸린 시간: ', end_time-start_time)

# 5. 제출 준비
submission = pd.read_csv(path + 'gender_submission.csv', index_col=0)

# y_submit = model.predict(test_set)
# y_submit = np.round(y_submit)
# y_submit = y_submit.astype(int)

# submission['Survived'] = y_submit
# submission.to_csv(path + 'gender_submission.csv', index=True)


# 1. 스케일러 하기 전
# loss :  0.5068862438201904
# acc스코어 :  0.776536312849162
# 걸린 시간:  6.674813747406006

# 2. MinMax scaler
# loss :  0.5094090700149536
# acc스코어 :  0.776536312849162
# 걸린 시간:  2.851576089859009

# 3. Standard scaler
# loss :  0.5106971859931946
# acc스코어 :  0.7653631284916201
# 걸린 시간:  2.454378843307495

# 4. MaxAbsScaler
# loss :  0.5017945170402527
# acc스코어 :  0.776536312849162
# 걸린 시간:  3.6580231189727783

# 5. RobustScaler
# loss :  0.5067141056060791
# acc스코어 :  0.770949720670391
# 걸린 시간:  4.105631113052368