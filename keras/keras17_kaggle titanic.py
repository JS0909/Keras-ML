# [실습]
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping

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

train_set = train_set.drop(columns='Cabin', axis=1) # Cabin 칼럼 삭제
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True) # Age의 nan값을 Age 평균으로 대체
print(train_set['Embarked'].mode()) # Embarked에 어떤 오브젝트들이 들어가 있는지 확인 // 'S', 'C', 'Q'
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True) # Embarked의 nan값을 모두 0으로 채움
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
# @@@@@ 여기서 인코딩을 쓰지 않고 리스트 내용 대체로 해결한 이유 @@@@@
# 성별을 남자면 0, 여자면 1로 대체 / Embarked에서 S면 0, C면 1, Q면 2로 모두 대체
# Onehot-encoding을 굳이 하지 않아도 되는 이유는 어차피 똑같이 0, 1로 대체할건데 굳이 행열 만들어서 체킹할 필요 없음 여기서는 가중치가 훈련에 영향 안줌
# 인코딩하면 인덱스 값으로 뿌려주는 것으로 오브젝트가 숫자로 변환되는데 필요 없는 연산을 한번 더 하는 셈임
# 그냥 리스트 내용을 숫자로 대체해서 실행하는 게 나음
# 근데 만약 오브젝트들이 너무 많다면 그냥 인코딩을 해버리는게 코드는 간결해질 것임

y = train_set['Survived'] # Survived를 제출해야되니까 train_set에서 Survived 칼럼을 떼와서 y에 넣음
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) # train_set에서 처리못할 데이터 PassengerId','Name','Ticket를 제외 + y값인 Survived를 제외한 나머지를 x에 넣음
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

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
model.add(Dense(1, activation='sigmoid')) # 이진 분류로 아웃풋 뽑음
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[Es], validation_split=0.2)
# log변수에 fit 내용 저장하는 건 나중에 그래프 그릴때 쓸려고 / 그래프 안그리면 필요x

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = np.round(y_predict) 
# y_predict는 살아남을 확률로 구해지기 때문에 반올림 시켜서 죽었다(0), 살았다(1)로 바꿔줘야 y_test와 비교가 가능
acc_sc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc_sc)

# 5. 제출 준비
submission = pd.read_csv(path + 'gender_submission.csv', index_col=0)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum()) # Fare에 nan값이 하나 들어있음 근데 train_set에는 거기에 nan값 없었음..
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True) # Fare에 nan값 하나 있음
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y_submit = model.predict(test_set) # 제출용 결과값 변수에 저장
y_submit = np.round(y_submit) # 확률에서 반올림을 시켜서 죽음(0), 살음(1)을 뽑아냄
y_submit = y_submit.astype(int) # 이상하게 subission 파일이 1.0, 0.0으로 표기되서 int타입으로 전환시키는 작업(소수점 아예 없애는 작업)

submission['Survived'] = y_submit
submission.to_csv(path + 'gender_submission.csv', index=True)