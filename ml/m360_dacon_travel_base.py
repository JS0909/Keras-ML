# 데이터를 불러오고 살펴보기 위한 pandas 라이브러리
import pandas as pd

filepath = 'D:\study_data\_data\dacon_travel/'

# train 데이터 불러오기
train = pd.read_csv(filepath+'train.csv')

# test 데이터 불러오기
test = pd.read_csv(filepath+'test.csv')

# sample_submission 불러오기
sample_submission = pd.read_csv(filepath+'sample_submission.csv')

# describe 메소드는 수치형 변수들의 기초 통계량을 보여줍니다.
train.describe()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#상관관계 분석도
plt.figure(figsize=(15,10))

heat_table = train.corr()
mask = np.zeros_like(heat_table)
mask[np.triu_indices_from(mask)] = True
heatmap_ax = sns.heatmap(heat_table, annot=True, mask = mask, cmap='coolwarm', vmin=-1, vmax=1)
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=15, rotation=90)
heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=15)
plt.title('correlation between features', fontsize=40)
# plt.show()


# 먼저 결측치가 얼마나 있는지 확인합니다.
na_check = train.isna().sum()

na_check[na_check > 0]


# 해당 열들의 값을 살펴봅시다.
train[na_check[na_check > 0].keys()].head()

# pandas의 fillna 메소드를 활용하여 NAN 값을 채워니다.
train_nona = train.copy()

# 0 으로 채우는 경우
train_nona.DurationOfPitch = train_nona.DurationOfPitch.fillna(0)

# mean 값으로 채우는 경우
mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome']
for col in mean_cols:
    train_nona[col] = train_nona[col].fillna(train[col].mean())

# "Unknown"으로 채우는 경우
train_nona.TypeofContact = train_nona.TypeofContact.fillna("Unknown")

# 결과를 확인합니다.
train_nona.isna().sum()

object_columns = train.columns[train.dtypes == 'object']
print('object 칼럼은 다음과 같습니다 : ', list(object_columns))

from sklearn.preprocessing import LabelEncoder

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])


# MinMaxScaler를 준비해줍니다.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train_scale = train_enc.copy()

# 학습된 scaler를 사용하여 변환해줍니다.
train_scale[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.fit_transform(train_scale[['Age', 'DurationOfPitch', 'MonthlyIncome']])

# 결측치 처리
# 0 으로 채우는 경우
test.DurationOfPitch = test.DurationOfPitch.fillna(0)

# mean 값으로 채우는 경우
mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar','NumberOfTrips','NumberOfChildrenVisiting','MonthlyIncome']
for col in mean_cols:
    test[col] = test[col].fillna(test[col].median())

# "Unknown"으로 채우는 경우
test.TypeofContact = test.TypeofContact.fillna("Unknown")

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    
    # test 데이터를 이용해 encoder를 학습하는 것은 Data Leakage 입니다! 조심!
    encoder.fit(train_nona[o_col])
    
    # test 데이터는 오로지 transform 에서만 사용되어야 합니다.
    test[o_col] = encoder.transform(test[o_col])

test[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test[['Age', 'DurationOfPitch', 'MonthlyIncome']])


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 모델 선언
model = RandomForestClassifier(random_state=9)


# 분석할 의미가 없는 칼럼을 제거합니다.
train = train_scale.drop(columns=['id'])
test = test.drop(columns=['id'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]

print(x.head())

x = x.drop(columns=['OwnCar', 'NumberOfChildrenVisiting','NumberOfPersonVisiting','PreferredPropertyStar'])
test = test.drop(columns=['OwnCar','NumberOfChildrenVisiting','NumberOfPersonVisiting','PreferredPropertyStar'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=666, shuffle=True)

# 모델 학습
model.fit(x_train,y_train)

# 학습된 모델을 이용해 결과값 예측후 상위 10개의 값 확인
prediction1 = model.predict(test)
print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
print(prediction1[:10])
print(model.score(x_test, y_test))


model.fit(x,y.values.ravel())
prediction = model.predict(test)

# 예측된 값을 정답파일과 병합
sample_submission['ProdTaken'] = prediction

# submission을 csv 파일로 저장합니다.
# index=False란 추가적인 id를 부여할 필요가 없다는 뜻입니다. 
# 정확한 채점을 위해 꼭 index=False를 넣어주세요.
sample_submission.to_csv(filepath+'submissionB.csv',index = False)


