import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

filepath = 'D:\study_data\_data\dacon_travel/'
train = pd.read_csv(filepath+'train.csv', index_col=0)
test = pd.read_csv(filepath+'test.csv', index_col=0)
submission = pd.read_csv(filepath+'submission.csv', index_col=0)

# print(train.head())
# print(train.info())
# print(train.isnull().sum())

# 결측치 TypeofContact 빼고 중간값으로 대체함, 데이터 수치들 보면 중간값이 제일 무난할거 같음--------------------
train['Age'].fillna(train['Age'].median(), inplace=True)
train['TypeofContact'].fillna('N', inplace=True) # N으로 채운 이유는 콘택 타입 없는 건 '없음'으로 주고 처리하기 위해
train['DurationOfPitch'].fillna(train['DurationOfPitch'].median(), inplace=True)
train['NumberOfFollowups'].fillna(train['NumberOfFollowups'].median(), inplace=True)
train['PreferredPropertyStar'].fillna(train['PreferredPropertyStar'].median(), inplace=True)
train['NumberOfTrips'].fillna(train['NumberOfTrips'].median(), inplace=True)
train['NumberOfChildrenVisiting'].fillna(train['NumberOfChildrenVisiting'].median(), inplace=True)
train['MonthlyIncome'].fillna(train['MonthlyIncome'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['TypeofContact'].fillna('N', inplace=True)
test['DurationOfPitch'].fillna(test['DurationOfPitch'].median(), inplace=True)
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].median(), inplace=True)
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True)
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True)
test['NumberOfChildrenVisiting'].fillna(test['NumberOfChildrenVisiting'].median(), inplace=True)
test['MonthlyIncome'].fillna(test['MonthlyIncome'].median(), inplace=True)
# print(train.isnull().sum())
#-----------------------------------------------------------------------------------------------------------

# object타입 라벨인코딩--------------------
le = LabelEncoder()
idxarr = train.columns
idxarr = np.array(idxarr)

for i in idxarr:
      if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])
# print(train.info())
# ------------------------------------------

# 피처임포턴스 그래프 보기 위해 데이터프레임형태의 x_, y_ 놔둠 / 훈련용 넘파이어레이형태의 x, y 생성-----------
x_ = train.drop(['ProdTaken','NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1)
y_ = train['ProdTaken']
x = np.array(x_)
y = np.array(y_)
y = y.reshape(-1, 1)

test = test.drop(['NumberOfChildrenVisiting','NumberOfPersonVisiting','OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3
test = np.array(test)
# print(x.shape, y.shape)
#-----------------------------------------------------------------------------------------------------------

'''
# 이상치 그래프로 확인 ------------------------------------------------------------------
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def outliers_printer(dataset):
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        
    plt.show()

outliers_printer(x)
# 이상치 그래프로 확인 ------------------------------------------------------------------
'''

'''
# PCA 반복문 - XGB 테스트 // 별로 안좋게 나옴, LDA는 y라벨 두개뿐이라 큰 의미 없음
for i in range(x.shape[1]):
    pca = PCA(n_components=i+1)
    x2 = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
    model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(i+1, '의 결과: ', results)
'''

parameters_xgb = {
            'n_estimators':[100,200,300,400,500],
            'learning_rate':[0.1,0.2,0.3,0.5,1,0.01,0.001],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0,1,2,3,4,5,7,10,100],
            'min_child_weight':[0,0.1,0.001,0.5,1,5,10,100],
            'subsample':[0,0.1,0.2,0.3,0.5,0.7,1],
            'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
            'reg_lambda':[0,0.1,0.01,0.001,1,2,10],
              } 

parameters_rnf = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}
]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=134, shuffle=True)
# print(np.unique(y_train, return_counts=True))

# 2. 모델
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
rnf = RandomForestClassifier()

# model = xgb
# model = rnf
# model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(xgb, parameters_xgb, cv=5, n_jobs=-1, verbose=2))
model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(rnf, parameters_rnf, cv=5, n_jobs=-1, verbose=2))
# model = make_pipeline(MinMaxScaler(), xgb)
# model = make_pipeline(MinMaxScaler(), rnf)

# 3. 훈련
'''
# 칼럼별 중요도 확인--------------------------------------------------
def plot_feature_importances(model):
    plt.figure(figsize=(13,8))
    n_features = x_.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
                # x                     y
    plt.yticks(np.arange(n_features), x_.columns) # 눈금 설정
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features) # ylimit : 축의 한계치 설정
    if str(models[i]).startswith('XGBClassifier'):
        plt.title('XGB()')
    else:
        plt.title(model)

plot_feature_importances(model)
plt.show()
# NumberOfChildrenVisiting, NumberOfPersonVisiting, OwnCar 가장 영향력 적음
# ----------------------------------------------------------------------
'''

start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('스코어: ', results)
print('걸린 시간: ', end-start)

# 5. 제출 준비
y_submit = model.predict(test)
submission['ProdTaken'] = y_submit

submission.to_csv(filepath + 'submission.csv', index = True)





# submission 1번파일
# 스코어:  0.8673469387755102
# 걸린 시간:  22.424696445465088

# submission 2번파일
# 스코어:  0.8567774936061381
# 걸린 시간:  147.84010410308838

# submission 3번파일
# 스코어:  0.8695652173913043
# 걸린 시간:  0.6045560836791992

# submission 4번파일 랜포
# 스코어:  0.8797953964194374
# 걸린 시간:  0.1614227294921875

# submission 5번파일 랜포+halving
# 스코어:  0.8746803069053708
# 걸린 시간:  4.660583972930908

# submission 6번파일 랜포+halving+랜덤시드 134로 바꿈
# 스코어:  0.887468030690537
# 걸린 시간:  4.499013185501099






'''
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Age                       1955 non-null   float64
 1   TypeofContact             1955 non-null   int32
 2   CityTier                  1955 non-null   int64
 3   DurationOfPitch           1955 non-null   float64
 4   Occupation                1955 non-null   int32
 5   Gender                    1955 non-null   int32
 6   NumberOfPersonVisiting    1955 non-null   int64
 7   NumberOfFollowups         1955 non-null   float64
 8   ProductPitched            1955 non-null   int32
 9   PreferredPropertyStar     1955 non-null   float64
 10  MaritalStatus             1955 non-null   int32
 11  NumberOfTrips             1955 non-null   float64
 12  Passport                  1955 non-null   int64
 13  PitchSatisfactionScore    1955 non-null   int64
 14  OwnCar                    1955 non-null   int64
 15  NumberOfChildrenVisiting  1955 non-null   float64
 16  Designation               1955 non-null   int32
 17  MonthlyIncome             1955 non-null   float64
 18  ProdTaken                 1955 non-null   int64
 '''