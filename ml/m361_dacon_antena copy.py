import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier,XGBRegressor


path = 'D:/study_data/_data/dacon_antena/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(704) # Seed 고정

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))


train_df = pd.read_csv(path + 'train.csv')
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])

# print(train_df.info())
# print(train_df.columns.values)
# print(train_df.isnull().sum())

 
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

cols = ["X_10","X_11"]
train_x[cols] = train_x[cols].replace(0, np.nan)

imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman'
                       )

train_x = pd.DataFrame(imp.fit_transform(train_x), columns=train_x.columns)

train_01 = outliers(train_x['X_01'])[0]
train_02 = outliers(train_x['X_02'])[0]
train_03 = outliers(train_x['X_03'])[0]
train_04 = outliers(train_x['X_04'])[0]
train_05 = outliers(train_x['X_05'])[0]
train_06 = outliers(train_x['X_06'])[0]
train_07 = outliers(train_x['X_07'])[0]
train_08 = outliers(train_x['X_08'])[0]
train_09 = outliers(train_x['X_09'])[0]
train_10 = outliers(train_x['X_10'])[0]

train_11 = outliers(train_x['X_11'])[0]
train_12 = outliers(train_x['X_12'])[0]
train_13 = outliers(train_x['X_13'])[0]
train_14 = outliers(train_x['X_14'])[0]
train_15 = outliers(train_x['X_15'])[0]
train_16 = outliers(train_x['X_16'])[0]
train_17 = outliers(train_x['X_17'])[0]
train_18 = outliers(train_x['X_18'])[0]
train_19 = outliers(train_x['X_19'])[0]
train_20 = outliers(train_x['X_20'])[0]

train_21 = outliers(train_x['X_21'])[0]
train_22 = outliers(train_x['X_22'])[0]
train_23 = outliers(train_x['X_23'])[0]
train_24 = outliers(train_x['X_24'])[0]
train_25 = outliers(train_x['X_25'])[0]
train_26 = outliers(train_x['X_26'])[0]
train_27 = outliers(train_x['X_27'])[0]
train_28 = outliers(train_x['X_28'])[0]
train_29 = outliers(train_x['X_29'])[0]
train_30 = outliers(train_x['X_30'])[0]

train_31 = outliers(train_x['X_31'])[0]
train_32 = outliers(train_x['X_32'])[0]
train_33 = outliers(train_x['X_33'])[0]
train_34 = outliers(train_x['X_34'])[0]
train_35 = outliers(train_x['X_35'])[0]
train_36 = outliers(train_x['X_36'])[0]
train_37 = outliers(train_x['X_37'])[0]
train_38 = outliers(train_x['X_38'])[0]
train_39 = outliers(train_x['X_39'])[0]
train_40 = outliers(train_x['X_40'])[0]

train_41 = outliers(train_x['X_41'])[0]
train_42 = outliers(train_x['X_42'])[0]
train_43 = outliers(train_x['X_43'])[0]
train_44 = outliers(train_x['X_44'])[0]
train_45 = outliers(train_x['X_45'])[0]
train_46 = outliers(train_x['X_46'])[0]
train_47 = outliers(train_x['X_47'])[0]
train_48 = outliers(train_x['X_48'])[0]
train_49 = outliers(train_x['X_49'])[0]
train_50 = outliers(train_x['X_50'])[0]

train_51 = outliers(train_x['X_51'])[0]
train_52 = outliers(train_x['X_52'])[0]
train_53 = outliers(train_x['X_53'])[0]
train_54 = outliers(train_x['X_54'])[0]
train_55 = outliers(train_x['X_55'])[0]
train_56 = outliers(train_x['X_56'])[0]

# tlist = [train_01,train_02,train_03,train_04,train_05,train_06,train_07,train_08,train_09,train_10,
#          train_11,train_12,train_13,train_14,train_15,train_16,train_17,train_18,train_19,train_20,
#          train_21,train_22,train_23,train_24,train_25,train_26,train_27,train_28,train_29,train_30,
#          train_31,train_32,train_33,train_34,train_35,train_36,train_37,train_38,train_39,train_40,
#          train_41,train_42,train_43,train_44,train_45,train_46,train_47,train_48,train_49,train_50,
#          train_51,train_52,train_53,train_54,train_55,train_56]
# for i, t in enumerate(tlist):
#     print(i+1, ':', len(t))

lead_outlier_index = np.concatenate((
                            train_01, #1 : 1145
                            # train_02, #2 : 6587
                            train_03, #3 : 699
                            
                            train_06, #6 : 419
                            # train_07, #7 : 2052
                            # train_08, #8 : 8193
                            train_09, #9 : 1400
                            train_10, #10 : 783
                            
                            train_11, #11 : 878
                            train_12, #12 : 315
                            train_13, #13 : 820
                            train_14, #14 : 282
                            train_15, #15 : 60
                            train_16, #16 : 257
                            train_17, #17 : 513
                            train_18, #18 : 247
                            train_19, #19 : 152
                            train_20, #20 : 18
                            
                            train_21, #21 : 61
                            train_22, #22 : 20
                            
                            train_24, #24 : 64
                            train_25, #25 : 135
                            train_26, #26 : 229
                            train_27, #27 : 589
                            train_28, #28 : 1034
                            train_29, #29 : 1168
                            # train_30, #30 : 5926
                            
                            train_31, #31 : 1848
                            train_32, #32 : 1862
                            # train_33, #33 : 3942
                           
                            train_38, #38 : 1524
                            train_39, #39 : 1499
                            train_40, #40 : 1449
                            
                            train_41, #41 : 550
                            train_42, #42 : 209
                            train_43, #43 : 246
                            train_44, #44 : 255
                            train_45, #45 : 59
                            # train_46, #46 : 5519
                            
                            train_49, #49 : 2826
                            train_50, #50 : 464
                            
                            train_51, #51 : 487
                            train_52, #52 : 442
                            train_53, #53 : 423
                            train_54, #54 : 411
                            train_55, #55 : 384
                            train_56  #56 : 433
    
    ),axis=None)

lead_not_outlier_index = []
for i in train_x.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
        
trainX_set_clean = train_x.loc[lead_not_outlier_index]      
trainX_set_clean = trainX_set_clean.reset_index(drop=True)
trainY_set_clean = train_y.loc[lead_not_outlier_index]      
trainY_set_clean = trainY_set_clean.reset_index(drop=True)

train_x = trainX_set_clean
train_y = trainY_set_clean
print(train_x.shape, train_y.shape)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, train_size=0.85, random_state=1234)

'''
# 베이지안옵티마이제이션-------------------------------------------------------------------------------------------------------------------
bayseian_params = {
    'colsample_bytree' : (0.5, 1),
    'max_depth' : (6,16),
    'min_child_weight' : (1, 50),
    'reg_alpha' : (0.01, 50),
    'reg_lambda' : (0.001, 1),
    'subsample' : (0.5, 1)
}

# bayseian_params = {
#     'colsample_bytree' : (0.7, 1.5),
#     'max_depth' : (5,15),
#     'min_child_weight' : (4, 11),
#     'reg_alpha' : (15, 35),
#     'reg_lambda' : (0.3, 1.5),
#     'subsample' : (0.2, 1.3)
# }


def lgb_function(max_depth, min_child_weight,subsample, colsample_bytree, reg_lambda,reg_alpha):
    params ={
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                    # 정수만
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                  # 0~1 사이값만
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'reg_lambda' : max(reg_lambda,0),                       # 양수만
        'reg_alpha' : max(reg_alpha,0),
    }
    
    # *여러개의인자를받겠다
    # **키워드받겠다(딕셔너리형태)
    model = MultiOutputRegressor(XGBRegressor(**params))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    
    return score

lgb_bo = BayesianOptimization(f=lgb_function, pbounds=bayseian_params, random_state=123)

lgb_bo.maximize(init_points=3, n_iter=50)
print(lgb_bo.max)

# {'target': 0.07854669971632096, 'params': {'colsample_bytree': 0.7408048655591737, 'max_depth': 9.816995154429137, 
#                                            'min_child_weight': 31.273860779507064, 'reg_alpha': 9.214580668533486, 
#                                            'reg_lambda': 0.24638597047759284, 'subsample': 0.87180566845496}}
#----------------------------------------------------------------------------------------------------------------------------------
'''

# 2. 모델
model = MultiOutputRegressor(XGBRegressor(n_estimators=500, learning_rate=0.02, 
                                          colsample_bytree= max(min(0.7408048655591737,1),0), max_depth=int(round(9.816995154429137)), 
                                          min_child_weight= int(round(31.273860779507064)), reg_alpha= max(9.214580668533486,0), 
                                           reg_lambda= max(0.24638597047759284,0), subsample= max(min(0.87180566845496,1),0)                                         ))

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
preds = model.predict(x_test)
print('r2:', r2_score(y_test, preds))
print(model.score(train_x, train_y))


# 5. 제출준비
model.fit(train_x, train_y)
preds = model.predict(test_x)

submit = pd.read_csv(path + 'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]

submit.to_csv(path + 'submission.csv', index=False)



# 0.28798862985210744

# 0.38385531397806155 / 08

# 0.3813003238320756

# 0.38400226638667195