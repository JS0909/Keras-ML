import pandas as pd
import random
import os
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier,XGBRegressor

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


path = 'D:/study_data/_data/dacon_antena/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(704) # Seed 고정

train_df = pd.read_csv(path + 'train.csv')
test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])
train = np.array(train_df)


# print(train_df.describe(percentiles=precent))
# print(train_df.info())
# print(train_df.columns.values)
# print(train_df.isnull().sum())

 
train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

train_x = train_x.drop(['X_04'], axis=1)
test_x = test_x.drop(['X_04'], axis=1)


cols = ["X_10","X_11"]
train_x[cols] = train_x[cols].replace(0, np.nan)

imp = IterativeImputer(estimator = LinearRegression(), 
                       tol= 1e-10, 
                       max_iter=30, 
                       verbose=2, 
                       imputation_order='roman')

train_x = pd.DataFrame(imp.fit_transform(train_x), columns=train_x.columns)

train_10 = outliers(train_x['X_10'])[0]
print(train_10)




model = MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.08, gamma = 1, subsample=0.75, colsample_bytree = 1, max_depth=7)).fit(train_x, train_y)

preds = model.predict(test_x)
print(preds.shape)
print(model.score(train_x, train_y))



# 5. 제출준비
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