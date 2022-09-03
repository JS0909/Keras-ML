import pandas as pd
import numpy as np 


path = 'D:\_data\dacon_travel/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)


train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)


print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())


train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
# print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['ProdTaken'])['PreferredPropertyStar'].mean())

alldata = [train_set,test_set]
for dataset in alldata:    
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4
    dataset.loc[ dataset['Age'] > 59, 'Age'] = 5


train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfChildrenVisiting'].notnull()].groupby(['MaritalStatus'])['NumberOfChildrenVisiting'].mean())

# print(train_set['Occupation'].unique()) # ['Small Business' 'Salaried' 'Large Business' 'Free Lancer']
train_set.loc[ train_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'
test_set.loc[ test_set['Occupation'] =='Free Lancer' , 'Occupation'] = 'Salaried'

train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = np.array(train_set.columns)
for col in cols:
      if train_set[col].dtype == 'object':
        train_set[col] = le.fit_transform(train_set[col])
        test_set[col] = le.fit_transform(test_set[col])
# print(train_set)

def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out, [25,50,75])
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
                     
print(outliers(train_set['DurationOfPitch'])[0])
  
x = train_set.drop(['NumberOfFollowups', 'OwnCar', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting', 'MonthlyIncome', 'ProdTaken'], axis=1)
# x = train_set.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfFollowups', 'OwnCar', 'NumberOfPersonVisiting', 'NumberOfChildrenVisiting', 'MonthlyIncome'], axis=1)
y = train_set['ProdTaken']
print(x.shape) #1911,13

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.91,shuffle=True,random_state=1234,stratify=y)


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=6,shuffle=True,random_state=123)
cat_paramets = {"learning_rate" : [0.01], 'depth' : [8], 'od_pval' : [0.12673190617341812], 
                # 'model_size_reg': [0.54233262345], 
                'fold_permutation_block': [142], 'l2_leaf_reg' :[0.33021257848638497]}

cat = CatBoostClassifier(verbose=0,random_state=96,n_estimators=1304)
model = GridSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1)


from sklearn.metrics import accuracy_score
model.fit(x_train,y_train)   
y_predict = model.predict(x_test)
accscore = accuracy_score(y_test,y_predict)
print('acc :', accscore)

model.fit(x,y) # fitting again for all data
sub_y = model.predict(test_set)
sub_y = np.round(sub_y,0)
submission = pd.read_csv(path + 'sample_submission.csv')
submission['ProdTaken'] = sub_y

submission.to_csv(path+'submission.csv',index=False)

import joblib as jb
jb.dump(model, path + 'model_save.dat')



# acc : 0.9772727272727273