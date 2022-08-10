import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]])

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
# imputer = SimpleImputer() # 디폴트는 평균값
# imputer = SimpleImputer(strategy='mean')
# imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(strategy='most_frequent') # 가장 빈번한 것 중 가장 앞에오는 값
# imputer = SimpleImputer(strategy='constant') # 상수, 디폴트는 0
# imputer = SimpleImputer(strategy='constant', fill_value=77)

# imputer = KNNImputer() # nan값의 가까운 값들의 평균값

imputer = IterativeImputer(max_iter = 10, random_state = 0) # 선형회귀방식을 써서 결측치를 예측한다, iter 수는 몇번 선형회귀를 돌릴지인듯


imputer.fit(data)
# imputer.fit(data['x1']) // 일부 칼럼만
data2 = imputer.transform(data)
print(data2)







