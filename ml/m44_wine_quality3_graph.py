import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. 데이터
file = 'D:\study_data\_data\wine/winequality-white.csv'

data = pd.read_csv(file, index_col=None, header=0,sep=';')

print(data.shape) # (4898, 12)
print(data.describe()) # std: 표준편차
print(data.info())

#=============== [실습] 그래프 그리기 ==================
# 1. value_counts 쓰지 말고
# 2. groupby, count(), plt.bar차트 쓰기
# plt.bar로 quality칼럼 그리기
import matplotlib.pylab as plt

count_data = data.groupby('quality')['quality'].count()

plt.bar(count_data.index, count_data)
           # x              # y
plt.show()
