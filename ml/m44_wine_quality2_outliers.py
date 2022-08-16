import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pylab as plt

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

# data = np.array(data)
data2 = data.values
# data = data.to_numpy()

print(type(data2)) # <class 'numpy.ndarray'>
print(data2.shape) # (4898, 12)

x = data2[:, :11]
y = data2[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True)) # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(data['quality'].value_counts())
# 6    2198
# 5    1457
# 7     880
# 8     175
# 4     163
# 3      20
# 9       5

# 분류 문제에서는 데이터의 분포때문에 스코어가 많이 갈린다. 분포를 꼭 확인하자.

# 아웃라이어 확인

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
    collist = []
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        plt.title(i)
        collist.append(i)
        collist.append(outliers_loc)
        
    plt.show()
    return collist

outwhere = outliers(x)
print(outwhere)
collist = outliers_printer(x)
print(np.array(collist))

# 1사분위:  0.42
# q2:  3.21
# 3사분위:  9.4
# 22.87
# (array([   0,    0,    1, ..., 4895, 4896, 4897], dtype=int64), array([5, 6, 6, ..., 6, 6, 6], dtype=int64))

# 아웃라이어 처리
a0 =[  98,  169,  207,  294,  358,  551,  555,  656,  774,  847,  873,
       1053, 1109, 1123, 1124, 1138, 1139, 1141, 1142, 1146, 1147, 1178,
       1205, 1210, 1214, 1228, 1239, 1263, 1300, 1307, 1308, 1309, 1312,
       1313, 1334, 1349, 1372, 1373, 1404, 1420, 1423, 1505, 1526, 1536,
       1544, 1561, 1564, 1580, 1581, 1586, 1621, 1624, 1626, 1627, 1690,
       1718, 1730, 1758, 1790, 1801, 1856, 1857, 1858, 1900, 1930, 1932,
       1936, 1951, 1961, 2014, 2017, 2028, 2030, 2050, 2083, 2127, 2154,
       2162, 2191, 2206, 2250, 2266, 2308, 2312, 2321, 2357, 2378, 2400,
       2401, 2404, 2535, 2540, 2541, 2542, 2607, 2625, 2639, 2668, 2872,
       3094, 3095, 3220, 3265, 3307, 3410, 3414, 3526, 3710, 3915, 4259,
       4446, 4470, 4518, 4522, 4679, 4786, 4787, 4792, 4847]
for i in range(len(a0)):
    x[a0[i]][0] = data['fixed acidity'].median()
    
a8 = [  72,  115,  250,  320,  507,  509,  830,  834,  892,  928, 1014,
       1095, 1214, 1250, 1255, 1335, 1352, 1361, 1385, 1482, 1575, 1578,
       1583, 1649, 1681, 1758, 1834, 1852, 1900, 1946, 1959, 1960, 1976,
       2036, 2063, 2075, 2078, 2099, 2104, 2162, 2211, 2238, 2247, 2280,
       2281, 2319, 2321, 2364, 2369, 2370, 2399, 2646, 2711, 2771, 2853,
       2862, 2864, 2872, 2895, 2956, 2964, 3025, 3128, 3556, 3598, 3762,
       4109, 4135, 4259, 4470, 4565, 4567, 4601, 4744, 4787]
for i in range(len(a8)):
    x[a8[i]][8] = data['pH'].median()
    
a9=[  80,  154,  209,  245,  339,  357,  411,  415,  530,  563,  701,
        757,  758,  759,  778,  782,  797,  852,  854,  855,  866,  868,
        879,  974, 1016, 1036, 1099, 1160, 1169, 1280, 1285, 1293, 1294,
       1386, 1394, 1407, 1412, 1455, 1482, 1515, 1590, 1807, 1809, 1843,
       1848, 1862, 1969, 1971, 1995, 1997, 1998, 2006, 2057, 2073, 2211,
       2234, 2264, 2267, 2348, 2403, 2441, 2594, 2634, 2637, 2642, 2656,
       2668, 2721, 2748, 2750, 2872, 2873, 2874, 2893, 2926, 2930, 2931,
       3057, 3079, 3206, 3207, 3231, 3423, 3425, 3426, 3429, 3430, 3431,
       3436, 3458, 3532, 3641, 3642, 3680, 3683, 3685, 3697, 3736, 3754,
       3764, 3904, 3915, 3975, 3982, 3998, 3999, 4000, 4012, 4023, 4026,
       4065, 4072, 4239, 4391, 4401, 4582, 4617, 4696, 4753, 4792, 4815,
       4818, 4886, 4887]
for i in range(len(a9)):
    x[a9[i]][9] = data['sulphates'].median()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, shuffle=True, train_size=0.85, stratify=y)

# 2. 모델
model = RandomForestClassifier(random_state=666)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
y_pred = model.predict(x_test)

score = model.score(x_test, y_test)
print('model.score: ', score) # 0.710204081632653
print('acc_score: ', accuracy_score(y_pred, y_test)) # 0.710204081632653

from sklearn.metrics import f1_score
# 기본적으로 이진분류에서만 사용함
# 정밀도와 재현율 = 프리시전과 리콜

print('f1_macro: ', f1_score(y_pred, y_test, average='macro')) # 칼럼별로 f1을 따져서 평균을 내서 비교함
print('f1_micro: ', f1_score(y_pred, y_test, average='micro')) # micro f1 = acc score 

# 아웃라이어 처리 안하고
# model.score:  0.710204081632653
# acc_score:  0.710204081632653
# f1_macro:  0.4556941766626644
# f1_micro:  0.710204081632653

# model.score:  0.7061224489795919
# acc_score:  0.7061224489795919
# f1_macro:  0.4594158754369309
# f1_micro:  0.7061224489795919




