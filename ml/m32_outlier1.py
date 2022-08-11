# ipad 그림 참고

import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
# aaa = np.array([10,1,2,4,5,6,7,8,10,5])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5) # -5.0
    upper_bound = quartile_3 + (iqr * 1.5) # 19
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound)) # 괄호 안의 조건을 만족하는 값의 인덱스를 반환함

outliers_loc = outliers(aaa)
print('이상치의 위치: ', outliers_loc)
outliers_loc = np.array(outliers_loc) # (1, 2)
print(outliers_loc[0]) # [0 12]
outliers_idx = []
outliers_idx.extend(outliers_loc[0])
print(outliers_idx)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()





