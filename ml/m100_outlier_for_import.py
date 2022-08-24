import matplotlib.pyplot as plt
import numpy as np
import math

# outlier 뽑을때 앞으로 이거 임포트해서 써먹기~~

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

outliers_list=[]
def outliers_printer(dataset):
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        plt.title(i)
        outliers_list.append([outliers_loc[0]])
        
    plt.show()
    
''' 아웃라이어 위치 리스트를 받아서 원하는 값으로 대체하는 방법
a3 =  [  89,  110,  120,  121,  137,  147,  172,  210,  211,  247,  310,
        365,  397,  398,  426,  448,  450,  567,  755,  802,  865,  935,
       1033, 1036, 1100, 1136, 1237, 1241, 1387, 1413, 1435, 1447, 1449,
       1461, 1471, 1521, 1540, 1652, 1665, 1722, 1782, 1919, 1936, 1947]

for i in range(len(a3)):
    x[a3[i]][3] = 20    
'''    


# outliers z스코어 처리 함수
def outliers(df, col):
    out = []
    m = np.mean(df[col])
    sd = np.std(df[col])
    
    for i in df[col]: 
        z = (i-m)/sd
        if np.abs(z) > 3: 
            out.append(i)
            
    print("Outliers:",out)
    print("med",np.median(out))
    return np.median(out)
