import matplotlib.pyplot as plt
import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)
print(aaa.shape)
print(aaa)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5) # -5.0
    upper_bound = quartile_3 + (iqr * 1.5) # 19
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
   
def outliers_printer(dataset):
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(dataset.shape[1],1,i+1)
        plt.boxplot(col)
        
    plt.show()

outliers_printer(aaa)

# 1사분위:  4.0
# q2:  7.0
# 3사분위:  10.0
# 19.0
# 0 열의 이상치의 위치:  (array([ 0, 12], dtype=int64),)

# 1사분위:  200.0
# q2:  400.0
# 3사분위:  600.0
# 1200.0
# 1 열의 이상치의 위치:  (array([6], dtype=int64),)