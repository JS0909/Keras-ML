import matplotlib.pyplot as plt
import numpy as np

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
   
def outliers_printer(dataset):
    for i in range(dataset.shape[1]):
        cols = dataset[:, i]
        outliers_loc = outliers(cols)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(dataset.shape[1],1,i+1)
        plt.boxplot(cols)
        
    plt.show()