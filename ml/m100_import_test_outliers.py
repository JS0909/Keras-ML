import matplotlib.pyplot as plt
import numpy as np

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)
print(aaa.shape)
print(aaa)

import m100_outlier_for_import as m100

m100.outliers_printer(aaa)