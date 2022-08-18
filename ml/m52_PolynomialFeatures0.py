import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)


# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]
#  [ 1.  6.  7. 36. 42. 49.]]
# (4, 6)

#================================================================
x = np.arange(12).reshape(4,3)
print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)


# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
# (4, 10)