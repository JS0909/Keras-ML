import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)

#================================================================
x = np.arange(12).reshape(4,3)
print(x)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)

print(x_pf)
print(x_pf.shape)

