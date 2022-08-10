import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

# 1. 데이터
# datasets = load_iris() # (150, 4) -> (150, 2)
# datasets = load_breast_cancer() # (569, 30) -> (569, 1)
# datasets = load_wine() # (178, 13) -> (178, 2)
# datasets = fetch_covtype() # (581012, 54) -> (581012, 6)
datasets = load_digits() # (1797, 64) -> (1797, 9)

x = datasets.data
y = datasets.target
print(x.shape)

lda = LinearDiscriminantAnalysis()
lda.fit(x,y)
x = lda.transform(x)
print(x.shape)

lda_EVR = lda.explained_variance_ratio_
cumsum = np.cumsum(lda_EVR)
print(cumsum)