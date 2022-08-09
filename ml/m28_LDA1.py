import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape) # (581012, 54)

pca = PCA(n_components=54)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
print(cumsum)