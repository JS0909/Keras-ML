import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target(Y)'] = y
print(df)

print('====================== 상관계수 히트 맵 ======================')
print(df.corr()) # 각 칼럼별로 서로 상관관계를 나타냄, 단순리니어모델로 쭉 돌려보고 나온 결과치니까 무조건 신뢰하기 힘듦, 신뢰도 7~80%
# 양의 상관계수 = 비례, 음의 상관계수 = 반비례
'''
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()