import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
# 2.8버전부터 tensorflow.keras 안써도 됨, 걍 keras.으로 쓰라고 나옴

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)
x = x.reshape(70000, 28*28) # (70000, 784)

# [실습]
# pca를 통해 0.95 이상인 n_componets는 몇개?
# 0.95
# 0.99
# 0.999
# 1.0
# np.argmax

a=[]
pca = PCA(n_components=x.shape[1])
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
a.append(np.argmax(cumsum >= 0.95)+1)
a.append(np.argmax(cumsum >= 0.99)+1)
a.append(np.argmax(cumsum >= 0.999)+1)
a.append(np.argmax(cumsum >= 1.0)+1) # arg시리즈에서는 리스트 내 값에 대해서 알아서 iter형태로 불러와 비교해준다

print(a)

# [154, 331, 486, 713]
