# [실습]
# 784개 DNN으로 만든거(최상의 성능인 거 // 0.978이상)과 비교!!

# time 체크 / fit에서

# 1. 나의 최고의 DNN
# 시간:  362.1198184490204
# acc스코어 :  0.9445

# 2. 나의 최고의 CNN
# 시간:  152.1847949028015
# acc스코어 :  0.9765

# 3. PCA 0.95
# 154 의 결과:  0.9649
# 시간:  9.673815965652466

# 4. PCA 0.99
# 331 의 결과:  0.9631
# 시간:  14.473661184310913

# 5. PCA 0.999
# 486 의 결과:  0.963
# 시간:  18.711859703063965

# 6. PCA 1.0
# 713 의 결과:  0.9633
# 시간:  24.653253316879272


from tabnanny import verbose
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.svm import LinearSVC, SVC
# 2.8버전부터 tensorflow.keras 안써도 됨, 걍 keras.으로 쓰라고 나옴
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x = np.append(x_train, x_test, axis=0)

a=[]
pca = PCA(n_components=x_train.shape[1])
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
a.append(np.argmax(cumsum >= 0.95)+1)
a.append(np.argmax(cumsum >= 0.99)+1)
a.append(np.argmax(cumsum >= 0.999)+1)
a.append(np.argmax(cumsum >= 1.0)+1)
print(a)

model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
for i in range(len(a)):
    n = a[i]
    pca = PCA(n_components=n)
    x2_train = pca.fit_transform(x_train)
    x2_test = pca.transform(x_test)
    start = time.time()
    model.fit(x2_train, y_train, verbose=True)
    end = time.time()
    
    results = model.score(x2_test, y_test)
    print(n, '의 결과: ', results)
    print('시간: ', end-start)
