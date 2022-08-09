# n_component > 0.95
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# n_jobs = -1
# tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1

from tabnanny import verbose
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import time
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x = np.append(x_train, x_test, axis=0)

parameters = [
{'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6]},
{'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6,0.9,1]},
{'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.7,0.9]}
]

model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                                         gpu_id=1), parameters, verbose=1, refit=True, n_jobs=-1)

pca = PCA(n_components=x_train.shape[1])
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)

pca = PCA(n_components=np.argmax(cumsum >= 0.999)+1)
x2_train = pca.fit_transform(x_train)
x2_test = pca.transform(x_test)
start = time.time()
model.fit(x2_train, y_train, verbose=1)
end = time.time()

results = model.score(x2_test, y_test)
print('결과: ', results)
print('시간: ', end-start)

# PCA 0.95 이상
# 결과:  0.9649
# 시간:  510.00438714027405

# PCA 0.99 이상
# 결과:  0.9307
# 시간:  527.3115980625153