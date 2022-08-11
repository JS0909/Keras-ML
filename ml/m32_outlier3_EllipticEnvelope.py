import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
aaa = aaa.reshape(-1, 1)
ellip = EllipticEnvelope(contamination=.1) # 해당 데이터 범위에서 .1(10%)를 이상치로 잡겠다
ellip.fit(aaa)
results = ellip.predict(aaa)
print(results)
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1] : 여기서 -1은 이상치