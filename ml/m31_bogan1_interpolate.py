# 결측치 처리
# 1. 행 또는 열 삭제
# 2. 임의의 값으로 채움
#       평균 : mean
#       중위 : median
#       앞 : ffill
#       뒤 : bfill
#       0 : fillna
#       특정값 : ...
# 3. 보간 - interpolate : linear, 즉 선형회귀방식으로 빈 자리의 값을 찾아냄
# 4. 모델 - predict : 보간과 비슷함, 훈련시킨 웨이트를 적용해서 predict를 해서 해당 값을 데이터에 집어넣음
# 5. 트리/부스팅계열 - 통상 결측치, 이상치에 대해 자유롭다, 구역으로 분할하는 방식이니까

import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022', '8/11/2022', '8/12/2022', '8/13/2022', '8/14/2022']

dates = pd.to_datetime(dates)
print(dates)

print('======================================')
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
print(ts)

print('======================================')
ts = ts.interpolate() # 리니어형태로 매꾼다
print(ts)
