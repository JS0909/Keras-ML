# [과제]
# 3가지 원핫인코딩 방식을 비교할 것
#
# 1. pandas의 get_dummies
#       카테고리 id 별로 그대로 정리
'''
y = pd.get_dummies(y)
print(y.shape)
print(y)
'''


# 2. tensorflow의 to_categorical
#       무조건 0부터 칼럼 생성
#       만약 데이터 레이블이 3부터 있다면 0, 1, 2 라벨(체킹 칼럼)이 생성되어버림
'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y) # 인코딩 방식때문에 쉐이프가 1개 더 생겼음, 그래서 OneHotEncoder나 get_dummies를 사용해줘야함, 차이는 keras16 파일에서
print(y.shape) #(581012, 8))
print(y)
'''


# 3. sklearn의 OneHotEncoder
#       get_dummies와 같은 방식인데 벡터였던 y 데이터를 행렬로 변환하는 작업시키고 사용해야됨
#       다른 두가지는 벡터도 칼럼 레이블로 인식을 해주는데 얘는 못함
'''
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
print(y.shape) # (581012,)
y = datasets.target.reshape(-1,1) # reshape 전은 벡터로, reshape 후에 행렬로
print(y.shape) # (581012, 1)
oh.fit(y)
y = oh.transform(y).toarray()
print(y)
print(y.shape)
'''