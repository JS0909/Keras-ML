param_bounds = {'x1':(-1, 5),
                'x2':(0, 4)}

def y_function(x1, x2):
    return -x1 **2 - (x2 - 2) **2 + 10

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(f=y_function, # 모델의 평가지표로 나온 결과를 함수화해서 넣는다
                                 pbounds=param_bounds, # 파라미터 생성은 꼭 딕셔너리 형태로
                                 random_state=1234)

optimizer.maximize(init_points=2, # 초기 포인트를 몇번
                   n_iter=20) # 아무튼 두 숫자 더한 횟수만큼 돈다고 생각하면 됨

print(optimizer.max)