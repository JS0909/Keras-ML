import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
# 하이퍼 파라미터들 값 지정
 
'''
optuna.trial.Trial.suggest_categorical() : 리스트 범위 내에서 값을 선택한다.
optuna.trial.Trial.suggest_int() : 범위 내에서 정수형 값을 선택한다.
optuna.trial.Trial.suggest_float() : 범위 내에서 소수형 값을 선택한다.
optuna.trial.Trial.suggest_uniform() : 범위 내에서 균일분포 값을 선택한다.
optuna.trial.Trial.suggest_discrete_uniform() : 범위 내에서 이산 균일분포 값을 선택한다.
optuna.trial.Trial.suggest_loguniform() : 범위 내에서 로그 함수 값을 선택한다.
'''
# learning_rate : float range: (0,1)
# depth : int, [default=6]   range: [1,+inf]
# od_pval : float, [default=None] range: [0,1]
# model_size_reg : float, [default=None] range: [0,+inf]
# l2_leaf_reg : float, [default=3.0]  range: [0,+inf]
# fold_permutation_block : int, [default=1] T[1, 256]. 
def objectiveCAT(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'depth' : trial.suggest_int('depth', 8, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
        'learning_rate' : 0.01,
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
        'random_state' : 1127
    }
    
    # 학습 모델 생성
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    
    # 모델 성능 확인
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    
    return score
# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())

# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 50)

print('Best trial : score {}, \nparams {}'.format(study.best_trial.value, study.best_trial.params))

# 하이퍼파라미터별 중요도를 확인할 수 있는 그래프
optuna.visualization.plot_param_importances(study)

# 하이퍼파라미터 최적화 과정을 확인
optuna.visualization.plot_optimization_history(study)