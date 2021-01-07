import warnings
import pandas as pd
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_model(train_data):
    traind, val = train_test_split(train_data, test_size=0.2, random_state=1)
    train_y = traind['tags']
    val_y = val['tags']
    train_x = traind.drop(['tags'], axis=1)
    val_x = val.drop(['tags'], axis=1)

    def LGB_bayesian(num_leaves, min_data_in_leaf, learning_rate, min_sum_hessian_in_leaf, feature_fraction,
                     lambda_l1, lambda_l2, min_gain_to_split, max_depth, bagging_freq, bagging_fraction, max_bin):
        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)
        bagging_freq = int(bagging_freq)
        max_bin = int(max_bin)

        assert type(num_leaves) == int
        assert type(min_data_in_leaf) == int
        assert type(max_depth) == int

        param = {
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'learning_rate': learning_rate,
            'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'max_depth': max_depth,
            'save_binary': True,
            'max_bin': max_bin,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'seed': 1,
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'metric': 'auc',
        }
        lgtrain = lgb.Dataset(train_x, label=train_y)
        lgval = lgb.Dataset(val_x, label=val_y)
        model = lgb.train(param, lgtrain, 20000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=3000)
        pred_val_y = model.predict(val_x, num_iteration=model.best_iteration)
        score = roc_auc_score(val_y, pred_val_y)
        return score

    bounds_LGB = {
        'num_leaves': (5, 20),
        'min_data_in_leaf': (5, 100),
        'learning_rate': (0.005, 0.3),
        'min_sum_hessian_in_leaf': (0.00001, 20),
        'feature_fraction': (0.001, 0.5),
        'lambda_l1': (0, 10),
        'lambda_l2': (0, 10),
        'min_gain_to_split': (0, 1.0),
        'max_depth': (3, 200),
        'bagging_fraction': (0.5, 1),
        'max_bin': (5, 256),
        'bagging_freq': (0, 90),
    }

    LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1)

    init_points = 10
    n_iter = 200
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

    print(LGB_BO.max['target'])
    print(LGB_BO.max['params'])

    param = {
        'num_leaves': int(round(LGB_BO.max['params']['num_leaves'])),
        'max_bin': int(round(LGB_BO.max['params']['max_bin'])),
        'min_data_in_leaf': int(round(LGB_BO.max['params']['min_data_in_leaf'])),
        'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
        'bagging_freq': int(round(LGB_BO.max['params']['bagging_freq'])),
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'lambda_l1': LGB_BO.max['params']['lambda_l1'],
        'lambda_l2': LGB_BO.max['params']['lambda_l2'],
        'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],
        'max_depth': int(round(LGB_BO.max['params']['max_depth'])),
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
    }

    model = lgb.LGBMClassifier(
        **param
    )
    model.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    return model, param


data_set1 = pd.read_table('col1a1_feature_matrix.txt', low_memory=False)
train1, val1 = train_test_split(data_set1, test_size=0.2, random_state=1)
model1, param1 = train_model(train1)

data_set2 = pd.read_table('col1a2_feature_matrix.txt', low_memory=False)
train2, val2 = train_test_split(data_set2, test_size=0.2, random_state=1)
model2, param2 = train_model(train2)
