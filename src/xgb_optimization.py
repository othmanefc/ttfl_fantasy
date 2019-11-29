import os
import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.model_selection import train_test_split

from constants import TARGETS, VARS

params = {
    'feature_fraction': (0.6, 0.9),
    'bagging_fraction': (0.8, 1),
    'lambda_l1': (0, 3),
    'lambda_l2': (0, 3),
    'max_depth': (5, 100),
    'num_leaves': (10, 300),
    'min_split_gain': (0.001, 0.1),
    'min_child_weight': (0, 1),
    'learning_rate': (0.01, 1),
    'n_estimators': (50, 5000),
}


def xgb_optimization(X, y, params, random_state=1337):
    training_data = xgb.DMatrix(X, y)

    def xgb_model(feature_fraction, bagging_fraction, lambda_l1, lambda_l2,
                  max_depth, num_leaves, min_split_gain, min_child_weight,
                  learning_rate, n_estimators):

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['max_depth'] = int(round(max_depth))
        params['num_leaves'] = int(round(num_leaves))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['learning_rate'] = learning_rate
        params['n_estimators'] = int(round(n_estimators))

        params.update({
            "objective": "reg:squarederror",
            "max_bin": 255,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "boosting": "gbdt",
            "verbosity": 1,
            "early_stopping_round": 200,
            "metric": 'mae'
        })

        clf = xgb.cv(params,
                     training_data,
                     nfold=5,
                     seed=random_state,
                     verbose_eval=1)
        return (-1 * np.array(clf['test-rmse-mean'])).max()

    optimizer = BayesianOptimization(f=xgb_model,
                                     pbounds=params,
                                     random_state=1337)
    logger_path = os.path.join(os.getcwd().replace('/src', ''), 'logs',
                               'logs_xgb.json')

    if os.path.exists(logger_path):
        load_logs(optimizer, logs=logger_path)

    logger = JSONLogger(path=logger_path)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.maximize(init_points=5, n_iter=25, acq='ucb')

    return optimizer.max['params']


if __name__ == '__main__':
    path_data = os.path.join(os.getcwd(), 'data', 'season_2018_cleaned.csv')
    df = pd.read_csv(path_data)

    X = df[VARS]
    y = df['ttfl']

    opt_params = xgb_optimization(X, y, params)
