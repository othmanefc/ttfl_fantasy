import pandas as pd
import os
import xgboost as xgb
from constants import VARS, IDS
from sklearn.metrics import mean_absolute_error
from utils import train_test_split
from src.constants import DATA_DIR, LOGS_DIR


def best_params(path):
    if os.path.exists(path):
        df_params = pd.read_json(path, lines=True)
        df_params = df_params.sort_values(
            'target', ascending=False).reset_index(drop=True)
        best_params = df_params.loc[0, 'params']
        min_max_params = ['feature_fraction', 'bagging_fraction']
        non_zero_params = ['lambda_l1', 'lambda_l2']
        int_params = ['max_depth', 'num_leaves', 'n_estimators']

        for param in min_max_params:
            best_params[param] = max(min(best_params[param], 1), 0)
        for param in non_zero_params:
            best_params[param] = max(best_params[param], 0)
        for param in int_params:
            best_params[param] = int(round(best_params[param]))

        return best_params
    else:
        print('No Logs Found')


def run_model(X_train, y_train, X_test, y_test, params):
    matrice = xgb.DMatrix(X_train, y_train)
    model = xgb.train(params, matrice)
    preds = model.predict(xgb.DMatrix(X_test))
    print('MAE:', mean_absolute_error(preds, y_test))
    return preds


def save_preds(X_test, y_test, preds, ids_test, suffix):
    arrays = [ids_test, X_test, y_test, preds]
    arrays = [pd.DataFrame(array).reset_index(drop=True) for array in arrays]
    df_test = pd.concat(arrays, axis=1)
    df_test.columns = IDS + VARS + ['ttfl', 'preds']
    df_test.to_csv(
        os.path.join(os.getcwd().replace('src', ''), 'data',
                     'season_2018_preds_{}.csv'.format(suffix)))


def main(df_path, dates, params_path, suffix):

    df = pd.read_csv(df_path)
    X_train, X_test, y_train, y_test, _, ids_test = train_test_split(df, dates)
    b_params = best_params(params_path)
    preds = run_model(X_train, y_train, X_test, y_test, b_params)
    save_preds(X_test, y_test, preds, ids_test, suffix)

    print('predictions_saved')


if __name__ == '__main__':
    df_path = os.path.join(DATA_DIR, 'season_2018_cleaned.csv')
    params_path = os.path.join(LOGS_DIR, 'logs_xgb.json')
    dates = [20190109, 20190120]
    suffix = 'xgb_00'
    main(df_path, dates, params_path, suffix)
