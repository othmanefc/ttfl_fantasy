import pandas as pd
import os
import numpy as np


def normalize_by_group(df, by):
    groups = df.groupby(by)
    # computes group-wise mean/std,
    # then auto broadcasts to size of group chunk
    mean = groups.transform(np.mean)
    std = groups.transform(np.std)
    return (df[mean.columns] - mean) / std


def train_test_split(df, dates):
    if isinstance(dates, list):
        start = dates[0]
        end = dates[1]
        if start > end:
            raise Exception("start shouldn't be higher than end date")
        df.date = df.date.astype(int)
        df_test = df[(df.date <= end) & (df.date >= start)]
        df_train = df[df.date < start]
        X_train, y_train, ids_train = df_train[VARS], df_train[
            'ttfl'], df_train[IDS]
        X_test, y_test, ids_test = df_test[VARS], df_test['ttfl'], df_test[IDS]

        return X_train, X_test, y_train, y_test, ids_train, ids_test
    else:

        raise Exception("dates should be of type 'list' not {}".format(
            type(dates)))
