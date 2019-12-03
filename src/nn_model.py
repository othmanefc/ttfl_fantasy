import os
import pandas as pd
import numpy as np
from time import time

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import SGD


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.decomposition import PCA

from constants import VARS, IDS, LOGS_DIR, DATA_DIR


def init_model(init='normal', learning_rate=0.005, 
               decay=0.001, momentum=0.9):
    model = Sequential()
    model.add(Dense(40, input_dim=40, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer=init, activation='linear'))
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=True)
    #tensorboard_dir = os.path.join(LOGS_DIR, "/Models/NNModel/{}".format(time))
    #tensorboard = TensorBoard(log_dir=tensorboard_dir)

    model.compile(loss=root_mean_squared_error, optimizer=sgd)

    return model


def root_mean_squared_error(y_true, y_pred):
    # Custom loss function for keras
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def run_models(X, y):
    estimators = []
    estimators.append(('standardize', MinMaxScaler()))
    estimators.append(('pca', PCA(n_components=40)))

    model = KerasRegressor(
        build_fn=init_model,
        epochs=100,
        batch_size=8,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )

    estimators.append(('mlp', model))
    pipeline = Pipeline(estimators, verbose=True)
    kfold = KFold(n_splits=3)
    results = cross_val_score(pipeline, X, y, cv=kfold)
    #estimators = []
    print("Wider: %.2f (%.2f) RMSE" % (results.mean(), results.std()))
    filepath = os.path.join(LOGS_DIR, 'Models/NNModel/weights',
                            'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1,
                                 mode='min')
    callbacks_list = [checkpoint]
    pipeline.fit(X, y, mlp__callbacks=callbacks_list)
    #pipeline.save(os.path.join(LOGS_DIR, 'Models/NNModel',
    #                              'saved_model.h5'))
    #return model


if __name__ == '__main__':
    df_path = os.path.join(DATA_DIR, 'season_2018_cleaned.csv')
    df = pd.read_csv(df_path)
    X, y = df[VARS], df['ttfl']
    hist = run_models(X, y)
