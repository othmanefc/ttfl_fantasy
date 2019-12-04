import os
import pandas as pd
import numpy as np
from time import time
import random

from keras import backend as K
from keras.models import Sequential, load_models
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import SGD
from gpopy import FlowTunning
import mlflow.keras

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.decomposition import PCA

from constants import VARS, IDS, LOGS_DIR, DATA_DIR

PARAMS = {
    'epochs': [20, 50],
    'batch_size': [8, 16, 32],
    'dense_layers_1': [32, 64, 128],
    'dense_layers_2': [32, 64, 128],
    'dense_layers_3': [32, 64, 128],
    'init': 'normal',
    'dropout_1': {
        'func': random.uniform,
        'params': [0, 0.5]
    },
    'dropout_2': {
        'func': random.uniform,
        'params': [0, 0.5]
    },
    'dropout_3': {
        'func': random.uniform,
        'params': [0, 0.5]
    },
    'activation': 'relu',
    'learning_rate': {
        'func': random.uniform,
        'params': [0.0001, 0.01]
    },
    'momentum': {
        'func': random.uniform,
        'params': [0.1, 0.9]
    },
    'decay': {
        'func': random.uniform,
        'params': [0.001, 0.01]
    },
}


class nn_model(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def init_model(self, learning_rate=0.005, decay=0.001, momentum=0.9):
        def i_m():

            model = Sequential()
            model.add(Dense(40, input_dim=40, activation='relu'))
            model.add(Dense(32, kernel_initializer='normal',
                            activation='relu'))
            model.add(Dense(32, kernel_initializer='normal',
                            activation='relu'))
            model.add(Dropout(0.2))
            model.add(
                Dense(1, kernel_initializer='normal', activation='linear'))
            sgd = SGD(lr=learning_rate,
                      momentum=momentum,
                      decay=decay,
                      nesterov=True)
            #tensorboard_dir = os.path.join(LOGS_DIR, "/Models/NNModel/{}".format(time))
            #tensorboard = TensorBoard(log_dir=tensorboard_dir)

            model.compile(loss=self.root_mean_squared_error, optimizer=sgd)
            return model

        return i_m

    def i_m(self, data):
        layer_1 = data['dense_layers_1']
        layer_2 = data['dense_layers_2']
        layer_3 = data['dense_layers_3']
        dropout_1 = data['dropout_1']
        dropout_2 = data['dropout_2']
        dropout_3 = data['dropout_3']
        activation = data['activation']
        learning_rate = data['learning_rate']
        momentum = data['momentum']
        decay = data['decay']
        init = data['init']

        model = Sequential()
        model.add(Dense(layer_1, input_dim=40, activation=activation))
        model.add(Dropout(dropout_1))
        model.add(
            Dense(layer_2, kernel_initializer=init, activation=activation))
        model.add(Dropout(dropout_2))
        model.add(
            Dense(layer_3, kernel_initializer=init, activation=activation))
        model.add(Dropout(dropout_3))
        model.add(Dense(1, kernel_initializer=init, activation='linear'))
        sgd = SGD(lr=learning_rate,
                  momentum=momentum,
                  decay=decay,
                  nesterov=True)
        #tensorboard_dir = os.path.join(LOGS_DIR, "/Models/NNModel/{}".format(time))
        #tensorboard = TensorBoard(log_dir=tensorboard_dir)

        model.compile(loss=self.root_mean_squared_error, optimizer=sgd)
        return model

    def root_mean_squared_error(self, y_true, y_pred):
        # Custom loss function for keras
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def run_models(self):
        estimators = []
        estimators.append(('standardize', MinMaxScaler()))
        estimators.append(('pca', PCA(n_components=40)))

        model = KerasRegressor(
            build_fn=self.init_model,
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            shuffle=True,
            verbose=1,
        )

        # estimators.append(('mlp', model))
        pipeline = Pipeline(estimators, verbose=True)
        X_transform = pipeline.fit_transform(self.X)
        # kfold = KFold(n_splits=3)
        # results = cross_val_score(pipeline, X, y, cv=kfold)
        #estimators = []
        # print("Wider: %.2f (%.2f) RMSE" % (results.mean(), results.std()))
        filepath = os.path.join(LOGS_DIR, 'Models/NNModel/weights',
                                'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     verbose=1,
                                     mode='min')
        callbacks_list = [checkpoint]
        model.fit(X_transform, self.y, callbacks=callbacks_list)
        model.model.save(
            os.path.join(LOGS_DIR, 'Models/NNModel', 'saved_model.h5'))
        #pipeline.save(os.path.join(LOGS_DIR, 'Models/NNModel',
        #                              'saved_model.h5'))
        #return model

    def run_saved_model(self, X=None, y=None):
        model = KerasRegressor(
            build_fn=self.init_model,
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            shuffle=True,
            verbose=1,
        )
        model.model = load_models(
            os.path.join(LOGS_DIR, 'Models/NNModel', 'saved_model.h5'))
        if np.any([X, y]) == None:
            X, y = self.X, self.y

        estimators = []
        estimators.append(('standardize', MinMaxScaler()))
        estimators.append(('pca', PCA(n_components=40)))
        pipeline = Pipeline(estimators, verbose=True)
        X_transform = pipeline.fit_transform(X)
        
        model.fit(X_transform, y)
        model.model.save(
            os.path.join(LOGS_DIR, 'Models/NNModel', 'saved_model.h5'))
        
        return model
        

    def run_models_mlflow(self, data):
        estimators = []
        estimators.append(('standardize', MinMaxScaler()))
        estimators.append(('pca', PCA(n_components=40)))

        # estimators.append(('mlp', model))
        pipeline = Pipeline(estimators, verbose=True)
        # kfold = KFold(n_splits=3)
        # results = cross_val_score(pipeline, X, y, cv=kfold)
        #estimators = []
        # print("Wider: %.2f (%.2f) RMSE" % (results.mean(), results.std()))
        X_transform = pipeline.fit_transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(X_transform, y)
        # filepath = os.path.join(LOGS_DIR, 'Models/NNModel/weights',
        #                         'weights_tunning.{epoch:02d}-{val_loss:.2f}.hdf5')
        # checkpoint = ModelCheckpoint(filepath,
        #                              monitor='val_loss',
        #                              save_best_only=True,
        #                              verbose=1,
        #                              mode='min')
        # callbacks_list = [checkpoint]
        model = self.i_m(data)
        model.fit(X_train,
                  y_train,
                  epochs=data['epochs'],
                  batch_size=data['batch_size'],
                  validation_data=(X_test, y_test),
                  verbose=1)
        score = model.evaluate(X_test, y_test)
        print(
            "#######################- RESULTS -####################################"
        )
        print('Test loss:', score)
        print(
            "######################################################################"
        )
        return (-score, model)

    def mlflow_run(self,
                   params,
                   maximum_generation=20,
                   population_size=10,
                   auto_track=True):
        tunning = FlowTunning(params=params,
                              population_size=population_size,
                              maximum_generation=maximum_generation,
                              auto_track=auto_track)
        tunning.set_score(self.run_models_mlflow)
        tunning.run()

    # filepath = os.path.join(LOGS_DIR, 'Models/NNModel/weights',
    #                         'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    # checkpoint = ModelCheckpoint(filepath,
    #                              monitor='val_loss',
    #                              save_best_only=True,
    #                              verbose=1,
    #                              mode='min')
    # callbacks_list = [checkpoint]
    # pipeline.fit(X, y, mlp__callbacks=callbacks_list)

    #pipeline.save(os.path.join(LOGS_DIR, 'Models/NNModel',
    #                              'saved_model.h5'))
    #return model


if __name__ == '__main__':
    df_path = os.path.join(DATA_DIR, 'season_2018_cleaned.csv')
    df = pd.read_csv(df_path)
    X, y = df[VARS], df['ttfl']
    # hist = run_models(X, y)
    model = nn_model(X, y)
    model.mlflow_run(params=PARAMS, population_size=5, maximum_generation=20)
