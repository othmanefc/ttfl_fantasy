import os
import pandas as pd
from time import time

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard

from constants import VARS, IDS


def init_model():
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    tensorboard_dir = os.path.join(os.getcwd().replace('/src', '/logs'),
                                   "/Models/NNModel/{}".format(time))
    tensorboard = TensorBoard(log_dir=tensorboard_dir)

    model.compile(loss=root_mean_squared_error, optimizer='adam')

    return model


def root_mean_squared_error(y_true, y_pred):
    # Custom loss function for keras
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def run_models(X, y):
    model = KerasRegressor(build_fn=init_model,
                           epochs=100,
                           batch_size=32,
                           validation_split=0.2,
                           shuffle=True,
                           verbose=1)

    filepath = os.path.join(
        os.getcwd().replace('/src', '/logs/Models/NNModel'),
        'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1,
                                 mode='min')
    callbacks_list = [checkpoint]

    model = model.fit(X, y, callbacks=callbacks_list)

    return model


if __name__ == '__main__':
    df_path = os.path.join(os.getcwd().replace('/src', ''), 'data',
                           'season_2018_cleaned.csv')
    df = pd.read_csv(df_path)
    X, y = df[VARS], df['ttfl']
    hist = run_models(X, y)
    print(hist)