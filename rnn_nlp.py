import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import ast
import os 

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from keras.layers import Dense, Activation, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, Input
from keras.models import Sequential


def build_model():


  model = Sequential()
  model.add(Input(shape=(1, 100,)))
  model.add(LSTM(4, return_sequences=True))
  model.add(Dropout(0.1))
  model.add(LSTM(8))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])


  return(model)

if __name__ == '__main__':
  # np.random.seed(7)
    
    df = pd.read_csv(os.getcwd() + '/data/pg_clean_10.csv')
    y = np.array(df['rating']).reshape(-1, 1)
    X = np.array([ast.literal_eval(i) for i in tqdm(df['sentiment'])])
    #X = np.resize(X, (X.shape[0],1,X.shape[1]))

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=7)


    c = 0
    maes = []
    rmses = []

    for train_index, val_index in kf.split(np.zeros(len(y)), y):

        X_train = X[train_index]
        X_train = np.resize(X_train, (X_train.shape[0], 1, X_train.shape[1]))

        X_val = X[val_index]
        X_val = np.resize(X_val, (X_val.shape[0], 1, X_val.shape[1]))

        y_train = y[train_index]
        y_val = y[val_index]

        model = None
        model = build_model()

        history = model.fit(
        X_train,
        y_train,
        batch_size=4,
        epochs=50,
        verbose=1,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_val, y_val),)

        hist = history.history

        # weights = [layer.get_weights() for layer in model.layers]
        # print(weights)


        plt.figure()
        plt.plot(hist['mae'], '--', label='train_mae')
        plt.plot(hist['root_mean_squared_error'], '--', label='train_rmse')
        plt.plot(hist['val_mae'], label='val_mae')
        plt.plot(hist['val_root_mean_squared_error'], label='val_rmse')
        plt.ylim(0.2, 0.75)
        plt.grid()
        plt.legend()

        plt.title('Learning rate')
        plt.xlabel('Epoch')

        plt.savefig(os.getcwd() + '/fig/rnn_nlp_{}'.format(c))
        # plt.show()

        maes.append(hist['val_mae'][-1])
        rmses.append(hist['val_root_mean_squared_error'][-1])

        c += 1


    print('------------------------')
    print(maes)
    print(rmses)

    print('MAE: {}'.format(np.mean(maes)))
    print('RMSE: {}'.format(np.mean(rmses)))

    # y_pred = model.predict(testX)

    # print('MAE: {}'.format(mean_absolute_error(testY, y_pred)))
    # print('RMSE: {}'.format(mean_squared_error(testY, y_pred, squared=False)))


    # x = [i for i in range(len(y_pred))]

    # plt.figure()
    # plt.plot(x, testY, color = 'red', linestyle='--', marker='o', label = 'Real data')
    # plt.plot(x, y_pred, color = 'blue', linestyle='--', marker='o', label = 'Predicted data')
    # plt.title('Prediction, MAE: {}'.format(mean_absolute_error(testY, y_pred)))
    # plt.legend()
    # plt.show()

    # diff = abs(y_pred - testY)
    # plt.figure()
    # plt.plot(x, diff)
    # plt.title('Prediction, Differences')
    # plt.show()