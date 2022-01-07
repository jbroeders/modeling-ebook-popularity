import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np 
import ast
import os

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation
from keras.models import Sequential

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import load
from tqdm import tqdm

def shorten_sequence(seq):
    n = len(seq)
    # print(n)

    res = []

    if n > 100:
        split = n/100

        for i in range(100):
            res.append(np.mean(seq[int(i*split): int(i*split + split)]))

    else:

        # print(seq)
        res = pad_sequences([seq], padding='post', dtype='float32', value =0, maxlen = 100)
        return(res[0])

    # x = [i for i in range(100)]
    # plt.figure()
    # plt.plot(x, res)
    # plt.show()
    return(np.array(res))

if __name__ == '__main__':

    df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')
    y = np.array(df['rating']).reshape(-1, 1)

    X = np.array([shorten_sequence(ast.literal_eval(i)) for i in tqdm(df['sentiment'], desc='Shortening sequences')])

    
 

    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = 100))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam',loss = 'mean_squared_error')
    kf = KFold(n_splits=5, shuffle=True, random_state=2)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.compile(optimizer = 'adam',loss = 'mean_squared_error')
        model.fit(X_train, y_train, batch_size = 10, epochs = 125, verbose=False)

        y_pred = model.predict(X_test)

        print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
        print('RMSE: {}'.format(mean_squared_error(y_test, y_pred, squared=False)))


    x = [i for i in range(len(y_pred))]

    plt.figure()
    plt.plot(x, y_test, color = 'red', label = 'Real data')
    plt.plot(x, y_pred, color = 'blue', label = 'Predicted data')
    plt.title('Prediction, MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
    plt.legend()
    plt.show()

    diff = abs(y_pred - y_test)
    plt.figure()
    plt.plot(x, diff)
    plt.title('Prediction, Differences')
    plt.show()