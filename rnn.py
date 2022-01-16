import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import ast
import os 

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from keras.layers import Dense, Activation, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM
from keras.models import Sequential

if __name__ == '__main__':
  # np.random.seed(7)

  df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')
  y = np.array(df['rating']).reshape(-1, 1)
  X = np.array([ast.literal_eval(i) for i in tqdm(df['sentiment'])])
  X = np.resize(X, (X.shape[0],1,X.shape[1]))

  look_back=1



  print(X.shape)
  #x_train, x_test, y_train, y_test = train_test_split(X, y)
  trainX, testX, trainY, testY = train_test_split(X, y, test_size= 0.3, random_state=2)

  # model = tf.keras.models.Sequential()
  # Dense = tf.keras.layers.Dense
  # Dropout = tf.keras.layers.Dropout
  # LSTM = tf.keras.layers.LSTM
  # model.add(LSTM(16, activation='relu', return_sequences=True))
  # model.add(Dropout(0.2))
  # model.add(LSTM(256, activation='relu'))
  # model.add(Dropout(0.1))
  # model.add(Dense(32, activation='relu'))
  # model.add(Dropout(0.2))

  model = Sequential()
  model.add(LSTM(16, input_shape=(1, 100), return_sequences=True))
  model.add(Dropout(0.1))
  model.add(LSTM(8))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=250, batch_size=10, verbose=1)

    
  y_pred = model.predict(testX)

  print('MAE: {}'.format(mean_absolute_error(testY, y_pred)))
  print('RMSE: {}'.format(mean_squared_error(testY, y_pred, squared=False)))


  x = [i for i in range(len(y_pred))]

  plt.figure()
  plt.plot(x, testY, color = 'red', linestyle='--', marker='o', label = 'Real data')
  plt.plot(x, y_pred, color = 'blue', linestyle='--', marker='o', label = 'Predicted data')
  plt.title('Prediction, MAE: {}'.format(mean_absolute_error(testY, y_pred)))
  plt.legend()
  plt.show()

  diff = abs(y_pred - testY)
  plt.figure()
  plt.plot(x, diff)
  plt.title('Prediction, Differences')
  plt.show()