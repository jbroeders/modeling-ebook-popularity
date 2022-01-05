import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import ast

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Activation
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from joblib import load

df = pd.read_csv('res.csv')
y = df['rating']
X = [ast.literal_eval(i) for i in df['sentiment']]

# X = np.asarray(load('sequences.arr')[0:3])
# y = load('ratings.arr')[0:3]
print(X)
X = pad_sequences(X, padding='post', value=-1000, dtype='float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

print(X_train[0:10])
print(y_train[0:10])
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 5418))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')

model.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()