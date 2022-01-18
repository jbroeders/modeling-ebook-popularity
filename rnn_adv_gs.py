import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import ast
import os 

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from keras.layers import Dense, Activation, Conv1D, Flatten, MaxPooling1D, Dropout, LSTM, Input, Embedding, Bidirectional, Concatenate, concatenate
from keras.models import Sequential, Model
from keras import regularizers





if __name__ == '__main__':

    df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')
    y = np.array(df['rating']).reshape(-1, 1)

    X = np.array([ast.literal_eval(i) for i in tqdm(df['sentiment'])])
    X_nlp = np.resize(X, (X.shape[0],1,X.shape[1]))

   

    df['date'] = df['date'].astype('datetime64').astype(int).astype(float)
    mcols = ['sentiment', 'nrc_fear', 'nrc_anger', 'nrc_trust', 'nrc_surprise', 'nrc_positive', 'nrc_negative', 'nrc_sadness', 'nrc_disgust', 'nrc_joy']

    for idx in tqdm(range(len(df))):
        for mcol in mcols:
            df[mcol][idx] = np.mean(ast.literal_eval(df[mcol][idx]))


    X_meta = df.drop(labels=['Unnamed: 0', 'genre', 'rating', 'id', 'title'], axis=1)
    genre_dummies = pd.get_dummies(df['genre'])
    X_meta = pd.concat([X_meta, genre_dummies], axis=1)


    # print(X_nlp[0:10])
    # print(X_meta[0:10])

    scaler = MinMaxScaler()
    X_meta = scaler.fit_transform(X_meta)
    # X_nlp = scaler.fit_transform(X_nlp)


    X_meta_train, X_meta_test = X_meta[:int(0.75*len(X_meta))], X_meta[int(0.75*len(X_meta)):]
    X_nlp_train, X_nlp_test = X_nlp[:int(0.75*len(X_meta))], X_nlp[int(0.75*len(X_meta)):]
    y_train, y_test = y[:int(0.75*len(y))], y[int(0.75*len(y)):]

    meta_input = Input(shape=(22,), name='meta_input')
    nlp_input = Input(shape=(1, 100,), name='nlp_input')



    LSTM1 = [4, 8, 16, 32, 64, 128, 256]
    LSTM2 = [4, 8, 16, 32, 64, 128, 256]
    
    for ls1 in LSTM1:
        for ls2 in LSTM2:

            print('ls1: {}'.format(ls1))
            print('ls2: {}'.format(ls2))
        

            nlp_out = LSTM(ls1, return_sequences=True )(nlp_input)
            nlp_out = Dropout(0.1)(nlp_out)
            nlp_out = LSTM(ls2)(nlp_out)

            x = concatenate([nlp_out, meta_input])
            x = Dense(1)(x)

            model = Model(inputs=[nlp_input , meta_input], outputs=x)
            model.compile(optimizer='adam', loss='mean_squared_error')



            
        
            model.fit({'nlp_input': X_nlp_train, 'meta_input': X_meta_train}, y_train, epochs=50, batch_size=4, verbose=0)
            y_pred = model.predict({'nlp_input': X_nlp_test, 'meta_input': X_meta_test})
   
            print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
            print('RMSE: {}'.format(mean_squared_error(y_test, y_pred, squared=False)))
            print ('------------------------------------------------------s')

    x = [i for i in range(len(y_pred))]

    plt.figure()
    plt.plot(x, y_test, color = 'red', linestyle='--', marker='o', label = 'Real data')
    plt.plot(x, y_pred, color = 'blue', linestyle='--', marker='o', label = 'Predicted data')
    plt.title('Prediction, MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
    plt.legend()
    plt.show()

    diff = abs(y_pred - y_test)
    plt.figure()
    plt.plot(x, diff)
    plt.title('Prediction, Differences')
    plt.show()