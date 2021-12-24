import pandas as pd
import numpy as np
import random
import tqdm
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
if __name__=='__main__':
        df = pd.read_csv(os.getcwd() + '/data/pg_clean_baseline.csv')
        print(len(df))
        y = df['rating']

        #all features
        X = df.drop(['rating'], axis=1)

        #random input
        Xr = np.array([random.random() for i in range(len(y))]).reshape(-1, 1)

        #nrc features only
        Xn = df.drop(['id', 'rating'], axis=1)

        #publish_date only
        Xp = np.array(df['id']).reshape(-1, 1)





        maes = np.array([])
        rmses = np.array([])

        for X in [Xr, Xp, Xn, X]:
                for i in tqdm.tqdm(range(50)):

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


                        rf = RandomForestRegressor()
                        rf.fit(X_train, y_train)

                        yhat = rf.predict(X_test)

                        maes = np.append(maes, mean_absolute_error(y_test, yhat))
                        rmses = np.append(rmses, mean_squared_error(y_test, yhat, squared=False))
                        # print('MAE: {}'.format(mae))
                        # print('RMSE: {}'.format(rmse))

                print('MAE: {}'.format(np.mean(maes)))
                print('RMSE: {}'.format(np.mean(rmses)))