import pandas as pd
import numpy as np
import random
import ast
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

def mean(seq):
    seq = [ast.literal_eval(i) for i in tqdm(seq)]
    return(np.mean(seq))


if __name__=='__main__':

    df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')

    y = np.array(df['rating']).reshape(-1, 1)

    df['date'] = df['date'].astype('datetime64').astype(int).astype(float)

    vectorizer = TfidfVectorizer()
    df['title'] = [i.lower() for i in df['title']]
    corpus = df['title'].values
    # print(corpus)

    df['title'] = vectorizer.fit_transform(corpus)

    genre_dummies = pd.get_dummies(df['genre'])
    mcols = ['sentiment', 'nrc_fear', 'nrc_anger', 'nrc_trust', 'nrc_surprise', 'nrc_positive', 'nrc_negative', 'nrc_sadness', 'nrc_disgust', 'nrc_joy']

    for idx in tqdm(range(len(df))):
        for mcol in mcols:
            df[mcol][idx] = np.mean(ast.literal_eval(df[mcol][idx]))

    X = df.drop(labels=['Unnamed: 0', 'genre', 'rating', 'id', 'title'], axis=1)
    X = pd.concat([X, genre_dummies], axis=1)

    feature_names = [f"feature {i}" for i in range(X.shape[1])]

    # print(X.head())
    # print(y[0:5])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)


    yhat = rf.predict(X_test)

    print(mean_squared_error(y_test, yhat, squared=False))
    print(mean_absolute_error(y_test, yhat))

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()



    x = [i for i in range(len(yhat))]
    plt.figure()
    plt.plot(x, y_test, color = 'red', linestyle='--', marker='o', label = 'Real data')
    plt.plot(x, yhat, color = 'blue', linestyle='--', marker='o', label = 'Predicted data')
    plt.title('Prediction, MAE: {}'.format(mean_absolute_error(y_test, yhat)))
    plt.legend()
    plt.show()

    diff = abs(yhat - y_test)
    plt.figure()
    plt.plot(x, diff)
    plt.title('Prediction, Differences')
    plt.show()