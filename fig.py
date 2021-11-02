import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.subplots_adjust(bottom=0.20)


def genre_bar():

    df = pd.read_csv('pg_clean.csv')
    vc = df.groupby(by='genre')[
        'genre'].value_counts().sort_values(ascending=False)

    x = vc.index.get_level_values(0).values
    y = vc.values

    ax = sns.barplot(x, y, palette='autumn')
    ax.set_xticklabels(x, rotation=45)
    ax.set_xlabel('Genre')
    ax.set_ylabel('Book amount')
    ax.set_title('Sample Size per Genre')
    ax.grid(axis='y')
    ax.figure.savefig(os.getcwd() + '/fig/genre_bar.png')


if __name__ == '__main__':

    genre_bar()
