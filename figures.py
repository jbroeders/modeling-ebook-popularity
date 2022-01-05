import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def genre_bar(df):

    vc = df.groupby(by='genre')[
        'genre'].value_counts().sort_values(ascending=False)

    x = vc.index.get_level_values(0).values
    y = vc.values

    plt.subplots_adjust(bottom=0.20)
    ax = sns.barplot(x, y, palette='autumn')

    ax.set_xticklabels(x, rotation=45)
    ax.set_xlabel('Genre')
    ax.set_ylabel('Book amount')
    ax.set_title('Sample Size per Genre')
    ax.grid(axis='y')
    ax.figure.savefig(os.getcwd() + '/fig/genre_bar.png')

def rating_distributions(df):

    plt.figure(figsize=(6.4, 4.8))
    ax = sns.displot(df['rating'], kde=True, bins=23)
    ax.figure.savefig(os.getcwd() + '/fig/rating_hist')

    plt.figure(figsize=(6.4, 4.8))
    ax = sns.displot(df['rating_amount'], kde=True, bins=23)
    ax.figure.savefig(os.getcwd() + '/fig/rating_amount_hist')

def nrc_values(df, idx):

    nrc_columns = ['anger', 'trust', 'surprise', 'positive', 'negative',
                   'sadness', 'disgust', 'joy']

    plt.figure(figsize=(6.4, 4.8))

    for val in nrc_columns:
        y = json.loads(df[val][idx])
        x = np.linspace(0, len(y), num=len(y), endpoint=True)

        lin = interp1d(x, y)
        xnew = np.linspace(0, len(y), num=50, endpoint=True)

        # plt.plot(xnew, lin(xnew), '-', alpha=0.5, label=val)

        ax = sns.lineplot(x=xnew, y=lin(xnew), alpha=0.2, label=val)

    ax.set_xlabel('Sentence')
    ax.set_ylabel('Emotion frequency')
    ax.set_title(df['title'][idx] + ' by ' + df['author'][idx])
    ax.legend(loc='best')
    ax.grid()

    ax.figure.savefig(os.getcwd() + '/fig/nrc_{}'.format(idx))

def raw_sequences(df, idx):

    val_columns = ['positive', 'negative']

    plt.figure(figsize=(6.4, 4.8))

    for val in val_columns:
        
        
        y = json.loads(df[val][idx])
        n = [i for i in range(len(y))]
        
        
        plt.plot(n, y, label=val)
        
        
        
        

    plt.xlabel('Sentence')
    plt.ylabel('Emotion frequency')
    plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.legend(loc='best')
    plt.grid()

def fft_transforms(df, idx, term):

    nrc_columns = ['positive', 'negative']

    plt.figure(figsize=(6.4, 4.8))

    for val in nrc_columns:
        
        
        y = json.loads(df[val][idx])
        n = [i for i in range(len(y))]
        
        rft = np.fft.rfft(y)
        rft[term:] = 0
        y_smooth = np.fft.irfft(rft)
        
        x = [i for i in range(len(y_smooth))]
        # plt.plot(n, y, label=val)
        plt.plot(x, y_smooth, label=val)
        
        
        
        

    plt.xlabel('Sentence')
    plt.ylabel('Emotion frequency')
    plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.legend(loc='best')
    plt.grid()

    


if __name__ == '__main__':

    df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')
    genre_bar(df)


    # for idx in range(5):
    #     nrc_values(df, idx)
