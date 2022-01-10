import os
import json
import joblib
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

def raw_sequence(y):
        
    x = [i for i in range(len(y))]
    plt.plot(x, y, label='sentiment')
        
    

    plt.xlabel('Sentence')
    plt.ylabel('Compound Score')
    # plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.title('Raw values')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()

def fft_transform(y, term):

        
    x = [i for i in range(len(y))]
    plt.plot(x, y, label='sentiment')
        
    

    plt.xlabel('Sentence')
    plt.ylabel('Compound Score')
    # plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.title('FFT')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()
    

def fft_scaled(y):

        
    x = [i for i in range(len(y))]
    plt.plot(x, y, label='sentiment')
        
    

    plt.xlabel('Narrative time')
    plt.ylabel('Compound Score')
    # plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.title('Scaled and Standardized')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()


def pp_plot(s1, s2, s3):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    x = [i for i in range(len(s1))]
    ax1.plot(x, s1)
    ax1.set_title('Sentiment Scores (Raw)')
    ax1.set_ylabel('Compound Score')
    ax1.set_xlabel('Sentence')
    ax1.grid()

    x = [i for i in range(len(s2))]
    ax2.plot(x, s2)
    ax2.set_title('Sentiment Scores (FFT)')
    ax2.set_ylabel('Compound Score')
    ax2.set_xlabel('Sentence')
    ax2.grid()

    x = [i for i in range(len(s3))]
    ax3.plot(x, s3)
    ax3.set_title('Sentiment Scores (scaled/standardized)')
    ax3.set_ylabel('Compound Score')
    ax3.set_xlabel('Narrative time')
    ax3.grid()

    fig.tight_layout(pad=1.5)
    #fig.suptitle('Sentiment Vector')
    plt.savefig(os.getcwd() + '/fig/pp_plot.png')

    plt.show()

    fig.savefig(os.getcwd() + '/fig/pp_plot.png')

def fft_terms(s1, s2, s3, s4):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

    x = [i for i in range(len(s1))]
    
    ax1.plot(x, s1)
    ax1.set_title('FFT: 2nd order')
    ax1.set_ylabel('Compound Score')
    ax1.set_xlabel('Narrative Time')

    ax2.plot(x, s2)
    ax2.set_title('FFT: 3rd order')
    ax2.set_ylabel('Compound Score')
    ax2.set_xlabel('Narrative Time')

    ax3.plot(x, s3)
    ax3.set_title('FFT: 4th order')
    ax3.set_ylabel('Compound Score')
    ax3.set_xlabel('Narrative Time')

    ax4.plot(x, s4)
    ax4.set_title('FFT: 5th order')
    ax4.set_ylabel('Compound Score')
    ax4.set_xlabel('Narrative Time')

    fig.tight_layout(pad=1.5)
    plt.show()

    fig.savefig(os.getcwd() + '/fig/fft_plot.png')

if __name__ == '__main__':

    s1 = joblib.load(os.getcwd() + '/lib/sequence_raw.arr')
    s2 = joblib.load(os.getcwd() + '/lib/sequence_fft.arr')
    s3 = joblib.load(os.getcwd() + '/lib/sequence_fft_scaled.arr')

    pp_plot(s1, s2, s3)

    s1 = joblib.load(os.getcwd() + '/lib/sequence_fft_2')
    s2 = joblib.load(os.getcwd() + '/lib/sequence_fft_3')
    s3 = joblib.load(os.getcwd() + '/lib/sequence_fft_4')
    s4 = joblib.load(os.getcwd() + '/lib/sequence_fft_5')