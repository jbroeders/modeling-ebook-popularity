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
    plt.ylabel('Sentiment')
    # plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.title('Raw values')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()

def fft_transform(y, term):

        
    x = [i for i in range(len(y))]
    plt.plot(x, y, label='sentiment')
        
    

    plt.xlabel('Sentence')
    plt.ylabel('Sentiment')
    # plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.title('FFT')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()
    

def fft_scaled(y):

        
    x = [i for i in range(len(y))]
    plt.plot(x, y, label='sentiment')
        
    

    plt.xlabel('standardized narrative time')
    plt.ylabel('Sentiment')
    # plt.title(df['title'][idx] + ' by ' + df['author'][idx])
    plt.title('Scaled and Standardized')
    # plt.legend(loc='best')
    plt.grid()
    plt.show()


def pp_plot(s1, s2, s3, s4, s5, s6, s7, s8):


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(4.5, 12))

    x1 = [i for i in range(len(s1))]
    x2 = [i for i in range(len(s5))]
    ax1.plot(x1, s1, label='Adaptation by M. Reynolds')
    ax1.plot(x2, s5, label='Adolescents Only by I.E. Cox')
    ax1.set_title('Sentiment Scores (Raw)')
    ax1.set_ylabel('Sentiment')
    ax1.set_xlabel('Sentence')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True)
    ax1.grid()

    x1 = [i for i in range(len(s2))]
    x2 = [i for i in range(len(s6))]
    ax2.plot(x1, s2)
    ax2.plot(x2, s6)
    ax2.set_title('Sentiment Scores (FFT)')
    ax2.set_ylabel('Sentiment')
    ax2.set_xlabel('Sentence')
    ax2.grid()


    x = [i for i in range(len(s3))]
   
    ax3.plot(x, s3)
    ax3.plot(x, s7)
    ax3.set_title('Sentiment Scores (standardized)')
    ax3.set_ylabel('Sentiment')
    ax3.set_xlabel('standardized narrative time')
    ax3.grid()

    ax4.plot(x, s4)
    ax4.plot(x, s8)
    ax4.set_title('Sentiment Scores (scaled)')
    ax4.set_ylabel('Sentiment (scaled)')
    ax4.set_xlabel('standardized narrative time')
    ax4.grid()

    fig.tight_layout(h_pad=10)
    plt.savefig(os.getcwd() + '/fig/pp_plot.png')

    plt.show()
    # fig.savefig(os.getcwd() + '/fig/pp_plot_final.png')

def fft_terms(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(6.4, 12))

    x = [i for i in range(len(s1))]
    
    ax1.plot(x, s1, label='The Blind Spot by H.E. Flint')
    ax1.plot(x, s6, label='The Aliens by M. Leinster')
    ax1.set_title('FFT: 2nd order')
    ax1.set_ylabel('Sentiment')
    ax1.set_xlabel('standardized narrative time')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True)
    ax1.grid()

    ax2.plot(x, s2)
    ax2.plot(x, s7)
    ax2.set_title('FFT: 3rd order')
    ax2.set_ylabel('Sentiment')
    ax2.set_xlabel('standardized narrative time')
    ax2.grid()

    ax3.plot(x, s3)
    ax3.plot(x, s8)
    ax3.set_title('FFT: 4th order')
    ax3.set_ylabel('Sentiment')
    ax3.set_xlabel('standardized narrative time')
    ax3.grid()

    ax4.plot(x, s4)
    ax4.plot(x, s9)
    ax4.set_title('FFT: 5th order')
    ax4.set_ylabel('Sentiment')
    ax4.set_xlabel('standardized narrative time')
    ax4.grid()

    ax5.plot(x, s5)
    ax5.plot(x, s10)
    ax5.set_title('FFT: 5th order')
    ax5.set_ylabel('Sentiment')
    ax5.set_xlabel('standardized narrative time')
    ax5.grid()

    fig.tight_layout(h_pad=4)
    plt.show()

    fig.savefig(os.getcwd() + '/fig/fft_plot.png')

if __name__ == '__main__':

    s1 = joblib.load(os.getcwd() + '/lib/sequence_raw_24749.arr')
    s2 = joblib.load(os.getcwd() + '/lib/sequence_fft_24749.arr')
    s3 = joblib.load(os.getcwd() + '/lib/sequence_fft_standardized_24749.arr')
    s4 = joblib.load(os.getcwd() + '/lib/sequence_fft_scaled_24749.arr')
    s5 = joblib.load(os.getcwd() + '/lib/sequence_raw_32651.arr')
    s6 = joblib.load(os.getcwd() + '/lib/sequence_fft_32651.arr')
    s7 = joblib.load(os.getcwd() + '/lib/sequence_fft_standardized_32651.arr')
    s8 = joblib.load(os.getcwd() + '/lib/sequence_fft_scaled_32651.arr')

    #pp_plot(s1, s2, s3, s4, s5, s6, s7, s8)

    s1 = joblib.load(os.getcwd() + '/lib/sequence_fft_4920_term_2.arr')
    s2 = joblib.load(os.getcwd() + '/lib/sequence_fft_4920_term_3.arr')
    s3 = joblib.load(os.getcwd() + '/lib/sequence_fft_4920_term_4.arr')
    s4 = joblib.load(os.getcwd() + '/lib/sequence_fft_4920_term_5.arr')
    s5 = joblib.load(os.getcwd() + '/lib/sequence_fft_4920_term_6.arr')
    s6 = joblib.load(os.getcwd() + '/lib/sequence_fft_24104_term_2.arr')
    s7 = joblib.load(os.getcwd() + '/lib/sequence_fft_24104_term_3.arr')
    s8 = joblib.load(os.getcwd() + '/lib/sequence_fft_24104_term_4.arr')
    s9 = joblib.load(os.getcwd() + '/lib/sequence_fft_24104_term_5.arr')
    s10 = joblib.load(os.getcwd() + '/lib/sequence_fft_24104_term_6.arr')

    fft_terms(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)