import pandas as pd
import ebooklib
import pickle
import os


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
from textblob import TextBlob
from ebooklib import epub
from nrclex import NRCLex
from joblib import dump, load
from tqdm import tqdm
import numpy as np


import matplotlib.pyplot as plt

def extract_metadata(df):

    titles = []
    authors = []
    langs = []

    for i in tqdm(range(len(df)), desc='Extracting metadata'):
        filename = df['filename'][i]

        book = epub.read_epub(os.getcwd() + '/data/pg/{}'.format(filename))

        author = book.get_metadata('DC', 'creator')
        title = book.get_metadata('DC', 'title')
        lang = book.get_metadata('DC', 'language')

        if author and title:
            authors.append(author[0][0])
            titles.append(title[0][0])

        else:
            authors.append('')
            titles.append('')

        langs.append(lang[0][0])

    df['author'] = authors
    df['title'] = titles
    df['lang'] = langs

    df = df[df['lang'] == 'en'].reset_index(drop=True)

    return(df)

def filter_df(df):

    initial_length = len(df)
    df = df[df['lang'] == 'en'].reset_index(drop=True)
    fl = len(df)
    print('Removed {} books which do not have English as its main language'.format(initial_length - fl))
    df = df[df['rating_amount'] > 20].reset_index(drop=True)
    print('Removed {} books which have not been rated at least 20 times'.format(fl - len(df)))

    return(df)

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


def scale_sequence(sequence):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sequence = np.array(sequence).reshape(-1, 1)
    sequence = scaler.fit_transform(sequence)
    x = [i for i in range(len(sequence))]

    sequence = shorten_sequence([i[0] for i in sequence])
    

    return(sequence)
    # plt.plot(x, sequence)
    # plt.show()

def fft(sequence):

    # plt.figure()

    n = [i for i in range(len(sequence))]

    rft = np.fft.rfft(sequence)
    rft[5:] = 0
    y_smooth = np.fft.irfft(rft)
    x = [i for i in range(len(y_smooth))]

    return(y_smooth)
    # plt.plot(x, y_smooth)
    # plt.show()

def sentiment(sequence):
    analyzer = SentimentIntensityAnalyzer()

    scores = []
    for s in sequence:
        scores.append(analyzer.polarity_scores(s)['compound'])


    return(scores)


def nrc(sequence):
    pass


def process_text(df):

    rf = load(os.getcwd() + '/lib/rf_textextraction.joblib')

    df_temp = pd.DataFrame()
    ids = []
    ps = []

    for idx in tqdm(range(len(df)), desc='Processing text'):

        filename = df['filename'][idx]
        book = epub.read_epub(os.getcwd() + '/data/pg/' + filename)

        for item in book.get_items():

            if item.get_type() == ebooklib.ITEM_DOCUMENT:

                html = item.get_content()
                soup = BeautifulSoup(html, features='lxml')
                paragraphs = soup.find_all('p')
            
                if len(paragraphs) > 1:

                    for p in paragraphs:
                        ps.append(p.text)
                        ids.append(df['id'][idx])

    df_temp['id'] = ids
    df_temp['paragraph'] = ps

    df_temp = df_temp.dropna(subset=['paragraph']).reset_index(drop=True)

    df_temp['paragraph'] = [i if isinstance(i, str) else str(i) for i in df_temp['paragraph']]

    corpus = ';'.join(tuple(df_temp['paragraph'].apply(lambda x: x.replace(';', '')))).lower().split(';')

    vocab = ['by', 'illustrated', 'illustrations', 'note', 'notes', 'transcriber', "transcriber's",
             'chapter', 'editor', 'u.s.', 'copyright', 'publication', 'publisher',
            'renewed', 'published', 'ace', 'york', 'end']


    tfidf = TfidfVectorizer(vocabulary = vocab)
    X = tfidf.fit_transform(corpus)

    df_temp['yhat'] = rf.predict(X)


    grouped = df_temp.groupby(by='id')
    df = df.set_index('id')
    df['sentences'] = 0
    df['words'] = 0


    ids = []
    sequences = []
    ratings = []

    for name, dfg in tqdm(grouped):
        dfg = dfg[dfg['yhat'] == 1]
        book_corpus = ' '.join(';'.join(tuple(dfg['paragraph'].apply(lambda x: x.replace(';', '')))).lower().split(';'))

        sentences = TextBlob(book_corpus).sentences
        words = TextBlob(book_corpus).words

        windows = []

        for i in range(len(words)):

            window = []

            if i < 3:
                window.append(words[0:i+4])

            elif i > len(words) - 3:
                window.append(words[i-3:i+1])

            else:
                window.append(words[i-3:i+4])

            windows.append(window)


        

        if len(sentences) > 1:

            s = sentiment(sentences)
            # dump(s, os.getcwd() + '/lib/sequence_raw.arr')
            # break
            s = fft(s)
            # dump(s, os.getcwd() + '/lib/sequence_fft.arr')
            # break
            s = scale_sequence(s)
            dump(s, os.getcwd() + '/lib/sequence_fft_2.arr')
            break

            sequences.append(s)
            rating = df['rating'][name]
            print(rating, name)

            try:
                assert isinstance(rating, float)
                ratings.append(rating)

            except:
                assert(isinstance(rating.values[0], float))
                ratings.append(rating.values[0])

            ids.append(int(name))


    res = pd.DataFrame()
    res['id'] = ids
    res['sentiment'] = sequences
    res['rating'] = ratings

    return(res)








def main():
    df = pd.read_csv(os.getcwd() + '/data/pg_ratings.csv')[0:50]
    df = extract_metadata(df)
    df = filter_df(df)

    res = process_text(df)
    res.to_csv(os.getcwd() + '/data/pg_clean.csv')



if __name__ == '__main__':

    main()

