import pandas as pd
import ebooklib
import pickle
import os


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
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
    dates = []

    for i in tqdm(range(len(df)), desc='Extracting metadata'):
        filename = df['filename'][i]

        book = epub.read_epub(os.getcwd() + '/data/pg/{}'.format(filename))

        author = book.get_metadata('DC', 'creator')
        title = book.get_metadata('DC', 'title')
        lang = book.get_metadata('DC', 'language')
        date = book.get_metadata('DC', 'date')

        # desc = book.get_metadata('DC', 'description')
        # print(desc)



        if author and title:
            authors.append(author[0][0])
            titles.append(title[0][0])

        else:
            authors.append('')
            titles.append('')

        if date:
            dates.append(date[0][0])

        else:
            dates.append['']

        langs.append(lang[0][0])

    df['author'] = authors
    df['title'] = titles
    df['lang'] = langs
    df['date'] = dates

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

def standardize_sequence(seq):
    n = len(seq)
    res = []

    if n > 100:

        split = n/100
        for i in range(100):
            res.append(np.mean(seq[int(i*split): int(i*split + split)]))

    else:

        res = pad_sequences([seq], padding='post', dtype='float32', value =0, maxlen = 100)
        return(res)

   
    return(res)


def scale_sequence(sequence):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sequence = np.array(sequence).reshape(-1, 1)
    sequence = scaler.fit_transform(sequence)

    sequence = ([i[0] for i in sequence])

    return(sequence)

def fft(sequence, term):

    # plt.figure()

    n = [i for i in range(len(sequence))]

    rft = np.fft.rfft(sequence)
    rft[term:] = 0
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
    df['sl'] = 0
    df['words'] = 0


    ids = []
    sequences = []
    ratings = []
    sl = []
    genres = []
    dates = []
    titles = []

    nrc_fears = []
    nrc_angers = []
    nrc_trusts = []
    nrc_surprises = []
    nrc_positives = []
    nrc_negatives = []
    nrc_sadnesss = []
    nrc_disgusts = []
    nrc_joys = []


    for name, dfg in tqdm(grouped):
        dfg = dfg[dfg['yhat'] == 1]
        book_corpus = ' '.join(';'.join(tuple(dfg['paragraph'].apply(lambda x: x.replace(';', '')))).lower().split(';'))

        sentences = TextBlob(book_corpus).sentences
        words = TextBlob(book_corpus).words

        if len(sentences) > 1:
            
            s1 = sentiment(sentences)
            # dump(s1, os.getcwd() + '/lib/sequence_raw_32651.arr')
            # for i in [2, 3, 4, 5, 6]:
            #     s2 = fft(s1, i)
            s2 = fft(s1, 2)
            # dump(s2, os.getcwd() + '/lib/sequence_fft_32651.arr')
        
            s3 = standardize_sequence(s2)
            # dump(s3, os.getcwd() + '/lib/sequence_fft_standardized_32651.arr')

            s4 = scale_sequence(s3)
            #dump(s4, os.getcwd() + '/lib/sequence_fft_scaled_32651.arr')
            #dump(s4, os.getcwd() + '/lib/sequence_fft_4920_term_{}.arr'.format(i))
            
            # break
            # pp_plot(s1, s2, s3, s4)

            sequences.append(s4)
            rating = df['rating'][name]
            genre = df['genre'][name]
            title = df['title'][name]
            date = df['date'][name]
           
            try:
                assert isinstance(rating, float)
                ratings.append(rating)
                genres.append(genre)
                titles.append(title)
                dates.append(date)

            except:
                assert(isinstance(rating.values[0], float))
                ratings.append(rating.values[0])
                genres.append(genre.values[0])
                titles.append(title.values[0])
                dates.append(date.values[0])

            sl.append(len(sentences))
            ids.append(int(name))



            nrc_fear = []
            nrc_anger = []
            nrc_trust = []
            nrc_surprise = []
            nrc_positive = []
            nrc_negative = []
            nrc_sadness = []
            nrc_disgust = []
            nrc_joy = []

            for sentence in sentences:

                sentence = str(sentence)

                # print('\t' + str(sentence))
                # print(str(sentence))
                nrc = NRCLex(sentence).affect_frequencies
                
                nrc_fear.append(nrc['fear'])
                nrc_anger.append(nrc['anger'])
                nrc_trust.append(nrc['trust'])
                nrc_surprise.append(nrc['surprise'])
                nrc_positive.append(nrc['positive'])
                nrc_negative.append(nrc['negative'])
                nrc_sadness.append(nrc['sadness'])
                nrc_disgust.append(nrc['disgust'])
                nrc_joy.append(nrc['joy'])


            nrc_fear = fft(nrc_fear, 2)
            nrc_fear = standardize_sequence(nrc_fear)
            nrc_fear = scale_sequence(nrc_fear)

            nrc_anger = fft(nrc_anger, 2)
            nrc_anger = standardize_sequence(nrc_anger)
            nrc_anger = scale_sequence(nrc_anger)

            nrc_trust = fft(nrc_trust, 2)
            nrc_trust = standardize_sequence(nrc_trust)
            nrc_trust = scale_sequence(nrc_trust)

            nrc_surprise = fft(nrc_surprise, 2)
            nrc_surprise = standardize_sequence(nrc_surprise)
            nrc_surprise = scale_sequence(nrc_surprise)

            nrc_positive = fft(nrc_positive, 2)
            nrc_positive = standardize_sequence(nrc_positive)
            nrc_positive = scale_sequence(nrc_positive)

            nrc_negative = fft(nrc_negative, 2)
            nrc_negative = standardize_sequence(nrc_negative)
            nrc_negative = scale_sequence(nrc_negative)

            nrc_sadness = fft(nrc_sadness, 2)
            nrc_sadness = standardize_sequence(nrc_sadness)
            nrc_sadness = scale_sequence(nrc_sadness)

            nrc_disgust = fft(nrc_disgust, 2)
            nrc_disgust = standardize_sequence(nrc_disgust)
            nrc_disgust = scale_sequence(nrc_disgust)

            nrc_joy = fft(nrc_fear, 2)
            nrc_joy = standardize_sequence(nrc_joy)
            nrc_joy = scale_sequence(nrc_joy)

            nrc_fears.append(nrc_fear)
            nrc_angers.append(nrc_anger)
            nrc_trusts.append(nrc_trust)
            nrc_surprises.append(nrc_surprise)
            nrc_positives.append(nrc_positive)
            nrc_negatives.append(nrc_negative)
            nrc_sadnesss.append(nrc_sadness)
            nrc_disgusts.append(nrc_disgust)
            nrc_joys.append(nrc_joy)

            


    res = pd.DataFrame()
    res['id'] = ids
    res['sentiment'] = sequences
    
    res['sl'] = sl
    res['genre'] = genres
    res['title'] = titles
    res['date'] = dates

    res['nrc_fear'] = nrc_fears
    res['nrc_anger'] = nrc_angers
    res['nrc_trust'] = nrc_trusts
    res['nrc_surprise'] = nrc_surprises
    res['nrc_positive'] = nrc_positives
    res['nrc_negative'] = nrc_negatives
    res['nrc_sadness'] = nrc_sadnesss
    res['nrc_disgust'] = nrc_disgusts
    res['nrc_joy'] = nrc_joys
    
    res['rating'] = ratings


    return(res)


def pp_plot(s1, s2, s3, s4):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6.4, 18))

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
    ax3.set_title('Sentiment Scores (standardized)')
    ax3.set_ylabel('Compound Score')
    ax3.set_xlabel('Narrative time')
    ax3.grid()

    ax4.plot(x, s4)
    ax4.set_title('Sentiment Scores (scaled)')
    ax4.set_ylabel('Compound Score (scaled)')
    ax4.set_xlabel('Narrative time')
    ax4.grid()

    fig.tight_layout(pad=3.5)
    #fig.suptitle('Sentiment Vector')
    plt.savefig(os.getcwd() + '/fig/pp_plot.png')

    plt.show()




def main():
    df = pd.read_csv(os.getcwd() + '/data/pg_ratings.csv').reset_index(drop=True)
    # df = df[df['id'] == 4920].reset_index(drop=True)
    df = extract_metadata(df)
    df = filter_df(df)

    res = process_text(df)
    res.to_csv(os.getcwd() + '/data/fft_2.csv')
    print(res.head())



    



if __name__ == '__main__':

    main()

