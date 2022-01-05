import pandas as pd
import pickle
import os

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import TextBlob


import ebooklib
from ebooklib import epub
from tqdm import tqdm
from joblib import load

from nrclex import NRCLex

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
    print('Removed {} books which do not have English as its main language'.format(initial_length - len(df)))
    df = df[df['rating_amount'] > 20].reset_index(drop=True)
    print('Removed {} books which have not been rated at least 20 times'.format(fl - len(df)))

    return(df)


def extract_text(df):

    df['corpus'] = ''
    book = epub.read_epub(os.getcwd() + '/data/pg/' + df['filename'][0])

    for idx in tqdm(range(len(df)), desc='Extracting text'):
        filename = df['filename'][idx]
        book = epub.read_epub(os.getcwd() + '/data/pg/' + filename)

        res = ''

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:

                html = item.get_content()
                soup = BeautifulSoup(html, features='lxml')
                paragraphs = soup.find_all('p')

                for p in paragraphs[0:5]:

                    text = p.text

                    if 'Illustrated' in text or 'Transcriber' in text or text[0:2] == 'By':
                        continue

                    res += text

                for p in paragraphs[5:-5]:

                    text = p.text
                    res += text

                for p in paragraphs[-5:]:
                    text = p.text

                    if ('Editor' in text) or ('END' in text) or ('errors' in text) or ('[1]' in text) or ('Transcriber' in text):
                        continue

                    res += text

        # df['corpus'][idx] = res
        df.loc[idx, 'corpus'] = res

    return(df)


def nrc(df):

    d = {'fear': [],
         'anger': [],
         'trust': [],
         'surprise': [],
         'positive': [],
         'negative': [],
         'sadness': [],
         'disgust': [],
         'joy': [],
         }

    for idx in tqdm(range(len(df['corpus'])), desc='Adding nrc features'):

        text = df['corpus'][idx]
        nrc = NRCLex(text).affect_frequencies

        for key in nrc:

            if 'anticip' in key:
                continue

            d[key].append(nrc[key])

    for key in d:
        print(len(d[key]))
        df[key] = d[key]

    return(df)

def main():
    df = pd.read_csv(os.getcwd() + '/data/pg_ratings.csv')[0:125]
    df = extract_metadata(df)
    df = filter_df(df)
    df = extract_text(df)
    df = nrc(df)
   

if __name__ == '__main__':

    main()

