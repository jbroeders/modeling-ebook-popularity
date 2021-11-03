import pandas as pd
import pickle
import os

from bs4 import BeautifulSoup

import ebooklib
from ebooklib import epub
from tqdm import tqdm

from nrclex import NRCLex


def add_filenames(df):

    df['filename'] = [str(i) + '.epub' for i in df['id']]

    for i in tqdm(range(len(df)), desc="Adding filenames"):
        filename = df['filename'][i]

        if not os.path.isfile(os.getcwd() + '/data/pg/{}'.format(filename)):
            df = df.drop(i, axis=0)

    df = df.reset_index(drop=True)

    return(df)


def add_meta(df):

    titles = []
    authors = []
    langs = []

    for i in tqdm(range(len(df)), desc='Adding metadata'):
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


def extract_text(df):

    df['corpus'] = ''
    df['sentences'] = [[] for i in range(len(df))]
    df['paragraphs'] = [[] for i in range(len(df))]

    book = epub.read_epub(os.getcwd() + '/data/pg/' + df['filename'][0])

    for idx in tqdm(range(len(df))):
        filename = df['filename'][idx]
        book = epub.read_epub(os.getcwd() + '/data/pg/' + filename)

        s = []
        ps = []
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

                    sentences = text.split('.')
                    for sentence in sentences:
                        s.append(sentence)

                    ps.append(text)
                    res += text

                for p in paragraphs[5:-5]:

                    text = p.text

                    sentences = text.split('.')
                    for sentence in sentences:
                        s.append(sentence)

                    ps.append(text)
                    res += text

                for p in paragraphs[-5:]:
                    text = p.text

                    if ('Editor' in text) or ('END' in text) or ('errors' in text) or ('[1]' in text) or ('Transcriber' in text):
                        continue

                    sentences = text.split('.')
                    for sentence in sentences:
                        s.append(sentence)

                    ps.append(text)
                    res += text

        df['sentences'][idx] = s
        df['paragraphs'][idx] = ps
        df['corpus'][idx] = res

    return(df)


def extract_text_simple(df):

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


def add_nrc_simple(df):

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


def add_nrc(df):

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

        ds = {'fear': [],
              'anger': [],
              'trust': [],
              'surprise': [],
              'positive': [],
              'negative': [],
              'sadness': [],
              'disgust': [],
              'joy': [],
              }

        text = df['corpus'][idx]
        nrc = NRCLex(text)

        for sentence in nrc.sentences:
            nrc_sent = NRCLex(str(sentence))

            nrc_dict = nrc_sent.affect_frequencies

            for key in nrc_dict:

                if 'anticip' in key:
                    continue

                ds[key].append(nrc_dict[key])

        for key in ds:
            d[key].append(ds[key])

    for key in d:
        print(len(d[key]))
        df[key] = d[key]

    return(df)


def clean_text(df):

    return(df)


def main():
    df = pd.read_csv(
        os.getcwd() + '/data/pg.csv').drop(['Unnamed: 0'], axis=1)
    df = add_filenames(df)
    df = add_meta(df)
    df = extract_text_simple(df)
    df = add_nrc(df)
    df = clean_text(df)
    df.to_csv(os.getcwd() + '/data/pg_clean.csv', index=False)


if __name__ == '__main__':

    main()
