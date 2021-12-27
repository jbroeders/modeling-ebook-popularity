import requests
import urllib
import time
import os

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from tqdm import tqdm


def get_fiction_identifiers():

    res = pd.DataFrame()

    bookshelves = {'science_fiction': 68,
                   'horror': 42,
                   'history': 41,
                   'adventure': 82,
                   'mystery': 51,
                   'fantasy': 36,
                   'comedy': 44,
                   'crime': 28,
                   'children': 18,
                   'erotic': 33,
                   'romance': 412}

    sort_orders = ['title', 'downloads', 'release_date']
    base_url = 'https://www.gutenberg.org/ebooks/bookshelf/'

    book_ids = []
    genres = []

    for bookshelve in tqdm(bookshelves):
        for order in tqdm(sort_orders):

            idx = 1

            while True:

                response = requests.get(
                    base_url + "{}?sort_order={}&start_index={}".format(bookshelves[bookshelve], order, idx))
                soup = BeautifulSoup(response.text, features="lxml")

                booklinks = soup.find_all("li", "booklink")

                if len(booklinks) > 1:

                    for val in booklinks:
                        link = val.find("a")
                        book_id = link['href'].split('/')[-1]

                        book_ids.append(book_id)
                        genres.append(bookshelve)
                else:
                    break

                idx += 25

    res['id'] = book_ids
    res['genre'] = genres
    res = res.drop_duplicates(keep='first').reset_index(drop=True)
    # res.to_csv(os.getcwd() + '/data/pg_ids.csv')

    return(res)


def get_content(df):

    df['filename'] = 0

    base_url = 'https://www.gutenberg.org/ebooks/'

    for idx in tqdm(range(len(df))):

        book_id = df['id'][idx]
        filename = '{}.epub'.format(book_id)


        if os.path.isfile(os.getcwd() + '/data/pg/{}'.format(filename)):
            df['filename'][idx] = filename
            continue

        resp = requests.get(
            base_url + "{}.epub.noimages".format(book_id), allow_redirects=True)


        if resp.headers.get('Content-Type') == 'application/epub+zip':

            with open(os.getcwd() + '/data/pg/{}'.format(filename), 'wb') as f:
                f.write(resp.content)

            df['filename'][idx] = filename

        else:

            df['filename'][idx] = 'none'


        time.sleep(12)

    df.to_csv(os.getcwd() + '/data/pg_files.csv')

def get_ratings():

    base_url = 'https://www.goodreads.com/search?q='

    rv = []
    rs = []

    for idx in tqdm(range(len(df)), desc='Getting book ratings'):

        if idx % 50 == 0 and idx != 0:
            time.sleep(220)

        title = df['title'][idx]
        auth = df['author'][idx]

        try:
            url = (base_url + title + ' ' + auth).replace(' ', '%20')

        except Exception as e:
            rv.append(0)
            rs.append(0)
            continue

        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, features="lxml")

        ratings = soup.find_all("span", {"class": "minirating"})
        titles = soup.find_all("span", {"role": "heading"})
        authors = soup.find_all("span", {"itemprop": "author"})

        ratings_dict = {
            'rating': [],
            'sample_size': []
        }

        try:
            for i in range(len(ratings)):
                jw = jaro.jaro_winkler_metric(
                    title+auth, titles[i].text + authors[i].text)

                if jw > 0.8:
                    text = ratings[i].text

                    for idx in range(len(text)):

                        if text[idx].isnumeric():
                            rval = text[idx:idx+3]

                            break

                    for idx in reversed(range(len(text))):

                        if text[idx].isnumeric():

                            sample = text[idx]

                            while text[idx-1].isnumeric():
                                sample = text[idx - 1] + sample
                                idx = idx - 1

                            break

                ratings_dict['rating'].append(float(rval))
                ratings_dict['sample_size'].append(float(sample))

            # print(ratings_dict)

            max_ss = max(ratings_dict['sample_size'])
            max_ss_idx = ratings_dict['sample_size'].index(max_ss)

            # print("max_ss: {}, max_ss_idx: {}".format(max_ss, max_ss_idx))

            rv.append(ratings_dict['rating'][max_ss_idx])
            rs.append(ratings_dict['sample_size'][max_ss_idx])

        except Exception as e:
            # print(e)
            rv.append(0)
            rs.append(0)

        time.sleep(12)

    df['rating'] = rv
    df['rating_amount'] = rs

    df.to_csv(os.getcwd() + '/data/pg_ratings.csv')


if __name__ == '__main__':

    df = get_fiction_identifiers()
    df = get_content(df)
    get_ratings(df)
