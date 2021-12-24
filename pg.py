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
    res.to_csv('pg.csv')

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

    df.to_csv('pg_complete.csv')

    return(df)


if __name__ == '__main__':

    df = pd.read_csv(os.getcwd() + '/data/pg.csv')
    get_content(df)
