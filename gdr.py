import pandas as pd
import requests
import time
import os

from bs4 import BeautifulSoup
from tqdm import tqdm

if __name__ == '__main__':

    df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')
    base_url = 'https://www.goodreads.com/search?q='

    r = []
    ra = []

    try:

        for idx in tqdm(range(759, len(df)), desc='Getting book ratings'):

            if idx % 50 == 0 and idx != 0:
                time.sleep(160)

            title = df['title'][idx]
            auth = df['author'][idx]

            try:
                url = (base_url + title + ' ' + auth).replace(' ', '%20')

            except Exception as e:
                r.append(0)
                ra.append(0)

            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, features="lxml")

            ratings = soup.find_all("span", {"class": "minirating"})

            try:
                text = ratings[0].text

                for idx in range(len(text)):

                    if text[idx].isnumeric():
                        rating = text[idx:idx+3]

                        break

                for idx in reversed(range(len(text))):

                    if text[idx].isnumeric():

                        sample = text[idx]

                        while text[idx-1].isnumeric():
                            sample = text[idx - 1] + sample
                            idx = idx - 1

                        break

                r.append(rating)
                ra.append(sample)

            except Exception as e:
                r.append(0)
                ra.append(0)

            time.sleep(5)

        df['rating'] = r
        df['rating_amount'] = ra

        df.to_csv(os.getcwd() + '/data/pg_ratings.csv')

    except Exception as e:

        df = pd.DataFrame()
        df['rating'] = r
        df['rating_amount'] = ra

        df.to_csv(os.getcwd() + '/data/ratings_temp2.csv')
