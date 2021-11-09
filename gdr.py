import pandas as pd
import requests
import jaro
import time
import os

from bs4 import BeautifulSoup
from tqdm import tqdm

if __name__ == '__main__':

    # df = pd.read_csv(os.getcwd() + '/data/pg_clean.csv')
    df = pd.read_csv('aws.csv')
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
