import pandas as pd
import requests
import os

from tqdm import tqdm

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium import webdriver

from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.utils import ChromeType


def collect_ol():

    subjects = ['fantasy', 'horror', 'humor', 'romance', 'science_fiction',
                'short_stories', 'thriller', 'young_adult']
    pi_vals = range(1950, 2022)

    for subject in subjects:
        for pi in pi_vals:

            response = requests.get(
                'https://openlibrary.org/subjects/{}.json?limit=1000&ebooks=true&published_in={}'
                .format(subject, pi))

            try:
                print(subject, pi, len(pd.read_json(response.text)))

            except Exception as e:
                print(e)

            with open(os.getcwd() + '/data/ol/{}_{}.json'.format(subject, pi), "w") as file:
                file.write(response.text)


def combine_ol():

    data_dir = os.getcwd() + '/data/ol/'
    files = os.listdir(data_dir)

    res = pd.DataFrame()

    for file in tqdm(files):

        df = pd.read_json(data_dir + file)
        works = pd.json_normalize(df['works'])
        works['genre'] = file.split('_')[0]

        res = pd.concat([res, works])

        # print('File: {}, n = {}'.format(file, len(works)))

    res.to_csv('ol.csv')
    print('succes!')


def download_ol(genre):

    df = pd.read_csv(os.getcwd() + '/data/ol.csv')
    df = df[df['genre'] == genre].reset_index(drop=True)
    df['filename'] = 0

    print('Items to lookup: {}'.format(len(df)))

    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': os.getcwd() + '/data/ol/epub/'}
    chrome_options.add_experimental_option('prefs', prefs)

    browser = webdriver.Chrome(ChromeDriverManager(
        chrome_type=ChromeType.CHROMIUM).install(), options=chrome_options)
    wait = WebDriverWait(browser, 20)

    base_url = 'https://openlibrary.org/'

    for i in tqdm(range(len(df))):
        browser.get(base_url + df['key'][i])

        try:
            availability = wait.until(EC.visibility_of_element_located(
                (By.XPATH, '//*[@id="read-options"]/div[3]/a[1]'))).text

            if availability == 'Read':

                dl = wait.until(EC.visibility_of_element_located(
                    (By.XPATH, '//*[@id="read-options"]/div[6]/ul/li[3]/a')))

                filename = dl.get_attribute('href').split('/')[-1]
                df['filename'][i] = filename

                dl.click()

        except Exception as e:
            print(e)


def dl_file(url, filename):

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(os.getcwd() + '/data/ol/epub/{}'.format(filename), 'wb') as file:
            for chunk in r.iter_content(chunk_size=8192):
                file.write(chunk)


def dl_ol(genre):

    df = pd.read_csv(os.getcwd() + '/data/ol.csv')
    df = df[df['genre'] == genre].dropna(
        subset=['availability.identifier']).reset_index(drop=True)
    df['filename'] = 0

    print('Items to lookup: {}'.format(len(df)))

    base_url = 'https://archive.org/download/'

    for idx in tqdm(range(len(df))):
        identifier = df['availability.identifier'][idx]

        dl_url = base_url + '{}/{}.epub'.format(identifier, identifier)
        filename = dl_url.split('/')[-1]

        try:
            dl_file(dl_url, filename)
            df['filename'][idx] = filename

        except Exception as e:
            # print(e)
            pass

    df.to_csv('ol_{}.csv'.format(genre))


if __name__ == '__main__':

    dl_ol('horror')
