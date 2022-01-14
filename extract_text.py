import pandas as pd
import numpy as np
import os

from textblob import TextBlob
from textblob.tokenizers import WordTokenizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from joblib import dump

cwd = os.getcwd()

df = pd.read_csv(cwd + '/data/paragraphs_labeled.csv')
df = df[['paragraph', 'nr']].reset_index(drop=True)
df = df.dropna(subset=['paragraph']).reset_index(drop=True)



df['paragraph'] = [i if isinstance(i, str) else str(i) for i in df['paragraph']]


corpus = ';'.join(tuple(df['paragraph'].apply(lambda x: x.replace(';', '')))).lower().split(';')


vocab = ['by', 'illustrated', 'illustrations', 'note', 'notes', 'transcriber', "transcriber's", 'chapter', 'editor', 'u.s.', 'copyright', 'publication', 'publisher',
'renewed', 'published', 'ace', 'york', 'end']


tfidf = TfidfVectorizer(vocabulary = vocab)
X = tfidf.fit_transform(corpus)
y = np.array(df['nr'])


print(X.shape, y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


rf = RandomForestClassifier()
rf.fit(X, y)

dump(rf, cwd + '/lib/rf_textextraction.joblib')


rf.fit(X_train, y_train)

yhat = rf.predict(X_test)

print(classification_report(y_test, yhat))
print(confusion_matrix(y_test, yhat))

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(rf, X_test, y_test, cmap=Noner)
plt.show()


