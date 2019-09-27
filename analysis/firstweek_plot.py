import re
import string
from collections import Counter
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns

import nltk
from nltk.stem.snowball import SnowballStemmer

def main():
    with open('calcstars.p', 'rb') as f:
        data = pickle.load(f)
    reviews = data['reviews']
    review_text = ' '.join([r['text'] for r in reviews])

    stemmer = SnowballStemmer("english")
    review_tokens = [stemmer.stem(r) for r in nltk.word_tokenize(review_text)]
    token_counts = Counter(review_tokens)
    stopwords = (nltk.corpus.stopwords.words('english')
                 + list(string.ascii_lowercase)
                 + list(string.punctuation)
                 + ['``', "''", "'s", '’', "n't", '...'])
    words = list(set(list(token_counts)) - set(stopwords))
    words.sort(key = lambda x: token_counts[x], reverse=True)

    words = words[:20]

    rcParams['font.sans-serif'] = ['Avantgarde', 'TeX Gyre Adventor', 'URW Gothic L']+rcParams['font.sans-serif']
    #sns.set(font='TeX Gyre Adventor')
    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(4,4))
    manywords = []
    for word in words:
        manywords += [word]*token_counts[word]
    sns.countplot(y=manywords, palette='winter')
    plt.tight_layout()
    plt.savefig('word_counts.png')






if __name__=='__main__':
    main()