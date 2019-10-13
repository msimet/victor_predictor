"""
predict.py: Take models trained by train.py and add new columns to a dataframe representing their predicted outputs.
"""
import pickle
import numpy as np
import scipy.spatial.distance
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import gensim
from utils import make_regression_array, make_genre_data, remove_proper_nouns

def predict_genres(df):
    """
    Predict genres using the trained k-means model.
    """
    genre_data_rescaled, _ = make_genre_data(df, keep_zeros=True)
    with open('trained_models/genre_rescaler.p', 'rb') as f:
        mms = pickle.load(f)
    data_transformed = mms.transform(genre_data_rescaled)

    with open('trained_models/genre_classifier.p', 'rb') as f:
        model = pickle.load(f)
    centroids = model.cluster_centers_
    distances = {}
    for i, genre in enumerate(centroids):
        distances['genre{}'.format(i)] = [scipy.spatial.distance.cosine(book, genre) for book in data_transformed]

    for key in distances:
        df[key] = distances[key]

    df = df.drop(columns=["fantasy", "sf", "horror", "dystopian", "romance",
                          "adventure", "urban_fantasy", "mystery", "historical",
                          "high_fantasy", "mythology", "humor", "literature",
                          "time_travel", "space", "lgbt"])
    return df

def predict_topics(df):
    """
    Take the trained topics we found earlier, and run the models on review sentences. Then run a pre-trained
    sentiment analysis on each sentence. Take the max topic for each review and add the sentiment to that element;
    then take the average over all sentences per book.
    """
    stopwords = (nltk.corpus.stopwords.words('english')
                 + ['book', 'read'])
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    lda_model = gensim.models.LdaMulticore.load('good_lda_model.gensim')

    sid = SentimentIntensityAnalyzer()
    lemmatizer = WordNetLemmatizer()

    def make_nvec(v, n=10):
        if len(v) == n:
            return [vv[1] for vv in v]
        result = [0.] * n
        for vv in v:
            result[vv[0]] = vv[1]
        return result

    sentiments = []
    nsentiments = []
    vectors = []
    for review in df['reviews']:
        sentences = []
        for sentence in review.split('.'):
            sentences.extend(sentence.split('||'))

        clean_sentences = [remove_proper_nouns(sentence)[0] for sentence in sentences]
        clean_sentences = [[lemmatizer.lemmatize(t) for t in nltk.word_tokenize(sentence)]
                           for sentence in clean_sentences]
        clean_sentences = [[word for word in sentence if word not in stopwords and len(word) > 3]
                           for sentence in clean_sentences]
        clean_sentences = [dictionary.doc2bow(sentence) for sentence in clean_sentences]
        sentence_vectors = [make_nvec(lda_model[sentence]) for sentence in clean_sentences]
        vectors.append(np.mean(sentence_vectors, axis=0))

        sentiment = np.zeros(10)
        nsentiment = np.zeros(10)
        for sentence, vector in zip(sentences, sentence_vectors):
            score = sid.polarity_scores(sentence)
            sentiment[np.argmax(vector)] += score['compound']
            nsentiment[np.argmax(vector)] += 1
        sentiments.append(sentiment / nsentiment)
        nsentiments.append(nsentiment)
    sentiments = np.array(sentiments)
    sentiments[np.isnan(sentiments)] = 0  # division by zero
    for i in range(10):
        df['review_embed{}'.format(i)] = sentiments[:, i]
    return df

def predict_score(df):
    """
    Take the pre-trained tree models and compute the A) nominee class and b) continuous score.
    Then turn those into a 0-1 score. Also do this marginalizing over authorship variables & reading level variables
    or both.
    """
    data, mask = make_regression_array(df, keep_short_stories=True)
    df = df[mask]

    cols = np.load('trained_data/columns.npy')
    model_list = []
    for i in range(20):
        with open(f'trained_data/gbm_ensemble_{i}.p', 'rb') as f:
            model_list.append(pickle.load(f))

    regression_model_list = []
    for i in range(10):
        with open(f'trained_data/gbm_regressor_ensemble_{i}.p', 'rb') as f:
            regression_model_list.append(pickle.load(f))
    thresh = np.load('trained_models/thresh.npy')


    yc_list = []
    for model in model_list:
        yc_list.append(model.predict(data[cols]))
    nominee = np.mean(yc_list, axis=0) >= thresh

    ys_list = []
    for model in regression_model_list:
        ys_list.append(model.predict(data[cols]))
    score = np.mean(ys_list, axis=0)
    yes_mask = nominee > 0
    ymin = score[yes_mask].min()
    ymax = score[yes_mask].max()
    nmin = score[~yes_mask].min()
    nmax = score[~yes_mask].max()
    score[yes_mask] = np.clip(0.5 + 0.5 * (score[yes_mask] - ymin) / (ymax - ymin), 0.5, 1.0)
    score[~yes_mask] = np.clip(0.5 * (score[~yes_mask] - nmin) / (nmax - nmin), 0, 0.5)

    df['pred_score'] = score

    def make_datasets_readinglevel(data):
        tdata = data.copy()
        for ya in [data['ya'].min(), data['ya'].max()]:
            for children in [data['children'].min(), data['children'].max()]:
                tdata['ya'] = ya
                tdata['children'] = children
                yield tdata

    def make_datasets_author(data):
        tdata = data.copy()
        for aviews in np.linspace(
                data['author_annualviews'].min(), data['author_annualviews'].max(), 10):
            for nprev in range(max(data['nprev']) // 2):
                tdata['author_annualviews'] = aviews
                tdata['nprev'] = nprev
                yield tdata

    # Marginalize over reading level
    yc_list = []
    for tdata in make_datasets_readinglevel(data):
        for model in model_list:
            yc_list.append(model.predict(tdata[cols]))
    nominee = np.mean(yc_list, axis=0) > 0.1

    ys_list = []
    for tdata in make_datasets_readinglevel(data):
        for model in regression_model_list:
            ys_list.append(model.predict(tdata[cols]))
    score = np.mean(ys_list, axis=0)
    yes_mask = nominee > 0
    ymin = score[yes_mask].min()
    ymax = score[yes_mask].max()
    nmin = score[~yes_mask].min()
    nmax = score[~yes_mask].max()
    score[yes_mask] = np.clip(0.5 + 0.5 * (score[yes_mask] - ymin) / (ymax - ymin), 0.5, 1.0)
    score[~yes_mask] = np.clip(0.5 * (score[~yes_mask] - nmin) / (nmax - nmin), 0, 0.5)

    df['pred_score_readinglevel'] = score

    # Marginalize over authorship
    yc_list = []
    for tdata in make_datasets_author(data):
        for model in model_list:
            yc_list.append(model.predict(tdata[cols]))
    nominee = np.mean(yc_list, axis=0) > 0.1

    ys_list = []
    for tdata in make_datasets_author(data):
        for model in regression_model_list:
            ys_list.append(model.predict(tdata[cols]))
    score = np.mean(ys_list, axis=0)
    yes_mask = nominee > 0
    ymin = score[yes_mask].min()
    ymax = score[yes_mask].max()
    nmin = score[~yes_mask].min()
    nmax = score[~yes_mask].max()
    score[yes_mask] = np.clip(0.5 + 0.5 * (score[yes_mask] - ymin) / (ymax - ymin), 0.5, 1.0)
    score[~yes_mask] = np.clip(0.5 * (score[~yes_mask] - nmin) / (nmax - nmin), 0, 0.5)

    df['pred_score_author'] = score

    # Marginalize over reading level and authorship
    yc_list = []
    for ttdata in make_datasets_readinglevel(data):
        for tdata in make_datasets_author(ttdata):
            for model in model_list:
                yc_list.append(model.predict(tdata[cols]))
    nominee = np.mean(yc_list, axis=0) > 0.2

    ys_list = []
    for ttdata in make_datasets_readinglevel(data):
        for tdata in make_datasets_author(ttdata):
            for model in regression_model_list:
                ys_list.append(model.predict(tdata[cols]))
    score = np.mean(ys_list, axis=0)
    yes_mask = nominee > 0
    ymin = score[yes_mask].min()
    ymax = score[yes_mask].max()
    nmin = score[~yes_mask].min()
    nmax = score[~yes_mask].max()
    score[yes_mask] = np.clip(0.5 + 0.5 * (score[yes_mask] - ymin) / (ymax - ymin), 0.5, 1.0)
    score[~yes_mask] = np.clip(0.5 * (score[~yes_mask] - nmin) / (nmax - nmin), 0, 0.5)

    df['pred_score_readinglevel_author'] = score

    return df
