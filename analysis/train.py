"""
train.py: Train all the elements of the VictorPredictor score.
"""

import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
import nltk
import gensim
from nltk.stem import WordNetLemmatizer
from utils import make_genre_data, remove_proper_nouns

def make_genres(df):
    """
    Fit genres by k-means clustering the bucketized shelf data.

    Parameters
    ==========
    df: pd.DataFrame
        Shelf data.
    """
    genre_data_rescaled, ids = make_genre_data(df)
    mms = MinMaxScaler()
    mms.fit(genre_data_rescaled)
    with open('trained_models/genre_rescaler.p', 'wb') as f:
        pickle.dump(mms, f)
    data_transformed = mms.transform(genre_data_rescaled)

    # Want to keep clusters at least a minimum size--3% of the dataset seems to work well.
    size_of_clusters = [1]
    min_size = int(0.03 * len(data_transformed))

    model = None
    masked_data = data_transformed
    # Now, we loop through a model that categorizes genres, and throw out any too-small genres until we converge.
    while np.min(size_of_clusters) < min_size:
        if model is not None:
            remove_clusters = [i for i, c in enumerate(size_of_clusters) if c < min_size]
            for i in remove_clusters:
                mask = labels != i
                labels = labels[mask]
                masked_data = masked_data[mask]
                ids = ids[mask]
            min_size = int(0.03 * len(masked_data))

        # Start a kmeans cluster model, and run it through the KElbowVisualizer to find the best-fit value of n_clusters
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(4, 24))

        visualizer.fit(masked_data)  # Fit the data to the visualizer
        visualizer.finalize()
        plt.savefig("eda/k_elbow_visualizer.png") # Okay to overwrite this, we only want the last one anyway
        ngenres = visualizer.elbow_value_

        model = KMeans(ngenres)
        model.fit(masked_data)
        labels = model.predict(masked_data)
        label_counter = Counter(labels)
        size_of_clusters = [label_counter[i] for i in range(ngenres)]
    with open('trained_models/genre_classifier.p', 'wb') as f:
        pickle.dump(model, f)
    np.save('trained_models/ngenres.npy', ngenres)

def make_topics(df):
    """
    Train the gensim topics that will be used to categorize reviews.
    """
    lemmatizer = WordNetLemmatizer()

    stopwords = (nltk.corpus.stopwords.words('english')
                 + ['book', 'read'])

    reviews = []
    for review in df['reviews']:
        reviews.extend(remove_proper_nouns(review))
    reviews_lemmatized = [[lemmatizer.lemmatize(t)
                           for t in nltk.word_tokenize(review)]
                          for review in reviews if review]
    clean_reviews = [[r for r in review if r not in stopwords and len(r) > 3] for review in reviews_lemmatized]
    dictionary = gensim.corpora.Dictionary(clean_reviews)
    dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in clean_reviews]
    dictionary.save('trained_models/dictionary.gensim')
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    lda_model.save('trained_models/good_lda_model.gensim')

def detrend_dates(df):
    """
    Build a model to detrend log10(nratings) vs date published, since older books have more ratings.
    """

    numerical_data = df[~df['pub_date'].isna()]
    numerical_data['pub_date'] = pd.to_datetime(numerical_data['pub_date'], errors='coerce').dt.date

    x = numerical_data['pub_date'] - datetime.date(1954, 1, 1)
    y = np.log10(numerical_data['nratings'])
    x = x.dt.days
    x = x.fillna(0)
    mask = (x >= 0) & (x < x.max())
    maxday = x[mask].max()
    x = x.values / x[mask].max()
    numerical_data = numerical_data[mask]

    from numpy.polynomial import Polynomial
    p = Polynomial.fit(x[mask], y[mask], 6)
    xp, yp = p.linspace()
    np.save("trained_models/maxday.npy", maxday)
    np.save("trained_models/daypolyfit.npy", [xp, yp])

def make_score(df):
    """
    Run a classifier and then a regressor to build the elements of the VictorPredictor score.  Returns a bunch
    of things for plotting purposes, used by analyze.py.
    """
    data, mask = make_regression_array(df)
    ymask = (all_data['pubyear'][mask] < 2018) & (all_data['pubyear'][mask] > 2008)
    mask[mask][~ymask.values] = False
    data = data[ymask]
    correlations = []
    for column in data.columns:
        correlations.append(spearmanr(data[column], data['true_score'])[1])
    cols = [c for c, p in zip(data.columns, correlations) if p < 1E-2]
    np.save('trained_models/columns.npy', cols)

    x_train = data.sample(frac=0.8, random_state=100)
    x_test = data.drop(x_train.index)
    data_train = x_train
    data_test = x_test
    x_train = x_train[cols]
    x_test = x_test[cols]
    y_train_continuous = x_train.pop("true_score")
    y_test_continuous = x_test.pop("true_score")
    thresh = 0
    y_train = y_train_continuous > thresh
    y_test = y_test_continuous > thresh
    x_train_true = x_train[y_train == 1]
    x_train_false = x_train[y_train == 0]

    yp_train_list = []
    yp_test_list = []
    model_list = []
    frac = 1.0 * len(x_train_true) / len(x_train_false)
    for i in range(20):
        xt = pd.concat((x_train_false.sample(frac=frac, replace=True), x_train_true))
        yt = y_train[xtr.index]
        gbm = GradientBoostingClassifier(n_estimators=200)
        gbm.fit(xt, yt)
        yp_train_list.append(gbm.predict(x_train))
        yp_test_list.append(gbm.predict(x_test))
        model_list.append(gbm)
        with open(f'trained_models/gbm_ensemble_{i}.p', 'wb') as f:
            pickle.dump(gbm, f)

    mean_train_prediction = np.mean(yp_train_list, axis=0)
    mean_test_prediction = np.mean(yp_test_list, axis=0)
    thresh = 0.95
    np.save('trained_models/thresh.npy', thresh)

    tp = np.sum(mean_test_prediction[y_test == 1] >= thresh)
    fn = np.sum(mean_test_prediction[y_test == 1] < thresh)
    tn = np.sum(mean_test_prediction[y_test == 0] < thresh)
    fp = np.sum(mean_test_prediction[y_test == 0] >= thresh)

    mean_train_prediction = mean_train_prediction >= thresh
    mean_test_prediction = mean_test_prediction >= thresh

    # Now try a regressor to rank things that were coded "winners"
    x_train_regression = x_train[mean_train_prediction > 0]
    y_train_regression = y_train_continuous[mean_train_prediction > 0]
    x_test_regression = x_test[mean_test_prediction > 0]
    y_test_regression = y_test_continuous[mean_test_prediction > 0]

    yp_train_regression_list = []
    yp_test_regression_list = []
    regression_model_list = []
    for i in range(10):
        xtr = x_train_regression.sample(frac=1, replace=True)
        ytr = y_train_regression[xtr.index]
        gbm = GradientBoostingRegressor(n_estimators=50)
        gbm.fit(xtr, ytr)
        yp_train_regression_list.append(gbm.predict(x_train_regression))
        yp_test_regression_list.append(gbm.predict(x_test_regression))
        regression_model_list.append(gbm)
        with open(f'trained_models/gbm_regressor_ensemble_{i}.p', 'wb') as f:
            pickle.dump(gbm, f)

    mean_train_regression_prediction = np.mean(yp_train_regression_list, axis=0)
    mean_test_regression_prediction = np.mean(yp_test_regression_list, axis=0)

    return (tp, tn, fp, fn,
            y_train, y_test, mean_train_prediction, mean_test_prediction,
            y_train_regression, y_test_regression, mean_train_regression_prediction, mean_test_regression_prediction)
