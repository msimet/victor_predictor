"""
analyze.py: produce some diagnostic plots from VictorPredictor analysis
"""
import pickle
import seaborn as sns
import numpy as np
import gensim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as skl_confusion_matrix
from sklearn.utils.multiclass import unique_labels

sns.set_style('whitegrid')

small_size = 18
medium_size = 20
bigger_size = 22

plt.rc('font', size=small_size)          # controls default text sizes
plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=small_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

def initial_exploration(df):
    """ Plot some basic quantities from the works dataframe. """

    sns.pairplot(df[["rating", "nratings", "nreviews", "to_read", "favs", "dnf"]])
    plt.savefig("eda/pairplot.png")
    plt.clf()

    sns.kdeplot(np.log10(np.clip(df['nratings'], 1, None)), df['rating'], shade=True,
                shade_lowest=False)
    plt.xlabel("log10(nratings)")
    plt.savefig("eda/ratings_dist.png")
    plt.clf()

    mask = np.log10(df['nratings']) > 3
    sns.regplot(np.log10(df['nratings'][mask]), df['rating'][mask])
    plt.xlabel("log(nratings)")
    plt.savefig("eda/ratings_scaling.png")
    plt.clf()

    mask = df['pubyear'] > 2010
    x = df['pubyear'][mask]
    y = np.log10(np.clip(df['nratings'][mask], 1, None))
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(x, y, 4)
    sns.boxplot(df['pubyear'][mask].astype(int), np.log10(df['nratings'][mask]))
    xp, yp = p.linspace()
    maskp = xp > 2010.5
    xp -= 2011
    plt.plot(xp[maskp], yp[maskp], color='black', lw=3)
    plt.xlim((plt.gca().get_xlim()[0], plt.gca().get_xlim()[1]-1))
    plt.ylabel("log10(nratings)")
    plt.savefig("eda/nratings_year_scaling.png")
    plt.clf()

def plot_nominees():
    """ Plot the number of nominations and final ballots """
    years = [2019, 2018, 2017, 2016, 2015, 2014]
    nnoms = [1800, 1534, 2078, 3695, 2122, 1923]
    nvotes = [3097, 2828, 3319, 3130, 5950, 3587]
    sns.lineplot(years, nnoms, label='Nominations', lw=3)
    sns.lineplot(years, nvotes, label='Votes', lw=3)
    plt.xlabel("Year")
    plt.savefig("eda/nominees.png")

def analyze_genres(df):
    """
    Produce some text and plots related to genre assignment
    """
    prettystring = df.apply(lambda x: '{} by {}'.format(x.title, x.author), axis=1)
    genre_data = df[[c for c in df.columns if c[:5] == 'genre']]
    labels = np.argmin(genre_data, axis=0)

    with open('eda/genre_booklist.txt', 'w') as f:
        for label in range(len(genre_data.columns)):
            f.write(f"Group {label}\n")
            mask = labels == label
            for row in prettystring[mask]:
                f.write("{}\n".format(row))

    with open('trained_models/genre_classifier.p', 'rb') as f:
        model = pickle.load(f)
    centroids = model.cluster_centers_
    genres = np.array(genre_data.columns)
    avg_centroid = np.mean(centroids, axis=0)
    with open('eda/genre_defs.txt', 'w') as f:
        for i, centroid in enumerate(centroids):
            f.write(f"Group {i}\n")
            print(f"Group {i}")
            cdiff = centroid - avg_centroid
            order = np.argsort(cdiff)[::-1]
            for j, (g, c) in enumerate(zip(genres[order], cdiff[order])):
                if j > 4:
                    continue
                f.write(f"{g}: {100 * c:.3f}\n")

    with sns.color_palette("husl", 5):
        for i, centroid in enumerate(centroids):
            cdiff = centroid - avg_centroid
            fig = plt.figure()
            order = np.argsort(cdiff)[::-1]
            tick_labels = genres[order][:5]
            for j, (g, c) in enumerate(zip(genres[order], cdiff[order])):
                if j > 4:
                    continue
                plt.barh(5-j, c)
            sns.despine(left=True, bottom=True)
            plt.yticks([5, 4, 3, 2, 1], tick_labels)
            fig.gca().get_xaxis().set_visible(False)
            plt.savefig("eda/genre_graph_{}.png".format(i))
            plt.clf()

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, learning_rate=300, perplexity=30, early_exaggeration=12, init='random',
                random_state=2019)
    X_tsne = tsne.fit_transform(genre_data)


    fig = plt.figure()
    figsize = fig.get_size_inches()
    plt.close(fig)
    with sns.color_palette("husl", len(genres)):
        fig = plt.figure(figsize=[2 * figsize[0], 2 * figsize[1]])
        for i in range(len(genres)):
            mask = labels == i
            sns.scatterplot(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.3, label=str(i))
        plt.yticks([], [])
        plt.xticks([], [])  # plt.xlim((-0.05, 0.2))
        plt.legend()
        plt.savefig("eda/genre_tsne.png")
        plt.clf()

def confusion_matrix(tp, tn, fp, fn):
    """ Make a confusion matrix for the outputs of the nominee classification."""
    # This code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        # Compute confusion matrix
        cm = skl_confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        cm = cm[:, ::-1]

        _, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes[::-1], yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
                 rotation_mode="anchor")
        ax.tick_params(axis='y', which='major', pad=15)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] < thresh else "black")
        ax.set_aspect('equal')
        ax.set_xlim((-0.5, 1.5))
        ax.set_ylim((-0.5, 1.5))
        return ax

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    y_true = [0]*(tn+fp) + [1]*(fn+tp)
    y_pred = [0]*(tn+fn) + [1]*(tp+fp)
    plot_confusion_matrix(y_true, y_pred, classes=np.array(['Not nominated', 'Nominated']), normalize=True,
                          cmap='inferno')

    plt.savefig('eda/confusion_matrix.png')

def analyze_topics():
    """
    Print out the most important words for each topic.
    """
    lda_model = gensim.models.LdaMulticore.load('good_lda_model.gensim')
    with open('eda/topic_words.txt', 'w') as f:
        for idx, topic in lda_model.print_topics(-1):
            f.write('Topic: {} \nWords: {}\n'.format(idx, topic))
