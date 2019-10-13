"""
pipeline.py: run all training routines (if desired) and then predict scores.
"""
import os
import pandas as pd
from sqlalchemy import create_engine
from train import make_genres, make_topics, detrend_dates, make_score
from predict import predict_genres, predict_topics, predict_score
from analyze import initial_exploration, plot_nominees, analyze_genres, confusion_matrix, analyze_topics

username = os.environ['rds_username']
password = os.environ['rds_password']
host = os.environ['rds_host']
dbname = 'goodreads_db'
port = '5432'


def main(train=False, analyze=False):
    """ Run training (if train=True), predict scores, and then run analysis routines (if analyze=True) """
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname))
    df = pd.read_sql_query("SELECT * FROM works", engine)

    if train:
        make_genres(df)
    df = predict_genres(df)
    if train:
        make_topics(df)
    df = predict_topics(df)
    if train:
        detrend_dates(df)
        (tp, tn, fp, fn,
         y_train, y_test, yp_train_class, yp_test_class,
         y_train_regress, y_test_regress, yp_train_regress, yp_test_regress) = make_score(df)
    df = predict_score(df)

    df.to_sql('works_flask', engine, if_exists='replace')

    if analyze:
        initial_exploration(df)
        plot_nominees()
        analyze_genres(df)
        confusion_matrix(tp, tn, fp, fn)
        analyze_topics()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--train', action="store_true", help="Perform training (default false)")
    p.add_argument('-a', '--analyze', action='store_true', help='Analyze data, ie make plots (default false)')
    args = p.parse_args()
    main(train=args.train, analyze=args.analyze)
