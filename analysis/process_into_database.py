import glob
import pickle
import os
import re
import difflib
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database, drop_database
import psycopg2

def clean_text(s):
    if not s:
        return s
    return re.sub('\s+', ' ', s.replace('\n', ' ')).strip()

def bucketize_shelves(shelves):
    to_read = shelves.get('to-read', 0) + shelves.get('currently-reading', 0)
    favorite_tags = ["favorites", "favourites", "all-time-favorites", "favorite-books", "favorite",
                     "faves", "my-favorites", "favorite-series", "favs", "favourite", "recommended"]
    favs = sum([shelves.get(t, 0) for t in favorite_tags])
    fantasy_tags = ["fantasy", "magic"]
    fantasy = sum([shelves.get(t, 0) for t in fantasy_tags])
    ya_tags = ["young-adult", "ya", "teen", "ya-fiction", "young-adult-fiction", "ya-books", "ya-lit", "coming-of-age",
               "teen-fiction", "middle-grade"]
    ya = sum([shelves.get(t, 0) for t in ya_tags])
    sf_tags = ["science-fiction", "sci-fi", "scifi", "sf"]
    sf = sum([shelves.get(t, 0) for t in sf_tags])
    horror = shelves.get('horror', 0)
    dystopian_tags = ["dystopian", "dystopian-fiction", "post-apocalyptic", "apocalyptic", "cyberpunk", "futuristic"]
    dystopian = sum([shelves.get(t, 0) for t in dystopian_tags])
    romance_tags = ["romance", "paranormal-romance", "new-adult", "love-triangle", "contemporary"]
    romance = sum([shelves.get(t, 0) for t in romance_tags])
    adventure_tags = ["adventure", "action", "action-adventure"]
    adventure = sum([shelves.get(t, 0) for t in adventure_tags])
    urban_fantasy_tags = ["paranormal", "supernatural", 'urban_fantasy',
                          "vampires", "vampire", "werewolves", "witches", "zombies", "steampunk"]
    urban_fantasy = sum([shelves.get(t, 0) for t in urban_fantasy_tags])
    children_tags = ["childrens", "children", "children-s", "childhood", "children-s-books", "kids", "childrens-books",
                     "children-s-literature", "childhood-favorites", "juvenile", "kids-books", "youth"]
    children = sum([shelves.get(t, 0) for t in children_tags])
    mystery_tags = ["mystery", "thriller", "suspense", "crime", "mystery-thriller"]
    mystery = sum([shelves.get(t, 0) for t in mystery_tags])
    historical_tags = ["historical-fiction", "historical"]
    historical = sum([shelves.get(t, 0) for t in historical_tags])
    high_fantasy_tags = ["high-fantasy", "epic-fantasy", "epic"]
    high_fantasy = sum([shelves.get(t, 0) for t in high_fantasy_tags])
    dnf_tags = ["dnf", "abandoned", "did-not-finish", "unfinished"]
    dnf = sum([shelves.get(t, 0) for t in dnf_tags])
    mythology_tags = ["mythology", "retellings", "fairy-tales", "fae", "retelling", "fairy-tale", "greek-mythology"]
    mythology = sum([shelves.get(t, 0) for t in mythology_tags])
    humor_tags = ["humor", "humour", "comedy", "funny", "satire"]
    humor = sum([shelves.get(t, 0) for t in humor_tags])
    literature_tags = ["literature", "classic-literature", "literary-fiction"]
    literature = sum([shelves.get(t, 0) for t in literature_tags])
    comics_tags = ["graphic-novels", "comics", "graphic-novel"]
    comics = sum([shelves.get(t, 0) for t in comics_tags])
    time_travel = shelves.get('time-travel', 0)
    short_stories = shelves.get('short-stories', 0)
    space_tags = ["space-opera", "space", "aliens"]
    space = sum([shelves.get(t, 0) for t in space_tags])
    lgbt_tags = ["lgbt", "lgbtq", "feminism"]
    lgbt = sum([shelves.get(t, 0) for t in lgbt_tags])

    return [to_read, favs, dnf, comics, short_stories, fantasy, ya, sf, horror, dystopian, romance,
            adventure, urban_fantasy, children, mystery, historical, high_fantasy, mythology, humor,
            literature, time_travel, space, lgbt]

def single_review(file_name):
    # Get just the file name, strip the .p and the bookdata_ for the id
    ident = os.path.splitext(os.path.split(file_name)[-1])[0][9:]
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print("Failed for file {} with {}".format(file_name, e))
        return None, []
    # Row item 0: id
    row = [ident]
    # Row item 1: title
    row.append(data['title'])
    # row item 2: series y/n (no checking for index within series right now)
    if data['series']:
        row.append(1)
    else:
        row.append(0)
    # row item 3: first authors
    row.append(data['authors'][0])
    #row item 4: all authors
    row.append(' & '.join(data['authors']))
    # row item 5: rating
    row.append(data['rating'])
    # row item 6: nratings
    row.append(data['nratings'])
    # row item 7: nreviews
    row.append(data['nreviews'])
    # row item 8: blurb
    row.append(clean_text(data['blurb']))
    # row item 9: publication year
    row.append(data['pub_year'])
    reviews = []
    # row item 10: language
    if 'language' in data:
        row.append(data['language'])
    else:
        row.append('English')
    # first, get rid of duplicate reviews
    for review in data['reviews']:
        del review['shelves']
    deduplicated_reviews = [dict(t) for t in {tuple(d.items()) for d in data['reviews']}]

    for review in deduplicated_reviews:
        if review['text']:
            # clean up spacing issues
            clean_review = clean_text(review['text'])
            # The review is in there twice--as a blurb of first few hundred words, then again as the complete review
            # So, pull only the last of the text
            clean_review = clean_review[clean_review.rfind(clean_review[:100]):]
            reviews.append(clean_review)
    # row item 11: top reviews
    row.append(' || '.join(reviews))
    shelf_rows = [(ident, shelf[0], shelf[1]) for shelf in data['shelves']]
    shelf_dict = {s[0]: s[1] for s in data['shelves']}
    row.extend(bucketize_shelves(shelf_dict))
    return row, shelf_rows

def get_hugos_data():
    return pd.read_csv('hugos_data/titleauthor_withscores.csv')

def get_all_hugos_winners():
    return pd.read_csv('hugos_data/hugo_votecounts_raw_collated_nomatches.txt.csv')

def get_isfdb_data():
    return pd.read_pickle('isfdb_hugos_eligible.df')

def make_database(work_rows, shelf_rows):

    with open('pg_credentials.p', 'rb') as f:
        credentials = pickle.load(f)
    # Set your postgres username/password, and connection specifics
    username = credentials['username']
    password = credentials['password']
    host = 'localhost'
    port = '5432'  # default port that postgres listens on
    db_name = 'goodreads_db'

    ## 'engine' is a connection to a database
    ## Here, we're using postgres, but sqlalchemy can connect to other things too.
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name))

    ## drop the database if we already made it
    drop_database(engine.url)
    if not database_exists(engine.url):
        create_database(engine.url)

    all_hugos_data = get_all_hugos_winners()
    prevdict = {}
    for author in all_hugos_data['Author'].unique():
        prevdict[author] = all_hugos_data[all_hugos_data['Author']==author]['Year'].values

    hugos_data = get_hugos_data()
    hid = list(hugos_data['ID'])
    for row in work_rows:
        if int(row[0]) in hid:
            print("found it")
            hd = hugos_data.iloc[hid.index(int(row[0]))]
            row += [hd['Winner'], hd['Score']]
        else:
            row += [None, 0.0]
        if row[9]:
            if row[3] in prevdict:
                row += [np.sum(prevdict[row[3]]<=row[9])]
            else:
                match = difflib.get_close_matches(row[3], all_hugos_data['Author'], cutoff=0.9)
                if match:
                    match = match[0]
                    row += [np.sum(prevdict[match]<=row[9])]
                else:
                    row += [0]
        else:
            row += [0]

    isfdb_data = get_isfdb_data()
    for row in work_rows:
        title_mask = row[1]==isfdb_data['pub_title']
        author_mask = row[4]==isfdb_data['author_canonical']
        if any(title_mask & author_mask):
            irow = isfdb_data[title_mask & author_mask]
        else:
            if any(author_mask):
                tisfdb_data = isfdb_data[author_mask]
            else:
                author_mask = difflib.get_close_matches(row[1], isfdb_data['author_canonical'], cutoff=0.9)
                if author_mask:
                    tisfdb_data = isfdb_data[isfdb_data['author_canonical']==author_mask[0]]
                else:
                    tisfdb_data = None
            if tisfdb_data is not None:
                title_mask = difflib.get_close_matches(row[1], tisfdb_data['pub_title'])
                if title_mask:
                    irow = tisfdb_data[tisfdb_data['pub_title'] == title_mask[0]]
                else:
                    if row[9] is not None:
                        irow = {'pub_year': datetime.date(int(row[9]), 1, 1), 'publisher_id': -1, 'pub_series_num': 0,
                                'author_annualviews': tisfdb_data['author_annualviews'].values[0],
                                'author_lastname': tisfdb_data['author_lastname'].values[0]}
                    else:
                        irow = {'pub_year': None, 'publisher_id': -1, 'pub_series_num': 0,
                                'author_annualviews': tisfdb_data['author_annualviews'].values[0],
                                'author_lastname': tisfdb_data['author_lastname'].values[0]}

            else:
                if row[9] is not None:
                    irow = {'pub_year': datetime.date(int(row[9]), 1, 1), 'publisher_id': -1, 'pub_series_num': 0, 'author_annualviews': None, 'author_lastname': row[4].split()[-1]}
                else:
                    irow = {'pub_year': None, 'publisher_id': -1, 'pub_series_num': 0, 'author_annualviews': None, 'author_lastname': row[4].split()[-1]}

        try:
            row += [irow['pub_year'].values[0], irow['publisher_id'].values[0], irow['pub_series_num'].values[0],
                    irow['author_annualviews'].values[0], irow['author_lastname'].values[0]]
        except AttributeError:
            row += [irow['pub_year'], irow['publisher_id'], irow['pub_series_num'],
                    irow['author_annualviews'], irow['author_lastname']]

    main_df = pd.DataFrame(work_rows,
                           columns=['id', 'title', 'is_series', 'author', 'all_authors',
                                    'rating', 'nratings', 'nreviews', 'blurb', 'pubyear', 'language', 'reviews',
                                    'to_read', 'favs', 'dnf', 'comics', 'short_stories', 'fantasy',
                                    'ya', 'sf', 'horror', 'dystopian', 'romance', 'adventure',
                                    'urban_fantasy', 'children', 'mystery', 'historical',
                                    'high_fantasy', 'mythology', 'humor', 'literature', 'time_travel',
                                    'space', 'lgbt', 'winner', 'true_score', 'nprev', 'pub_date',
                                    'publisher_id', 'series_num', 'author_annualviews', 'author_lastname']).set_index('id')
    pubyear = main_df['pub_date']
    def pyr(y):
        try:
            return py.year
        except:
            return 3000

    pubyear = [pyr(pb) for pb in pubyear]
    main_df['pubyear'] = np.min([main_df['pubyear'], pubyear], axis=0)
    shelf_df = pd.DataFrame(shelf_rows, columns=['work_id', 'shelf', 'nshelves'])

    main_df.to_sql('works', engine, if_exists='replace')
    print (len(engine.execute("SELECT * FROM works").fetchall()))
    shelf_df.to_sql('shelves', engine, index=True, if_exists='replace')

def main():
    main_rows = []
    shelf_rows = []
    if not os.path.exists('processed.p'):
        files = glob.glob('good_data/bookdata*.p')
        for file_name in files:
            row, shelf_row = single_review(file_name)
            if row:
                main_rows.append(row)
            shelf_rows.extend(shelf_row)

        with open('processed.p', 'wb') as f:
            pickle.dump([main_rows, shelf_rows], f)
    else:
        with open('processed.p', 'rb') as f:
            main_rows, shelf_rows = pickle.load(f)

    make_database(main_rows, shelf_rows)

def make_csv():
    main_rows = []
    files = glob.glob('good_data/bookdata*.p')
    titles = []
    authors = []
    ids = []
    for file_name in files:
        row, shelf_row = single_review(file_name)
        if row:
            titles.append(row[1])
            ids.append(row[0])
            authors.append(row[4])
    df = pd.DataFrame({"Author": authors, 'Title': titles, 'ID': ids})
    df.to_csv('hugos_data/titleauthor.csv')

if __name__=='__main__':
    main()
#    make_csv()