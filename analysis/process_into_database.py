"""
process_into_database.py: take all the Goodreads pickle files in directory new_data and process them for addition
to the VictorPredictor database.
"""

import glob
import pickle
import os
import re
import difflib
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

def clean_text(s):
    """ Remove newlines and multiple spaces from reviews and replace with just a single space."""
    if not s:
        return s
    return re.sub(r'\s+', ' ', s.replace('\n', ' ')).strip()

def bucketize_shelves(shelves):
    """
    The shelf data contains some items that are strongly correlated but often not used together.  E.g., some people
    use 'favorites', others 'favorite', others 'favourite'. This function removes synonyms to reduce the
    dimensionality of the shelf data.
    """
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
    """
    Generate data for a single book from scraped Goodreads data
    """
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
    # The set comprehension gets rid of duplicate reviews, then the list comprehension turns it back into
    # a list of dicts.
    deduplicated_reviews = [dict(t) for t in {tuple(d.items()) for d in data['reviews']}]

    for review in deduplicated_reviews:
        if review['text']:
            # clean up spacing issues
            clean_review = clean_text(review['text'])
            # The review is in there twice--as a blurb of first few hundred words, then again as the complete review
            # So, pull only the last iteration of duplicated text
            clean_review = clean_review[clean_review.rfind(clean_review[:100]):]
            reviews.append(clean_review)
    # row item 11: top reviews
    row.append(' || '.join(reviews))
    shelf_rows = [(ident, shelf[0], shelf[1]) for shelf in data['shelves']]
    shelf_dict = {s[0]: s[1] for s in data['shelves']}
    row.extend(bucketize_shelves(shelf_dict))
    return row, shelf_rows

def get_hugos_data():
    """ Return already-processed Hugos crossmatch data data """
    return pd.read_csv('hugos_data/titleauthor_withscores.csv')

def get_all_hugos_winners():
    """ Return already-processed Hugos winner data """
    return pd.read_csv('hugos_data/hugo_votecounts_raw_collated_nomatches.txt.csv')

def get_isfdb_data():
    """ Return already-processed isfdb data """
    return pd.read_pickle('isfdb_data/isfdb_withsites.df')

def add_to_database(work_rows, shelf_rows):
    """ Take two lists of tuples describing works (work_rows) and shelves (shelf_rows) and add then to the 'works'
        and 'shelves' tables of database goodreads_db.
    """
    # Set up database connections
    username = os.environ['rds_username']
    password = os.environ['rds_password']
    host = os.environ['rds_host']
    dbname = 'goodreads_db'
    port = '5432'
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname))
    con = psycopg2.connect(database=dbname, user=username, host=host, password=password)
    cur = con.cursor()

    # create the database if it doesn't exist yet
    if not database_exists(engine.url):
        create_database(engine.url)

    # Generate a list of every time an author won or was nominated
    all_hugos_data = get_all_hugos_winners()
    prevdict = {}
    for author in all_hugos_data['Author'].unique():
        prevdict[author] = all_hugos_data[all_hugos_data['Author'] == author]['Year'].values

    # Now, crossmatch with the hugos data -- both winners and authors
    hugos_data = get_hugos_data()
    hid = list(hugos_data['ID'])
    for row in work_rows:
        if int(row[0]) in hid:
            hd = hugos_data.iloc[hid.index(int(row[0]))]
            row += [hd['Winner'], hd['Score']]
        else:
            row += [None, 0.0]
        if row[9]:
            if row[3] in prevdict:
                row += [np.sum(prevdict[row[3]] <= row[9])]
            else:
                match = difflib.get_close_matches(row[3], all_hugos_data['Author'], cutoff=0.9)
                if match:
                    match = match[0]
                    row += [np.sum(prevdict[match] <= row[9])]
                else:
                    row += [0]
        else:
            row += [0]

    # Then crossmatch with the ISFDB. Get both title+author if we can, but just the author info is helpful otherwise.
    isfdb_data = get_isfdb_data()
    for row in work_rows:
        title_mask = row[1] == isfdb_data['pub_title']
        author_mask = row[4] == isfdb_data['author_canonical']
        if any(title_mask & author_mask):
            irow = isfdb_data[title_mask & author_mask]
        else:
            if any(author_mask):
                tisfdb_data = isfdb_data[author_mask]
            else:
                author_mask = difflib.get_close_matches(row[1], isfdb_data['author_canonical'], cutoff=0.9)
                if author_mask:
                    tisfdb_data = isfdb_data[isfdb_data['author_canonical'] == author_mask[0]]
                else:
                    tisfdb_data = None
            if tisfdb_data is not None:
                title_mask = difflib.get_close_matches(row[1], tisfdb_data['pub_title'])
                if title_mask:
                    irow = tisfdb_data[tisfdb_data['pub_title'] == title_mask[0]]
                else:
                    irow = {'author_annualviews': tisfdb_data['author_annualviews'].values[0],
                            'author_lastname': tisfdb_data['author_lastname'].values[0],
                            'pub_id': -1,
                            'pub_ctype': 'NOVEL',
                            'pub_isbn': -1,
                            'author_id': tisfdb_data['author_id'].values[0],
                            'author_canonical': tisfdb_data['author_canonical'].values[0],
                            'title_id': -1,
                            'title_storylen': None,
                            'title_graphic': 'No',
                            'title_annualviews': None,
                            'debut_year': tisfdb_data['debut_year'].values[0],
                            'identifier_value_amazon': None,
                            'identifier_value_goodreads': None,
                            'identifier_value_oclc': None,
                            'identifier_value_bn': None,
                            'identifier_value_audible': None}
                    if row[9] is not None:
                        irow['pub_year'] = datetime.date(int(row[9]), 1, 1)
                    else:
                        irow['pub_year'] = None
            else:
                irow = {'author_annualviews': None,
                        'author_lastname': row[4].split()[-1],
                        'pub_id': -1,
                        'pub_ctype': 'NOVEL',
                        'pub_isbn': -1,
                        'author_id': -1,
                        'author_canonical': row[4],
                        'title_id': -1,
                        'title_storylen': None,
                        'title_graphic': 'No',
                        'title_annualviews': None,
                        'debut_year': None,
                        'identifier_value_amazon': None,
                        'identifier_value_goodreads': None,
                        'identifier_value_oclc': None,
                        'identifier_value_bn': None,
                        'identifier_value_audible': None}
                if row[9] is not None:
                    irow['pub_year'] = datetime.date(int(row[9]), 1, 1)
                else:
                    irow['pub_year'] = None

        # Sometimes these are lists, sometimes single values
        if hasattr(irow['pub_year'], 'len'):
            row += [irow['pub_year'].values[0], irow['author_annualviews'].values[0], irow['author_lastname'].values[0],
                    irow['pub_id'].values[0], irow['pub_ctype'].values[0], irow['pub_isbn'].values[0],
                    irow['author_id'].values[0], irow['author_canonical'].values[0], irow['title_id'].values[0],
                    irow['title_storylen'].values[0], irow['title_graphic'].values[0],
                    irow['title_annualviews'].values[0], irow['debut_year'].values[0],
                    irow['identifier_value_amazon'].values[0], irow['identifier_value_goodreads'].values[0],
                    irow['identifier_value_oclc'].values[0], irow['identifier_value_bn'].values[0],
                    irow['identifier_value_audible'].values[0]]
        else:
            row += [irow['pub_year'], irow['author_annualviews'], irow['author_lastname'],
                    irow['pub_id'], irow['pub_ctype'], irow['pub_isbn'], irow['author_id'],
                    irow['author_canonical'], irow['title_id'], irow['title_storylen'],
                    irow['title_graphic'], irow['title_annualviews'], irow['debut_year'],
                    irow['identifier_value_amazon'], irow['identifier_value_goodreads'],
                    irow['identifier_value_oclc'], irow['identifier_value_bn'], irow['identifier_value_audible']]

    main_df = pd.DataFrame(work_rows,
                           columns=['id', 'title', 'is_series', 'author', 'all_authors',
                                    'rating', 'nratings', 'nreviews', 'blurb', 'pubyear', 'language', 'reviews',
                                    'to_read', 'favs', 'dnf', 'comics', 'short_stories', 'fantasy',
                                    'ya', 'sf', 'horror', 'dystopian', 'romance', 'adventure',
                                    'urban_fantasy', 'children', 'mystery', 'historical',
                                    'high_fantasy', 'mythology', 'humor', 'literature', 'time_travel',
                                    'space', 'lgbt', 'winner', 'true_score', 'nprev', 'pub_date',
                                    'author_annualviews', 'author_lastname', 'pub_id', 'pub_ctype', 'pub_isbn',
                                    'author_id', 'author_canonical', 'title_id', 'title_storylen',
                                    'title_graphic', 'title_annualviews', 'debut_year', 'identifier_value_amazon',
                                    'identifier_value_goodreads', 'identifier_value_oclc',
                                    'identifier_value_bn', 'identifier_value_audible']).set_index('id')
    pubyear = main_df['pub_date']
    def pyr(py):
        try:
            return py.year
        except:
            return 3000

    pubyear = [pyr(pb) for pb in pubyear]
    main_df['pubyear'] = np.min([main_df['pubyear'], pubyear], axis=0)
    shelf_df = pd.DataFrame(shelf_rows, columns=['work_id', 'shelf', 'nshelves'])

    try:
        old_data = list(pd.read_sql_query("SELECT id FROM works", engine)['id'])
        old_data = [o for o in old_data if o in main_df['id']]
        cur.execute_batch("DELETE FROM works WHERE id=%s", old_data)
        cur.execute_batch("DELETE FROM shelves WHERE id=%s", old_data)
        con.commit()
    except pd.io.sql.DatabaseError:
        # This error just means the table doesn't exist, typically.
        pass


    main_df.to_sql('works', engine, if_exists='append')
    shelf_df.to_sql('shelves', engine, if_exists='append')
    con.close()
    engine.close()

def main():
    """ Process each file in the new_data directory, then add those books to the initial database. """
    main_rows = []
    shelf_rows = []
    files = glob.glob('new_data/bookdata*.p')
    for file_name in files:
        row, shelf_row = single_review(file_name)
        if row:
            main_rows.append(row)
        shelf_rows.extend(shelf_row)

    add_to_database(main_rows, shelf_rows)

def make_csv():
    """ Generate a .csv file of all titles and authors for hand-matching troublesome Hugos winners. """
    files = glob.glob('new_data/bookdata*.p')
    titles = []
    authors = []
    ids = []
    for file_name in files:
        row, _ = single_review(file_name)
        if row:
            titles.append(row[1])
            ids.append(row[0])
            authors.append(row[4])
    df = pd.DataFrame({"Author": authors, 'Title': titles, 'ID': ids})
    df.to_csv('hugos_data/titleauthor.csv')

if __name__ == '__main__':
    import sys
    if '--csv' in sys.argv:
        make_csv()
    else:
        main()
