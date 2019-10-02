from flask import render_template
from flask import request
from flask import Markup
from vp import app
import pandas as pd
import psycopg2
import os
import difflib
import numpy as np
from matplotlib import cm
from matplotlib.colors import rgb2hex

username = os.environ['rds_username']
password = os.environ['rds_password']
host = os.environ['rds_host']
dbname = 'goodreads_db'
port = '5432'

con = None
con = psycopg2.connect(database=dbname, user=username, host=host, password=password)

max_nrows = 25

genres = ['Hard SF', "Space SF", 'Dystopia', 'High fantasy', 'Adventure', 'Urban fantasy', 'Literary', 'Mystery/horror', 'Paranormal romance', "Fantasy romance"]
genre_indices = [2,6,3,0,9,7,8,4,1,5]
genredict = [{'index': i, 'name': g, "checked": "checked"} for i, g in zip(genre_indices, genres)]
levels = ['Adult', 'YA', 'Children']
leveldict = [{'index': l, 'name': l, "checked": "checked"} for l in levels]
ignore = ['Ignore author characteristics', 'Ignore reading levels', 'Include short story collections and comics']
ignore_alts = ["Authors who have won or been nominated for an award are more likely to win or be nominated in the future. If you want to find something new, check this box to ignore this effect.", "YA and children's books are less likely to win awards aimed at adult fiction. Check this box to ignore this effect.", "Short story collections and graphic novels are not eligible for best novel awards, but some of them appear in our data set. By default we don't show them, but if you check this box we will--but note that our data set for these formats is not complete!"]
ignoredict = [{'index': ig.split()[1], "name": ig, "checked": "", "alt": alt}
              for ig, alt in zip(ignore, ignore_alts)]

@app.route('/')
@app.route('/index.html')
def index():
    do_genres = request.args.getlist('genres')
    do_levels = request.args.getlist('levels')
    do_ignore = request.args.getlist('ignore')
    where_clauses = []
    if len(do_genres) < len(genres) and len(do_genres)>0:
        genre_retlist = [{'index': gd['index'], 'name': gd['name'], 'checked': 'checked' if f'{gd["index"]}' in do_genres else ''} for gd in genredict]
        do_genres = [f'is_genre{dg} is True' for dg in do_genres if int(dg) in genre_indices]
    else:
        do_genres = []
        genre_retlist = genredict

    if len(do_levels) < len(levels) and len(do_levels)>0:
        level_retlist = [{'index': ld['index'], 'name': ld['name'], 'checked': 'checked' if ld["index"] in do_levels else ''} for ld in leveldict]
        dl = []
        if 'Adult' in do_levels:
            dl += ['(1.0*children/nreviews < 0.1 AND 1.0*ya/nreviews < 0.1)']
        if 'YA' in do_levels:
            dl += ['1.0*ya/nreviews >= 0.1']
        if 'Children' in do_levels:
            dl += ['1.0*children/nreviews >= 0.1']
        do_levels = dl
    else:
        do_levels = []
        level_retlist = leveldict

    scoretype = 'pred_score'
    if len(do_ignore):
        idict = [i for i in ignoredict]
        if 'reading' in do_ignore:
            scoretype += '_readinglevel'
            idict[1] = {'index': idict[1]['index'], 'name': idict[1]['name'],
                        'checked': 'checked', 'alt': idict[1]['alt']}
        if 'author' in do_ignore:
            scoretype += '_author'
            idict[0] = {'index': idict[0]['index'], 'name': idict[0]['name'],
                        'checked': 'checked', 'alt': idict[0]['alt']}
    else:
        idict = ignoredict
        
    where_clause = 'WHERE pubyear=2019'

    if len(do_genres):
        where_clause += ' AND (' + ' OR '.join(do_genres) +')'
    if len(do_levels):
        where_clause += ' AND (' + ' OR '.join(do_levels) +')'
    if 'booktype' not in do_ignore and False:
        where_clause += "AND comics!=0 AND short_stories!=0 AND pub_ctype!='COLLECTION' AND pub_ctype!='OMNIBUS' AND title_storylen!='short story'"
        where_clause += "AND title_storylen!='novelette' AND title_graphic!='Yes'"
        
    
    sql_query = f"SELECT title, all_authors as author, author_lastname, {scoretype} as score FROM works_flask {where_clause}"
    print(sql_query)
    query_results = pd.read_sql_query(sql_query, con)
#    query_results['color'] = [rgb2hex(cm.viridis(s)) for s in query_results['score']]
    query_results['color'] = ["hsl(214, {:.2f}%, 50%)".format(score*100) for score in query_results['score']]
    query_results['link'] = ['']*len(query_results)
    
    
    sortby = request.args.get('sortby', 'score')
    direction = request.args.get('direction', 'desc')
    if sortby=='score':
        if direction == 'desc':
            score_sort_direction = 'asc'
            score_arrow = Markup(" &#9652;")
        else:
            score_sort_direction = 'desc'
            score_arrow = Markup(" &#9662;")
        sort_key = 'score'
    else:
        score_sort_direction = "desc"
        score_arrow=''
    if sortby=='title':
        if direction == 'desc':
            title_sort_direction = 'asc'
            title_arrow = Markup(" &#9652;")
        else:
            title_sort_direction = 'desc'
            title_arrow = Markup(" &#9662;")
        sort_key = 'title'
    else:
        title_sort_direction = 'asc'
        title_arrow=""
    if sortby=='author':
        if direction == 'desc':
            author_sort_direction = 'asc'
            author_arrow = Markup(" &#9652;")
        else:
            author_sort_direction = 'desc'
            author_arrow = Markup(" &#9662;")
        sort_key = 'author_lastname'
    else:
        author_sort_direction = 'asc'
        author_arrow = ""

    sort_dir = True if direction=='asc' else False
    urlkeys = ['{}={}'.format(key, item) for key, item in request.args.items(multi=True) if key!="current_page" and key!="direction" and key!='sortby']
    title_urlkeys = urlkeys + ['sortby=title&direction={}'.format(title_sort_direction)]
    author_urlkeys = urlkeys + ['sortby=author&direction={}'.format(author_sort_direction)]
    score_urlkeys = urlkeys + ['sortby=score&direction={}'.format(score_sort_direction)]
    title_url = 'index.html?' + '&'.join(title_urlkeys)
    author_url = 'index.html?' + '&'.join(author_urlkeys)
    score_url = 'index.html?' + '&'.join(score_urlkeys)

    query_results['author_lastname'] = query_results['author_lastname'].str.lower()
    query_results = query_results.sort_values(by=sort_key, ascending=sort_dir)#.drop(columns='author_lastname')
    nrows = len(query_results)
    curr_page = int(request.args.get('current_page', 1))
    if nrows > max_nrows:
        npages = int(np.ceil(1.0*nrows/max_nrows))
        curr_page = max([1, min([npages, curr_page])])
        query_results = query_results[max_nrows*(curr_page-1):max_nrows*curr_page]
    else:
        npages=1
    urlkeys = ['{}={}'.format(key, item) for key, item in request.args.items(multi=True) if key!="current_page"]
    next_urlkeys = urlkeys + ['current_page={}'.format(curr_page+1)]
    prev_urlkeys = urlkeys + ['current_page={}'.format(curr_page-1)]
    next_url = 'index.html?' + '&'.join(next_urlkeys)
    prev_url = 'index.html?' + '&'.join(prev_urlkeys)

    return render_template('index.html', 
                           books=query_results.to_dict('records'),
                           genres=genre_retlist,
                           levels=level_retlist,
                           ignore=idict,
                           curr_page=curr_page,
                           npages=npages,
                           prev_url=prev_url,
                           next_url=next_url,
                           title_url=title_url, title_arrow=title_arrow,
                           author_url=author_url, author_arrow=author_arrow,
                           score_url=score_url, score_arrow=score_arrow)
