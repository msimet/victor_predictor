"""
utils.py: some basic data processing for VictorPredictor training and prediction.
"""

import numpy as np
import pandas as pd
import datetime
from scipy.interpolate import interp1d

def make_regression_array(data, keep_short_stories=False):
    """
    Take the outputs of a SQL database call, drop non-numeric columns, do some feature engineering, drop bad rows,
    and return the data and a mask that will map the new data array back into the original dataframe.

    Parameters
    ==========
    data: pd.DataFrame
        Data pulled from a specific SQL table
    keep_short_stories: bool
        For training data, set this to False (the default). To predict scores for short stories (for display only),
        set to True.

    Returns
    =======
    arr: a pd.DataFrame containing numerical features, engineered features, and good rows only
    mask: a pd.Series of bools such that data[mask].index == arr.index.
    """
    # Get rid of non-numeric columns
    data = data.drop(columns=['id', 'title', 'author', 'all_authors', 'pubyear', 'winner', 'blurb',
                              'author_lastname', 'pub_id', 'pub_isbn', 'author_id', 'author_canonical', 'title_id',
                              'identifier_value_amazon', 'identifier_value_goodreads', 'identifier_value_oclc',
                              'identifier_value_bn', 'identifier_value_audible'])

    if keep_short_stories:
        mask = ((data['language'] == 'English') | (data['language'] == ''))
    else:
        mask = ((data['comics'] == 0) & (data['short_stories'] == 0)
                & (data['pub_ctype'] != 'COLLECTION') & (data['pub_ctype'] != 'OMNIBUS')
                & (data['title_storylen'] != 'short story') & (data['title_storylen'] != 'novelette')
                & (data['title_graphic'] != 'Yes')
                & ((data['language'] == 'English') | (data['language'] == '')))

    data = data[mask]
    data = data.drop(columns=['language', 'pub_ctype', 'title_storylen', 'short_stories', 'title_graphic', 'comics'])

    data['pub_date'] = pd.to_datetime(data['pub_date'], errors='coerce')
    year_mask = (data['debut_year'].isna()) | (data['pub_date'] >= datetime.date(2030, 1, 1)) | (data['pub_date'].isna())
    data['debut_year'][year_mask] = np.nan
    data['debut_year'][~year_mask] = (
        (pd.to_datetime(data['debut_year'][~year_mask])-data['pub_date'][~year_mask]).dt.days//365)
    print(data['debut_year'][~year_mask])
    data = data.rename(columns={'debut_year': 'years_since_debut'})

    # Get rid of NaNs
    for column in data:
        if data[column].isna().any():
            if column == 'pub_date':
                data[column] = data[column].fillna(datetime.date(1900, 1, 1))
            else:
                data[column] = data[column].fillna(np.median(data[column][~(data[column].isna())]))

    # idk where these came from
    if 'level_0' in data:
        data = data.drop(columns='level_0')
    if 'index' in data:
        data = data.drop(columns='index')

    if True:
        # relative strength of shelves
        data['nreviews'] = 1.*data['nreviews']/data['nratings']
        data['to_read'] = 1.*data['to_read']/data['nratings']
        data['favs'] = 1.*data['favs']/data['nratings']
        data['dnf'] = 1.*data['dnf']/data['nratings']
        data['ya'] = 1.*data['ya']/data['nratings']
        data['children'] = 1.*data['children']/data['nratings']

    if True:
        maxdate = np.load('maxday.npy')
        # Trend out the year dependence
        data['pub_date'] = pd.to_datetime(data['pub_date'], errors='coerce').dt.date-datetime.date(1954, 1, 1)
        data['pub_date'] = data['pub_date'].dt.days.values
        data['pub_date'] /= maxdate
        data.loc[data['pub_date'] > 1, 'pub_date'] = 0
        data.loc[data['pub_date'] < 0, 'pub_date'] = 0

        x, y = np.load('daypolyfit.npy')
        interpolator = interp1d(x, y)

        data['nratings'] /= 10**interpolator(data['pub_date'])
    else:
        data = data.drop(columns='pub_date')

    data['nratings'] = np.log10(data['nratings'])
    return data, mask
