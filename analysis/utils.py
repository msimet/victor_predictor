import numpy as np
import datetime
from scipy.interpolate import interp1d

def make_regression_array(data, keep_short_stories=False):
    # Get rid of non-numeric columns
    data = data.drop(columns=['id', 'title', 'author', 'all_authors', 'pubyear', 'winner', 'blurb', 'publisher_id',
                              'author_lastname', 'series_num'])

    print("Languages:", data['language'].unique())
    # Get rid of ineligible books & save the mask
    data['comics'] /= data['nratings']
    data['short_stories'] /= data['nratings']

    if keep_short_stories:
        mask = (data['language']=='English')
    else:
        mask = (data['comics']<0.0005) & (data['short_stories']<0.0005) & (data['language']=='English')

    data = data[mask]
    data = data.drop(columns='language')

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
        data['to_read'] = 1.*data['nreviews']/data['nratings']
        data['favs'] = 1.*data['nreviews']/data['nratings']
        data['dnf'] = 1.*data['nreviews']/data['nratings']
        data['ya'] = 1.*data['nreviews']/data['nratings']
        data['children'] = 1.*data['nreviews']/data['nratings']

    if True:
        maxdate = np.load('maxday.npy')
        # Trend out the year dependence
        data['pub_date'] = data['pub_date']-datetime.date(1954,1,1)
        data['pub_date'] = data['pub_date'].dt.days.values
        data['pub_date'] /= maxdate
        data.loc[data['pub_date']>1,'pub_date'] = 0
        data.loc[data['pub_date']<0,'pub_date'] = 0

        x, y = np.load('daypolyfit.npy')
        interpolator = interp1d(x, y)

        data['nratings'] /= 10**interpolator(data['pub_date'])
    else:
        data = data.drop(columns='pub_date')

    data['nratings'] = np.log10(data['nratings'])
    return data, mask
