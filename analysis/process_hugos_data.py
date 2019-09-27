import pickle
import re
import glob
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import difflib

""" Pull best novel data from a Hugo Awards html file."""

def parse_html(filename):
    year = re.search('(\d\d\d\d)', filename)
    year = year.group(1)
    results = []
    with open(filename) as f:
        soup = BeautifulSoup(f.read(), features="html.parser")
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.text.strip()
        if text == "Best Novel" or text[:11] == "Best Novel " or text[:12] == "Best Novella": # Next item is the best novel ul
            ul = p.find_next('ul')
            nominees = ul.find_all('li') # And here are the nominees themselves
            for nominee in nominees:
                text = nominee.text.split('by')
                if text[0] == 'No Award':
                    continue
                if len(text)==1:
                    text = text[0].split(',')
                title = text[0].strip()
                if title[-1] == ',': # Some say "Ender's Game, by Orson Scott Card"
                    title = title[:-1]
                author = text[1].strip()
                if author[-1] == ')':
                    # Here, keep any parenthesis section except the last which is the publisher
                    author = '('.join(author.split('(')[:-1])
                if author[-1] == ']':
                    author = '['.join(author.split('[')[:-1])
                if 'class' in nominee.attrs and 'winner' in nominee['class']:
                    results.append({'Year': year, 'Author': author, 'Title': title, "Winner": "W"})
                else:
                    results.append({'Year': year, 'Author': author, 'Title': title, "Winner": "N"})
    return results

def get_winners():
    html_files = glob.glob("hugos_data/*.html")
    res = []
    for file_name in html_files:
        res.extend(parse_html(file_name))
    return res

""" Code to parse the Hugos PDF data I copied and pasted into a text file """

def nip_percent(s):
    res = re.search("(.*) \([\d.]+\%\)", s)
    if res:
        return res.group(1)
    res = re.search("(.*) [\d.]+\%\s*$", s)
    if res:
        return res.group(1)
    return s

# checked
def t1(s):
    # Section header
    res = re.search('^(\d\d\d\d) (novel|novella) (nomination|vote)\s*$', s.lower())
    if not res:
        return None
    return {"year": res.group(1), "booktype": res.group(2), "datatype": res.group(3)+'s'}

#checked
def t2(s):
    # No Award count
    res = re.search("no award ([\d]+)\s*$", s.lower())
    if not res:
        return None
    return {"Title": "No Award", "Author": "None", "Count": res.group(1)}

def t2a(s):
    res = re.search("no award", s.lower())
    if not res:
        return None
    return {"Title": "No Award", "Author": "None"}

# checked
def t3(s):
    # Multiple numbers
    res = re.search("^([\d]+)\s+(?:\d+\s+)*\d+\s*$", s)
    if not res:
        return None
    return {"Count": res.group(1)}

#checked
def t4(s):
    # Count Authorlast, Authorfirst. Title (Publisher)
    res = re.search("^([\d]+) (.*?), (.*)[.] (.*?) \(.*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(4), "Author": "{} {}".format(res.group(3), res.group(2)), "Count": res.group(1)}

#checked
def t5(s):
    # Title, by Author - Count
    res = re.search("(.*?), by (.*?) (?:-|–) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2), "Count": res.group(3)}

# checked
def t6(s):
    # Count Title by Author (Publisher)
    res = re.search("^([\d]+) (.*?) by (.*?) \(.*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(3), "Count": res.group(1)}

#checked
def t7(s):
    # Title by Author Count
    res = re.search("(.*?) by (.*?) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2), "Count": res.group(3)}

#checked
def t8(s):
    # Title - Author - Count
    res = re.search("(.*?) (?:-|–) (.*?) (?:-|–) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2), "Count": res.group(3)}

#checked
def t9(s):
    # Count Title (Authorlast) Nominationcount
    res = re.search("^([\d]+) (.*?) \((.*?)\) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(3), "NominationCount": res.group(4), "Count": res.group(1)}

# Checked
def t10(s):
    # Count Title (Author; Publisher)
    res = re.search("^([\d]+)\s+(.*?) \((.*?); .*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(3), "Count": res.group(1)}

# checked
def t11(s):
    # Count Title by Author
    res = re.search("^([\d]+)(.*?)by(.*)", s)
    if not res:
        return None
    return {"Title": res.group(2).strip(), "Author": res.group(3).strip(), "Count": res.group(1)}

#checked
def t12(s):
    # Count Count2 TitleAuthor
    res = re.search("^(\d+) [\d.]+ (.*)", s)
    if not res:
        return None
    return {"TitleAuthor": res.group(2), "Count": res.group(1)}

# checked
def t13(s):
    # Title Count
    res = re.search("(.*?) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Count": res.group(2)}

# checked
def t14(s):
    # Count
    res = re.search("^([\d]+)\s*$", s)
    if not res:
        return None
    return {"Count": res.group(1)}

#checked
def t15(s):
    # (count) Title
    res = re.search("^\(([\d]+)\) (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Count": res.group(1)}

#checked
def t16(s):
    # (Count)
    res = re.search("^\(([\d]+)\)\s*$", s)
    if not res:
        return None
    return {"Count": res.group(1)}

# checked
def t17(s):
    # Count Title, Author
    res = re.search("^([\d]+)(.*), (.*)", s)
    if not res:
        return None
    return {"Title": res.group(2).strip(), "Author": res.group(3).strip(), "Count": res.group(1)}

# checked
def t18(s):
    # Count\tTitle\tAuthor
    res = re.search("^([\d]+)\s*\t(.*)\t(.*)", s)
    if not res:
        return None
    return {"Title": res.group(2).strip(), "Author": res.group(3).strip(), "Count": res.group(1)}

# checked
def t19(s):
    # Title\tby\tAuthor\tCount
    res = re.search("(.*)\tby\t(.*?)\t([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1).strip(), "Author": res.group(2).strip(), "Count": res.group(3)}

#checked
def t20(s):
    # Count Title
    res = re.search("^([\d]+) (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Count": res.group(1)}

# checked
def t21(s):
    # Title by Author (Publisher)
    res = re.search("(.*)\s*by\s*(.*?) \(.*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1).strip(), "Author": res.group(2)}

#checked
def t22(s):
    # Title by Author
    res = re.search("(.*?) by (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2).strip()}

#checked
def t23(s):
    # Title (Author; Publisher)
    res = re.search("(.*?) \((.*?); .*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2)}

#checked
def t24(s):
    # Authorlast: Title
    res = re.search("^(.*?): (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(1)}

#checked
def t25(s):
    # Title, Author
    res = re.search("(.*), (.*)", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2)}

#checked
def t26(s):
    # Title
    return {"Title": s.strip().title()}

def process_textfile():
    with open('hugos_data/hugos_voting_data.txt') as f:
        data = f.readlines()
    res = []
    for s in data:
        s = nip_percent(s)
        for _ in range(1):
            r = t1(s)
            if r:
                res.append(r)
                break
            r = t2(s)
            if r:
                res.append(r)
                break
            r = t2a(s)
            if r:
                res.append(r)
                break
            r = t3(s)
            if r:
                res.append(r)
                break
            r = t4(s)
            if r:
                res.append(r)
                break
            r = t5(s)
            if r:
                res.append(r)
                break
            r = t6(s)
            if r:
                res.append(r)
                break
            r = t7(s)
            if r:
                res.append(r)
                break
            r = t8(s)
            if r:
                res.append(r)
                break
            r = t9(s)
            if r:
                res.append(r)
                break
            r = t10(s)
            if r:
                res.append(r)
                break
            r = t11(s)
            if r:
                res.append(r)
                break
            r = t12(s)
            if r:
                res.append(r)
                break
            r = t13(s)
            if r:
                res.append(r)
                break
            r = t14(s)
            if r:
                res.append(r)
                break
            r = t15(s)
            if r:
                res.append(r)
                break
            r = t16(s)
            if r:
                res.append(r)
                break
            r = t17(s)
            if r:
                res.append(r)
                break
            r = t18(s)
            if r:
                res.append(r)
                break
            r = t19(s)
            if r:
                res.append(r)
                break
            r = t20(s)
            if r:
                res.append(r)
                break
            r = t21(s)
            if r:
                res.append(r)
                break
            r = t22(s)
            if r:
                res.append(r)
                break
            r = t23(s)
            if r:
                res.append(r)
                break
            r = t24(s)
            if r:
                res.append(r)
                break
            r = t25(s)
            if r:
                res.append(r)
                break
            res.append(t26(s))
    return res

def list_get(list, i, default):
    if len(list)>i:
        return list[i]
    return default

def main():
    winners = get_winners()
    votes = process_textfile()

    voting_data = {}
    year = None
    booktype = None
    for i in range(1989,2020):
        voting_data[str(i)] = {}
        for datatype in ['votes', 'nominations']:
            voting_data[str(i)][datatype] = {}
            for booktype in ['novel', 'novella']:
                voting_data[str(i)][datatype][booktype] = {'Author': [], 'Title': [], 'TitleAuthor': [], 'NominationCount': [], 'VoteCount': [], 'BookType': []}
    for vote in votes:
        if 'booktype' in vote:
            df = voting_data[vote['year']][vote['datatype']][vote['booktype']]
            datatype = vote['datatype']
        else:
            for key in vote:
                val = vote[key]
                if key == 'Count':
                    if datatype[:4] == 'vote':
                        key = 'VoteCount'
                    else:
                        key = 'NominationCount'
                df[key].append(val)
    for year in voting_data:
        for dtype in voting_data[year]:
            for btype in voting_data[year][dtype]:
                vd = voting_data[year][dtype][btype]
                if vd['Title']:
                    vd['BookType'] = [btype]*len(vd['Title'])
                elif vd['TitleAuthor']:
                    vd['BookType'] = [btype] * len(vd['TitleAuthor'])
    list_of_dfs = []
    for year in range(1989, 2020):
        year = str(year)
        df = pd.DataFrame(columns=['Year', 'Author', 'Title', 'NominationCount', 'VoteCount', 'Winner', "BookType"])
        for dtype in voting_data[year]:
            for btype in voting_data[year][dtype]:
                this_vd = voting_data[year][dtype][btype]
                if 'TitleAuthor' in this_vd and this_vd['TitleAuthor']:
                    this_vd['Title'] = this_vd['TitleAuthor']
                    del this_vd['TitleAuthor']
                this_vd["Year"] = [year] * len(this_vd['Title'])
                for key in list(this_vd.keys()):
                    if not this_vd[key]:
                        del this_vd[key]
                if not this_vd:
                    continue
                for i in range(len(this_vd['Title'])):
                    df = df.append({key: list_get(this_vd[key], i, None) for key in this_vd}, ignore_index=True)
                if False:
                    scores = np.zeros(len(df))
                    for booktype in ['novel', 'novella']:
                        bmask = df['BookType'] == booktype
                        winner_mask = (df['Winner']=='W')
                        nominee_mask = ((df['Winner']=='N') | winner_mask)
                        df = df.astype({'VoteCount': float, 'NominationCount': float, 'Score': float, 'Year': int})
                        if np.any(~(df['VoteCount']).isna()):
                            try:
                                win_votecount = df['VoteCount'][winner_mask & bmask].values[0]
                            except IndexError:
                                if year == 2015:
                                    win_votecount = 3495 # Noah Ward!
                            scores[nominee_mask & bmask] = 1.0*df['VoteCount'][nominee_mask & bmask].values/win_votecount
                            if np.any(~np.isnan(df['NominationCount'])):
                                nominee_minscore = scores[nominee_mask & bmask].min()
                                nominee_nomcount = df['NominationCount'][nominee_mask & bmask].values.min()
                                scores[bmask & ~nominee_mask] = nominee_minscore*df['NominationCount'][bmask & ~nominee_mask]/nominee_nomcount
                            if year == 2015:
                                print("Pause here")
                        else:
                            scores[nominee_mask & bmask] = 0.7
                            scores[winner_mask & bmask] = 1
                    df['Score'] = scores
                list_of_dfs.append(df)


    df = pd.concat(list_of_dfs, ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv('hugos_data/hugo_votecounts_raw.csv')
#    print(df)

    df = pd.DataFrame(columns=['Author', 'Title', 'Year', 'Winner'])
    for w in winners:
        df = df.append({'Author': w['Author'], 'Title': w['Title'], 'Year': w['Year'], 'Winner': w['Winner']}, ignore_index=True)

    df.to_csv('hugos_data/winners.csv')

if __name__=='__main__':
    main()