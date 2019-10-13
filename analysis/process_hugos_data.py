"""
process_hugos_data.py: Pull best novel data from a Hugo Awards html file and from a .txt file prepared by hand,
and combine it into one dataset of all publicly available winners, nominees, and longlist nominees.
"""
import re
import glob
from bs4 import BeautifulSoup
import pandas as pd

def parse_html(filename):
    """
    Process an HTML file of Hugo winners and nominees.

    Parameters
    ==========
    filename: str
        The filename to process
    Return
    =======
    A list of dicts, one dict per winner or nominee
    """
    year = re.search(r'(\d\d\d\d)', filename)
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
                if len(text) == 1:
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
    """ Grab all the HTML files listing the winners, and return a list of dicts of all the information
        contained in each file. """
    html_files = glob.glob("hugos_data/*.html")
    res = []
    for file_name in html_files:
        res.extend(parse_html(file_name))
    return res

def nip_percent(s):
    """
    Some of the elements in the text file have percentages that mess up our assumed patterns. This nips that substring
    from the end of the main book string.

    Parameters
    ==========
    s: str
        The line of text from the Hugos text file
    Returns
    =======
    A string with any percent value removed from the end
    """
    res = re.search(r"(.*) \([\d.]+\%\)", s)
    if res:
        return res.group(1)
    res = re.search(r"(.*) [\d.]+\%\s*$", s)
    if res:
        return res.group(1)
    return s

def t1(s):
    """ regexp search for section header """
    res = re.search(r'^(\d\d\d\d) (novel|novella) (nomination|vote)\s*$', s.lower())
    if not res:
        return None
    return {"year": res.group(1), "booktype": res.group(2), "datatype": res.group(3)+'s'}

def t2(s):
    """ regexp search for No Award count """
    res = re.search(r"no award ([\d]+)\s*$", s.lower())
    if not res:
        return None
    return {"Title": "No Award", "Author": "None", "Count": res.group(1)}

def t2a(s):
    """ regexp search for No Award """
    res = re.search("no award", s.lower())
    if not res:
        return None
    return {"Title": "No Award", "Author": "None"}

def t3(s):
    """ Regexp search for multiple numbers """
    res = re.search(r"^([\d]+)\s+(?:\d+\s+)*\d+\s*$", s)
    if not res:
        return None
    return {"Count": res.group(1)}

def t4(s):
    """ regexp search for Count Authorlast, Authorfirst. Title (Publisher) """
    res = re.search(r"^([\d]+) (.*?), (.*)[.] (.*?) \(.*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(4), "Author": "{} {}".format(res.group(3), res.group(2)), "Count": res.group(1)}

def t5(s):
    """ regexp search for Title, by Author - Count """
    res = re.search(r"(.*?), by (.*?) (?:-|–) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2), "Count": res.group(3)}

def t6(s):
    """ regexp search for Count Title by Author (Publisher) """
    res = re.search(r"^([\d]+) (.*?) by (.*?) \(.*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(3), "Count": res.group(1)}

def t7(s):
    """ regexp search for Title by Author Count """
    res = re.search(r"(.*?) by (.*?) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2), "Count": res.group(3)}

def t8(s):
    """ regexp search for Title - Author - Count """
    res = re.search(r"(.*?) (?:-|–) (.*?) (?:-|–) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2), "Count": res.group(3)}

def t9(s):
    """ regexp search for Count Title (Authorlast) Nominationcount """
    res = re.search(r"^([\d]+) (.*?) \((.*?)\) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(3), "NominationCount": res.group(4), "Count": res.group(1)}

def t10(s):
    """ regexp search for Count Title (Author; Publisher) """
    res = re.search(r"^([\d]+)\s+(.*?) \((.*?); .*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(3), "Count": res.group(1)}

def t11(s):
    """ regexp search for Count Title by Author """
    res = re.search(r"^([\d]+)(.*?)by(.*)", s)
    if not res:
        return None
    return {"Title": res.group(2).strip(), "Author": res.group(3).strip(), "Count": res.group(1)}

def t12(s):
    """ regexp search for Count Count2 TitleAuthor """
    res = re.search(r"^(\d+) [\d.]+ (.*)", s)
    if not res:
        return None
    return {"TitleAuthor": res.group(2), "Count": res.group(1)}

def t13(s):
    """ regexp search for Title Count """
    res = re.search(r"(.*?) ([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Count": res.group(2)}

def t14(s):
    """ regexp search for Count """
    res = re.search(r"^([\d]+)\s*$", s)
    if not res:
        return None
    return {"Count": res.group(1)}

def t15(s):
    """ regexp search for (count) Title """
    res = re.search(r"^\(([\d]+)\) (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Count": res.group(1)}

def t16(s):
    """ regexp search for (Count) """
    res = re.search(r"^\(([\d]+)\)\s*$", s)
    if not res:
        return None
    return {"Count": res.group(1)}

def t17(s):
    """ regexp search for Count Title, Author """
    res = re.search(r"^([\d]+)(.*), (.*)", s)
    if not res:
        return None
    return {"Title": res.group(2).strip(), "Author": res.group(3).strip(), "Count": res.group(1)}

def t18(s):
    """ regexp search for Count\tTitle\tAuthor """
    res = re.search(r"^([\d]+)\s*\t(.*)\t(.*)", s)
    if not res:
        return None
    return {"Title": res.group(2).strip(), "Author": res.group(3).strip(), "Count": res.group(1)}

def t19(s):
    """ regexp search for Title\tby\tAuthor\tCount """
    res = re.search(r"(.*)\tby\t(.*?)\t([\d]+)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1).strip(), "Author": res.group(2).strip(), "Count": res.group(3)}

def t20(s):
    """ regexp search for Count Title """
    res = re.search(r"^([\d]+) (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Count": res.group(1)}

def t21(s):
    """ regexp search for Title by Author (Publisher) """
    res = re.search(r"(.*)\s*by\s*(.*?) \(.*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1).strip(), "Author": res.group(2)}

def t22(s):
    """ regexp search for Title by Author """
    res = re.search(r"(.*?) by (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2).strip()}

def t23(s):
    """ regexp search for Title (Author; Publisher) """
    res = re.search(r"(.*?) \((.*?); .*?\)\s*$", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2)}

def t24(s):
    """ regexp search for Authorlast: Title """
    res = re.search(r"^(.*?): (.*?)$", s)
    if not res:
        return None
    return {"Title": res.group(2), "Author": res.group(1)}

def t25(s):
    """ regexp search for Title, Author """
    res = re.search(r"(.*), (.*)", s)
    if not res:
        return None
    return {"Title": res.group(1), "Author": res.group(2)}

def t26(s):
    """ Simply return title, if no other pattern has worked """
    return {"Title": s.strip().title()}

def process_textfile():
    """
    Process the Hugos voting data text file produced from PDFs. Essentially, turn all the strings into dicts of
    year, author, title, and vote or nominee count (not all keys will be populated for any given dict, given the
    structure of the files).
    """
    with open('hugos_data/hugos_voting_data.txt') as f:
        data = f.readlines()
    res = []
    # There are a huge number of patterns that the lines in the text file may follow. This extremely long statement
    # checks them from most complex to least complex, returning the results for the most complex pattern that matches
    # the string. I.e., 25 King, Stephen. Cujo would return {"Count": 25, "Author": "Stephen King", "Title": "Cujo"}
    # and not the string itself; only strings that do not match any pattern will be returned as-is (to handle titles
    # on single lines).  So step through each pattern. If it doesn't match it returns None, so move onto the next
    # pattern. Otherwise, add the new dict onto the `res` list and continue to the next string.
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

def list_get(l, i, default):
    """ Produce a .get method for lists that works like the .get method for dicts.
        We need this because we are, essentially, zipping lists together, but we need the max length not the
        min length of any string, and we're not actually zipping anything because we're making row dicts for a
        DataFrame."""
    if len(l) > i:
        return l[i]
    return default

def main():
    """ Parse all the data in HTML and text files describing the Hugos winners, and produce a giant dataframe
        that contains all of the relevant information."""
    winners = get_winners()
    votes = process_textfile()

    voting_data = {}
    year = None
    booktype = None
    for i in range(1989, 2020):
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
                list_of_dfs.append(df)


    df = pd.concat(list_of_dfs, ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv('hugos_data/hugo_votecounts_raw.csv')

    df = pd.DataFrame(columns=['Author', 'Title', 'Year', 'Winner'])
    for w in winners:
        df = df.append({'Author': w['Author'], 'Title': w['Title'], 'Year': w['Year'], 'Winner': w['Winner']}, ignore_index=True)

    df.to_csv('hugos_data/winners.csv')

if __name__ == '__main__':
    main()
