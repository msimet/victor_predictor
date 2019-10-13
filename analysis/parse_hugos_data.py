""" Pull best novel data from a Hugo Awards html file."""

import pickle
from bs4 import BeautifulSoup

def main(filename):
    """
    Run through an HTML file from the Hugos website and pull the winner and nominee data, then save it to a pickle file.

    Parameters
    ==========
    filename: str
        The filename to process
    """
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
                    results.append((author, title, 'winner'))
                else:
                    results.append((author, title, 'nominee'))

    with open('hugos_data/hugo_awards_{}.p'.format(filename.split('/')[-1][:4]), 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
