#  Idan Achituve    300083029
import numpy as np

RARE_THRESHOLD = 3


# read data and return 2 lists, a header (classes) list and body words list
def read_data(file):
    articles = []
    articles_headers = []
    with open(file) as f:
        content = f.readlines()  # read file
        at_header = True
        for i in range(len(content)):
            line = content[i]  # get current line
            if line == "\n":
                at_header = not at_header
            else:
                if at_header:
                    article_header = line.replace('>', '\t').split('\t')[2:-1]  # take only the classes from the header line
                    articles_headers.append(article_header)
                else:
                    article_body = line.split()
                    articles.append(article_body)

    return articles, articles_headers


# remove rare words from the set
def remove_rare_words(articles):
    word_counts = get_word_counts(articles)  # counts per word
    articles_filtered = []
    for article in articles:
        article_words = []
        for word in article:
            if word_counts[word] > RARE_THRESHOLD:  # get word only if number of occurrences > 3
                article_words.append(word)
        articles_filtered.append(article_words)

    return articles_filtered


# create a dictionary with the articles words and their counts
def get_word_counts(articles):
    events_counts = {}
    for article in articles:
        for word in article:
            events_counts[word] = events_counts[word] + 1 if word in events_counts else 1

    return events_counts


# get word counts in each article
def word_count_per_article(articles):
    counts_in_article = []
    for article in articles:
        events_counts = {}
        for word in article:
            events_counts[word] = events_counts[word] + 1 if word in events_counts else 1
        counts_in_article.append(events_counts)

    return counts_in_article


def words_per_doc_vec(articles):

    all_words = list(get_word_counts(articles).keys())  # get all words
    V = len(all_words)  # vocabulary size
    N = len(articles)  # number of documents
    n = np.zeros((N, V))  # each cell is n_tk

    for d in range(N):
        for word in articles[d]:
            word_idx = all_words.index(word)  # get word index
            n[d, word_idx] += 1  # increment cell value by 1

    return n