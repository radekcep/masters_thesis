from sklearn.datasets import fetch_20newsgroups
from collections import namedtuple


MLData = namedtuple("MLData", "data a_class")


english_allowed_characters = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', ',', ';',
    '.', '!', '?', ':', '’', '’', '/', '|', '_', '%', '@', '#', '$',
    '%', 'ˆ', '&', '*', '‘', '+', '-', '=', '<', '>', '(', ')', '[',
    ']', '{', '}', '\\', '\''
]


def one_shot_of(char, allowed_characters):
    one_shot = [0] * len(allowed_characters)
    try:
        index_in_allowed_characters = allowed_characters.index(char)
        one_shot[index_in_allowed_characters] = 1
    except ValueError:
        pass
    return one_shot


def array_of_one_shots_from(text, allowed_characters):
    res = []
    for char in text:
        char = char.lower()
        res.append(one_shot_of(char, allowed_characters))
    return res


def newsgroups_data(subset, data_width, allowed_characters):
    newsgroups = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    res = []

    for article_index, article in enumerate(newsgroups.data):
        if len(article) > data_width:
            continue
        for index_of_subset_of_article in range(0, int(len(article) / data_width)):
            first_char_index  = index_of_subset_of_article * data_width
            last_char_index   = index_of_subset_of_article * data_width + data_width
            subset_of_article = article[first_char_index:last_char_index]
            one_shots         = array_of_one_shots_from(subset_of_article, allowed_characters)

            res.append(MLData(data=one_shots, a_class=newsgroups.target[article_index]))
    return res


# data = newsgroups_data(subset='train', data_width=256, allowed_characters=english_allowed_characters)
# print(data[0].a_class)
#
# import numpy as np
# np.set_printoptions(threshold=np.nan, linewidth=200)
# print(np.matrix(data[0].data))
