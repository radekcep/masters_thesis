# -*- coding: utf-8 -*-

import os
import base64
import json
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

def category_id_for(category):
    # names_category_table = {
    #     'comp.graphics': 0,
    #     'comp.os.ms-windows.misc': 0,
    #     'comp.sys.ibm.pc.hardware': 0,
    #     'comp.sys.mac.hardware': 0,
    #     'comp.windows.x': 0,
    #     'rec.autos': 1,
    #     'rec.motorcycles': 1,
    #     'rec.sport.baseball': 1,
    #     'rec.sport.hockey': 1,
    #     'sci.crypt': 2,
    #     'sci.electronics': 2,
    #     'sci.med': 2,
    #     'sci.space': 2,
    #     'misc.forsale': 3,
    #     'talk.politics.misc': 4,
    #     'talk.politics.guns': 4,
    #     'talk.politics.mideast': 4,
    #     'talk.religion.misc': 5,
    #     'alt.atheism': 5,
    #     'soc.religion.christian': 5,
    # }

    names_category_table = {
        'comp.graphics': 0,
        'comp.os.ms-windows.misc': 1,
        'comp.sys.ibm.pc.hardware': 2,
        'comp.sys.mac.hardware': 3,
        'comp.windows.x': 4,
        'rec.autos': 5,
        'rec.motorcycles': 6,
        'rec.sport.baseball': 7,
        'rec.sport.hockey': 8,
        'sci.crypt': 9,
        'sci.electronics': 10,
        'sci.med': 11,
        'sci.space': 12,
        'misc.forsale': 13,
        'talk.politics.misc': 14,
        'talk.politics.guns': 15,
        'talk.politics.mideast': 16,
        'talk.religion.misc': 17,
        'alt.atheism': 18,
        'soc.religion.christian': 19,
    }
    return names_category_table[category]

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
        one_shot = one_shot_of(char, allowed_characters)
        res.append(one_shot)
    return res


def newsgroups_data(subset, data_width, allowed_characters):
    newsgroups = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    res = []

    print('Processing dataset...')
    for article_index, article in enumerate(newsgroups.data):
        if len(article) < data_width:
            continue
        for index_of_subset_of_article in range(0, int(len(article) / data_width)):
            first_char_index  = index_of_subset_of_article * data_width
            last_char_index   = index_of_subset_of_article * data_width + data_width
            subset_of_article = article[first_char_index:last_char_index]
            one_shots         = array_of_one_shots_from(subset_of_article, allowed_characters)

            category_name = newsgroups.target_names[newsgroups.target[article_index]]
            res.append(MLData(data=one_shots, a_class=category_id_for(category_name)))

    return res
