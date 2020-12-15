import pandas as pd
import json
import numpy as np
import random
from collections import defaultdict


def get_second(x):
    if pd.isnull(x):
        return ""
    elif len(x.split('/')) > 2:
        return x.split('/')[2]
    else:
        return ""


def get_third(x):
    if pd.isnull(x):
        return ""
    elif len(x.split('/')) > 3:
        return x.split('/')[3]
    elif x.isdigit():
        return x
    else:
        return ""


def define_dictionary_words(docs, modals, start=1):
    print("define_dictionary_words")
    return {}


def define_dictionary_n_grams(docs, start=1):
    print("define_dictionary_n_grams")

    import topmine_src.phrase_mining as phrase_mining
    import sys
    import topmine_src.utils as utils

    file_name = "vw_remont-i-stroitel_stvo_only_text"
    output_path = "remont_n-grams"

    # represents the minimum number of occurences you want each phrase to have.
    min_support = 10

    # represents the threshold for merging two words into a phrase. A lower value
    # alpha leads to higher recall and lower precision,
    alpha = 4

    # length of the maximum phrase size
    max_phrase_size = 10

    phrase_miner = phrase_mining.PhraseMining(file_name, min_support, max_phrase_size, alpha);
    partitioned_docs, index_vocab = phrase_miner.mine()
    frequent_phrases = phrase_miner.get_frequent_phrases(min_support)
    utils.store_partitioned_docs(partitioned_docs)
    utils.store_vocab(index_vocab)
    utils.store_frequent_phrases(frequent_phrases, output_path)
    return {}


def define_dicts(docs, modals):
    print("define_dicts")
    dictionary_words = define_dictionary_words(docs, modals, start=1)
    dictionary_n_grams = define_dictionary_n_grams(docs, start=len(dictionary_words) + 1)
    return dictionary_words, dictionary_n_grams


def calc_n_dw(docs, dictionary_words, dictionary_n_grams, modals):
    print("calc_n_dw")
    return 1


def make_docword_file(n_dw, name):
    print("make_docword_file")
    pass


def make_vocab_file(dictionary_words, dictionary_n_grams, name):
    print("make_vocab_file")
    pass


def make_bow_uci_files(docs, name="bow_uci", modals=["text"]):
    # посмотреть https: // gist.github.com / persiyanov / e58d37bfd0894612593fa36930bd56fb и не изобретать велосипед
    dictionary_words, dictionary_n_grams = define_dicts(docs, modals)
    n_dw = calc_n_dw(docs, dictionary_words, dictionary_n_grams, modals)
    make_docword_file(n_dw, name=name)
    make_vocab_file(dictionary_words, dictionary_n_grams, name=name)


def read_and_prepared(name, path=""):
    data = pd.read_table(path + name)

    docs = pd.DataFrame()
    docs = data[(pd.notnull(data['name']) | pd.notnull(data['description'])) & (data.status == 3)]
    docs['text'] = docs['name'].map(lambda x: x if pd.notnull(x) else "") + " " + docs['description'].map(
        lambda x: x if pd.notnull(x) else "")

    docs['first'] = docs['parent_id'].map(lambda x: x.split('/')[1] if pd.notnull(x) else None)
    docs['first'][pd.isnull(docs['first'])] = docs['category_id'][pd.isnull(docs['first'])].map(
        lambda x: x.split('/')[1] if pd.notnull(x) else "")

    docs['second'] = docs['parent_id'].map(get_second)
    docs['second'][docs['second'] == ""] = docs['category_id'][docs['second'] == ""].map(get_second)

    docs['third'] = docs['parent_id'].map(get_third)
    docs['third'][docs['third'] == ""] = docs['category_id'][docs['third'] == ""].map(get_third)


    # профильтруем по первой категории и возьмем только remont-i-stroitel_stvo
    docs = docs[docs['first'] == "remont-i-stroitel_stvo"]

    docs.to_csv("docs_csv" + name + ".csv")

    return docs

def read_and_prepared_marking(name):
    data = pd.read_table(name)

    docs = pd.DataFrame()
    docs = data[(pd.notnull(data['name']) | pd.notnull(data['description'])) & (data.status == 3)]
    docs['text'] = docs['name'].map(lambda x: x if pd.notnull(x) else "") + " " + docs['description'].map(
        lambda x: x if pd.notnull(x) else "")

    docs['first'] = docs['parent_id'].map(lambda x: x.split('/')[1] if pd.notnull(x) else None)
    docs['first'][pd.isnull(docs['first'])] = docs['category_id'][pd.isnull(docs['first'])].map(
        lambda x: x.split('/')[1] if pd.notnull(x) else "")

    docs['second'] = docs['parent_id'].map(get_second)
    docs['second'][docs['second'] == ""] = docs['category_id'][docs['second'] == ""].map(get_second)

    docs['third'] = docs['parent_id'].map(get_third)
    docs['third'][docs['third'] == ""] = docs['category_id'][docs['third'] == ""].map(get_third)


    # профильтруем по первой категории и возьмем только remont-i-stroitel_stvo
    docs = docs[docs['first'] == "remont-i-stroitel_stvo"]

    docs.to_csv("docs_csv" + name + ".csv")

    return docs

def form_test_train_set(docs=None, name="", test_size=0.1, format_out="vw"):
    rich_docs = docs[docs['third'] != ""]

    random.seed(23, version=2)
    rich_docs['rand'] = rich_docs['card_id'].map(lambda x: random.uniform(0, 1))


    if format_out == "vw":
        f = open('vw_train_rich' + name, 'w')
        for index, row in rich_docs.iterrows():
            if row['rand'] >= test_size:
                string = ""
                string += str(row['card_id']).replace(":", "")
                # if len(str(row['text']).replace(":", "")) > 0:
                string += " |@text " + str(row['text']).replace(":", "")
                # if len(str(row['first']).replace(":", "")) > 0:
                string += " |@first " + str(row['first']).replace(":", "")
                # if len(str(row['second']).replace(":", "")) > 0:
                string += " |@second " + str(row['second']).replace(":", "")
                # if len(str(row['third']).replace(":", "")) > 0:
                string += " |@third " + str(row['third']).replace(":", "")
                # string += " |worker " + str(row['worker_id'])
                # if string != str(row['card_id']).replace(":", ""):
                f.write(string + '\n')
        f.close()

        f = open('vw_test_rich' + name, 'w')
        for index, row in rich_docs.iterrows():
            if row['rand'] < test_size:
                string = ""
                string += str(row['card_id']).replace(":", "")
                # if len(str(row['text']).replace(":", "")) > 0:
                string += " |@text " + str(row['text']).replace(":", "")
                # if len(str(row['first']).replace(":", "")) > 0:
                string += " |@first " + str(row['first']).replace(":", "")
                # if len(str(row['second']).replace(":", "")) > 0:
                string += " |@second " + str(row['second']).replace(":", "")
                # if len(str(row['third']).replace(":", "")) > 0:
                #string += " |@third " + ""
                # string += " |worker " + str(row['worker_id'])
                # if string != str(row['card_id']).replace(":", ""):
                f.write(string + '\n')
        f.close()

        return 'vw_train_rich' + name, 'vw_test_rich' + name

    elif format_out == "bow_uci":
        train_docs = rich_docs[rich_docs['rand'] >= test_size]
        test_docs = rich_docs[rich_docs['rand'] < test_size]
        make_bow_uci_files(train_docs, name="bow_uci_train_rich", modals=["text", "first", "second", "third"])
        make_bow_uci_files(test_docs, name="bow_uci_test_rich", modals=["text", "first", "second"])
        return "bow_uci_train_rich", "bow_uci_test_rich"


def form_castom_set(docs=None, name=""):
    poor_docs = docs[docs['category_id'] == "#"]
    f = open('vw_castom' + name, 'w')
    for index, row in poor_docs.iterrows():
        string = ""
        string += str(row['card_id']).replace(":", "")
        # if len(str(row['text']).replace(":", "")) > 0:
        string += " |@text " + str(row['text']).replace(":", "")
        # if len(str(row['first']).replace(":", "")) > 0:
        string += " |@first " + str(row['first']).replace(":", "")
        # if len(str(row['second']).replace(":", "")) > 0:
        string += " |@second " + str(row['second']).replace(":", "")
        # if len(str(row['third']).replace(":", "")) > 0:
        #string += " |@third " + str(row['third']).replace(":", "")
        # string += " |worker " + str(row['worker_id'])
        # if string != str(row['card_id']).replace(":", ""):
        f.write(string + '\n')
    f.close()
    return 'vw_castom' + name