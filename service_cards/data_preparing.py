import pandas as pd
import json
import numpy as np
import random


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

def read_and_prepared(name):
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

def form_test_train_set(docs=None, name="", test_size=0.1):
    rich_docs = docs[docs['third'] != ""]

    random.seed(23, version=2)
    rich_docs['rand'] = rich_docs['card_id'].map(lambda x: random.uniform(0, 1))

    f = open('vw_train_rich' + name, 'w')
    for index, row in rich_docs.iterrows():
        if row['rand'] >= test_size:
            string = ""
            string += str(row['card_id']).replace(":", "")
            # if len(str(row['text']).replace(":", "")) > 0:
            string += " |text " + str(row['text']).replace(":", "")
            # if len(str(row['first']).replace(":", "")) > 0:
            string += " |first " + str(row['first']).replace(":", "")
            # if len(str(row['second']).replace(":", "")) > 0:
            string += " |second " + str(row['second']).replace(":", "")
            # if len(str(row['third']).replace(":", "")) > 0:
            string += " |third " + str(row['third']).replace(":", "")
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
            string += " |text " + str(row['text']).replace(":", "")
            # if len(str(row['first']).replace(":", "")) > 0:
            string += " |first " + str(row['first']).replace(":", "")
            # if len(str(row['second']).replace(":", "")) > 0:
            string += " |second " + str(row['second']).replace(":", "")
            # if len(str(row['third']).replace(":", "")) > 0:
            ###string += " |third " + str(row['third']).replace(":", "")
            # string += " |worker " + str(row['worker_id'])
            # if string != str(row['card_id']).replace(":", ""):
            f.write(string + '\n')
    f.close()

    return 'vw_train_rich' + name, 'vw_test_rich' + name

def form_castom_set(docs=None, name="", test_size=0.1):
    pass