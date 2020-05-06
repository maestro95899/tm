import importlib
from IPython.display import clear_output
import random

import artm
import random

import data_preparing
import artm_model

importlib.reload(data_preparing)
importlib.reload(artm_model)

class exp_info():
    def __init__():
        self.model = None
        self.theta_train = None
        self.theta_test = None
        self.theta_train = None
        self.theta_castom = None
        self.docs = None
        self.topic_number = {}
        self.card2rubric = {}
        self.card2name = {}
        self.card2text = {}
        self.card2topic = {}
        self.card2topic_castom = {}
        self.card2topic_train = {}
        self.card2topic_test = {}
        self.topic2rubric = {}
    def extract_values(self, old_info):
        pass


def make_docs_dicts(inf):
    inf['card2rubric'] = {}
    for index, row in inf['docs'].iterrows():
        inf['card2rubric'][row['card_id']] = str(row['first']) + '/' + str(row['second']) + '/' + str(row['third'])
    inf['card2name'] = {}
    for index, row in inf['docs'].iterrows():
        inf['card2name'][row['card_id']] = row['name']
    inf['card2text'] = {}
    for index, row in inf['docs'].iterrows():
        inf['card2text'][row['card_id']] = row['text']

def get_theta_from_vw(inf, path):
    batch_vectorizer = None
    batch_vectorizer = artm.BatchVectorizer(data_path='./' + path,
                                                  data_format='vowpal_wabbit',
                                                  target_folder='folder' + path)
    return inf['model'].transform(batch_vectorizer=batch_vectorizer)

def form_card2topic(inf,):
    inf['card2topic_castom'] = {}
    for card_id in inf['theta_castom'].columns:
        inf['card2topic_castom'][card_id] = inf['theta_castom'][card_id].argmax()

    inf['card2topic_train'] = {}
    for card_id in inf['theta_train'].columns:
        inf['card2topic_train'][card_id] = inf['theta_train'][card_id].argmax()

    inf['card2topic_test'] = {}
    for card_id in inf['theta_test'].columns:
        inf['card2topic_test'][card_id] = inf['theta_test'][card_id].argmax()

    inf['card2topic'] = dict(list(inf['card2topic_test'].items()) +
                        list(inf['card2topic_train'].items()) +
                        list(inf['card2topic_castom'].items()))

def show_top_topic_for_documents(inf, docs_list):
    hist = {}
    for doc in docs_list:
        topic = inf['card2topic'][doc]
        if topic in hist:
            hist[topic] += 1
        else:
            hist[topic] = 1
    return hist

def show_docs_by_topic(inf, n):
    docs_by_topic_n = [card_id for card_id, value in inf['card2topic'].items() if value==n]
    for i in range(20):
        if i < len(docs_by_topic_n) - 1:
            row = inf['docs'][inf['docs']['card_id'] == docs_by_topic_n[i]]
            print(str(row['first'].values), str(row['second'].values), str(row['third'].values), str(row['text'].values))

def find_category(inf, category, available_card_ids):
    category_docs_list = inf['docs'][(inf['docs']['card_id'].map(lambda x: x in available_card_ids)) & (inf['docs']['third']==category)]['card_id']
    return show_top_topic_for_documents(inf, category_docs_list)

# для каждой темы определим, какая рубрика в ней преобладает
# для этого посчитаем для каждого документа, какая тема имеет наибольшее влияние на него,
# будем говорить, что этот документ имеет такую тему, вспомним, что у каждого документа есть рубрика.
# для каждой темы найдем моду по рубрикам среди документов с такой темой
def rubric_for_topic(inf, topic_num):
    cards_with_rubric = {}
    for key, value in inf['card2topic_train'].items():
        if value == topic_num:
            if inf['card2rubric'][key] in cards_with_rubric:
                cards_with_rubric[inf['card2rubric'][key]] += 1
            else:
                cards_with_rubric[inf['card2rubric'][key]] = 1
    if len(cards_with_rubric) > 0:
        max_val = max(cards_with_rubric.values())
        for key, value in cards_with_rubric.items():
            if value == max_val:
                return key
    else:
        return None

def get_topic2rubric(inf):
    inf['topic2rubric'] = {}
    for num in range(inf['topic_number']):
        inf['topic2rubric'][num] = rubric_for_topic(inf, num)

def measure_accuracy_on_test(inf):
    # accuracy точность на тестовой выборке
    n = 0
    pos = 0
    for key, value in inf['card2topic_test'].items():
        n += 1

        if inf['topic2rubric'][value] == inf['card2rubric'][key]:
            pos += 1
        else:
            if n % 1000 == 0:
                pass
                #print("prediction: \t", inf['topic2rubric'][value], "!= \n", "val \t\t", inf['card2rubric'][key], "\n on ",
                #      inf['card2text'][key], "\n")
    print(pos, " / ", n, pos / n)

def accuracy_on_test(inf, prediction, debug_mode=False):
    # accuracy точность на тестовой выборке
    n = 0
    pos = 0
    for key, value in inf['card2topic_test'].items():
        n += 1
        if prediction[key] == inf['card2rubric'][key]:
            pos += 1
        else:
            if debug_mode:
                if n % 1000 == 0:
                    print("prediction: \t", prediction[key],
                            "!= \n", "val \t\t", inf['card2rubric'][key], "\n on ",
                            inf['card2text'][key], "\n")
    print(pos, " / ", n, pos / n)
    return pos / n

def get_answers_on_castom(inf):
    # посмотрим как классифицировались кастомные услуги
    f = open("разметка_кастомных_услуг.csv", 'w')
    n = 0
    pos = 0
    for key, value in inf['card2topic_castom'].items():
        n += 1

        if inf['topic2rubric'][value] == inf['card2rubric'][key]:
            pos += 1
        else:
            if n % 1000 == 0:
                print(n / 1000, " prediction: \t", inf['topic2rubric'][value], "\t настоящий: \t ", inf['card2rubric'][key],
                      "\n name: ", inf['card2name'][key], "\n", inf['card2text'][key], "\n")
                f.write(str(inf['topic2rubric'][value]) + "\t" + str(inf['card2name'][key]) + "\t" + str(inf['card2text'][key]) + '\n')
    f.close()
    print(pos, " / ", n, pos / n)

def get_answers_on_castom_advanced(inf, hand_marking=True, count=10, cards_list=None):
    # разметим классификацию кастомных услуг
    f = open("разметка_кастомных_услуг.csv", 'w')
    if not cards_list:
        random.seed(23)
        cards_list = random.sample(list(inf['card2topic_castom'].keys()), count)

    pos = 0
    n = 0
    for card in cards_list:
        n += 1
        print(n, " \nprediction: \t", inf['topic2rubric'][inf['card2topic_castom'][card]],
              "\n настоящий: \t ", inf['card2rubric'][card],
              "\n name: ", inf['card2name'][card], "\n", inf['card2text'][card], "\n")
        verdict = input()
        if verdict == '1':
            pos += 1

        f.write(
            str(inf['topic2rubric'][inf['card2topic_castom'][card]]) + "\t" +
            str(inf['card2name'][card]) + "\t" +
            str(inf['card2text'][card]) + '\t' +
            verdict + '\n'
        )
    f.close()
    print(pos, " / ", count, pos / count)

def do_marking_on_castom(inf, pred_card2rubric, count=10):
    f = open("разметка_кастомных_услуг.csv", 'w')
    random.seed(23)
    cards_list = random.sample(list(pred_card2rubric.keys()), count)

    pos = 0
    n = 0
    for card in cards_list:
        n += 1
        print(n, " \nprediction: \t", pred_card2rubric[card],
              "\n настоящий: \t ", inf['card2rubric'][card],
              "\n name: ", inf['card2name'][card], "\n", inf['card2text'][card], "\n")
        verdict = input()
        if verdict == '1':
            pos += 1

        f.write(
            str(pred_card2rubric[card]) + "\t" +
            str(inf['card2name'][card]) + "\t" +
            str(inf['card2text'][card]) + '\t' +
            verdict + '\n'
        )
    f.close()
    print(pos, " / ", count, pos / count)

def get_p_cd_third(inf, path):
    batch_vectorizer = None
    batch_vectorizer = artm.BatchVectorizer(data_path='./' + path,
                                            data_format='vowpal_wabbit',
                                            target_folder='folder' + path)
    return inf['model'].transform(batch_vectorizer=batch_vectorizer, predict_class_id='@third')

def get_prediction_p_cd(inf, p_cd):
    card2rubric = {}
    rubric_list = p_cd.T.columns
    for card_id in p_cd.columns:
        pre_rubric = inf['card2rubric'][card_id]
        card2rubric[card_id] = pre_rubric[0: pre_rubric.rfind("/")] + "/" + rubric_list[p_cd[card_id].argmax()]
    return card2rubric

def make_prediction_by_p_cd(inf, mode='castom'): #castom/train/test
    inf['p_cd_' + mode] = get_p_cd_third(inf, inf['path_' + mode])
    inf['p_cd_card2rubric_' + mode] = get_prediction_p_cd(inf, inf['p_cd_' + mode])

def make_prediction_by_topic2rubric(inf, mode='test'): #castom/train/test
    card2rubric = {}
    for card_id, topic in inf['card2topic_' + mode].items():
        card2rubric[card_id] = inf['topic2rubric'][topic]
    return card2rubric

def compare_quality(inf):
    prediction = make_prediction_by_topic2rubric(inf, mode='test')
    acc = accuracy_on_test(inf, prediction)
    print('prediction_by_topic2rubric:', acc)

    make_prediction_by_p_cd(inf, mode='test')
    acc = accuracy_on_test(inf, inf['p_cd_card2rubric_test'], debug_mode=False)
    print('prediction_by_p_cd:', acc)

def exp_weigth(exp_name='weight_1_0_1_1', weigths=[1., 0., 1., 1.], ):
    inf = {}
    inf['exp_name'] = exp_name
    inf['test_size'] = 0.1
    inf['num_collection_passes'] = 15
    inf['topic_number'] = 750
    inf['data_path'] = "service_cards_tokenised_remont_only"

    inf['docs'] = data_preparing.read_and_prepared(inf['data_path'])
    inf['path_train'], inf['path_test'] = data_preparing.form_test_train_set(docs=inf['docs'], name=inf['exp_name'], test_size=inf['test_size'])
    inf['path_castom'] = data_preparing.form_castom_set(inf['docs'], name=inf['exp_name'])
    inf['model'], inf['theta_train'] = artm_model.create_and_learn_PLSA_class_ids_weigth(
                                                                inf['path_train'],
                                                                topic_number=inf['topic_number'],
                                                                num_collection_passes=inf['num_collection_passes'],
                                                                weigths=weigths
                                                             )

    inf['theta_test'] = get_theta_from_vw(inf, inf['path_test'])
    inf['theta_castom'] = get_theta_from_vw(inf, inf['path_castom'])
    clear_output(True)

    make_docs_dicts(inf)
    form_card2topic(inf)
    get_topic2rubric(inf)
    #wwm.measure_accuracy_on_test(inf)
    make_prediction_by_p_cd(inf, mode='castom')
    inf['method_topic2rubric_card2rubric_castom'] = make_prediction_by_topic2rubric(inf, mode='castom')
    make_prediction_by_p_cd(inf, mode='test')
    inf['method_topic2rubric_card2rubric_test'] = make_prediction_by_topic2rubric(inf, mode='test')
    accuracy_on_test(inf, inf['method_topic2rubric_card2rubric_test'])
    accuracy_on_test(inf, inf['p_cd_card2rubric_test'])



