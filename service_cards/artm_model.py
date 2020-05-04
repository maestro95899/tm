import artm
print('artm.version()', artm.version())

def create_and_learn_PLSA(name="", topic_number=750, num_collection_passes=1):

    batch_vectorizer_train = None
    batch_vectorizer_train = artm.BatchVectorizer(data_path='./' + name,
                                                  data_format='vowpal_wabbit',
                                                  target_folder='folder' + name)
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=batch_vectorizer_train.data_path)
    topic_names = ['topic_{}'.format(i) for i in range(topic_number)]

    model_plsa = artm.ARTM(topic_names=topic_names,
                           class_ids={'@text': 1.0, '@first': 1.0, '@second': 1.0, '@third': 1.0},
                           cache_theta=True,
                           theta_columns_naming='title',
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary)])

    model_plsa.initialize(dictionary=dictionary)

    model_plsa.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model_plsa.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    model_plsa.scores.add(artm.TopicKernelScore(name='TopicKernelScore', class_id='@text', probability_mass_threshold=0.3))
    model_plsa.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6, class_id='@text'))

    model_plsa.num_document_passes = 1

    model_plsa.fit_offline(batch_vectorizer=batch_vectorizer_train, num_collection_passes=num_collection_passes)

    theta_train = model_plsa.transform(batch_vectorizer=batch_vectorizer_train)

    return model_plsa, theta_train

def create_and_learn_PLSA_class_ids_weigth(name="", topic_number=750, num_collection_passes=1, weigths=[1., 1., 1., 1.]):

    batch_vectorizer_train = None
    batch_vectorizer_train = artm.BatchVectorizer(data_path='./' + name,
                                                  data_format='vowpal_wabbit',
                                                  target_folder='folder' + name)
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=batch_vectorizer_train.data_path)
    topic_names = ['topic_{}'.format(i) for i in range(topic_number)]

    model_plsa = artm.ARTM(topic_names=topic_names,
                           class_ids={'@text': weigths[0], '@first': weigths[1], '@second': weigths[2], '@third': weigths[3]},
                           cache_theta=True,
                           theta_columns_naming='title',
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary)])

    model_plsa.initialize(dictionary=dictionary)

    model_plsa.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model_plsa.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    model_plsa.scores.add(artm.TopicKernelScore(name='TopicKernelScore', class_id='@text', probability_mass_threshold=0.3))
    model_plsa.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6, class_id='@text'))
    model_plsa.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score', class_id='@third'))

    model_plsa.num_document_passes = 1

    model_plsa.fit_offline(batch_vectorizer=batch_vectorizer_train, num_collection_passes=num_collection_passes)

    theta_train = model_plsa.transform(batch_vectorizer=batch_vectorizer_train)

    return model_plsa, theta_train

