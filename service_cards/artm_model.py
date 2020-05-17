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


def create_and_learn_ARTM_decorPhi_text(name="", topic_number=750, num_collection_passes=1, weigths=[1., 1., 1., 1.], decorTau=1.0):

    batch_vectorizer_train = None
    batch_vectorizer_train = artm.BatchVectorizer(data_path='./' + name,
                                                  data_format='vowpal_wabbit',
                                                  target_folder='folder' + name)
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=batch_vectorizer_train.data_path)
    topic_names = ['topic_{}'.format(i) for i in range(topic_number)]

    model = artm.ARTM(topic_names=topic_names,
                           class_ids={'@text': weigths[0], '@first': weigths[1], '@second': weigths[2], '@third': weigths[3]},
                           cache_theta=True,
                           theta_columns_naming='title',
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary)])
    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi_text', tau=decorTau, class_ids='@text'))

    model.initialize(dictionary=dictionary)

    model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    model.scores.add(artm.TopicKernelScore(name='TopicKernelScore', class_id='@text', probability_mass_threshold=0.3))
    model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6, class_id='@text'))
    model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score', class_id='@third'))

    model.num_document_passes = 1

    model.fit_offline(batch_vectorizer=batch_vectorizer_train, num_collection_passes=num_collection_passes)

    theta_train = model.transform(batch_vectorizer=batch_vectorizer_train)

    return model, theta_train

def create_and_learn_ARTM_decorPhi_modal(name="", topic_number=750, num_collection_passes=1, weigths=[1., 1., 1., 1.], decorTau=1.0):

    batch_vectorizer_train = None
    batch_vectorizer_train = artm.BatchVectorizer(data_path='./' + name,
                                                  data_format='vowpal_wabbit',
                                                  target_folder='folder' + name)
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=batch_vectorizer_train.data_path)
    topic_names = ['topic_{}'.format(i) for i in range(topic_number)]

    model = artm.ARTM(topic_names=topic_names,
                           class_ids={'@text': weigths[0], '@first': weigths[1], '@second': weigths[2], '@third': weigths[3]},
                           cache_theta=True,
                           theta_columns_naming='title',
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary)])
    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi_modals', tau=decorTau, class_ids=['@first', '@second', '@third']))

    model.initialize(dictionary=dictionary)

    model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    model.scores.add(artm.TopicKernelScore(name='TopicKernelScore', class_id='@text', probability_mass_threshold=0.3))
    model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6, class_id='@text'))
    model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score', class_id='@third'))

    model.num_document_passes = 1

    model.fit_offline(batch_vectorizer=batch_vectorizer_train, num_collection_passes=num_collection_passes)

    theta_train = model.transform(batch_vectorizer=batch_vectorizer_train)

    return model, theta_train

def create_and_learn__ARTM_background_topics(name="", topic_number=750, num_collection_passes=1,
                                             weigths=[1., 1., 1., 1.],
                                             decorTau=1.0,
                                             background_topics_count=2,
                                             back_PhiReg=0.1,
                                             back_ThetaReg=2.0,
                                             obj_PhiReg=-0.1,
                                             obj_ThetaReg=-2.0):

    batch_vectorizer_train = None
    batch_vectorizer_train = artm.BatchVectorizer(data_path='./' + name,
                                                  data_format='vowpal_wabbit',
                                                  target_folder='folder' + name)
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=batch_vectorizer_train.data_path)
    objective_topics = ['topic_{}'.format(i) for i in range(topic_number - background_topics_count)]
    background_topics = ['background_topic_{}'.format(i) for i in range(background_topics_count)]
    topic_names = objective_topics.copy()
    topic_names.extend(background_topics)

    model = artm.ARTM(topic_names=topic_names,
                           class_ids={'@text': weigths[0], '@first': weigths[1], '@second': weigths[2], '@third': weigths[3]},
                           cache_theta=True,
                           theta_columns_naming='title',
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary)])
    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi_modals', topic_names=objective_topics, tau=decorTau, class_ids=['@first', '@second', '@third']))

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='BackgroundSparsePhi',
                                                      topic_names=background_topics, class_ids=['@text'], tau=back_PhiReg))
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='BackgroundSparseTheta',
                                                        topic_names=background_topics, tau=back_ThetaReg))

    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='ObjectiveSparsePhi',
                                                      topic_names=objective_topics, class_ids=['@text'], tau=obj_PhiReg))
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='ObjectiveSparseTheta',
                                                        topic_names=objective_topics, tau=obj_ThetaReg))

    model.initialize(dictionary=dictionary)

    model.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
    model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
    model.scores.add(artm.TopicKernelScore(name='TopicKernelScore', class_id='@text', probability_mass_threshold=0.3))
    model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6, class_id='@text'))
    model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score', class_id='@third'))

    model.num_document_passes = 1

    model.fit_offline(batch_vectorizer=batch_vectorizer_train, num_collection_passes=num_collection_passes)

    theta_train = model.transform(batch_vectorizer=batch_vectorizer_train)

    return model, theta_train
