# Primitive Python paths (Python paths under which primitives registers themselves) have to adhere to namespace rules.
# Those rules describe that the Python path consists of multiple segments, one of them being "primitive name". Those
# names should be a general name to describe the logic of a primitive with the idea that multiple implementations
# of the same logic share the same name. This file contains a list of known and allowed primitive names.
# Names should be descriptive and something which can help people understand what the primitive is about.
# You can assume general understanding of data science concepts and names.
#
# Everyone is encouraged to help currate this list and suggest improvements (merging, removals, additions)
# of values in that list by submitting a merge request. We are not strict about names here, the main purpose of
# this list is to encourage collaboration and primitive name reuse when that is reasonable. Please check the list
# first when deciding on a Python path of your primitive and see if it can fit well under an existing name.
#
# On Linux, you can sort the list by running:
#
#     grep "^ *'" d3m/metadata/primitive_names.py | env LC_COLLATE=C sort -u
#
# See: https://gitlab.com/datadrivendiscovery/d3m/issues/3

PRIMITIVE_NAMES = [
    'ada_boost',
    'adaptive_simultaneous_markov_blanket',
    'add',
    'add_semantic_types',
    'adjacency_spectral_embedding',
    'ape',
    'ard',
    'arima',
    'audio_featurization',
    'audio_reader',
    'audio_slicer',
    'audio_transfer',
    'average_pooling_1d',
    'average_pooling_2d',
    'average_pooling_3d',
    'bagging',
    'batch_normalization',  # To be used with "layer" primitive family.
    'bayesian_logistic_regression',  # To be used with "classification" primitive family.
    'bernoulli_naive_bayes',
    'bert_classifier',
    'binarizer',
    'binary_crossentropy',
    'binary_encoder',
    'cast_to_type',
    'categorical_accuracy',
    'categorical_crossentropy',
    'categorical_hinge',
    'categorical_imputer',
    'channel_averager',
    'clean_augmentation',
    'cleaning_featurizer',
    'cluster',
    'cluster_curve_fitting_kmeans',
    'column_fold',
    'column_map',
    'column_parser',
    'column_type_profiler',
    'compute_scores',
    'concat',
    'conditioner',
    'construct_predictions',
    'convolution_1d',
    'convolution_2d',
    'convolution_3d',
    'convolutional_neural_net',
    'corex_continuous',
    'corex_supervised',
    'corex_text',
    'cosine_proximity',
    'count_vectorizer',
    'cover_tree',
    'croc',
    'csv_reader',
    'cut_audio',
    'data_conversion',
    'dataframe_to_list',
    'dataframe_to_list_of_list',
    'dataframe_to_list_of_ndarray',
    'dataframe_to_ndarray',
    'dataframe_to_tensor',
    'datamart_augmentation',
    'datamart_download',
    'dataset_map',
    'dataset_sample',
    'dataset_text_reader',
    'dataset_to_dataframe',
    'datetime_field_compose',
    'datetime_range_filter',
    'decision_tree',
    'deep_feature_synthesis',
    'deep_markov_bernoulli_forecaster',
    'deep_markov_categorical_forecaster',
    'deep_markov_gaussian_forecaster',
    'denormalize',
    'dense',
    'diagonal_mvn',
    'dict_vectorizer',
    'dimension_selection',
    'discriminative_structured_classifier',
    'do_nothing',
    'do_nothing_for_dataset',
    'doc_2_vec',
    'dropout',
    'dummy',
    'echo_ib',
    'echo_linear',
    'edge_list_to_graph',
    'ekss',
    'elastic_net',
    'encoder',
    'enrich_dates',
    'ensemble_forest',
    'ensemble_voting',
    'esrnn',
    'extra_trees',
    'extract_columns',
    'extract_columns_by_semantic_types',
    'extract_columns_by_structural_types',
    'fast_ica',
    'fast_lad',
    'feature_agglomeration',
    'feed_forward_neural_net',
    'fixed_split_dataset_split',
    'flatten',
    'forward',
    'gaussian',  # To be used with "classification" or "clustering" primitive family.
    'gaussian_naive_bayes',
    'gaussian_process',
    'gaussian_random_projection',
    'gcn_mixhop',
    'general_relational_dataset',
    'generative_structured_classifier',
    'generic_univariate_select',
    'geocoding',
    'glda',
    'global_average_pooling_1d',
    'global_average_pooling_2d',
    'global_average_pooling_3d',
    'global_causal_discovery',
    'global_structure_imputer',
    'gmm',
    'go_dec',
    'goturn',
    'gradient_boosting',
    'graph_node_splitter',
    'graph_to_edge_list',
    'graph_transformer',
    'grasta',
    'grasta_masked',
    'greedy_imputation',
    'grouping_field_compose',
    'grouse',
    'hdbscan',
    'hdp',
    'high_rank_imputer',
    'hinge',
    'horizontal_concat',
    'huber_pca',
    'i3d',
    'i_vector_extractor',
    'ibex',
    'identity_parentchildren_markov_blanket',
    'image_reader',
    'image_transfer',
    'image_transfer_learning_transformer',
    'imputer',
    'inceptionV3_image_feature',
    'increment',
    'iqr_scaler',
    'iterative_labeling',
    'iterative_regression_imputation',
    'joint_mutual_information',
    'k_means',
    'k_neighbors',
    'kernel_pca',
    'kernel_ridge',
    'kfold_dataset_split',
    'kfold_time_series_split',
    'kss',
    'kullback_leibler_divergence',
    'l1_low_rank',
    'label_decoder',
    'label_encoder',
    'labler',
    'laplacian_spectral_embedding',
    'largest_connected_component',
    'lars',
    'lasso',
    'lasso_cv',
    'lda',
    'light_gbm',
    'linear',
    'linear_discriminant_analysis',
    'linear_svc',
    'linear_svr',
    'link_prediction',  # To be used with "collaborative_filtering" or "graph_matching" primitive family.
    'list_to_dataframe',
    'list_to_ndarray',
    'load_edgelist',
    'load_graphs',
    'load_single_graph',
    'local_structure_imputer',
    'log_mel_spectrogram',
    'logcosh',
    'logistic_regression',  # To be used with "classification" primitive family.
    'loss',
    'low_rank_imputer',
    'lstm',
    'lupi_svm',
    'max_abs_scaler',
    'max_pooling_1d',
    'max_pooling_2d',
    'max_pooling_3d',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'mean_baseline',
    'mean_imputation',
    'mean_squared_error',
    'mean_squared_logarithmic_error',
    'merge_partial_predictions',
    'metafeature_extractor',
    'mice_imputation',
    'min_max_scaler',
    'missing_indicator',
    'mlp',
    'model',
    'monomial',
    'multinomial_naive_bayes',
    'multitable_featurization',
    'mutual_info',  # To be used with "classification" or "regression" primitive family.
    'n_beats',
    'naive_bayes',
    'ndarray_to_dataframe',
    'ndarray_to_list',
    'nearest_centroid',
    'nk_sent2vec',
    'no_split_dataset_split',
    'non_parametric',  # To be used with "clustering" primitive family.
    'normalize_column_references',
    'normalize_graphs',
    'normalizer',
    'null',
    'number_of_clusters',
    'numeric_range_filter',
    'nystroem',
    'one_hot_encoder',
    'ordinal_encoder',
    'out_of_sample_adjacency_spectral_embedding',
    'out_of_sample_laplacian_spectral_embedding',
    'output_dataframe',
    'owl',  # To be used with "regression" primitive family.
    'parser',  # To be used with "collaborative_filtering", "graph_matching", "vertex_nomination", or "community_detection" primitive family.
    'pass_to_ranks',
    'passive_aggressive',
    'pca',
    'pca_features',
    'pcp_ialm',
    'poisson',
    'polynomial_features',
    'primitive_sum',
    'profiler',
    'quadratic_discriminant_analysis',
    'quantile_transformer',
    'ragged_dataset_reader',
    'random',
    'random_classifier',
    'random_forest',
    'random_projection_time_series_featurization',
    'random_sampling_imputer',
    'random_trees_embedding',
    'rank',  # To be used with "classification" primitive family.
    'ravel',
    'rbf_sampler',
    'recommender_system',  # To be used with "collaborative_filtering" primitive family.
    'redact_columns',
    'regex_filter',
    'relational_time_series',
    'remote_sensing_pretrained',
    'remove_columns',
    'remove_duplicate_columns',
    'remove_semantic_types',
    'rename_duplicate_name',
    'replace_semantic_types',
    'replace_singletons',
    'resnet50_image_feature',
    'resnext101_kinetics_video_features',
    'retina_net',
    'reverse',
    'rfd',
    'rfe',
    'rffeatures',
    'rfm_precondition_ed_gaussian_krr',
    'rfm_precondition_ed_polynomial_krr',
    'ridge',
    'rnn_time_series',
    'robust_scaler',
    'rpca_lbd',
    'satellite_image_loader',
    'score_based_markov_blanket',
    'sdne',
    'search',
    'search_hybrid',
    'search_hybrid_numeric',
    'search_numeric',
    'seeded',  # To be used with "graph_matching" primitive family.
    'seeded_graph_matching',  # To be used with "vertex_nomination" primitive family.
    'segment_curve_fitter',
    'select_fwe',
    'select_percentile',
    'sequence_to_bag_of_tokens',
    'sgd',
    'shapelet_learning',
    'signal_dither',
    'signal_framer',
    'signal_mfcc',
    'simon',
    'simple_imputer',
    'simultaneous_markov_blanket',
    'sparse_categorical_crossentropy',
    'sparse_pca',
    'sparse_random_projection',
    'spectral',  # To be used with "vertex_nomination" primitive family.
    'spectral_graph',  # To be used with "clustering" primitive family.
    'splitter',
    'squared_hinge',
    'ssc_admm',
    'ssc_cvx',
    'ssc_omp',
    'stack_ndarray_column',
    'stacking',  # To be used with "operator" primitive family.
    'standard_scaler',
    'string_imputer',
    'structured',  # To be used with "classification" primitive family.
    'subtract',
    'sum',
    'svc',
    'svr',
    't_distributed_stochastic_neighbor_embedding',
    'tabular_extractor',
    'targets_reader',
    'tensor_machines_binary',  # To be used with "classification" primitive family.
    'tensor_machines_regularized_least_squares',
    'term_filter',
    'text_classifier',
    'text_encoder',
    'text_reader',
    'text_summarization',
    'text_to_vocabulary',
    'text_tokenizer',
    'tfidf_vectorizer',
    'time_series_binner',
    'time_series_forecasting',
    'time_series_formatter',
    'time_series_neighbours',
    'time_series_reshaper',
    'time_series_to_list',
    'to_numeric',
    'topic_vectorizer',
    'train_score_dataset_split',
    'trecs',
    'tree_augmented_naive_bayes',
    'trim_regressor',
    'truncated_svd',
    'unary_encoder',
    'unfold',
    'unicorn',
    'uniform_segmentation',
    'update_semantic_types',
    'variance_threshold',
    'vector_autoregression',
    'vertical_concatenate',
    'vgg16',
    'vgg16_image_feature',
    'video_reader',
    'voter',
    'voting',
    'wikifier',
    'word_2_vec',
    'word_embedding_builder',
    'xgboost_dart',
    'xgboost_gbtree',
    'yolo',
    'zero_count',
]
