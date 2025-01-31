from labeling_funcs.tweets_lfs import *
from labeling_funcs.bbc_lfs import *

tweets_config = {'experiment_name': 'tweets_opinion_lexicon_matcher_test_set',
          'x_train_filepath': 'data/augmented/tweets_x_train_x2.csv',
          'y_train_filepath': 'data/augmented/tweets_y_train_x2.csv',
          'x_dev_filepath' : 'data/processed/tweets_x_test.csv',
          'y_dev_filepath' : 'data/processed/tweets_y_test.csv',
          'cardinality' : 3,
          'lfs' : [opinion_lexicon_pos, opinion_lexicon_neg, opinion_lexicon_neu,
                   lf_contains_awesome, lf_contains_cancelled, lf_contains_complaint,
                   lf_contains_delayed,lf_contains_thank_you,],
          'stop_words' : ['united', 'usairways', 'southwestair', 'americanair', 'jetblue', 'virginamerica', 'flight'],}

bbc_config = {'experiment_name': 'bbc_bigrams_and_clustering_lgreg',
          'x_train_filepath': 'data/processed/bbc_x_train.csv',
          'y_train_filepath': 'data/processed/bbc_y_train.csv',
          'x_dev_filepath' : 'data/processed/bbc_x_dev.csv',
          'y_dev_filepath' : 'data/processed/bbc_y_dev.csv',
          'cardinality' : 5,
          'lfs' : [lf_contains_tech_terms, lf_contains_business_terms, lf_contains_sport_terms, 
                    lf_contains_entertainment_terms, lf_contains_politics_terms, lf_kmeans_label_0, lf_kmeans_label_1,
                    lf_kmeans_label_2, lf_kmeans_label_3, lf_kmeans_label_4],
          'stop_words' : ['said', 'people', 'new', 'mr', 'quarter', '2004', 'month', 'year', 'ago', 'old', 'don', 'know', 'recent', 'months']}

