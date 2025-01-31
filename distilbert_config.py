from labeling_funcs.tweets_lfs import *
from labeling_funcs.bbc_lfs import *

# tutorial on constructing labeling functions from https://www.snorkel.org

tweets_config = {'experiment_name': 'tweets_opinion_lexicon_label_model',
          'x_train_filepath': 'data/processed/tweets_x_train.csv',
          'y_train_filepath': 'data/processed/tweets_y_train.csv',
          'x_dev_filepath' : 'data/processed/tweets_x_dev.csv',
          'y_dev_filepath' : 'data/processed/tweets_y_dev.csv',
          'cardinality' : 3,
          'lfs' : [lf_contains_cancelled,
                    lf_contains_delayed, lf_contains_complaint, lf_contains_thank_you, lf_contains_awesome,],
          'num_labels' : 3}

bbc_config = {'experiment_name': 'bbc_eda_augmentation',
          'x_train_filepath': 'data/processed/bbc_x_train.csv',
          'y_train_filepath': 'data/processed/bbc_y_train.csv',
          'x_dev_filepath' : 'data/processed/bbc_x_dev.csv',
          'y_dev_filepath' : 'data/processed/bbc_y_dev.csv',
          'cardinality' : 5,
          'lfs' : [lf_contains_tech_terms, lf_contains_business_terms, lf_contains_sport_terms, 
                    lf_contains_entertainment_terms, lf_contains_politics_terms],
          'num_labels' : 5}
