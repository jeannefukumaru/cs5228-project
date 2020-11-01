import torch
import torch.nn as nn
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from augment_funcs import * 
from snorkel.augmentation import RandomPolicy, MeanFieldPolicy, PandasTFApplier
import mlflow
from mlflow import log_metric, log_param
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
from run_lgreg_utils import read_data_from_config
from lgreg_config import bbc_config as config

tfs = [swap_adjectives, replace_verb_with_synonym, replace_noun_with_synonym, 
        replace_adjective_with_synonym]

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True)

X_train, y_train, _, _ = read_data_from_config(config)

print(preview_tfs(X_train,tfs))

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=3,
    keep_original=True,
    p=[0.25, 0.25, 0.25, 0.25],
)

X_y_combined = pd.concat([X_train, y_train], axis=1)
tf_applier = PandasTFApplier(tfs, mean_field_policy)
aug = tf_applier.apply(X_y_combined)
df_train_aug = aug.drop('sentiment_id', axis=1)
y_train_aug= aug["sentiment_id"]

print(f'Original training set size: {len(X_train)}')
print(f'Augmented training set size {len(df_train_aug)}')

df_train_aug.to_csv('data/augmented/tweets_x_train_augmented_x3.csv', index=False)
y_train_aug.to_csv('data/augmented/tweets_y_train_augmented_x3.csv', index=False)