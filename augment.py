import torch
from torchtext import data
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

mlflow.set_experiment('augmentation_x3')

tfs = [swap_adjectives, replace_verb_with_synonym, replace_noun_with_synonym, 
        replace_adjective_with_synonym]

random_policy = RandomPolicy(
    len(tfs), sequence_length=2, n_per_original=2, keep_original=True)

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
train = pd.concat([X_train, y_train], axis=1)

X_dev = pd.read_csv('data/processed/X_dev.csv')
y_dev = pd.read_csv('data/processed/y_dev.csv')

print(preview_tfs(train,tfs))

mean_field_policy = MeanFieldPolicy(
    len(tfs),
    sequence_length=2,
    n_per_original=3,
    keep_original=True,
    p=[0.1, 0.3, 0.3, 0.3],
)

tf_applier = PandasTFApplier(tfs, mean_field_policy)
df_train_augmented = tf_applier.apply(train)
y_train_aug = df_train_augmented['sentiment_id']
df_train_aug = df_train_augmented.drop('sentiment_id', axis=1)

print(f'Original training set size: {len(train)}')
print(f'Augmented training set size {len(df_train_aug)}')

df_train_aug.to_csv('data/augmented/X_train_augmented.csv', index=False)
y_train_aug.to_csv('data/augmented/y_train_augmented.csv', index=False)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

airline_stop_words = ['united', 'usairways', 'southwestair', 'americanair', 'jetblue', 'virginamerica', 'flight']
custom_stop_words = text.ENGLISH_STOP_WORDS.union(airline_stop_words)
# vectorizer = CountVectorizer(ngram_range=(1, 5))
vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
X_train_aug = vectorizer.fit_transform(df_train_augmented.text.tolist())
X_dev_vectorized = vectorizer.transform(X_dev.text.tolist())
X_train_no_aug = vectorizer.fit_transform(X_train.text.tolist())

from sklearn.linear_model import LogisticRegressionCV
log_param('model', 'log_reg_cv_5')
sklearn_model_aug = LogisticRegressionCV(max_iter=500, cv=5, random_state=0, solver='liblinear').fit(X_train_aug, np.array(y_train_aug))
sklearn_model_no_aug = LogisticRegressionCV(max_iter=500, cv=5, random_state=0, solver='liblinear').fit(X_train_no_aug, np.array(y_train))
cv_aug = sklearn_model_aug.score(X_train_aug, y_train_aug)
cv_no_aug = sklearn_model_no_aug.score(X_train_no_aug, y_train)

dev_accuracy = sklearn_model_aug.score(X=X_dev_vectorized, y=y_dev) * 100
log_metric('train_augmented_cv_score', cv_aug)
log_metric('train_non_augmented_cv_score', cv_no_aug)
log_metric('dev_set_accuracy', dev_accuracy)
print(f'train set augmented score: {cv_aug}')
print(f'train set non augmented score: {cv_no_aug}')
print(f"Dev set accuracy: {dev_accuracy}%")

predictions = sklearn_model_aug.predict(X_dev_vectorized)
precision, recall, fscore, support = precision_recall_fscore_support(y_dev, predictions, average='micro')

log_metric('precision', precision)
log_metric('recall',recall)
log_metric('fscore',fscore)
# log_metric('support',support)
precision, recall, fscore, support = precision_recall_fscore_support(y_dev, predictions)
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'fscore: {fscore}')
print(f'support: {support}')
metrics = pd.DataFrame({'precision':precision, 'recall':recall, 'fscore':fscore, 'support': support})
metrics_csv_name = datetime.now().strftime('%Y%m%d%M') + 'metrics.csv'
metrics.to_csv('outputs/' + metrics_csv_name)