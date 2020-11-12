from random import seed
from random import randrange
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from mlflow import log_metric 
import mlflow 
from lgreg_utils import read_data_from_config
from baseline_config import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(dest='experiment_name', type=str, help="mlflow experiment name")
parser.add_argument(dest='config', type=str, help="'tweets_config' or 'bbc_config'")
args = parser.parse_args()

print(args)
print('parsing arguments..')
if args.config == 'tweets_config':
  config = tweets_config
else:
  config = bbc_config

print(f"setting up experiment: {args.experiment_name}")
mlflow.set_experiment(args.experiment_name)

print('reading in data...')
X_train, y_train, X_dev, y_dev = read_data_from_config(config)

def zero_rule_algorithm_classification(zero_rule, y_dev):
    prediction = zero_rule
    predicted = [prediction for i in range(len(y_dev))]
    return predicted

predicted = zero_rule_algorithm_classification(1, y_dev)

precision, recall, fscore, support = precision_recall_fscore_support(y_dev, predicted, average='macro')
accuracy_score = accuracy_score(list(y_dev.values.reshape(-1)), predicted)
log_metric('precision', precision)
log_metric('recall',recall)
log_metric('fscore',fscore)
# log_metric('support',support)
log_metric('accuracy', accuracy_score)
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'fscore: {fscore}')
print(f'support: {support}')
print(f'accuracy score: {accuracy_score}')
