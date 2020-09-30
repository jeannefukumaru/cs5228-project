from random import seed
from random import randrange
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from mlflow import log_metric 
import mlflow 

mlflow.set_experiment('zero_rule_baseline')

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

X_dev = pd.read_csv('data/processed/X_dev.csv')
y_dev = pd.read_csv('data/processed/y_dev.csv')

def zero_rule_algorithm_classification(zero_rule, y_dev):
    prediction = zero_rule
    predicted = [prediction for i in range(len(y_dev))]
    return predicted

predictions = zero_rule_algorithm_classification(0, y_dev)

precision, recall, fscore, support = precision_recall_fscore_support(y_dev, predictions, average='macro')
accuracy_score = accuracy_score(y_dev, predictions)
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