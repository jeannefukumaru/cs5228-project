from mlflow import log_metric
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd 

def print_and_log_accuracies(dev_weak_sup_accuracy, dev_full_sup_accuracy):
    # log_metric('train_weak_sup_cv_score', cv_weak)
    # log_metric('train_full_sup_cv_score', cv_sup)
    log_metric('dev_weak_sup_accuracy', dev_weak_sup_accuracy)
    log_metric('dev_full_sup_accuracy', dev_full_sup_accuracy)
    # print(f'train set weakly supervised score: {cv_weak}')
    # print(f'train set fully supervised score: {cv_sup}')
    print(f"dev weak sup accuracy: {dev_weak_sup_accuracy}%")
    print(f"dev full sup accuracy: {dev_full_sup_accuracy}%")

def get_and_log_precision_recall_f1(sklearn_model, X_dev, y_dev):
    predictions = sklearn_model.predict(X_dev)
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
    return precision, recall, fscore, support

def read_data_from_config(config):
    X_train = pd.read_csv(config['x_train_filepath'])
    y_train = pd.read_csv(config['y_train_filepath'])
    X_dev = pd.read_csv(config['x_dev_filepath'])
    y_dev = pd.read_csv(config['y_dev_filepath'])
    return X_train, y_train, X_dev, y_dev
