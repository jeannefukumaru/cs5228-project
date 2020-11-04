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

def print_and_log_precision_recall_f1(sklearn_model, model_name, X_dev, y_dev):
    predictions = sklearn_model.predict(X_dev)
    precision, recall, fscore, support = precision_recall_fscore_support(y_dev, predictions, average='micro')
    precision_metric = f'{model_name}_precision'
    recall_metric = f'{model_name}_recall'
    fscore_metric = f'{model_name}_fscore'
    print(precision_metric)
    log_metric(precision_metric, precision)
    log_metric(recall_metric,recall)
    log_metric(fscore_metric,fscore)
    precision, recall, fscore, support = precision_recall_fscore_support(y_dev, predictions)
    print(f'{model_name} precision: {precision}')
    print(f'{model_name} recall: {recall}')
    print(f'{model_name} fscore: {fscore}')
    print(f'{model_name} support: {support}')
    return precision, recall, fscore, support

def read_data_from_config(config):
    X_train = pd.read_csv(config['x_train_filepath'])
    y_train = pd.read_csv(config['y_train_filepath'])
    X_dev = pd.read_csv(config['x_dev_filepath'])
    y_dev = pd.read_csv(config['y_dev_filepath'])
    return X_train, y_train, X_dev, y_dev
