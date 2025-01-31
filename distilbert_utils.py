from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from mlflow import log_metric
import torch
import pandas as pd 

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

ddef log_and_print_metrics(model_name, results_dict):
    log_metric(f'{model_name}_loss', results_dict['eval_loss'])
    log_metric(f'{model_name}_accuracy', results_dict['eval_accuracy'] * 100)
    log_metric(f'{model_name}_precision', results_dict['eval_precision'])
    log_metric(f'{model_name}_recall', results_dict['eval_recall'])
    log_metric(f'{model_name}_fscore', results_dict['eval_f1'])
    print(f"dev weak sup accuracy: {results_dict['eval_accuracy']}%")
    print(f"precision: {results_dict['eval_precision']}")
    print(f"recall: {results_dict['eval_recall']}")
    print(f"fscore: {results_dict['eval_f1']}")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

def read_data_from_config(config):
    X_train = pd.read_csv(config['x_train_filepath'])
    y_train = pd.read_csv(config['y_train_filepath'])
    X_dev = pd.read_csv(config['x_dev_filepath'])
    y_dev = pd.read_csv(config['y_dev_filepath'])
    return X_train, y_train, X_dev, y_dev
