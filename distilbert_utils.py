from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from mlflow import log_metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def log_and_print_metrics(results_dict):
    log_metric('dev_weak_sup_accuracy', results_dict['eval_accuracy'] * 100)
    # log_metric('dev_full_sup_accuracy', dev_full_sup_accuracy)
    print(f"dev weak sup accuracy: {results_dict['eval_accuracy']}%")
    # print(f"dev full sup accuracy: {dev_full_sup_accuracy}%")
    log_metric('precision', results_dict['precision'])
    log_metric('recall', results_dict['recall'])
    log_metric('fscore', results_dict['fscore'])
    # log_metric('support',support)
    print(f"precision: {results_dict['precision']}")
    print(f"recall: {results_dict['recall']}")
    print(f"fscore: {results_dict['fscore']}")
    print(f"support: {results_dict['support']}")

class TweetsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)