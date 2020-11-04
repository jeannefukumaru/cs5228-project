# (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html),
# https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4
# https://github.com/cjhutto/vaderSentiment

import pandas as pd 
import nltk
from snorkel.utils import probs_to_preds
from snorkel.labeling import PandasLFApplier, labeling_function, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds
import numpy as np
import matplotlib.pyplot as plt 
from distilbert_utils import *
from mlflow import log_metric, log_param, log_artifacts
import mlflow
from datetime import datetime
import os 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from distilbert_config import *
from distilbert_utils import read_data_from_config
import argparse
import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument(dest='experiment_name', type=str, help="mlflow experiment name")
parser.add_argument(dest='config', type=str, help="'tweets_config' or 'bbc_config'")
args = parser.parse_args()

print(args)
print('parsing arguments..')
if args.config == 'tweets_config':
  config = tweets_config
elif args.config == 'bbc_config':
  config = bbc_config

print(f"setting up experiment: {args.experiment_name}")
mlflow.set_experiment(args.experiment_name)

lfs = config['lfs']

print('reading in data...')
X_train, y_train, X_dev, y_dev = read_data_from_config(config)

print('applying labelling functions to data...')
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=X_train)
L_dev = applier.apply(df=X_dev)

print('fitting Label Model')
label_model = LabelModel(cardinality=config['cardinality'], verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
label_model_acc = label_model.score(L=L_dev, Y=y_dev, tie_break_policy="random")["accuracy"]
print(f'label model acc: {label_model_acc}')

print('fitting Majority Label Voter model')
majority_model = MajorityLabelVoter(cardinality=config['num_labels'])
majority_acc = majority_model.score(L=L_dev, Y=np.array(y_dev).reshape(-1,1), tie_break_policy="random")["accuracy"]
print(f'majority_label_acc: {majority_acc}')

log_metric('majority_label_acc', majority_acc)
log_metric('label_model_acc', label_model_acc)

probs_train = label_model.predict_proba(L=L_train)
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=X_train, y=probs_train, L=L_train
)
preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

print('setting up model that will take in noise-aware labels from Label Model')
print('tokenizing and encoding texts')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
full_sup_train_encodings = tokenizer(X_train['text'].values.tolist(), truncation=True, padding=True)
weak_suptrain_encodings = tokenizer(df_train_filtered['text'].values.tolist(), truncation=True, padding=True)
dev_encodings = tokenizer(X_dev['text'].values.tolist(), truncation=True, padding=True)

train_weak_sup_dataset = Dataset(weak_sup_train_encodings, preds_train_filtered)
train_full_sup_dataset = Dataset(full_sup_train_encodings, y_train.value.tolist())
dev_weak_sup_dataset = Dataset(dev_encodings, y_dev.values.tolist())

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=config['num_labels'])
print('start training!')
weak_sup_trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_weak_sup_dataset,# training dataset
    eval_dataset=dev_weak_sup_dataset,   # evaluation dataset
    compute_metrics = compute_metrics        
)

weak_sup_trainer.train()
weak_sup_results_dict = weak_sup_trainer.evaluate()

results_dict = trainer.evaluate()

full_sup_trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_full_sup_dataset,# training dataset
    eval_dataset=dev_dataset,   # evaluation dataset
    compute_metrics = compute_metrics        
)

full_sup_trainer.train()

full_sup_results_dict = full_sup_trainer.evaluate()

log_and_print_metrics('distilbert_weak_sup', weak_sup_results_dict)
log_and_print_metrics('distilbert_full_sup', full_sup_results_dict)