# (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html),
# https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4
# https://github.com/cjhutto/vaderSentiment

import pandas as pd 
import nltk
from snorkel.labeling import PandasLFApplier, labeling_function, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds
import numpy as np
import matplotlib.pyplot as plt 
from labeling_funcs import * 
from distilbert_utils import *
from plotting_funcs import plot_label_frequency, plot_probabilities_histogram
from mlflow import log_metric, log_param, log_artifacts
import mlflow
from datetime import datetime
import os 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

mlflow.set_experiment('bert_sentiment_and_matchers_with_augmentation')

lfs = [opinion_lexicon_neg, opinion_lexicon_pos, opinion_lexicon_neu, textblob_pos, textblob_neg, 
        textblob_neu, vader_lexicon, lf_contains_delayed,
        lf_contains_thank_you, lf_contains_awesome]

print('reading in data...')
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

X_dev = pd.read_csv('data/processed/X_dev.csv')
y_dev = pd.read_csv('data/processed/y_dev.csv')

print('applying labelling functions to data...')
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=X_train)
L_dev = applier.apply(df=X_dev)

print('fitting Label Model')
label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
label_model_acc = label_model.score(L=L_dev, Y=y_dev, tie_break_policy="random")["accuracy"]
print(f'label model acc: {label_model_acc}')

print('fitting Majority Label Voter model')
majority_model = MajorityLabelVoter(cardinality=3)
# preds_train = majority_model.predict(L=L_train)
majority_acc = majority_model.score(L=L_dev, Y=np.array(y_dev).reshape(-1,1), tie_break_policy="random")["accuracy"]
print(f'majority_label_acc: {majority_acc}')


log_metric('majority_label_acc', majority_acc)
log_metric('label_model_acc', label_model_acc)

probs_train = label_model.predict_proba(L=L_train)
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=X_train, y=probs_train, L=L_train
)
preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

print('setting up discriminative model that will take in noise-aware labels from Label Model')
print('tokenizing and encoding texts')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train['text'].values.tolist(), truncation=True, padding=True)
dev_encodings = tokenizer(X_dev['text'].values.tolist(), truncation=True, padding=True)

train_weak_sup_dataset = TweetsDataset(train_encodings, preds_train_filtered)
dev_weak_sup_dataset = TweetsDataset(dev_encodings, y_dev.values.tolist())

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

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
print('start training!')
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_weak_sup_dataset,# training dataset
    eval_dataset=dev_weak_sup_dataset,   # evaluation dataset
    compute_metrics = compute_metrics        
)

trainer.train()

save_model_folder = "results/datetime.now().strftime('%Y%m%d%M')/"
os.makedirs(save_model_folder)
trainer.save_model(save_model_folder)

results_dict = trainer.evaluate()
log_and_print_metrics(results_dict)


metrics = pd.DataFrame({'precision':results_dict['precision'], 'recall':results_dict['recall'], 'fscore':results_dict['fscore'], 'support': results_dict['support']})
metrics_csv_name = datetime.now().strftime('%Y%m%d%M') + 'metrics.csv'
metrics.to_csv('outputs/' + metrics_csv_name)
print('overall metrics saved to outputs/csv_name')

print('plotting summary graphs of labeling functions')
plot_probabilities_histogram(probs_train[:, POS])
plot_label_frequency(L_train)

lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
csv_name = datetime.now().strftime('%Y%m%d%M') + 'lf_summary.csv'
lf_summary.to_csv('outputs/' + csv_name)
print('labelling functions summary csv saved to outputs/{csv_name}')
print(lf_summary)

print(lf_summary)
log_artifacts("outputs")
