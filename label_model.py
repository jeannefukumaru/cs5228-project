import numpy as np
import matplotlib.pyplot as plt 
from mlflow import log_metric, log_param, log_artifacts
import mlflow
from datetime import datetime
import os 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from snorkel.labeling import PandasLFApplier, labeling_function, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.labeling import filter_unlabeled_dataframe
from snorkel.utils import probs_to_preds
from label_model_config import *
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
majority_model = MajorityLabelVoter(cardinality=5)
majority_acc = majority_model.score(L=L_dev, Y=np.array(y_dev).reshape(-1,1), tie_break_policy="random")["accuracy"]
print(f'majority_label_acc: {majority_acc}')

log_metric('majority_label_acc', majority_acc)
log_metric('label_model_acc', label_model_acc)