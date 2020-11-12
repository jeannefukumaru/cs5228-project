# CS5228 Final Project AY2020 Semester 1  

# Introduction

Supervised learning has given us state-of-the-art results on many machine learning tasks, from sentiment analysisi, text categorizationii, image classificationiii to question and answeringiv. However, supervised learning needs data to be labelled by human annotators, a task that is time consuming and laborious. To alleviate this bottleneck, researchers have developed techniques to programmatically build training datasets, techniques that bypass the need for annotators and save timev. Collectively, these techniques are grouped under the term 'Data Programming'vi. 

In this project, we investigate to what extent data programming techniques are able to replace hand-crafted, "gold" labels for supervised learning. We take two datasets, the Twitter US Airline Sentiment datasetvii, and the BBC News datasetviii, which from several linguistic aspects are different from each other. We then separately apply two data programming techniques on each dataset - labelling functionsix and transformation functionsx - to obtain a weakly labeled datasets. The resulting weakly labeled datasets are then used to train a Logistic Regression model and a Distilbert Transformer model. 

# Setup python environment 
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
``` 

# Add labelling functions 
modify `labeling_funcs/bbc_lfs.py` and / or `labeling_funcs/tweets_lfs.py` to add / modify labeling functions that programmatically generated labels. 
example: 

```
@labeling_function()
def lf_contains_cancelled(x):
    return NEG if "cancelled" in x.text.lower() else ABSTAIN
```

# Edit config files 
- `label_model_config.py` contains training and testing configuration for Label Model
- `baseline_config.py` contains training and testing configuration for baseline model 
- `lgreg_config.py` contains training and testing configuration for Logistic Regression model 
- `distilbert_config.py` contains training and testing configuration for Distilbert model

# Run label model to denoise and reweight labelling functions 
`python label_model.py <experiment-name> <config-name>`
example: `python label_model.py tweets_opinion_lexicon tweets_config`

# Discriminative models 
## run baseline 
`python baseline.py <experiment-name> <config-name>`
example: `python baseline.py tweets_baseline tweets_config`

## Run Logistic Regression Model on Weak / Gold Label Dataset
`python lgreg.py <experiment-name> <config-name>`
example: `python lgreg.py tweets_weak_lgreg tweets_config`

## Run Distilbert Model on Weak / Gold Label Dataset
`python distilbert.py <experiment-name> <config-name>`
example: `python distilbert.py tweets_weak_distilbert tweets_config`

## View experiment results using MLFlow 
`mlflow ui` 

# Create processed datasets 
```
python data/create_bbc_dataset.py  # process BBC News dataste and perform train/dev/test splits
python data/create_tweet_dataset.py  # process Twitter US Airline Sentiment dataset and perform train/dev/test splits 
```

# Create augmented datasets 
`python data/eda_nlp/code/augment.py --input=data/bbc_train_all.csv --output_x_file=../augmented/eda_bbc_x_train.csv --output_y_file=../augmented/eda_bbc_y_train.csv`

