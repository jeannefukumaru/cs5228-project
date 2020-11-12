# CS5228 Final Project AY2020 Semester 1  

# Setup python environment 

# Add labelling functions 

# Edit config files 

# Run label model to denoise and reweight labelling functions 
python label_model.py <experiment-name> <config-name>
example: python label_model.py tweets_opinion_lexicon tweets_config

# Discriminative models 
## run baseline 
python baseline.py <experiment-name> <config-name>
example: python baseline.py tweets_baseline tweets_config

## Run logistic regression
python lgreg.py <experiment-name> <config-name>
example: python lgreg.py tweets_weak_lgreg tweets_config

## Run distilbert 
python distilbert.py <experiment-name> <config-name>
example: python distilbert.py tweets_weak_distilbert tweets_config

## View experiment results using MLFlow 
mlflow ui 

# Create processed datasets 
python data/create_bbc_dataset.py  # process BBC News dataste and perform train/dev/test splits 
python data/create_tweet_dataset.py  # process Twitter US Airline Sentiment dataset and perform train/dev/test splits 

# Create augmented datasets 
python data/eda_nlp/code/augment.py --input=data/bbc_train_all.csv --output_x_file=../augmented/eda_bbc_x_train.csv --output_y_file=../augmented/eda_bbc_y_train.csv

