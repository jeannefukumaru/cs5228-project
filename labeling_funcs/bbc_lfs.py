 # supposedly trained on social media data 
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from textblob import TextBlob
from collections import defaultdict
from sklearn.feature_extraction import text
from sklearn.decomposition import LatentDirichletAllocation
import joblib

POLITICS = 4
ENTERTAINMENT = 3
SPORT = 2
BUSINESS = 1
TECH = 0
ABSTAIN = -1

bbc_stop_words = ['said', 'people', 'new', 'mr']
custom_stop_words = text.ENGLISH_STOP_WORDS.union(bbc_stop_words)

bbc_knn = joblib.load('bbc_knn.joblib')
bbc_vectorizer = joblib.load('bbc_vec.joblib')

@labeling_function()
def lf_contains_tech_terms(x):
    words = ['mobile', 'games', 'music', 'software', 'game', 'technology', 'phone',
                    'users', 'digital', 'microsoft', 'net', 'broadband', 'tv','computer',
                    'search', 'service', 'online', 'security', 'phones', 'video', 'apple',
                    'virus', 'data', 'internet', 'sony', 'information', 'services', 'pc',]
    return TECH if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_business_terms(x):
    words = ['growth', 'economy', 'sales', 'bank', 'market', 'oil', 'company', 'economic',
                 'firm', 'prices', 'dollar', 'shares', 'quarter', 'deal', 'figures', 'rates',
                 'companies', 'expected', 'business', 'trade', 'firms', 'spending', 'analysts',
                 'profits', 'group', 'stock', 'financial', 'demand', 'president', 'month', 'country',
                 'tax', 'jobs', 'euros']
    return BUSINESS if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_sport_terms(x):
    words = ['england', 'game', 'win', 'wales', 'play', 'cup', 'ireland', 'players', 'team',
                'match', 'chelsea', 'injury', 'final','rugby', 'coach', 'won', 'set', 'league', 'united',
                'champion', 'nations', 'liverpool', 'arsenal', 'scotland', 'race']
    return SPORT if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_entertainment_terms(x):
    words = ['film', 'music', 'awards', 'festival', 'album', 'star', 'bbc', 'films', 'oscar', 'chart',
                'actress', 'singer', 'comedy', 'series', 'box', 'song', 'rock', 'stars', 'movie',
                'musical', 'nominations', 'nominated']
    return ENTERTAINMENT if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_politics_terms(x):
    words = ['labour', 'blair', 'election', 'party', 'brown', 'government', 'howard', 'tax', 'minister',
            'chancellor', 'lord', 'prime', 'tory', 'public', 'tories', 'lib', 'britain', 'campaign', 'leader',
            'secretary', 'police', 'eu', 'kennedy', 'law', 'vote', 'tony', 'lords', 'blunkett', 'ukip', 'immigration',
            'budget', 'council', 'country', 'michael', 'spokesman']
    return POLITICS if any(t in x.text.lower() for t in words) else ABSTAIN

# labeling functions for topic modelling results 

# not to mention concordance or other sentence structures 

# and other clustering options 
# knn model option
@labeling_function()
def lf_knn_pred_tech(x):
    vecs = bbc_vectorizer.transform(x)
    return TECH if bbc_knn.predict(vecs) == 0 else ABSTAIN

@labeling_function()
def lf_knn_pred_business(x):
    vecs = bbc_vectorizer.transform(x)
    return BUSINESS if bbc_knn.predict(vecs) == 1 else ABSTAIN

@labeling_function()
def lf_knn_pred_sport(x):
    vecs = bbc_vectorizer.transform(x)
    return SPORT if bbc_knn.predict(vecs) == 2 else ABSTAIN

@labeling_function()
def lf_knn_pred_entertainment(x):
    vecs = bbc_vectorizer.transform(x)
    return ENTERTAINMENT if bbc_knn.predict(vecs) == 3 else ABSTAIN

@labeling_function()
def lf_knn_pred_politics(x):
    vecs = bbc_vectorizer.transform(x)
    return POLITICS if bbc_knn.predict(vecs) == 4 else ABSTAIN
    