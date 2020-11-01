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

kmeans = joblib.load('bbc_kmeans.joblib')
vectorizer = joblib.load('tfidf.joblib')

bbc_stop_words = ['said', 'people', 'new', 'mr']
custom_stop_words = text.ENGLISH_STOP_WORDS.union(bbc_stop_words)

@labeling_function()
def lf_contains_tech_terms(x):
    words = ['mobile phone', 'high definition','anti virus', 'bbc news', 'wi fi', 'mobile phones', 'digital music',
             'mac mini', 'peer peer', 'high speed', 'ask jeeves', 'file sharing', 'open source', 'consumer electronics',
             'music players', 'blu ray', 'search engine', 'camera phone', 'mp3 players', 'ms simonetti', 'hard drive',
             'half life', 'legal action', 'digital cameras', 'vice president', 'mail messages', 'net users', 'desktop search',
             'security firm','chief executive', 'hi tech', 'video games', 'search engines', 'video games', 'internet explorer',
             'hip hop', 'anti spyware', 'san andreas', 'windows xp', 'pc pro', 'european parliment', 'anti spam', 'sony psp',
             'media player', 'junk mail', 'battery life', 'computer users', 'jupiter research', 'internet access', 'mobile tv']
    return TECH if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_business_terms(x):
    words = ['chief executive', 'deutsche boerse', 'economic growth', 'stock market', 'oil prices', 'fourth quarter', 'stock exchange',
            'consumer spending', 'bank england', 'sri lanka', 'house prices', 'told reuters', 'housing market', 'news agency', '000 jobs',
            'central bank', 'deutsche bank', 'told bbc', 'south korea', 'long term', 'prime minister', 'world biggest',
            'manchester united', 'retail sales', 'bankruptcy protection', 'state owned', 'chief economist', 'wal mart', 'federal reserve',
            'european union', 'wall street', 'president bush', 'previous year', 'russian government', 'united states', 'low cost', 'financial times',
            'south africa', 'fannie mae', 'domestic demand', 'news corp', 'securities exchange', 'exchange commission', 'foreign firms',
            'job creation', 'london stock', 'budget deficit', 'car maker', 'saudi arabia', 'general motors', 'finance minister', 'pre tax']
    return BUSINESS if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_sport_terms(x):
    words = ['champions league', 'world cup', 'australian open', 'grand slam', 'told bbc', 'davis cup', 'manchester united', 'cross country',
            'bbc sport', 'world number', 'second half', 'real madrid', 'half time', 'fly half', 'south africa', 'cup final', 'french open', 
            'fa cup', 'lewis francis', 'european indoor', 'andy robinson', 'second set', 'coach andy', 'olympic champion', 'world record', 'carling cup',
            'subs used', 'drugs test', 'little bit', 'international rugby', 'open champion', 'england coach', 'anti doping', 'uefa cup', 'andy roddick',
            'lleyton hewitt', 'scrum half', 'world indoor', 'indoor championships', 'alex ferguson', 'world cross', 'free kick', 'stade touloussain', 
            'roger federer', 'jose mourinho', 'rbs nations', 'long term', 'grand prix', 'sir alex']
    return SPORT if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_entertainment_terms(x):
    words = ['box office', 'million dollar', 'dollar baby', 'los angeles', 'best film', 'film festival', 'named best', 'won best', 'vera drake', 
            'best director', '50 cent', 'best actress', 'best supporting', 'best actor', 'big brother', 'band aid', 'ray charles', 'singles chart',
            'super bowl', 'hip hop', 'martin scorsese', 'meet fockers', 'academy awards', 'harry potter', 'franz ferdinand', 'jamie foxx', 'ticket sales',
            'nominated best', 'imelda staunton', 'golden globe', 'spider man', 'mike leigh', 'da vinci', 'win best', 'music industry', 'clint eastwood',
            'award best', 'clive owen', 'rock band', 'berlin film', 'william hit', 'green day', 'alicia keys', 'west end', 'best british', 'second place',
            'oscar winning', 'finding neverland', 'uk film', 'joss stone']
    return ENTERTAINMENT if any(t in x.text.lower() for t in words) else ABSTAIN

@labeling_function()
def lf_contains_politics_terms(x):
    words = ['prime minister', 'general election', 'tony blair', 'kilroy silk', 'told bbc', 'michael howard', 'human rights', 'lib dems', 'home secretary',
            'gordon brown', 'liberal democrats', 'lib dem', 'labour party', 'council tax', 'id cards', 'tory leader', 'bbc radio', 'election campaign', 
            'foreign secretary', 'liberal democrat', 'leader michael', 'bbc news', 'public services', 'house lords', 'downing street', 'home office', 
            'charles kennedy', 'today programme', 'radio today', 'lord chancellor', 'local government', 'lord goldsmith', 'iraq war', 'political parties',
            'house arrest', 'law lords', 'income tax', 'david blunkett', 'tax cuts', 'jack straw', 'public sector', 'leader charles', 'lord falconer',
            'conservative party', 'alan milburn', 'england wales', 'pre election', 'blair told', 'labour election', 'terror suspects', 'oliver letwin',
            'attorney general', 'chancellor gordon']
    return POLITICS if any(t in x.text.lower() for t in words) else ABSTAIN

# labeling functions for topic modelling results 

# not to mention concordance or other sentence structures 

# and other clustering options 

@labeling_function()
def lf_kmeans_label_0(x):
    vecs = vectorizer.transform(x)
    predicted_cluster = kmeans.predict(vecs)
    return SPORT if predicted_cluster == 0 else ABSTAIN

@labeling_function()
def lf_kmeans_label_1(x):
    vecs = vectorizer.transform(x)
    predicted_cluster = kmeans.predict(vecs)
    return TECH if predicted_cluster == 1 else ABSTAIN

@labeling_function()
def lf_kmeans_label_2(x):
    vecs = vectorizer.transform(x)
    predicted_cluster = kmeans.predict(vecs)
    return ENTERTAINMENT if predicted_cluster == 2 else ABSTAIN

@labeling_function()
def lf_kmeans_label_3(x):
    vecs = vectorizer.transform(x)
    predicted_cluster = kmeans.predict(vecs)
    return BUSINESS if predicted_cluster == 3 else ABSTAIN


@labeling_function()
def lf_kmeans_label_4(x):
    vecs = vectorizer.transform(x)
    predicted_cluster = kmeans.predict(vecs)
    return POLITICS if predicted_cluster == 4 else ABSTAIN

    