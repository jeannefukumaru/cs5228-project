import nltk
from nltk.corpus import opinion_lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # supposedly trained on social media data 
from snorkel.labeling import labeling_function
from snorkel.preprocess import preprocessor
from textblob import TextBlob
import joblib
from sentence_transformers import SentenceTransformer

NEU = 2
POS = 1
NEG = 0
ABSTAIN = -1

pos_list = set(opinion_lexicon.positive())
neg_list = set(opinion_lexicon.negative())

vader = SentimentIntensityAnalyzer()

kmeans = joblib.load('tweets_kmeans.joblib')
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

@labeling_function()
def kmeans_neu(x):
    sentence = model.encode(str(x))
    label = kmeans.predict(sentence)
    return NEU if label == 0 else ABSTAIN

@labeling_function()
def kmeans_pos(x):
    sentence = model.encode(str(x))
    label = kmeans.predict(sentence)
    return POS if label == 1 else ABSTAIN

@labeling_function()
def kmeans_neg(x):
    sentence = model.encode(str(x))
    label = kmeans.predict(sentence)
    return NEG if label == 2 else ABSTAIN

def opinion_lexicon_sentiment(sentence):
    senti = 0 
    words = str(sentence).split(' ')
    for word in words:
        if word in pos_list:
            senti += 1
        elif word in neg_list:
            senti -= 1
    return senti

@labeling_function()
def opinion_lexicon_pos(x):
    senti = opinion_lexicon_sentiment(x)
    return POS if senti > 0 else ABSTAIN

@labeling_function()
def opinion_lexicon_neg(x):
    senti = opinion_lexicon_sentiment(x)
    return NEG if senti < 0 else ABSTAIN

@labeling_function()
def opinion_lexicon_neu(x):
    senti = opinion_lexicon_sentiment(x)
    return NEU if senti ==0 and senti <=1 else ABSTAIN

@labeling_function()
def vader_lexicon_neg(x):
    polarity = vader.polarity_scores(str(x))
    if polarity['neg'] > polarity['pos']:
        return NEG 
    else: return ABSTAIN

@labeling_function()
def vader_lexicon_pos(x):
    polarity = vader.polarity_scores(str(x))
    if polarity['pos'] > polarity['neg']:
        return POS 
    else: return ABSTAIN

@labeling_function()
def vader_lexicon_neu(x):
    polarity = vader.polarity_scores(str(x))
    if polarity['neu'] > polarity['pos'] and polarity['neu'] > polarity['neg']:
        return NEU
    else: return ABSTAIN

# @preprocessor(memoize=True)
# def textblob_sentiment(x):
#     scores = TextBlob(str(x))
#     return scores

@labeling_function()
def textblob_pos(x):
    scores = TextBlob(x.text)
    return NEG if scores.polarity < 0.0 else ABSTAIN

@labeling_function()
def textblob_neg(x):
    scores = TextBlob(x.text)
    return POS if scores.polarity > 0.0 else ABSTAIN

@labeling_function()
def textblob_neu(x):
    scores = TextBlob(x.text)
    return NEU if scores.polarity == 0.0 else ABSTAIN

# keyword lfs based on neg reasons
@labeling_function()
def lf_contains_cancelled(x):
    return NEG if "cancelled" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_contains_delayed(x):
    return NEG if "delayed" in x.text.lower() else ABSTAIN

@labeling_function()
def lf_contains_complaint(x):
    return NEG if "complaint" in x.text.lower() else ABSTAIN

# adding more words after eda of token frequency distributions grouped by sentiment
@labeling_function()
def lf_contains_thank_you(x):
    thanks = ['thank you', 'thanks', 'thank']
    return POS if any(t in x.text.lower() for t in thanks) else ABSTAIN

@labeling_function()
def lf_contains_awesome(x):
    pos_words = ['awesome', 'great', 'good', 'amazing','appreciate', 'helpful']
    return POS if any(t in x.text.lower() for t in pos_words) else ABSTAIN