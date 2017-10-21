import os
import time
import string
import pickle
import pandas as pd

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


def timeit(func):
   
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        delta  = time.time() - start
        return result, delta
    return wrapper

def identity(arg):
    
    return arg


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        
        return self

    def inverse_transform(self, X):
        
        return X

    def transform(self, X):
        
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        
        for sent in sent_tokenize(document):
            
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)



@timeit
def build_and_evaluate(X, y, classifier=LinearSVC, outpath=None, verbose=True):

    @timeit
    def build(classifier, X, y=None):
        
        if isinstance(classifier, type):
            classifier = classifier(C=0.7)

        model = Pipeline([
            ('preprocessor', NLTKPreprocessor()),
            ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
            ('classifier', classifier),
        ])

        model=model.fit(X, y)
        return model
    labels = LabelEncoder()
    y = labels.fit_transform(y)
    if verbose: print("Building for evaluation")

    if verbose: print("Building complete model and saving ...")
    model, secs = build(classifier, X, y)
    model.labels_ = labels

    if verbose: print("Complete model fit in {:0.3f} seconds".format(secs))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model
os.chdir("/home/prajjwal/Downloads/lang")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
add=train[train.label==1]
train=train.append(add)
train=train.append(add)
train=train.append(add)

X=train.tweet
y=train.label
y=y.astype('str')
xtest=test.tweet
model=build_and_evaluate(X,y,outpath="model")
with open("model", 'rb') as f:
    model = pickle.load(f)
yhat=model.predict(xtest)
    
sub=pd.DataFrame({'id':test.id,"label":yhat})
sub.to_csv("submit1.csv",index=False)