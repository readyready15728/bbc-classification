import collections
import os
import pickle
import random
import sys
from featx import high_information_words
from nltk.classify import MaxentClassifier, NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
from nltk.metrics.scores import accuracy
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import word_tokenize

random.seed(42)

basedir = 'news'
stopwords = set(stopwords.words('english'))
labeled_featuresets = []
# This is going to be used later for extracting informative words
words_by_category = collections.defaultdict(set)

for category in os.listdir(basedir):
    for item in os.listdir(os.path.join(basedir, category)):
        with open(os.path.join(basedir, category, item)) as f:
            raw = f.read()
            # Lowercase all words because doing otherwise probably doesn't
            # reflect too many meaningful distinctions and also because 
            # stopwords are all lowercase. Additionally get rid of all really
            # short words that would remain even after stopword removal.
            tokens = set(word.lower() for word in word_tokenize(raw) if len(word) > 3)
            tokens = tokens.difference(stopwords)
            words_by_category[category] |= tokens

            labeled_featuresets.append((dict((token, True) for token in tokens), category))

# Get rid of relatively uninformative words
informative_words = high_information_words(words_by_category.items(), min_score=1)

for featureset, _ in labeled_featuresets:
    # This is being done because there would be an error about dictionary
    # size changing during iteration otherwise
    for key in list(featureset.keys()):
        if key not in informative_words:
            featureset.pop(key)

# Make sure featuresets are well mixed
random.shuffle(labeled_featuresets)
train_test_split = int(len(labeled_featuresets) * 0.8)
train_set = labeled_featuresets[:train_test_split]
test_set = labeled_featuresets[train_test_split:]

if os.path.isfile('fit'):
    with open('fit', 'rb') as f:
        fit = pickle.load(f)
else:
    fit = NaiveBayesClassifier.train(train_set)
    # fit = MaxentClassifier.train(train_set, algorithm='megam')

    with open('fit', 'wb') as f:
        pickle.dump(fit, f)

actual = [label for (_, label) in test_set]
prediction = [fit.classify(featureset) for (featureset, _) in test_set]

print('Accuracy: %f' % accuracy(actual, prediction))
print('Confusion Matrix:')
print(ConfusionMatrix(actual, prediction))

