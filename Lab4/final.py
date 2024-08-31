from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.datasets import load_files
from pprint import pprint
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


moviedir = r'/Users/klarabratteby/Desktop/Skola/TNM108/movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)

len(movie.data)
print(len(movie.data))

# target names ("classes") are automatically generated from subfolder names
movie.target_names
print(movie.target_names)

# Split data into training and test sets
docs_train, docs_test, y_train, y_test = train_test_split(
    movie.data, movie.target, test_size=0.5, random_state=12)

# Building a pipeline
'''
# Bayes
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# Train model
text_clf.fit(docs_train, y_train)

predicted = text_clf.predict(docs_test)
print("multinomialBC accuracy ", np.mean(predicted == y_train))

print(metrics.classification_report(y_train, predicted,
                                    target_names=movie.target_names))

# Confusion matrix
print(metrics.confusion_matrix(y_train, predicted))
'''
# SVM
text_clf = Pipeline([
    # Convert text to the number of times indiviudal words or tokens appear
    ('vect', CountVectorizer()),
    #
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
     alpha=1e-6, random_state=42, max_iter=5, tol=None)),
])
# Train model
text_clf.fit(docs_train, y_train)

predicted = text_clf.predict(docs_test)
print("SVM accuracy ", np.mean(predicted == y_train))

print(metrics.classification_report(y_train, predicted,
                                    target_names=movie.target_names))

# Confusion matrix
print(metrics.confusion_matrix(y_train, predicted))

# Parameter tuning using grid search
parameters = {
    # both unigrams and bigrams will be considered
    'vect__ngram_range': [(1, 1), (1, 2)],
    # when to use IDF in tf-idf
    'tfidf__use_idf': (False, True),
    # controlls regularization, helps prevent overfitting, lower value on alfa -> better score
    'clf__alpha': (1e-5, 1e-6),
    # 'clf__random_state': (0, 42),
    # 'clf__max_iter': (0, 5),
}

# Detect how many cores are installed and use them all
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

# Smaller subset of the training data
gs_clf = gs_clf.fit(docs_train[:600], y_train[:600])

# Trying the classifier on fake movie reviews

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

# Cross-validation score
print('\nBest score:', gs_clf.best_score_, '\n')
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# have classifier make a prediction
pred = gs_clf.predict(reviews_new)

# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))
