from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1, d2, d3, d4)

# Term-frequency
vectorizer = CountVectorizer()

print(vectorizer)

# Our own vocabulary
my_stop_words = {"the", "is"}
my_vocabulary = {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
vectorizer = CountVectorizer(
    stop_words=my_stop_words, vocabulary=my_vocabulary)

# Print stop words
print(vectorizer.vocabulary)
print(vectorizer.stop_words)

# Create sparse matrix
smatrix = vectorizer.transform(Z)
print(smatrix)

# Convert sparse matrix with coordinate format into a dense format
matrix = smatrix.todense()
print(matrix)

# Calculate the tf-idf value
tfidf_transformer = TfidfTransformer(norm="l2")
tfidf_transformer.fit(smatrix)

# Print idf values
feature_names = vectorizer.get_feature_names_out()
df_idf = pd.DataFrame(tfidf_transformer.idf_,
                      index=feature_names, columns=["idf_weights"])
# Sort ascending
df_idf.sort_values(by=['idf_weights'])
print(df_idf.sort_values(by=['idf_weights']))

# tf-idf scores
tf_idf_vector = tfidf_transformer.transform(smatrix)

# get tfidf vector for first document
first_document = tf_idf_vector[0]  # first document "The sky is blue."
# print the scores
df = pd.DataFrame(first_document.T.todense(),
                  index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"], ascending=False)
print(df.sort_values(by=["tfidf"], ascending=False))

# Document Similarity

# Transform each document into a count-vectorised form
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

# Cosine similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)

# Calculate the angle  between the first and third documents
# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0][2])
print(math.degrees(angle_in_radians))

# Classifying Text
data = fetch_20newsgroups()
data.target_names
print(data.target_names)

# Consider only four categories, and fetch the training and testing data set
my_categories = ['rec.sport.baseball',
                 'rec.motorcycles', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=my_categories)
test = fetch_20newsgroups(subset='test', categories=my_categories)

print(len(train.data))
print(len(test.data))
print(train.data[9])

# Compute first the sparse tf-idf matrix of the training data set
cv = CountVectorizer()
# Tf matrix
X_train_counts = cv.fit_transform(train.data)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Pass it to the multinomial naive Bayes classifier to create a predictive model
model = MultinomialNB().fit(X_train_tfidf, train.target)

# Apply the model, testing
docs_new = ['Pierangelo is a really good baseball player', 'Maria rides her motorcycle', 'OpenGL on the GPU is fast',
            'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train.target_names[category]))
