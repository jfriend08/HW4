import sys
from sklearn.datasets import load_files
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

data_folder = sys.argv[1]
dataset = load_files(data_folder, shuffle=False)
print("n_samples: %d" % len(dataset.data))


docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
docs_train_counts = count_vect.fit_transform(docs_train)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
docs_train_tfidf = tfidf_transformer.fit_transform(docs_train_counts)
print "len(docs_train)", len(docs_train)
print "docs_train_counts[0]", docs_train_counts.toarray()
print "docs_train_counts[0]", docs_train_counts[0]

print "docs_train_counts.shape", docs_train_counts.shape
print "docs_train_counts[0].shape", docs_train_counts[0].shape

print "docs_train_tfidf", docs_train_tfidf
print "docs_train_tfidf.shape", docs_train_tfidf.shape

#Training a classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(docs_train_tfidf, y_train)
print "clf", clf

#Predict
import numpy as np
docs_test_counts = count_vect.transform(docs_test)
docs_test_tfidf = tfidf_transformer.transform(docs_test_counts)
predicted = clf.predict(docs_test_tfidf)
print "naive bayes", np.mean(predicted == y_test)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()) ])
text_clf = text_clf.fit(docs_train, y_train)
predicted = text_clf.predict(docs_test)
print "naive bayes pipeline", np.mean(predicted == y_test)


pipeline = Pipeline([('vect', TfidfVectorizer(min_df=3, max_df=0.95)),('clf', LinearSVC(C=1000)),])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
grid_search.fit(docs_train, y_train)
print(grid_search.grid_scores_)
y_predicted = grid_search.predict(docs_test)
print "SVC grid_search", np.mean(y_predicted == y_test)

from sklearn import metrics
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))
