from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from sklearn.metrics import accuracy_score
import pickle

import csv

def read_training_data():
	training_set = []
	labels = []
	training_data = pickle.load(open('training.pickle','rb'))
	count = 0
	for lines in training_data:
		for line in lines[1][0]:
			count = count + 1
			training_set.append(line)
			labels.append(lines[1][1])
		
	print("Total training data loaded: ",count)

	return training_set, labels

	
# a dummy function that just returns its input
def identity(x):
	return x

# split corpus in train and test
X, Y = read_training_data()
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
	vec = TfidfVectorizer(preprocessor = identity,
						  tokenizer = identity)
else:
	vec = CountVectorizer(preprocessor = identity,
						  tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
						('cls', MultinomialNB())] )
# train the classifier
classifier.fit(Xtrain, Ytrain)

# test
Yguess = classifier.predict(Xtest)
# evaluate
print(accuracy_score(Ytest, Yguess))

#################class SentenceLength(TransformerMixin):
	
##########	def transform(self, X, **transform_params):
		##########for status in train_feats:
			############c = status[0].count('!')

