#!/usr/bin/python3

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

from personality import read_corpus, network_prop, read_training_data_s, pos_neg_words



### Postagger: http://nlpforhackers.io/training-pos-tagger/
### Early studie: https://www.gsb.stanford.edu/sites/gsb/files/conf-presentations/miningfacebook.pdf
### http://pbpython.com/categorical-encoding.html
### http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

from nltk import word_tokenize, pos_tag
def pos_tagger(X):
	pos_list = []
	for i in X:
		pos_list.append(pos_tag(word_tokenize(i)))
	
	return pos_list

def features(sentence, index):
	pos_wordlist, neg_wordlist = pos_neg_words()
	""" sentence: [w1, w2, ...], index: the index of the word """
	return {
		'word': sentence[index],
		'is_capitalized': sentence[index][0].upper() == sentence[index][0],
		'is_all_caps': sentence[index].upper() == sentence[index],
		'is_numeric': sentence[index].isdigit(),
		'is_pos': sentence[index] in pos_wordlist,
		'is_neg': sentence[index] in neg_wordlist}
	
def untag(tagged_sentence):
	return [w for w, t in tagged_sentence]

def transform_to_dataset(tagged_sentence):
	X, y = [],[]
	for tagged in tagged_sentence:
		for index in range(len(tagged)):
			X.append(features(untag(tagged), index))
			y.append(tagged[index][1])
	
	return X,y

class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))
    
def classify(dataset):
	X = dataset['text'].values
	colum = ['text','label','features']
	cols = [col for col in dataset if col not in colum]
	featur = dataset.ix[:,cols]
	fe = featur.to_dict('records')
	data = dataset[['text','features','netw_size','betw','norm_betw','brok','norm_brok']]
	y = dataset['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=2)
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	classifier = Pipeline([
				('features', FeatureUnion([
				('textt',Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',vec),
				('tf', TfidfTransformer())
				])),
				('netw_size',Pipeline([
				('sele', FunctionTransformer(lambda x: x['netw_size'], validate=False)),
				('ar',ArrayCaster())
				])),
				])),
				('clf', DecisionTreeClassifier())
				])
	classifier.fit(X_train,y_train)
	predict = classifier.predict(X_test)
	print('accuracy: ',metrics.accuracy_score(y_test, predict))
	print(metrics.confusion_matrix(y_test, predict))
	labell = ['extra','intro']
	print(metrics.classification_report(y_test,predict, target_names=labell))

	
def main():
	training_set, labels, data_with_labels, with_id = read_training_data_s()
	dataset = network_prop(training_set, with_id)
	classify(dataset)
	

if __name__ == "__main__":
	main()
