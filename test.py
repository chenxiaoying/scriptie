#!/usr/bin/python3

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, scale
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score
from personality import read_corpus, network_prop, read_training_data_s, pos_neg_words
from personality import PosNeg, TextStats, Pronouns, Punt, ExtraWord, Adj, Afi, H4Lvd

from personality import read_test_data_s

### Postagger: http://nlpforhackers.io/training-pos-tagger/
### Early studie: https://www.gsb.stanford.edu/sites/gsb/files/conf-presentations/miningfacebook.pdf
### http://pbpython.com/categorical-encoding.html
### http://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
### Graph decisiontree: http://webgraphviz.com/


from nltk import word_tokenize, pos_tag

class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))
    
def classify(dataset):
	X = dataset['text'].values
	#colum = ['text','label','features']
	#cols = [col for col in dataset if col not in colum]
	#featur = dataset.ix[:,cols]
	data = dataset[['text','netw_size','betw','norm_betw','brok','norm_brok']]
	y = dataset['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=2)
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	print('Start Classfy.....')
	classifier = Pipeline([
				('features', FeatureUnion([
				('statn', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',vec),
				('tf',TfidfTransformer())
				])),
				('pos_neg', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Adj()),
				('ar', DictVectorizer())
				])),
				('text', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',TextStats()),
				('ar', DictVectorizer())
				])),
				('prop', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Afi()),
				('ar', DictVectorizer())
				])),
				('pun', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Punt()),
				('ar', DictVectorizer())
				])),
				('ex', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',ExtraWord()),
				('ar', DictVectorizer())
				])),
				('pro', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Pronouns()),
				('ar', DictVectorizer())
				])),
				('adj', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',PosNeg()),
				('ar', DictVectorizer())
				])),
				('h4', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',H4Lvd()),
				('ar', DictVectorizer())
				])),
				('brok', Pipeline([
				('sele', FunctionTransformer(lambda x: x['brok'], validate=False)),
				('con',ArrayCaster())
				])),
				('netw_size', Pipeline([
				('sele', FunctionTransformer(lambda x: x['netw_size'], validate=False)),
				('con',ArrayCaster())
				])),
				('betw', Pipeline([
				('sele', FunctionTransformer(lambda x: x['betw'], validate=False)),
				('con',ArrayCaster())
				])),
				('norm_betw', Pipeline([
				('sele', FunctionTransformer(lambda x: x['norm_betw'], validate=False)),
				('con',ArrayCaster())
				])),
				])),
				('clf', LinearSVC())
				])
	classifier.fit(X_train,y_train)
	predict = classifier.predict(X_test)
	print('accuracy: ',metrics.accuracy_score(y_test, predict))
	print(metrics.confusion_matrix(y_test, predict))
	labell = ['extra','intro']
	print(metrics.classification_report(y_test,predict, target_names=labell))
	tree.export_graphviz(classifier.named_steps['clf'],out_file='tree.dot')

def cla_ling(dataset):
	X = dataset['text'].values
	data = dataset[['text','netw_size','betw','norm_betw','brok','norm_brok']]
	y = dataset['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=2)
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	print('Start Classfy.....')
	classifier = Pipeline([
				('features', FeatureUnion([
				('statn', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',vec),
				('tf',TfidfTransformer())
				])),
				('netw_size', Pipeline([
				('sele', FunctionTransformer(lambda x: x['netw_size'], validate=False)),
				('con',ArrayCaster())
				])),
				('Textlength', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats', TextStats()),
				('vect', DictVectorizer())
				])),
				])),
				('clf', KNeighborsClassifier())
				])
	classifier.fit(X_train,y_train)
	predict = classifier.predict(X_test)
	print('accuracy: ',metrics.accuracy_score(y_test, predict))
	print(metrics.confusion_matrix(y_test, predict))
	labell = ['extra','intro']
	print(metrics.classification_report(y_test,predict, target_names=labell))

def clasi_test(dataset, test_data):
	X_train = dataset[['text','netw_size','betw','norm_betw','brok','norm_brok']]
	y_train = dataset['label'].map({'extra':0,'intro':1})
	X_test = test_data[['text','netw_size','betw','norm_betw','brok','norm_brok']]
	y_test = test_data['label'].map({'extra':0, 'intro': 1})
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	print('Start Classfy.....')
	classifier = Pipeline([
				('features', FeatureUnion([
				('statn', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',vec),
				('tf',TfidfTransformer())
				])),
				('pos_neg', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Adj()),
				('ar', DictVectorizer())
				])),
				('text', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',TextStats()),
				('ar', DictVectorizer())
				])),
				('prop', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Afi()),
				('ar', DictVectorizer())
				])),
				('pun', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Punt()),
				('ar', DictVectorizer())
				])),
				('ex', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',ExtraWord()),
				('ar', DictVectorizer())
				])),
				('pro', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',Pronouns()),
				('ar', DictVectorizer())
				])),
				('adj', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',PosNeg()),
				('ar', DictVectorizer())
				])),
				('h4', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('stats',H4Lvd()),
				('ar', DictVectorizer())
				])),
				('brok', Pipeline([
				('sele', FunctionTransformer(lambda x: x['brok'], validate=False)),
				('con',ArrayCaster())
				])),
				('netw_size', Pipeline([
				('sele', FunctionTransformer(lambda x: x['netw_size'], validate=False)),
				('con',ArrayCaster())
				])),
				('betw', Pipeline([
				('sele', FunctionTransformer(lambda x: x['betw'], validate=False)),
				('con',ArrayCaster())
				])),
				('norm_betw', Pipeline([
				('sele', FunctionTransformer(lambda x: x['norm_betw'], validate=False)),
				('con',ArrayCaster())
				])),
				])),
				('clf', GaussianNB())
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
	#classify(dataset)
	#cla_ling(dataset)
	
	### For final test with test set
	test_set, with_id_test = read_test_data_s()
	test_data = network_prop(test_set, with_id_test)
	clasi_test(dataset, test_data)
	

if __name__ == "__main__":
	main()
