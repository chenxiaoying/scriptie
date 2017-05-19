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
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegressionCV

from personality import read_corpus, network_prop, read_training_data_s, pos_neg

class TextStats(BaseEstimator, TransformerMixin):
	"""Extract features from each document for DictVectorizer"""

	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		return posts
		

def classify(dataset):
	X = dataset['text'].values
	colum = ['text','label','features']
	cols = [col for col in dataset if col not in colum]
	featur = dataset.ix[:,cols]
	fe = featur.to_dict('records')
	data = dataset[['text','features','netw_size','betw','norm_betw','brok','norm_brok']]
	dictt = DictVectorizer()
	featre = dictt.fit_transform(fe)
	y = dataset['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=2)
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	#f = np.hstack((dataset['netw_size'],dataset['betw']))
	#tt = dataset['text'].values
	classifier = Pipeline([
				('features', FeatureUnion([
				('textt',Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',vec),
				('tf', TfidfTransformer())
				])),
				('text',Pipeline([
				('sele', FunctionTransformer(lambda x: x[['netw_size','betw']], validate=False)),
				('imp',Imputer())
				])),
				])),
				('clf', MultinomialNB())
				])
	classifier.fit(X_train,y_train)
	predict = classifier.predict(X_test)
	print('accuracy: ',metrics.accuracy_score(y_test, predict))
	print(metrics.confusion_matrix(y_test, predict))
	labell = ['extra','intro']
	print(metrics.classification_report(y_test,predict, target_names=labell))

def classifying(dataset):
	columns = ['text','label','features','netw_size','betw','norm_betw','brok','norm_brok']
	y = dataset['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.33, random_state=42)
	mapper = DataFrameMapper([
			(['text'], LabelBinarizer()),
			(['features'], LabelBinarizer()),
			(['netw_size'], StandardScaler())
			])

	data = mapper.fit_transform(dataset)
	pipe = Pipeline([
			('features',data),
			('cls', LogisticRegressionCV())
			])
	print(cross_val_score(pipe, y_test, X_test))
	
	
def main():
	training_set, labels, data_with_labels, with_id = read_training_data_s()
	dataset = network_prop(training_set, with_id)
	classify(dataset)
	

if __name__ == "__main__":
	main()
