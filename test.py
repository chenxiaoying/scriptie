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

from personality import read_corpus, network_prop, read_training_data_s, pos_neg

def classify(dataset):
	X = dataset['text'].values
	y = dataset['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	#cc = c.fit_transform(X_train)
	#ccc = c.transform(X_test)
	classifier = Pipeline([
					('features', FeatureUnion([
					('text_pipe', Pipeline([
					('vec',vec),
					('tf', TfidfTransformer())
					])),
					('netw',Pipeline([
					('s_text', FunctionTransformer(dataset['netw_size'], validate=False))
					]))
					])),
					('cls',MultinomialNB())
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
			('cls', MultinomialNB())
			])
	print(cross_val_score(pipe, y_test, X_test))
	
	
def main():
	training_set, labels, data_with_labels, with_id = read_training_data_s()
	dataset = network_prop(training_set, with_id)
	classify(dataset)
	

if __name__ == "__main__":
	main()
