#!/usr/bin/python3
import csv
import nltk
from random import shuffle
import pickle

import collections, itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import statistics
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import words



""" Prepare training and test data"""

def read_corpus():
	userinfo = {}
	statusAll = {}
	with open('mypersonality_final.csv', encoding='utf-8', errors='ignore') as data_file:
		readfile = csv.DictReader(data_file)
		for line in readfile:
			tokens = nltk.word_tokenize(line['STATUS'])
			try:
				statusAll[line['#AUTHID']].append(tokens)
			except KeyError:
				statusAll[line['#AUTHID']] = [tokens]
				
			userinfo[line['#AUTHID']] = ([line['NETWORKSIZE'],line['BETWEENNESS'],line['NBETWEENNESS'],line['BROKERAGE'],line['NBROKERAGE']],
									line['sEXT'],line['sNEU'],line['sAGR'],line['sOPN'],line['sCON'],
									line['cEXT'],line['cNEU'],line['cAGR'],line['cOPN'],line['cCON'])

	
	return statusAll,userinfo
	

""" Calculate data distribution
	return list with userid wich scores for ext, agr all higher than 3, 
	in total 128 users seen as extraversion """
def calc(userinfo):
	tot = 0
	t = 0
	extagr = 0
	d = []
	for i in userinfo:
		if userinfo[i][6] == 'y':
			tot = tot + 1
		if float(userinfo[i][1]) > 3 and float(userinfo[i][3]) > 3:
			extagr = extagr + 1
			d.append(i)
		if userinfo[i][6] == 'n':
			t = t + 1
			
	print("Total yes: {}\nTotal no: {}\nTotal users with scores higher than 3: {}".format(tot,t,extagr))
	
	s_Ex = []
	s_Ag = []
	s_t = []
	c = 0
	for i in userinfo:
		s_Ex.append(float(userinfo[i][1]))
		s_Ag.append(float(userinfo[i][3]))
		gem = (float(userinfo[i][1]) + float(userinfo[i][2]) + float(userinfo[i][3]) + float(userinfo[i][4]) + float(userinfo[i][5])) / 5
		s_t.append(gem)
		if float(userinfo[i][3]) < 2:
			c += 1
	print('c',c)
	
	score_Ex = sum(s_Ex) / len(s_Ex)
	score_Ag = sum(s_Ag) / len(s_Ag)
	score_T = sum(s_t) / len(s_t)
	print('Max extraversion: ',max(s_Ex))
	print('Max agreeableness: ',max(s_Ag))
	print('Max gemiddeld scores: ',max(s_t))
	print('Gemiddeld score extraversion: {}\nGemiddeld score Agreeableness: {}\nGemiddeld score alle personaliteit: {}\n'.format(score_Ex, score_Ag, score_T))
	return d
	

""" Split data into extraversion and introversion """
def splitdata(d,statusAll):
	totExtra = 0
	totIntro = 0
	labeld_data = {}
	for i in statusAll:
		if i in d:
			totExtra = totExtra + 1
			labeld_data[i] = (statusAll[i],'extra')
		else:
			totIntro = totIntro + 1
			labeld_data[i] = (statusAll[i],'intro')
				
	print("Total Extraversion User: ", totExtra)
	print("Total Introversion User: ", totIntro)
	
	return labeld_data
	
def calc_status(labeld_data):
	t_s = []
	total = 0
	st_p_user_in = []
	st_p_user_ex = []
	for users in labeld_data:
		st_p_user = len(labeld_data[users][0])
		t_s.append(st_p_user)
		if labeld_data [users][1] == 'intro':
			st_p_user_in.append(len(labeld_data[users][0]))
		else:
			st_p_user_ex.append(len(labeld_data[users][0]))
		
	for i in t_s:
		total = total + i
	aantal_user = len(t_s)
	gemiddeld = total / aantal_user
	
	intro = sum(st_p_user_in)
	average_in = intro / len(st_p_user_in)
	extra = sum(st_p_user_ex)
	average_ex = extra / len(st_p_user_ex)

	### Calculate status distribution
	c = 0
	for i in st_p_user_ex:
		if i < 30:
			c +=1
	print(c)
	
	print('Aantal users: ', aantal_user)
	print('Total status all users: ', total)
	print('Average status per user: ',gemiddeld)
	print('Total extroverts status: ', extra)
	print('Total introverts status: ', intro)
	print('Average status introverts: ',average_in)
	print('Average status extroverts: ',average_ex)

			
		
"""code comes from information retrieval assignment 1"""
# splits a labelled dataset into two disjoint subsets train and test
def split_train_test(labeld_data, split=0.76):
	train_feats = []
	test_feats = []

	feats = []
	for i in labeld_data:
		feats.append((i,labeld_data[i]))
		
	shuffle(feats) # randomise dataset before splitting into train and test
	cutoff = int(len(feats) * split)
	train_feats, test_feats = feats[:cutoff], feats[cutoff:]    
	
	print("\n##### Splitting datasets...")
	print("  Training set: %i" % len(train_feats))
	print("  Test set: %i" % len(test_feats))
	
	with open('training.pickle','wb') as train_file:
		pickle.dump(train_feats,train_file)
	with open('test.pickle','wb') as test_file:
		pickle.dump(test_feats,test_file)
			
	print('Data already write to file!')
	
	return train_feats, test_feats


#############################################################
""" Prepare for training and test data """
def prepare_data():
	### read from original dataset
	statusAll,userinfo = read_corpus()
	### calculate the distribution of dataset
	d = calc(userinfo)
	### split dataset into introversion and extraversion
	labeld_data = splitdata(d, statusAll)
	### Calculate status
	calc_status(labeld_data)
	### split into training data and test data
	#split_train_test(labeld_data)

##############################################################

""" Ready for classifier """

###### dataset for nltk classifier
def read_training_data_nltk():
	training_set = []
	training_data = pickle.load(open('training.pickle','rb'))
	count = 0
	for lines in training_data:
		count = count + 1
		for line in lines[1][0]:
			feat = dict([(word, True) for word in line])
			training_set.append((feat,lines[1][1]))
		
	print("Total training data loaded: ",count)

	return training_set
	
## dataset for scikit
def read_training_data_s():
	training_set = []
	labels = []
	training_data = pickle.load(open('training.pickle','rb'))
	count = 0
	data_with_labels = []
	for lines in training_data:
		for line in lines[1][0]:
			count = count + 1
			training_set.append(line)
			labels.append(lines[1][1])
			data_with_labels.append((line,lines[1][1]))
			
	print("Total training data loaded: ",count,"\n")

	return training_set, labels, data_with_labels

""" Dummy classifier as baseiline
	Accuracy: 0.5487 """
def baseline(training_set, labels):
	
	# split corpus in train and test
	X = np.array(training_set)
	Y = np.array(labels)
	split_point = int(0.75*len(X))
	Xtrain = X[:split_point]
	Ytrain = Y[:split_point]
	Xtest = X[split_point:]
	Ytest = Y[split_point:]
	
	# dummy classifier
	dummy_classifier = DummyClassifier(strategy='most_frequent',random_state=0)
	dummy_classifier.fit(Xtrain,Ytrain)
	guess = []
	for i in Xtest:
		Yguess = dummy_classifier.predict([i])
		guess.append(Yguess)
		
	# evaluate
	c = 0
	for i in guess:
		if i == ['extra']:
			c += 1
			
	print("Total tested in baseline                : ",len(Ytest))
	print("Total extraversion predicted in baseline: ",c,"\n")
	print("Accuracy baseline                       : ",accuracy_score(Ytest, guess),"\n")
	

###############################################################
""" Classes for features extraction """
class TextStats(BaseEstimator, TransformerMixin):
	"""Extract features from each document for DictVectorizer"""

	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		return [{'length': len(text),
				 'num_sentences': text.count('.')}
				for text in posts]

class PosNeg(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, posts):
		score = []
		for lines in posts:
			for word in lines:
				pos_s = 0
				neg_s = 0
				if word in words.words():
					print(word)
					word_list = list(swn.senti_synsets(word))
					if word_list != []:
						word0 = word_list[0]
						print(word0.pos_score())
						if word0.pos_score() > 0.5:
							pos_s = word0.pos_score() + pos_s
						else:
							neg_s = word0.neg_score() + neg_s
			score.append({'pos':pos_s, 'neg': neg_s})
			print(score)
		return score
				
		

################################################################

def pos_neg_words():
	pos_wordlist = []
	neg_wordlist = []
	with open('positive-words_1.txt') as pos_file:
		for lines in pos_file:
			pos_wordlist.append(lines.strip())
	with open('negative-words_1.txt', encoding='utf-8', errors='ignore') as neg_file:
		for lines in neg_file:
			neg_wordlist.append(lines.strip())
	
	return pos_wordlist, neg_wordlist




""" Extract linguistic features """ 
def classify(training_set,labels):
	
	# split corpus in train and test
	X = np.array(training_set)
	Y = np.array(labels)
	split_point = int(0.75*len(X))
	Xtrain = X[:split_point]
	Ytrain = Y[:split_point]
	Xtest = X[split_point:]
	Ytest = Y[split_point:]
	print('Training set: ', len(Xtrain))
	print('Test set: ', len(Xtest))
	# combine the vectorizer with a Naive Bayes classifier
	classifier = Pipeline([
				('features', FeatureUnion([
				('posneg', PosNeg()),
				('stats', TextStats()),
				('vect', DictVectorizer())
				])),
				('cls', MultinomialNB())
				])
	# train the classifier
	classifier.fit(Xtrain, Ytrain)
	# test
	Yguess = classifier.predict(Xtest)
	# evaluate
	print(accuracy_score(Ytest, Yguess))


""" Split data in training and development set randomly """
def split_train_dev(data_with_labels,split=0.9):
	train_feats = []
	test_feats = []

	shuffle(data_with_labels) # randomise dataset before splitting into train and test
	cutoff = int(len(data_with_labels) * split)
	train_feats, test_feats = data_with_labels[:cutoff], data_with_labels[cutoff:]  
	
	print("\n##### Splitting datasets...")
	print("  Training set: %i" % len(train_feats))
	print("  Develop set: %i" % len(test_feats))
	return train_feats, test_feats

## Main program
def main():
	score = ['extra','intro']
	
	training_set,labels, data_with_labels = read_training_data_s()
	### Calculate baseline
	#baseline(training_set,labels)
	
	#train_feats,test_feats = split_train_dev(data_with_labels)
	classify(training_set,labels)
	
	#classifier = train(train_feats)
	#evaluation(classifier,test_feats,score)
	


if __name__ == "__main__":
	#main()
	pos_neg_words()
	#prepare_data()
