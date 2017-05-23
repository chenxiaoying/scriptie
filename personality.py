#!/usr/bin/python3
import csv
import nltk
from random import shuffle
import pickle
import nltk.classify

import collections, itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn import metrics
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from classification import precision_recall
from sklearn import svm
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from scipy.sparse import hstack
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from afinn import Afinn
from sklearn.ensemble import AdaBoostClassifier

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
		if float(userinfo[i][1]) < 3 and float(userinfo[i][3]) < 3:
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
	#labeld_data = splitdata(d, statusAll)
	### Calculate status
	#calc_status(labeld_data)
	### split into training data and test data
	#split_train_test(labeld_data)

##############################################################

""" Ready for classifier """

###### dataset for nltk classifier
def read_training_data_nltk():
	training_set = []
	bigr_set = []
	tri_set = []
	training_data = pickle.load(open('training.pickle','rb'))
	count = 0
	for lines in training_data:
		for line in lines[1][0]:
			count = count + 1
			feat = dict([(word, True) for word in line])
			training_set.append((feat,lines[1][1]))
			bigr = list(nltk.bigrams(line))
			bigr_feats = dict([(word[0] + ' ' + word[1], True) for word in bigr])
			bigr_set.append((bigr_feats, lines[1][1]))
			trigr = list(nltk.trigrams(line))
			trigr_feats = dict([(word[0] + ' ' + word[1], True) for word in trigr])
			tri_set.append((trigr_feats, lines[1][1]))
			
	

	print("Total training data loaded: ",count)

	return training_set, bigr_set, tri_set
	
## dataset for scikit
def read_training_data_s():
	training_set = []
	labels = []
	training_data = pickle.load(open('training.pickle','rb'))
	count = 0
	data_with_labels = []
	with_id = []
	status, network = read_corpus()
	for lines in training_data:
		allstatus = []
		for line in lines[1][0]:
			count = count + 1
			training_set.append(line)
			labels.append(lines[1][1])
			for word in line:
				allstatus.append(word)
			if lines[0] in network:
				with_id.append((network[lines[0]][0],line,lines[1][1]))
		data_with_labels.append((allstatus,lines[1][1]))
			
	print("Total training data loaded: ",count,"\n")
	
	return training_set, labels, data_with_labels, with_id

def read_test_data_s():
	test_set = []
	test_data = pickle.load(open('test.pickle','rb'))
	count = 0
	with_id = []
	status, network = read_corpus()
	for lines in test_data:
		for line in lines[1][0]:
			count = count + 1
			test_set.append(line)
			if lines[0] in network:
				with_id.append((network[lines[0]][0],line,lines[1][1]))
			
	print("Total test data loaded: ",count,"\n")
	
	return test_set, with_id
	
	
""" Dummy classifier as baseiline
	Accuracy: 0.5487 with tokenized 0.9split
	Accuracy: 0.43 0.75split"""
def baseline(training_set, labels):

	# split corpus in train and test
	X = np.array(training_set)
	Y = np.array(labels)
	split_point = int(0.9*len(X))
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
	print("Accuracy baseline                       : ",metrics.accuracy_score(Ytest, guess),"\n")
	

###############################################################
""" Extract linguistic features """ 
""" Classes for features extraction """
	
class TextStats(BaseEstimator, TransformerMixin):
	"""Extract features from each document for DictVectorizer"""

	def fit(self, x, y=None):
		return self

	def transform(self, posts):
		return [{'length': len(text.split()),
				 'word_length': np.mean([len(word) for word in text.split()])}
				for text in posts]

def features(sentence, index):
	pos_wordlist, neg_wordlist = pos_neg_words()
	sentence[index] = sentence[index].lower()
	""" sentence: [w1, w2, ...], index: the index of the word """
	return {
		'is_pos': sentence[index] in pos_wordlist,
		'is_neg': sentence[index] in neg_wordlist}
		
class PosNeg(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		feat = []
		for lines in data:
			feature = []
			line = nltk.word_tokenize(lines)
			for index in range(len(line)):
				feature.append(features(line, index))
			pos = 0
			neg = 0
			for diction in feature:
				if diction['is_pos'] == True:
					pos += 1
				if diction['is_neg'] == True:
					neg += 1
			if pos != 0 and neg != 0:
				feat.append({'is_pos':True, 'is_neg': True, 'pos_n': pos, 'neg_n':neg})
			elif pos != 0 and neg == 0:
				feat.append({'is_pos':True, 'is_neg': False, 'pos_n': pos, 'neg_n':neg})
			elif pos == 0 and neg != 0:
				feat.append({'is_pos':False, 'is_neg': True, 'pos_n': pos, 'neg_n':neg})
			else:
				feat.append({'is_pos':False, 'is_neg': False, 'pos_n': pos, 'neg_n':neg})	
		
		return feat
		
class Punt(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		score = []
		punctuation = ['.','!','^_^','<','3',':']
		for lines in data:
			punt = 0
			for word in lines:
				if word in punctuation:
					punt += 1
			score.append({'punt':punt})
		
		return score

class Adj(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		score = []
		for lines in data:
			line = nltk.pos_tag(nltk.word_tokenize(lines))
			adjec, verb, nn, pro = 0, 0, 0, 0
			for word in line:
				if word[1] == 'JJ':
					adjec += 1
				elif 'VB' in word[1]:
					verb += 1
				elif 'NN' in word[1]:
					nn += 1
				elif 'PRP' in word[1]:
					pro += 1
			score.append({'ajec':adjec, 'verb':verb, 'noun':nn, 'prop':pro})
		
		return score


class Afi(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		sco = []
		for lines in data:
			afinn = Afinn(emoticons=True)
			sc = afinn.score(lines)
			sco.append({'afinn':sc})
		
		return sco
		

def h4_list():
	h4 = open('inqdict.txt','r')
	h4l = {}
	for lines in h4:
		line = lines.split()
		word = line[0].lower()
		if '#' in word:
			word = word.split('#')[0]
		try:
			h4l[word].append(line[1:])
		except KeyError:
			h4l[word] = [line[1:]]
	
	for i in h4l:
		new_l = []
		for li in h4l[i]:
			if type(li) is list:
				new_l += li
			else:
				print(li)
				new_l = h4l[i]
		h4l[i] = new_l

	return h4l

class H4Lvd(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		sco = []
		h_list = h4_list()
		for lines in data:
			line = nltk.word_tokenize(lines)
			for word in line:
				pos, neg, nn, ad = 0,0,0,0
				word = word.lower()
				if word in h_list:
					if 'Neg' in h_list[word]:
						neg += 1
					elif 'Pos' in h_list[word]:
						pos += 1
					elif 'Noun' in h_list[word] or 'noun' in h_list[word]:
						nn += 1
					elif 'adj' in h_list[word]:
						ad += 1
			sco.append({'ajec':ad, 'pos':pos, 'noun':nn, 'neg':neg})
						
		return sco
		
class ExtraWord(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		score = []
		extra_wordlist = ['sang','hotel','kissed','shots','golden','dad','girls','restaurant','eve','best',
					'proud','miss','soccer','met','brother','cheers', 'friends','tickets','concern','friday',
					'aka','haha','drinks','Ryan','countless','bar','request','cats','football','checking','excitement',
					'love','kidding','hot','spend','glory','sing','perfect','every','sweet','dance','summer','afternoon',
					'exploring','finishing','early','evening','Reagan','visiting','year','spring','two','through','rest','gray',
					'book','until','hug','blast','chips','greeted','minutes','rest','times','cup','beach','seconds','Olympic',
					'following','dinner','participants','sharing','unusual','particular','lake','seemed','adventure','determined',
					'activity','doctor','toys','infection','box','cherry','strength','require','providing','increased','health',
					'conversation','ways','children','fewer','fascinating']
		for lines in data:
			line = nltk.word_tokenize(lines)
			extra = 0
			for word in line:
				word = word.lower()
				if word in extra_wordlist:
					extra += 1
			score.append({'extra':extra})
		
		return score
		
class Pronouns(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		score = []
		self_referencing = ['I', 'me', 'my','mine']
		social_ref = ['you','she','he', 'we','they','*PROPNAME*','them','his','her','your','anyone','our','everyone']
		for lines in data:
			self_s = 0
			social_s = 0
			line = nltk.word_tokenize(lines)
			for word in line:
				word = word.lower()
				if word in self_referencing:
					self_s += 1
				elif word in social_ref:
					social_s += 1
			score.append({'self':self_s, 'social':social_s})
		
		return score		
		
					
		

################################################################
""" Get list of positive and negative words"""
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

def pos_neg(training_set):
	pos_wordlist, neg_wordlist = pos_neg_words()
	features = []
	punctuation = ['!','!!','!!!','^_^','<','3',':']
	extra_wordlist = ['sang','hotel','kissed','shots','golden','dad','girls','restaurant','eve','best',
					'proud','miss','soccer','met','brother','cheers', 'friends','tickets','concern','friday',
					'aka','haha','drinks','Ryan','countless','bar','request','cats','football','checking','excitement',
					'love','kidding','hot','spend','glory','sing','perfect','every','sweet','dance','summer','afternoon',
					'exploring','finishing','early','evening','Reagan','visiting','year','spring','two','through','rest','gray',
					'book','until','hug','blast','chips','greeted','minutes','rest','times','cup','beach','seconds','Olympic',
					'following','dinner','participants','sharing','unusual','particular','lake','seemed','adventure','determined',
					'activity','doctor','toys','infection','box','cherry','strength','require','providing','increased','health',
					'conversation','ways','children','fewer','fascinating']
	self_referencing = ['I', 'me', 'my']
	social_ref = ['you','she','he', 'we','they','*PROPNAME*']
	for lines in training_set:
		info_f = []
		for word in lines:
			if word in pos_wordlist or word in neg_wordlist:
				info_f.append(word)
			elif word in punctuation or word in extra_wordlist:
				info_f.append(word)
			elif word in self_referencing or word in social_ref:
				info_f.append(word)
		features.append(' '.join(info_f))
		
	return features
		

def network_prop(training_set, with_id):
	net_size, betw, norm_betw, brok, norm_brok, status, label, length = [], [], [], [], [], [], [], []
	features = pos_neg(training_set)
	for lines in with_id:
		length.append(len(lines[1]))
		net_size.append(int(lines[0][0]))
		betw.append(float(lines[0][1]))
		norm_betw.append(float(lines[0][2]))
		brok.append(int(lines[0][3]))
		norm_brok.append(float(lines[0][4]))
		status.append(' '.join(lines[1]))
		label.append(lines[2])
	print('Average per status :',sum(length)/len(length))
	print('Totaal woorden :',sum(length))
	l_10, l_20, l_30, l_40, l_50, l = [], [],[],[],[],[]
	for i in length:
		if i <= 10:
			l_10.append(i)
		elif 10 < i <= 20:
			l_20.append(i)
		elif 20 < i <= 30:
			l_30.append(i)
		elif 30 < i <= 40:
			l_40.append(i)
		elif 40 < i <= 50:
			l_50.append(i)
		else:
			l.append(i)
	
	"""print(max(betw))
	print(max(brok))
	print(max(length))
	print(len(l_10),len(l_20),len(l_30),len(l_40),len(l_50),len(l))"""
	data = pd.DataFrame({'text':status,'label':label, 'features':features, 'netw_size':net_size,'betw':betw,
						'norm_betw':norm_betw,'brok':brok,'norm_brok':norm_brok})

	return data
		

#####################################################################

## Tutorial: http://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/

class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))
    
    
### http://weslack.com/question/1854200000002145488
"""Scikit Classifiers """
def classify(df):
	X = df['text'].values
	Y = df['label'].map({'extra':0,'intro':1})
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
	vec = TfidfVectorizer(min_df=1,ngram_range=(1,4),smooth_idf=False)
	print('Start Classifier.....')
	classifier = Pipeline([
				('features', FeatureUnion([
				('ADJ', Pipeline([
				('stats',Afi()),
				('ar', DictVectorizer())
				])),
				('pr', Pipeline([
				('stats',Punt()),
				('ar', DictVectorizer())
				])),
				('vect',Pipeline([
				('vec',vec),
				('tf', TfidfTransformer())
				])),
				])),
				('clf', AdaBoostClassifier())
				])

	#classifier = MultinomialNB()
	classifier.fit(X_train,y_train)
	predict = classifier.predict(X_test)
	print(metrics.accuracy_score(y_test, predict))
	print(metrics.confusion_matrix(y_test, predict))
	labell = ['extra','intro']
	print(metrics.classification_report(y_test,predict, target_names=labell))

##############################################################################
"""Nltk classifiers"""

def bag_of_words(words):
	'''
	>>> bag_of_words(['the', 'quick', 'brown', 'fox'])
	{'quick': True, 'brown': True, 'the': True, 'fox': True}
	'''
	return dict([(word, True) for word in words])
	
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

def get_info(train_feats):
	c  = 0
	intro = 0
	pos_wordlist, neg_wordlist = pos_neg_words()
	self_ref = ['I', 'me', 'my']
	social_ref = ['you','she','he', 'we','they','*PROPNAME*']
	punctuation = ['!','!!','!!!','^_^','<','3',':']
	extra_wordlist = ['sang','hotel','kissed','shots','golden','dad','girls','restaurant','eve','best',
					'proud','miss','soccer','met','brother','cheers', 'friends','tickets','concern','friday',
					'aka','haha','drinks','Ryan','countless','bar','request','cats','football','checking','excitement',
					'love','kidding','hot','spend','glory','sing','perfect','every','sweet','dance','summer','afternoon',
					'exploring','finishing','early','evening','Reagan','visiting','year','spring','two','through','rest','gray',
					'book','until','hug','blast','chips','greeted','minutes','rest','times','cup','beach','seconds','Olympic',
					'following','dinner','participants','sharing','unusual','particular','lake','seemed','adventure','determined',
					'activity','doctor','toys','infection','box','cherry','strength','require','providing','increased','health',
					'conversation','ways','children','fewer','fascinating']
	for statuses in train_feats:
		for status in statuses:
			status = dict([(word, False) for word in status])
			for word in status:
				word = word.lower()
				if word in pos_wordlist or word in neg_wordlist:
					status[word] = True
				if word in self_ref or word in social_ref:
					status[word] = True
				if word in extra_wordlist:
					status[word] = True
				if word in punctuation:
					status[word] = True
				#if word in pos_wordlist and statuses[1] == 'extra':
				#	c += 1
				#if word in neg_wordlist and statuses[1] == 'intro':
				#	intro += 1
					
	print(c, intro)
	return train_feats
	

def train(train_feats):
	classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)
	#classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_feats)
	#classifier = nltk.classify.DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=300)
	return classifier

# prints accuracy, precision and recall, f-score
def evaluation(classifier, test_feats, score):
	print ("\n##### Evaluation...")
	print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))

def split_label(data_with_labels):
	tr_set = []
	labels_set = []
	for i in data_with_labels:
		tr_set.append(i[0])
		labels_set.append(i[1])
		
	return tr_set, labels_set

			
## Main program
def Scikit_classify():
	score = ['extra','intro']

	training_set,labels, data_with_labels, with_id = read_training_data_s()
	### Calculate baseline
	#baseline(training_set,labels)
	
	### Get network properties
	dataset = network_prop(training_set, with_id)
	#train_set, labels = split_label(data_with_labels)
	
	####### Scikit ##############
	classify(dataset)
	

def nltk_classify():
	score = ['extra','intro']
	#######  NLTK  ##############
	acc = []
	dataset, bigr_set, tri_set = read_training_data_nltk()
	for N in range(5):
		train_feats,test_feats = split_train_dev(bigr_set)
		train_info = get_info(train_feats)
		classifier = train(train_info)
		#evaluation(classifier,test_feats,score)
		accuracy = nltk.classify.accuracy(classifier, test_feats)
		print(accuracy)
		acc.append(accuracy)
	average = sum(acc) / len(acc)
	print('Accuracy after N-fold cross validation: ',average)
	

### nltk bayes without features: 0.649
if __name__ == "__main__":
	Scikit_classify()
	#nltk_classify()
	#prepare_data()
