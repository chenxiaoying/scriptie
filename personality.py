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
from sklearn import metrics
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from classification import precision_recall
from sklearn import svm
from sklearn.svm import LinearSVC, LinearSVR, SVC
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
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

""" Prepare training and test data"""

def read_corpus():
	userinfo = {}
	statusAll = {}
	with open('mypersonality_final.csv', encoding='utf-8', errors='ignore') as data_file:
		readfile = csv.DictReader(data_file)
		for line in readfile:
			try:
				statusAll[line['#AUTHID']].append(line['STATUS'])
			except KeyError:
				statusAll[line['#AUTHID']] = [line['STATUS']]
				
			userinfo[line['#AUTHID']] = ([line['NETWORKSIZE'],line['BETWEENNESS'],line['NBETWEENNESS'],line['BROKERAGE'],line['NBROKERAGE'],line['TRANSITIVITY'],line['DENSITY']],
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
	count_ex = 0
	d = []
	for i in userinfo:
		if userinfo[i][6] == 'y' and userinfo[i][8] == 'y':
			tot = tot + 1
		if float(userinfo[i][1]) > 3 and float(userinfo[i][3]) > 3:
			extagr = extagr + 1
			d.append(i)
		if userinfo[i][6] == 'n' and userinfo[i][8] == 'n':
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
	

""" Split data into high sociability and low sociability """
def splitdata(d,statusAll):
	totExtra = 0
	totIntro = 0
	labeld_data = {}
	for i in statusAll:
		if i in d:
			totExtra = totExtra + 1
			labeld_data[i] = (statusAll[i],'high-social')
		else:
			totIntro = totIntro + 1
			labeld_data[i] = (statusAll[i],'low-social')
				
	print("Total High sociability User: ", totExtra)
	print("Total Low sociability User: ", totIntro)
	
	return labeld_data
	
def calc_status(labeld_data):
	t_s = []
	total = 0
	st_p_user_in = []
	st_p_user_ex = []
	for users in labeld_data:
		st_p_user = len(labeld_data[users][0])
		t_s.append(st_p_user)
		if labeld_data [users][1] == 'low-social':
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
	print('Total high sociable status: ', extra)
	print('Total low sociable status: ', intro)
	print('Average status low sociable: ',average_in)
	print('Average status high sociable: ',average_ex)

def whole_data(labeld_data):
	feats = []
	for i in labeld_data:
		feats.append((i,labeld_data[i]))
	
	with open('whole_data.pickle','wb') as d_file:
		pickle.dump(feats, d_file)
	print('{} Data already write to file!'.format(len(feats)))
#############################################################
""" Prepare data """
def prepare_data():
	### read from original dataset
	statusAll,userinfo = read_corpus()
	### calculate the distribution of dataset
	d = calc(userinfo)
	##### split dataset into introversion and extraversion
	labeld_data = splitdata(d, statusAll)
	##### Calculate status
	calc_status(labeld_data)
	#### write data in file
	whole_data(labeld_data)

##############################################################

""" Ready for classifier """
	
## dataset for scikit
def read_training_data_s():
	training_set = []
	labels = []
	training_data = pickle.load(open('whole_data.pickle','rb'))
	count = 0
	data_with_labels = []
	with_id = []
	status, network = read_corpus()
	for lines in training_data:
		for line in lines[1][0]:
			count = count + 1
			training_set.append((line,lines[1][1]))
			labels.append(lines[1][1])
			if lines[0] in network:
				with_id.append((network[lines[0]][0],line,lines[1][1]))
			
	print("Total data loaded: ",count,"\n")
	
	return training_set, with_id
		

def read_whole():
	dataset = pickle.load(open('whole_data.pickle','rb'))
	net_size, betw, norm_betw, brok, norm_brok, sta, label, tr, den = [],[], [], [], [], [], [], [], []
	status, network = read_corpus()
	for users in dataset:
		user_status = ' '.join([line for lines in users[1][0] for line in lines])
		if users[0] in network:
			lines = network[users[0]]
		net_size.append(int(lines[0][0]))
		betw.append(float(lines[0][1]))
		norm_betw.append(float(lines[0][2]))
		brok.append(int(lines[0][3]))
		norm_brok.append(float(lines[0][4]))
		sta.append(user_status)
		label.append(users[1][1])
		tr.append(float(lines[0][5]))
		den.append(float(lines[0][6]))
	data = pd.DataFrame({'text':sta,'label':label,'netw_size':net_size,'betw':betw,
						'norm_betw':norm_betw,'brok':brok,'norm_brok':norm_brok,'trans':tr, 'density':den})

	return data


###############################################################################

""" Prepare data for experiment 1 """
def status(training_set):
	text, label = [],[]
	for i in training_set:
		text.append(i[0])
		label.append(i[1])

	data = pd.DataFrame({'text':text, 'label':label})
	
	return data


""" Get social network properties for experiment 2 """
def network_prop(training_set, with_id):
	net_size, betw, norm_betw, brok, norm_brok, status, label, length, tr, den = [],[],[], [], [], [], [], [], [], []
	for lines in with_id:
		length.append(len(nltk.word_tokenize(lines[1])))
		net_size.append(int(lines[0][0]))
		betw.append(float(lines[0][1]))
		norm_betw.append(float(lines[0][2]))
		brok.append(int(lines[0][3]))
		norm_brok.append(float(lines[0][4]))
		status.append(lines[1])
		label.append(lines[2])
		tr.append(float(lines[0][5]))
		den.append(float(lines[0][6]))
	"""print('Average per status :',sum(length)/len(length))
	print('Totaal woorden :',sum(length))"""
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
	print(max(length))"""
	print(len(l_10),len(l_20),len(l_30),len(l_40),len(l_50),len(l))
	data = pd.DataFrame({'text':status,'label':label,'netw_size':net_size,'betw':betw,
						'norm_betw':norm_betw,'brok':brok,'norm_brok':norm_brok,'trans':tr,'den':den})

	return data
	
##############################################################################
"""Baseline"""

def identity(x):
	return x
""" Dummy classifier as baseiline """
def baseline(df):

	# split corpus in train and test
	X = df['text'].values
	Y = df['label'].map({'high-social':0,'low-social':1})
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.1, random_state=1)
	
	# dummy classifier
	tfidf = True
	if tfidf:
		vec = TfidfVectorizer(tokenizer = tok)
	else:
		vec = CountVectorizer(tokenizer = tok)
				
	dummy_classifier = DummyClassifier(random_state=0)
	pipeline = Pipeline([('vec',vec),
						('clf', dummy_classifier)
						])
	pipeline.fit(Xtrain,Ytrain)
	predict = pipeline.predict(Xtest)
	print('accuracy: ',metrics.accuracy_score(Ytest, predict))
	print(metrics.confusion_matrix(Ytest, predict))
	labell = ['extra','intro']
	print(metrics.classification_report(Ytest,predict, target_names=labell))

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
		
class PosNeg(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		feat = []
		pos_wordlist, neg_wordlist = pos_neg_words()
		for lines in data:
			line = nltk.word_tokenize(lines)
			pos, neg = 0, 0
			for word in line:
				if word in pos_wordlist:
					pos += 1
				if word in neg_wordlist:
					neg += 1
			feat.append({'is_pos': pos, 'is_neg': neg})
		return feat
		
class Punt(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		score = []
		punctuation = ['.','!','^_^','-','_','^','<','3',':',',',')',';','"','>','@','[',']','{','}','/','&']
		for lines in data:
			punt = 0
			for word in lines:
				if word in punctuation:
					punt += 1
			score.append({'punt':punt})
		
		return score

class PosTag(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		score = []
		for lines in data:
			line = nltk.pos_tag(nltk.word_tokenize(lines))
			score.append({word[0]:word[1] for word in line})
		
		return score


class Afi(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		sco = []
		for lines in data:
			afinn = Afinn(emoticons=True)
			sc = afinn.score(lines)
			if sc > 0:
				sco.append({'afinn':True})
			else:
				sco.append({'afinn:':False})
		
		return sco
		
class H4Lvd(BaseEstimator, TransformerMixin):
	
	def fit(self, x, y=None):
		return self
		
	def transform(self, data):
		sco = []
		h_list = h4_list()
		for lines in data:
			line = nltk.word_tokenize(lines)
			pos, neg, nn, ad, ac, ps, inc, dec, qua, so = 0,0,0,0,0,0,0,0,0,0
			for word in line:
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
					elif 'Actv' in h_list[word]:
						ac += 1
					elif 'Psv' in h_list[word]:
						ps += 1
					elif 'Incr' in h_list[word]:
						inc += 1
					elif 'Decr' in h_list[word]:
						dec += 1
					elif 'Qual' in h_list[word]:
						qua += 1
					elif 'Social' in h_list[word]:
						so += 1
					
			sco.append({'ajec':ad, 'pos':pos, 'noun':nn, 'neg':neg, 'active':ac,'passive':ps,'increase':inc,
						'decrease':dec,'Quality':qua, 'social':so})
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
		social_ref = ['you','she','he', 'we','they','*PROPNAME*','them','his','her',
						'your','anyone','our','everyone','us','it','him','ours','yours','theirs',
						'their','myself','yourself','herself','itself','ourselves','yourselves',
						'themselves','all','another','any','anybody','anything','both','each',
						'either','everybody','everything','few','many','most','neither','nobody',
						'none','nothing','other','others','somebody','some','someone','something','such']		
		for lines in data:
			self_s, social_s= 0,0
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

""" Additional word lists """

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

""" H4LVD """
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
		

#####################################################################


class ArrayCaster(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    return self

  def transform(self, data):
    return np.transpose(np.matrix(data))
    

def tok(x):
	tokens = nltk.word_tokenize(x)
	return tokens
 
"""Scikit Classifiers  Experiment 1"""

def classify(df):
	X = df['text'].values
	y = df['label'].map({'high-social':0,'low-social':1})
	vec = TfidfVectorizer(tokenizer=tok, ngram_range=(1,4))
	print('Start Classifier.....')
	classifier = Pipeline([
				('features', FeatureUnion([
				('statn', Pipeline([
				('vec',vec),
				('tf',TfidfTransformer())
				])),
				('pos', Pipeline([
				('vec', PosTag()),
				('dic',DictVectorizer())
				])),
				('ex', Pipeline([
				('vec', ExtraWord()),
				('dic',DictVectorizer())
				])),
				('pro', Pipeline([
				('vec', Pronouns()),
				('dic',DictVectorizer())
				])),
				('h4', Pipeline([
				('vec', H4Lvd()),
				('dic',DictVectorizer())
				])),
				('punt', Pipeline([
				('vec', Punt()),
				('dic',DictVectorizer())
				])),
				('poss', Pipeline([
				('vec', PosNeg()),
				('dic',DictVectorizer())
				])),
				('text', Pipeline([
				('vec', TextStats()),
				('dic',DictVectorizer())
				])),
				])),
				('clf', LinearSVC())
				])
	cv = KFold(n_splits = 10, shuffle=True, random_state=0)
	acc_score, pre_score, rec_score, f1_s = [],[],[],[]
	k = 0
	for train, test in cv.split(X):
		X_train, X_test = X[train], X[test]
		y_train, y_test = y[train], y[test]
		classifier.fit(X_train,y_train)
		predict = classifier.predict(X_test)
		accuracy = metrics.accuracy_score(y_test, predict)
		print('accuracy: ',accuracy)
		precision = metrics.precision_score(y_test, predict)
		print('precision: ',precision)
		recall = metrics.recall_score(y_test, predict)
		print('recall',recall)
		f1 = metrics.f1_score(y_test, predict)
		print('F1-score: ', f1)
		acc_score.append(accuracy)
		pre_score.append(precision)
		rec_score.append(recall)
		f1_s.append(f1)
		print(metrics.confusion_matrix(y_test, predict))
		labell = ['high-social','low-social']
		print(metrics.classification_report(y_test,predict, target_names=labell))
		miss = np.where(predict != y_test)
		k = k + 1
		print('{} time of testing'.format(k))
	average_acc = sum(acc_score) / len(acc_score)
	print('Average accuracy: ', average_acc)
	average_pre = sum(pre_score) / len(pre_score)
	print('Average precision: ', average_pre)
	average_rec = sum(rec_score) / len(rec_score)
	print('Average recall: ', average_rec)
	average_f1 = sum(f1_s) / len(f1_s)
	print('Average f1-score: ', average_f1)

def cla_network(dataset):
	data = dataset[['text','netw_size','betw','norm_betw','brok','norm_brok','trans','den']]
	y = dataset['label'].map({'high-social':0,'low-social':1})
	vec = TfidfVectorizer(tokenizer=tok, ngram_range=(1,4))
	print('Start Classfy.....')
	classifier = Pipeline([
				('features', FeatureUnion([
				('statn', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',vec),
				('tf',TfidfTransformer())
				])),
				('posT', Pipeline([
				('sele', FunctionTransformer(lambda x: x['text'], validate=False)),
				('vec',PosTag()),
				('di',DictVectorizer())
				])),
				('norm_betw', Pipeline([
				('sele', FunctionTransformer(lambda x: x['den'], validate=False)),
				('con',ArrayCaster())
				])),
				])),
				('clf', MultinomialNB(alpha=0.01))
				])
	k = 0
	cv = KFold(n_splits = 10, shuffle=True, random_state=0)
	acc_score, pre_score, rec_score, f1_s = [],[],[],[]
	for train, test in cv.split(data):
		X_train, X_test = data.iloc[train], data.iloc[test]
		y_train, y_test = y[train], y[test]
		classifier.fit(X_train,y_train)
		predict = classifier.predict(X_test)
		accuracy = metrics.accuracy_score(y_test, predict)
		print('accuracy: ',accuracy)
		precision = metrics.precision_score(y_test, predict)
		print('precision: ',precision)
		recall = metrics.recall_score(y_test, predict)
		print('recall',recall)
		f1 = metrics.f1_score(y_test, predict)
		print('F1-score: ', f1)
		acc_score.append(accuracy)
		pre_score.append(precision)
		rec_score.append(recall)
		f1_s.append(f1)
		print(metrics.confusion_matrix(y_test, predict))
		labell = ['high-social','low-social']
		print(metrics.classification_report(y_test,predict, target_names=labell))
		k = k + 1
		print('{} time of testing'.format(k))
	average_acc = sum(acc_score) / len(acc_score)
	print('Average accuracy: ', average_acc)
	average_pre = sum(pre_score) / len(pre_score)
	print('Average precision: ', average_pre)
	average_rec = sum(rec_score) / len(rec_score)
	print('Average recall: ', average_rec)
	average_f1 = sum(f1_s) / len(f1_s)
	print('Average f1-score: ', average_f1)
			
## Main program
def Scikit_classify():
	score = ['extra','intro']

	training_set, with_id = read_training_data_s()
	
	### Calculate baseline
	#data = status(training_set)
	#baseline(data)
	
	## Experiment 1: Only linguistic features
	#data = status(training_set)
	#classify(data)
	
	### Experiment 2: Get network properties
	dataset = network_prop(training_set, with_id)
	cla_network(dataset)
	
	### All status of a user to one
	#dataset = read_whole()
	

if __name__ == "__main__":
	## Data preprocessing
	#prepare_data()

	## Classification
	Scikit_classify()
