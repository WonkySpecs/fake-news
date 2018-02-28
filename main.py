import csv
import time
import random
import math

import numpy as np

from feature_extraction import FeatureExtractor

from nltk import word_tokenize
from pandas import read_csv, DataFrame
from sklearn.naive_bayes import MultinomialNB


from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

word2vec_file = "glove.6B.100d.txt"

def load_word2vec_dict(word2vec_file):
	w2vd = dict()
	with open(word2vec_file, encoding = "utf8") as w2vf:
		for line in w2vf:
			items = line.replace("\n", "").replace("\r", "").split(" ")
			word = items[0]
			vec = np.array([float(n) for n in items[1:]])
			w2vd[word] = vec

	return w2vd

def text_to_word2vec(tokenized_text, preloaded_w2v = None, word2vec_file = None, length = None):
	if preloaded_w2v and word2vec_file:
		print("Warning: text_to_word2vec received values for both word2vec_dict and word2vec_file, expected only one. Defaulting to use word2vec_dict")

	if not (preloaded_w2v or word2vec_file):
		print("Must specify word2vec_dict or word_2vec_file in text_to_word2vec()")
		exit()

	if length:
		if len(tokenized_text) > length:
			tokenized_text = tokenized_text[:length]

	vectors = []

	if preloaded_w2v:
		w2vdict = preloaded_w2v
	else:
		vectors_needed = set(tokenized_text)
		w2vdict = dict()
		with open(word2vec_file,  encoding = "utf8") as f:
			for line in f:
				if vectors_needed:
					items = line.replace("\n", "").replace("\r", "").split(" ")
					word = items[0]
					if word in vectors_needed:
						w2vdict[word] = np.array([float(n) for n in items[1:]])
						vectors_needed.remove(word)
				else:
					break

			if vectors_needed:
				print("Not all words found HANDLE THIS ITS GONNA CRASH FOR NOW")

	for word in tokenized_text:
		vectors.append(w2vdict[word])
	#If vector is too short, pad with 0s
	if length:
		if len(vectors) < length:
			vectors.extend([0 for i in range(length - len(vectors))])

	return np.array(vectors)

def set_length(tokens, length):
	if len(tokens) < length:
		tokens.extend([0 for i in range(length - len(tokens))])
		return tokens
	else:
		return tokens[:length]

#Input: list/array of predicted 0/1 labels and the actual labels
#Outputs: accuracy, precision, recall and f-measure of predictions
def compute_metrics(predictions, actual):
	tp = tn = fp = fn = 0

	for pred, ac in zip(predictions, actual):
		if ac == 0:
			if pred == 0:
				tn += 1
			else:
				fp += 1
		else:
			if pred == 1:
				tp += 1
			else:
				fn += 1
	recall = tp / float(tp + fn)
	precision = tp / float(tp + fp)

	#Accuracy, precision, recall, f-measure
	return (tp + tn) / len(predictions), precision, recall, 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
	print("Reading dataset")
	df = read_csv("news_ds.csv")

	#36 of the included samples are blank, we remove them
	df.drop([i for i in range(len(df)) if df.TEXT[i] == " "], inplace = True)
	df.reset_index(inplace = True, drop = True)

	print("Randomizing test set")
	indices = [n for n in range(len(df))]
	random.shuffle(indices)
	k = 5
	slice_size = len(indices) // k

	test_indices_lists = []
	for i in range(k):
		test_indices_lists.append([index for index in indices[i * slice_size: (i + 1) * slice_size]])


	print("Loading word2vec")
	word2vec_dict = load_word2vec_dict(word2vec_file)

	print("Tokenizing texts")
	df.TEXT = df.TEXT.apply(word_tokenize)

	print("Stripping unknown words")
	df.TEXT = df.TEXT.apply(lambda t : [w for w in t if w in word2vec_dict.keys()])

	print(max([len(t) for t in df.TEXT]))

	print("Example text to vectors")
	train_vecs = df.TEXT.apply(lambda t : text_to_word2vec(t, preloaded_w2v = word2vec_dict, length = 1000))
	for v in train_vecs:
		print(v.shape)

	#This heres a nn woo
	model = Sequential()
	model.add(LSTM(100, input_shape = train_vecs.shape, dropout = 0.2, recurrent_dropout = 0.2))
	model.add(Dense(1, activation = 'sigmoid'))
	input()
	exit()

	#Multinomial naive bayes

	extractor_name_dict = { "tf"			: FeatureExtractor("tf"),
							"tfidf" 		: FeatureExtractor("tfidf"),
							"2gram tf"		: FeatureExtractor("tf", ngram_range = (2, 2)),
							"2gram tfidf"	: FeatureExtractor("tfidf", ngram_range = (2, 2))}

	for name, feature_extractor in extractor_name_dict.items():
		print("Training and testing {} extractor with {}-fold cross validation".format(name, k))
		avg_accuracy =  avg_precision =  avg_recall =  avg_fmeasure = 0
		for test_indices in test_indices_lists:
			train_indices = [n for n in indices if n not in test_indices]
			test_text, train_text, test_labels, train_labels = df.TEXT[test_indices], df.TEXT[train_indices], df.LABEL[test_indices], df.LABEL[train_indices]

			#print("Computing train tf")
			train_freq_mat = feature_extractor.compute_freq_mat(train_text)

			#print("Training Naive Bayes")
			clf = MultinomialNB().fit(train_freq_mat, train_labels)

			#print("Computing test tf")
			test_freq_mat = feature_extractor.compute_freq_mat(test_text)
			predictions = clf.predict(test_freq_mat)

			accuracy, precision, recall, fmeasure = compute_metrics(predictions, test_labels)
			avg_accuracy += accuracy
			avg_precision += precision
			avg_recall += recall
			avg_fmeasure += fmeasure
			#print("{:10}: {}\n{:10}: {}\n{:10}: {}\n{:10}: {}\n".format("Accuracy", accuracy, "Precision", precision, "Recall", recall, "F-Measure", fmeasure))

		avg_accuracy = avg_accuracy / k
		avg_precision = avg_precision / k
		avg_recall = avg_recall / k
		avg_fmeasure = avg_fmeasure / k
		print("Averages:\n--------------\n{:10}: {}\n{:10}: {}\n{:10}: {}\n{:10}: {}\n".format("Accuracy", avg_accuracy, "Precision", avg_precision, "Recall", avg_recall, "F-Measure", avg_fmeasure))