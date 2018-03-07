import csv
import time
import random
import math

import numpy as np

from feature_extraction import FeatureExtractor

from nltk.corpus import stopwords
from pandas import read_csv, DataFrame
from sklearn.naive_bayes import MultinomialNB

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM

def load_word2vec_dict(embedding_length):
	w2vd = dict()
	w2vfile = None
	valid_embedding_lengths = [50, 100, 200, 300]
	if embedding_length in valid_embedding_lengths:
		w2vfile = "glove.6B." + str(embedding_length) + "d.txt"
	else:
		print("Invalid embedding length {} past to load_word2vec_dict, must be one of {}".format(embedding_length, valid_embedding_lengths))

	with open(w2vfile, encoding = "utf8") as w2vf:
		for line in w2vf:
			items = line.replace("\n", "").replace("\r", "").split(" ")
			word = items[0]
			vec = np.array([float(n) for n in items[1:]])
			w2vd[word] = vec

	for word in stopwords.words("english"):
		try:
			del w2vd[word]
			print("removed {}".format(word))
		except KeyError:
			pass
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
			vectors.extend([np.zeros(vectors[0].shape) for i in range(length - len(vectors))])

	return np.array(vectors)

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
	test_mode = True
	SEQ_LENGTH = 1000
	EMBEDDING_LENGTH = 100

	print("Reading dataset")
	df = read_csv("news_ds.csv")

	#36 of the included samples are blank, we remove them
	df.drop([i for i in range(len(df)) if df.TEXT[i] == " "], inplace = True)
	df.reset_index(inplace = True, drop = True)

	#Test mode = use small set of data
	if test_mode:
		df.drop([i for i in range(1000, len(df))], inplace = True)
		df.reset_index(inplace = True, drop = True)

	print("Tokenizing texts")
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(list(df.TEXT))
	sequences = tokenizer.texts_to_sequences(list(df.TEXT))
	word_index = tokenizer.word_index

	#Pads too short sequences with 0s, truncates too long sequences
	fixed_length_sequences = pad_sequences(sequences, maxlen = SEQ_LENGTH)

	# print("Randomizing test set")
	# indices = [n for n in range(len(df))]
	# random.shuffle(indices)
	# k = 5
	# slice_size = len(indices) // k

	# vector_length = 1000

	# test_indices_lists = []
	# for i in range(k):
	# 	test_indices_lists.append([index for index in indices[i * slice_size: (i + 1) * slice_size]])

	print("Loading word2vec")
	word2vec_dict = load_word2vec_dict(EMBEDDING_LENGTH)

	#We build a matrix of ()
	print("Building embedding mat")
	embedding_mat = np.zeros(((len(word_index) + 1), EMBEDDING_LENGTH))

	for word, i in word_index.items():
		embedding = word2vec_dict.get(word)
		if embedding is not None:
			embedding_mat[i] = embedding

	model = Sequential()
	#The embedding layer has fixed weights which just provide an efficient way to transform the input tokens into the provided word embeddings
	model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_LENGTH,
                        weights = [embedding_mat],
                        input_length = SEQ_LENGTH,
                        trainable = False))
	model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))
	model.add(Dense(1, activation = 'sigmoid'))

	model.compile(loss = 'binary_crossentropy',
	              optimizer = 'adam',
	              metrics = ['accuracy'])
	
	train_split = 0.1
	train_samples = math.floor(len(fixed_length_sequences) * (1 - train_split))
	print(train_samples)
	train_text, train_labels = fixed_length_sequences[:train_samples], df.LABEL[:train_samples]
	test_text, test_labels = fixed_length_sequences[train_samples:], df.LABEL[train_samples:]

	batch_size = 64

	model.fit(train_text, train_labels, epochs = 5, batch_size = batch_size, validation_data = (test_text, test_labels))

	score, acc = model.evaluate(test_text, test_labels,
                            	batch_size = batch_size)

	print(score, acc)
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