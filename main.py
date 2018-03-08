import csv
import time
import random
import math

import numpy as np

import matplotlib.pyplot as plt

from bayes import mn_bayes

from nltk.corpus import stopwords
from pandas import read_csv, DataFrame
from sklearn.naive_bayes import MultinomialNB

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, SimpleRNN, LSTM, Dropout
from keras.optimizers import Adam

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

def build_model(model_type, embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH):
	model = Sequential()

	#Embedding layer used to translate words to vectors
	if embedding_mat is not None:
		#If using prebuilt embeddings (ie Glove), fix embedding weights as the given weights
		model.add(Embedding(max_i + 1,
                        EMBEDDING_LENGTH,
                        weights = [embedding_mat],
                        input_length = SEQ_LENGTH,
                        trainable = False))
	else:
		#Otherwise embeddings will be learnt during fitting
		model.add(Embedding(max_i + 1, EMBEDDING_LENGTH, input_length = SEQ_LENGTH))

	if model_type == "LSTM":
		model.add(LSTM(100))
		model.add(Dropout(0.3))

	elif model_type == "RNN":
		model.add(SimpleRNN(100))
		model.add(Dropout(0.3))
	else:
		print("Invalid model_type {} passed to build_model".format(model_type))
		exit()

	#Single neuron dens layer provides the output
	model.add(Dense(1, activation = 'sigmoid'))

	model.compile(loss = 'binary_crossentropy',
	              optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False),
	              metrics = ['accuracy'])

	return model

if __name__ == "__main__":
	test_mode = False
	prebuilt_embeddings = True
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

	embedding_mat = None

	if prebuilt_embeddings:
		print("Loading word2vec")
		word2vec_dict = load_word2vec_dict(EMBEDDING_LENGTH)

		for word in stopwords.words("english"):
			if word in word_index.keys():
				del word_index[word]

		max_i = max(word_index.values())

		print("Building embedding mat")
		embedding_mat = np.zeros(((max_i + 1), EMBEDDING_LENGTH))

		for word, i in word_index.items():
			embedding = word2vec_dict.get(word)
			if embedding is not None:
				embedding_mat[i] = embedding
	
	train_split = 0.1
	num_train_samples = math.floor(len(fixed_length_sequences) * (1 - train_split))

	train_text, train_labels = fixed_length_sequences[:num_train_samples], df.LABEL[:num_train_samples]
	test_text, test_labels = fixed_length_sequences[num_train_samples:], df.LABEL[num_train_samples:]

	batch_size = 64
	num_epochs = 15

	model = build_model("RNN", embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH)
	rnn_history = model.fit(train_text, train_labels, epochs = num_epochs, batch_size = batch_size, validation_data = (test_text, test_labels))
	model = build_model("LSTM", embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH)
	lstm_history = model.fit(train_text, train_labels, epochs = num_epochs, batch_size = batch_size, validation_data = (test_text, test_labels))


	epoch_list = [i for i in range(1, num_epochs + 1)]
	plt.plot(epoch_list, lstm_history.history["val_acc"], 'g^', epoch_list, rnn_history.history["val_acc"], 'bs')
	plt.xylabel('Epoch')
	plt.ylabel('Validation Accuracy')
	plt.savefig("output")
	plt.show()
	input()
	exit()

	#Multinomial naive bayes
	mn_bayes(df, 5)