import csv
import time
import random
import math

import numpy as np

import matplotlib.pyplot as plt

from bayes import mn_bayes
from neural_nets import compare_rnn_lstm

from pandas import read_csv, DataFrame

#Given a list of tokens
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
	vec_len = None
	for v in w2vdict.values():
		if not vec_len:
			vec_len = len(v)
		else:
			if len(v) != vec_len:
				print("Vectors found during text_to_word2vec are nott he same length, aborting")
				exit()

	for word in tokenized_text:
		vectors.append(w2vdict.get(word, np.zeros((vec_len))))

	return np.array(vectors)

if __name__ == "__main__":
	test_mode = False
	prebuilt_embeddings = True

	print("Reading dataset")
	df = read_csv("news_ds.csv")

	#36 of the included samples are blank, we remove them
	df.drop([i for i in range(len(df)) if df.TEXT[i] == " "], inplace = True)
	df.reset_index(inplace = True, drop = True)

	#Test mode uses a subset of data
	if test_mode:
		df.drop([i for i in range(1000, len(df))], inplace = True)
		df.reset_index(inplace = True, drop = True)

	rnn_history, lstm_history = compare_rnn_lstm(df, prebuilt_embeddings)	

	#Plot performance graphs
	epoch_list = [i for i in range(1, len(rnn_history["acc"]) + 1)]
	plt.plot(epoch_list, lstm_history["acc"], color = (0, 0.6, 0))
	plt.plot(epoch_list, rnn_history["acc"], color = (0, 0, 0.6))
	plt.plot(epoch_list, lstm_history["val_acc"], color = (0.05, 1, 0.05))
	plt.plot(epoch_list, rnn_history["val_acc"], color = (0.05, 0.05, 1))
	plt.xlabel('Epoch')
	plt.ylabel('Validation Accuracy')
	plt.title("Accuracy and validation accuracy over 20 epochs")
	plt.savefig("output2")
	plt.show()

	#Multinomial naive bayes
	mn_bayes(df, 5)