import math
import numpy as np

from nltk.corpus import stopwords

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

def build_model(model_type, embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH):
	model = Sequential()

	#Embedding layer used to translate words to vectors
	if embedding_mat is not None:
		#If using prebuilt embeddings (ie Glove), fix embedding weights as the given weights
		model.add(Embedding(embedding_mat.shape[0],
                        EMBEDDING_LENGTH,
                        weights = [embedding_mat],
                        input_length = SEQ_LENGTH,
                        trainable = False))
	else:
		#Otherwise embeddings will be learnt during fitting
		model.add(Embedding(embedding_mat.shape[0], EMBEDDING_LENGTH, input_length = SEQ_LENGTH))

	if model_type == "LSTM":
		model.add(LSTM(100))
		model.add(Dropout(0.2))
	elif model_type == "RNN":
		model.add(SimpleRNN(150))
		model.add(Dropout(0.4))
	else:
		print("Invalid model_type {} passed to build_model".format(model_type))
		exit()

	#Single neuron dens layer provides the output
	model.add(Dense(1, activation = 'sigmoid'))

	model.compile(loss = 'binary_crossentropy',
	              optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False),
	              metrics = ['accuracy'])

	return model

def compare_rnn_lstm(df, prebuilt_embeddings):
	SEQ_LENGTH = 1000
	EMBEDDING_LENGTH = 100
	
	print("Tokenizing texts")
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(list(df.TEXT))
	sequences = tokenizer.texts_to_sequences(list(df.TEXT))
	word_index = tokenizer.word_index

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

	fixed_length_sequences = pad_sequences(sequences, maxlen = SEQ_LENGTH)

	train_split = 0.1
	num_train_samples = math.floor(len(fixed_length_sequences) * (1 - train_split))

	train_text, train_labels = fixed_length_sequences[:num_train_samples], df.LABEL[:num_train_samples]
	test_text, test_labels = fixed_length_sequences[num_train_samples:], df.LABEL[num_train_samples:]

	batch_size = 64
	num_epochs = 5

	model = build_model("RNN", embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH)
	rh = model.fit(train_text, train_labels, epochs = num_epochs, batch_size = batch_size, validation_data = (test_text, test_labels))
	model = build_model("LSTM", embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH)
	lh = model.fit(train_text, train_labels, epochs = num_epochs, batch_size = batch_size, validation_data = (test_text, test_labels))

	return rh.history, lh.history