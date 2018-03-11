import math
import numpy as np

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

	return w2vd

def build_model(model_type, embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH):
	model = Sequential()

	#Embedding layer used to translate words to vectors
	#Layer weights are fixed as the given word2vec vectors
	#Inputs to this layer then "select" the correct row from the embedding matrix for the next layer
	model.add(Embedding(embedding_mat.shape[0],
                    EMBEDDING_LENGTH,
                    weights = [embedding_mat],
                    input_length = SEQ_LENGTH,
                    trainable = False))
		

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

def compare_rnn_lstm(df):
	SEQ_LENGTH = 1000
	EMBEDDING_LENGTH = 100

	print("Tokenizing texts")
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(list(df.TEXT))
	sequences = tokenizer.texts_to_sequences(list(df.TEXT))
	word_index = tokenizer.word_index
	
	print("Loading word2vec")
	word2vec_dict = load_word2vec_dict(EMBEDDING_LENGTH)

	max_i = max(word_index.values())

	print("Building embedding mat")
	embedding_mat = np.zeros(((max_i + 1), EMBEDDING_LENGTH))

	for word, i in word_index.items():
		embedding = word2vec_dict.get(word)
		if embedding is not None:
			embedding_mat[i] = embedding

	#Truncate sequences longer than SEQ_LENGTH, pad shorter sequences with 0s
	fixed_length_sequences = pad_sequences(sequences, maxlen = SEQ_LENGTH)

	train_split = 0.1
	num_train_samples = math.floor(len(fixed_length_sequences) * (1 - train_split))

	train_text, train_labels = fixed_length_sequences[:num_train_samples], df.LABEL[:num_train_samples]
	test_text, test_labels = fixed_length_sequences[num_train_samples:], df.LABEL[num_train_samples:]

	batch_size = 64
	num_epochs = 20

	#Model.fit returns a history of the model which includes its performance on the given (unseen) test data
	model = build_model("RNN", embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH)
	rh = model.fit(train_text, train_labels, epochs = num_epochs, batch_size = batch_size, validation_data = (test_text, test_labels))
	model = build_model("LSTM", embedding_mat, SEQ_LENGTH, EMBEDDING_LENGTH)
	lh = model.fit(train_text, train_labels, epochs = num_epochs, batch_size = batch_size, validation_data = (test_text, test_labels))

	return rh.history, lh.history