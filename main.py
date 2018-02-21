import csv
import time
import random
import math

from count_vec_handler import CVHandler

from nltk import word_tokenize
from pandas import read_csv, DataFrame
from sklearn.naive_bayes import MultinomialNB

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print("Reading dataset")
df = read_csv("news_ds.csv")

#36 of the included samples are blank, we remove them
df.drop([i for i in range(len(df)) if df.TEXT[i] == " "], inplace = True)
df.reset_index(inplace = True, drop = True)

print("Randomizing test set")
indices = [n for n in range(len(df))]
random.shuffle(indices)
test_indices = indices[:len(indices) // 10]
train_indices = indices[len(indices) // 10 :]

train_text, test_text, train_labels, test_labels = df.iloc[train_indices, 1], df.iloc[test_indices, 1], df.iloc[train_indices, 2], df.iloc[test_indices, 2]

print("Computing train tf")
classifier = CVHandler()

train_tf = classifier.compute_tf_mat(train_text)

print("Training Naive Bayes")
clf = MultinomialNB().fit(train_tf, train_labels)

print("Computing test tf")
test_tf = classifier.compute_tf_mat(test_text)
predictions = clf.predict(test_tf)

correct = 0
for pred, actual in zip(predictions, test_labels):
	if pred == actual:
		correct += 1
print("{} correct out of {}  ({})%".format(correct, len(test_indices), correct * 100 / len(test_indices)))