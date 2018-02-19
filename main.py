import csv
import time
import random

from count_vec_handler import CVHandler

from nltk import word_tokenize
from pandas import read_csv, DataFrame
import sklearn.feature_extraction.text as sk
from sklearn.naive_bayes import MultinomialNB

print("Reading dataset")
df = read_csv("news_ds.csv")

print("Randomizing test set")
indices = [n for n in range(len(df))]
random.shuffle(indices)
test_indices = indices[:len(indices) // 10]
train_indices = indices[len(indices) // 10 :]

print("Computing train tf")
classifier = CVHandler()

train_tf = classifier.compute_tf_mat(df.iloc[train_indices, 1])

print("Training Naive Bayes")
clf = MultinomialNB().fit(train_tf, df.iloc[train_indices, 2])

print("Computing test tf")
test_tf = classifier.compute_tf_mat(df.iloc[test_indices, 1])
predictions = clf.predict(test_tf)

correct = 0
for pred, actual in zip(predictions, df.iloc[test_indices, 2]):
	if pred == actual:
		correct += 1
print("{} correct out of {}  ({})%".format(correct, len(test_indices), correct * 100 / len(test_indices)))