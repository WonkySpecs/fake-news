import random
import math

from feature_extraction import FeatureExtractor
from sklearn.naive_bayes import MultinomialNB
from common_functions import compute_metrics

def mn_bayes(df, k):
	extractor_name_dict = { "tf"			: FeatureExtractor("tf"),
							"tfidf" 		: FeatureExtractor("tfidf"),
							"2gram tf"		: FeatureExtractor("tf", ngram_range = (2, 2)),
							"2gram tfidf"	: FeatureExtractor("tfidf", ngram_range = (2, 2))}

	indices = [n for n in range(len(df))]
	random.shuffle(indices)
	slice_size = len(indices) // k

	test_indices_lists = []
	for i in range(k):
		test_indices_lists.append([index for index in indices[i * slice_size: (i + 1) * slice_size]])

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