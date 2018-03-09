import random
import math

from common_functions import compute_metrics

from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text as sk

#Wrapper class for sk countvectorizer and tfidftransformer
class FeatureExtractor:
	def __init__(self, feature_type, ngram_range = None):
		if feature_type.lower() in ["tf", "tfidf"]:
			self.feature_type = feature_type.lower()
		else:
			raise KeyError("FeatureExtractor feature_type expected 'tf' or 'tfidf', received {}".format(feature_type))

		self.count_vect = None
		if ngram_range == None:
			self.ngram_range = (1, 1)
		else:
			self.ngram_range = ngram_range

	def compute_freq_mat(self, input_texts):
		if self.count_vect:
			word_doc_freq_mat = self.count_vect.transform(input_texts)
		else:
			self.count_vect = sk.CountVectorizer(self.ngram_range)
			word_doc_freq_mat = self.count_vect.fit_transform(input_texts)

		if self.feature_type == "tf":
			tf_transformer = sk.TfidfTransformer(use_idf = False).fit(word_doc_freq_mat)
		else:
			tf_transformer = sk.TfidfTransformer(use_idf = True).fit(word_doc_freq_mat)

		tf_mat = tf_transformer.transform(word_doc_freq_mat)

		return tf_mat

def mn_bayes(df, k):
	extractor_name_dict = { "tf"			: FeatureExtractor("tf"),
							"tfidf" 		: FeatureExtractor("tfidf"),
							"2gram tf"		: FeatureExtractor("tf", ngram_range = (2, 2)),
							"2gram tfidf"	: FeatureExtractor("tfidf", ngram_range = (2, 2))}

	#Randomize test/training sets
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

			train_freq_mat = feature_extractor.compute_freq_mat(train_text)

			#print("Training Naive Bayes")
			clf = MultinomialNB().fit(train_freq_mat, train_labels)

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