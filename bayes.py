import random
import math

from sklearn.naive_bayes import MultinomialNB
import sklearn.feature_extraction.text as sk

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

#Wrapper class for sk countvectorizer and tfidftransformer
class FeatureExtractor:
	def __init__(self, feature_type, ngram_range = None):
		if feature_type.lower() in ["tf", "tfidf"]:
			self.feature_type = feature_type.lower()
		else:
			raise KeyError("FeatureExtractor feature_type expected 'tf' or 'tfidf', received {}".format(feature_type))

		self.count_vect = None
		if ngram_range is None:
			self.ngram_range = (1, 1)
		else:
			self.ngram_range = ngram_range

	def compute_freq_mat(self, input_texts):
		if self.count_vect:
			word_doc_freq_mat = self.count_vect.transform(input_texts)
		else:
			self.count_vect = sk.CountVectorizer(ngram_range = self.ngram_range)
			word_doc_freq_mat = self.count_vect.fit_transform(input_texts)

		if self.feature_type == "tf":
			freq_transformer = sk.TfidfTransformer(use_idf = False).fit(word_doc_freq_mat)
		else:
			freq_transformer = sk.TfidfTransformer(use_idf = True).fit(word_doc_freq_mat)

		freq_mat = freq_transformer.transform(word_doc_freq_mat)

		return freq_mat

def mn_bayes(df, k):
	try_combinations = False
	if try_combinations:
		extractor_name_dict = { "tf"			: ("tf", (1, 1)),
								"tfidf" 		: ("tfidf", (1, 1)),
								"2gram tf"		: ("tf", (2, 2)),
								"2gram tfidf"	: ("tfidf", (2, 2)),
								"3gram tf"		: ("tf", (3, 3)),
								"3gram tfidf"	: ("tfidf", (3, 3)),
								"4gram tf"		: ("tf", (4, 4)),
								"4gram tfidf"	: ("tfidf", (4, 4)),
								"1-3gram tfidf"	: ("tfidf", (1, 3))}
	else:
		extractor_name_dict = {	"3gram tfidf"	: ("tfidf", (3, 3))}

	#Randomize test/training sets
	indices = [n for n in range(len(df))]
	random.shuffle(indices)
	slice_size = len(indices) // k

	test_indices_lists = []
	for i in range(k):
		test_indices_lists.append([index for index in indices[i * slice_size: (i + 1) * slice_size]])

	for name, (freq_type, ngram) in extractor_name_dict.items():
		print("Training and testing {} extractor with {}-fold cross validation".format(name, k))
		avg_accuracy =  avg_precision =  avg_recall =  avg_fmeasure = 0
		for test_indices in test_indices_lists:
			feature_extractor = FeatureExtractor(freq_type, ngram_range = ngram)
			train_indices = [n for n in indices if n not in test_indices]
			test_text, train_text, test_labels, train_labels = df.TEXT[test_indices], df.TEXT[train_indices], df.LABEL[test_indices], df.LABEL[train_indices]

			train_freq_mat = feature_extractor.compute_freq_mat(train_text)
			test_freq_mat = feature_extractor.compute_freq_mat(test_text)

			classifier = MultinomialNB()
			#Train
			classifier.fit(train_freq_mat, train_labels)
			#Predict
			predictions = classifier.predict(test_freq_mat)
			#Evaluate
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