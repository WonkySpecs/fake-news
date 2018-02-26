import sklearn.feature_extraction.text as sk

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

	def compute_tf_mat(self, input_texts):
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