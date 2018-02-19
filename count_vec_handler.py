import sklearn.feature_extraction.text as sk

class CVHandler:
	def __init__(self):
		self.count_vect = None

	def compute_tf_mat(self, input_texts):
		if self.count_vect:
			word_doc_freq_mat = self.count_vect.transform(input_texts)
		else:
			self.count_vect = sk.CountVectorizer()
			word_doc_freq_mat = self.count_vect.fit_transform(input_texts)

		tf_transformer = sk.TfidfTransformer(use_idf = False).fit(word_doc_freq_mat)
		tf_mat = tf_transformer.transform(word_doc_freq_mat)

		return tf_mat