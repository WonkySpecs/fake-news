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