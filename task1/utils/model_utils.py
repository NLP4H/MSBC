import numpy as np 

def eval_f1(predictions, labels, threshold=0.5, average="micro"):
	"""
	Takes in numpy arrays of probabilities and labels and computes Micro / Macro F1 scores
	"""
	mask = labels.astype(bool)
	thresholded_predictions = predictions
	thresholded_predictions[predictions>=threshold] = 1
	thresholded_predictions[predictions<threshold] = 0

	gold_labels = labels.astype(float)
	tp = (gold_labels == thresholded_predictions) * mask
	
	per_document_accuracies = sum(tp == labels)

	tp_sum = tp.sum(axis=0).astype(float)
	pred_sum = thresholded_predictions.sum(axis=0).astype(float)
	true_sum = gold_labels.sum(axis=0).astype(float)
	total_sum = np.ones_like(true_sum).astype(float)

	true_sum_per_document = gold_labels.sum(axis=1).astype(float)
	tp_sum_per_document = tp.sum(axis=1).astype(float)
	pred_sum_per_document = thresholded_predictions.sum(axis=1).astype(float)


	precision_per_document = _prf_divide(tp_sum_per_document, pred_sum_per_document)
	# precision_per_document = tp_sum_per_document / pred_sum_per_document
	recall_per_document = _prf_divide(tp_sum_per_document, true_sum_per_document)
	# recall_per_document = tp_sum_per_document / true_sum_per_document
	f_score_per_document = _prf_divide((2 * precision_per_document * recall_per_document), (precision_per_document + recall_per_document))
	# f_score_per_document = 2 * precision_per_document * recall_per_document / ( precision_per_document + recall_per_document)

	if average=="micro":
		tp_sum = tp_sum.sum()
		pred_sum = pred_sum.sum()
		true_sum = true_sum.sum()
	
	precision = _prf_divide(tp_sum, pred_sum)
	recall = _prf_divide(tp_sum, true_sum)
	fscore = 2 * precision * recall / ( precision + recall)

	if average == "macro":
		precision = precision.mean()
		recall = recall.mean()
		fscore = fscore.mean()

	return {
		"precision": precision,
		"recall": recall,
		"fscore": fscore,
		"precision_per_document": precision_per_document.tolist(),
		"recall_per_document": recall_per_document.tolist(),
		"f_score_per_document": f_score_per_document.tolist()
	}

def _prf_divide(numerator, denominator):
	"""Performs division and handles divide-by-zero.
	On zero-division, sets the corresponding result elements to zero.
	"""
	result = numerator / denominator
	mask = denominator == 0.0
	if not mask.any():
		return result

	# remove nan
	result[mask] = 0.0
	return result