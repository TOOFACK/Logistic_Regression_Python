import numpy as np


def get_numeric_grad(f, x, eps):
	"""
	Function to calculate numeric gradient of f function in x.

	Parameters
	----------
	f : callable
	x : numpy.ndarray
		1d array, function argument
	eps : float
		Tolerance

	Returns
	-------
	: numpy.ndarray
		Numeric gradient.
	"""
	ones = np.zeros_like(x)
	ans = np.zeros_like(x)
	for i in range(x.shape[0]):
		ones[i] = 1
		ans[i] = (f(x + eps * ones) - f(x)) / eps
		ones[i] = 0
	return ans

def compute_balanced_accuracy(true_y, pred_y):
	"""
	Get balaced accuracy value

	Parameters
	----------
	true_y : numpy.ndarray
		True target.
	pred_y : numpy.ndarray
		Predictions.
	Returns
	-------
	: float
	"""
	possible_y = set(true_y)
	value = 0
	for current_y in possible_y:
		mask = true_y == current_y
		value += (pred_y[mask] == current_y).sum() / mask.sum()
	return value / len(possible_y)
