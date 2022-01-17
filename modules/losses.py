import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w."""
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef
        self.is_multiclass_task = False

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w.
        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray
        Returns
        -------
        : float
        """
        matr = X.dot(w) * -y
        matr = np.logaddexp(0, matr)
        return 1 / X.shape[0] * np.sum(matr) + self.l2_coef * np.dot(w[1:], w[1:])

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w.
        Parameters
        ----------
        X : scipy.sparse.csr_matrix or numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray
        Returns
        -------
        : 1d numpy.ndarray
        """

        denominator = X.dot(w) * -y
        denominator = expit(denominator)
        numerator = scipy.sparse.csr_matrix(-y.reshape(-1, 1)).multiply(scipy.sparse.csr_matrix(X))
        new_m = scipy.sparse.csr_matrix(numerator).multiply(scipy.sparse.csr_matrix(denominator.reshape(-1, 1)))
        f_sum = np.asarray(np.sum(new_m, axis=0))
        new_w = np.zeros_like(w)
        new_w[1:] = w[1:]
        return (1 / X.shape[0] * f_sum + self.l2_coef * 2 * new_w).ravel()



