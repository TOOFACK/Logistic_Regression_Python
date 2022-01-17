import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
            self,
            loss_function,
            batch_size=None,
            step_alpha=1,
            step_beta=0,
            tolerance=1e-5,
            max_iter=1000,
            random_seed=153,
            model_weights=1,
            **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.model_weights = model_weights

    def SGD(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):

        history = {}
        history['time'] = []
        history['func'] = []
        history['func_val'] = []

        if w_0 is None:  # Если начальный вектор весов не дан, создаем свой
            w_0 = np.random.rand(X.shape[1])

        permutation_X = np.random.permutation(X.shape[0])  # Рандомная перестановка строк
        new_X = X[permutation_X]  # Переставляем строки у X
        new_y = y[permutation_X]  # Перетсвляем также лейбы
        split_idx = np.arange(self.batch_size, new_X.shape[0],
                              self.batch_size)  # Индексы начало и конца каждого батча

        epoch_step = 1  # Номер эпохи
        lr = self.step_alpha / (epoch_step ** self.step_beta)

        for k in range(self.max_iter):
            batched_idx = 0  # Индекс для split_idx

            start = time.time()

            while batched_idx < len(split_idx):

                if batched_idx == 0:
                    new_w = w_0 - lr * self.loss_function.grad(new_X[:split_idx[batched_idx]],
                                                               new_y[:split_idx[batched_idx]], w_0)

                    self.model_weights = new_w  # Обновляем веса
                    w_0 = new_w

                else:
                    new_w = w_0 - lr * self.loss_function.grad(
                        new_X[split_idx[batched_idx - 1]:split_idx[batched_idx]]
                        , new_y[split_idx[batched_idx - 1]:split_idx[batched_idx]], w_0)

                    self.model_weights = new_w  # Обновляем веса
                    w_0 = new_w

                batched_idx += 1

            # Конец эпохи, нужно обучить оставшиеся элементы и обновить веса, а также сделать новую перестановку объектов

            new_w = w_0 - lr * self.loss_function.grad(new_X[split_idx[batched_idx - 1]:],
                                                       new_y[split_idx[batched_idx - 1]:], w_0)

            end = time.time()

            if trace:  # Если нужно отследить

                history['func'].append(self.loss_function.func(X, y, new_w))
                history['time'].append(end - start)
                if X_val is not None:
                    history['func_val'].append(self.loss_function.func(X_val, y_val, new_w))

            if np.linalg.norm(
                    new_w - w_0) < self.tolerance:  # Если веса перестали меняться, выходим из цикла
                break

            self.model_weights = new_w  # Обновляем веса
            w_0 = new_w

            permutation_X = np.random.permutation(X.shape[0])  # Рандомная перестановка строк
            new_X = X[permutation_X]  # Переставляем строки у X
            new_y = y[permutation_X]  # Перетсвляем также лейбы
            epoch_step += 1  # Увеличваем эпоху
            lr = self.step_alpha / (epoch_step ** self.step_beta)  # Обновляем lr

        if trace:
            return history

    def GD(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):

        if w_0 is None:  # Если начальный вектор весов не дан, создаем свой
            w_0 = np.random.rand(X.shape[1])

        history = {}
        history['time'] = []
        history['func'] = []
        history['func_val'] = []

        epoch_step = 0  # Шаг для lr

        for k in range(self.max_iter):
            start = time.time()
            epoch_step += 1  # После каждой эпохи увеличиваем

            lr = self.step_alpha / (epoch_step ** self.step_beta)  # Вычисляем lr

            new_w = w_0 - lr * self.loss_function.grad(X, y, w_0)
            end = time.time()

            if trace:  # Если нужно отследить

                history['func'].append(self.loss_function.func(X, y, new_w))
                history['time'].append(end - start)
                if X_val is not None:
                    history['func_val'].append(self.loss_function.func(X_val, y_val, new_w))

            if np.linalg.norm(
                    new_w - w_0) < self.tolerance:  # Если веса перестали меняться, выходим из цикла
                break

            self.model_weights = new_w
            w_0 = new_w

        if trace:
            return history

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):

        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """

        if not self.loss_function.is_multiclass_task:  # Если бинарная классификация
            if self.batch_size is None or self.batch_size >= X.shape[
                0]:  # Если полный градиентный спуск или размер батча превыщает кол-во объектов
                return self.GD(X, y, w_0, trace, X_val, y_val)
            else:
                return self.SGD(X, y, w_0, trace, X_val, y_val)

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        return np.where(X.dot(self.get_weights()) > threshold, 1, -1)

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.model_weights

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.model_weights)
