import numpy as np
import numpy.testing as npt
from modules.losses_solution import MultinomialLoss


def test_function_multiclass():
    loss_function = MultinomialLoss(l2_coef=0.5)
    X = np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ])
    y = np.array([0, 2, 1])
    w = np.array([
        [1, -1, 1],
        [2, -1.5, 0],
        [0.5, 0, 1.2],
    ])
    npt.assert_almost_equal(loss_function.func(X, y, w), 4.15554, decimal=5)


def test_gradient_multiclass():
    loss_function = MultinomialLoss(l2_coef=0.5)
    X = np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ])
    y = np.array([0, 2, 1])
    w = np.array([
        [1, -1, 1],
        [2, -1.5, 0],
        [0.5, 0, 1.2],
    ])
    right_gradient = np.array([
        [ 0.07326395, -2.72842572,  2.38716246],
        [-0.28803268,  0.06239323, -1.80924359],
        [ 0.21476874,  0.16603273,  1.62208144]])
    npt.assert_almost_equal(loss_function.grad(X, y, w), right_gradient, decimal=5)
    