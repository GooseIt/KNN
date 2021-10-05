import numpy as np


def euclidean_distance(X, Y):
    X_sq = np.dot(X, X.transpose())
    X_sq = np.diagonal(X_sq)
    X_sq = X_sq.reshape(-1, 1)
    X_sq = np.repeat(X_sq, Y.shape[0], axis=1)
    Y_sq = np.dot(Y, Y.transpose())
    Y_sq = np.diagonal(Y_sq)
    Y_sq = Y_sq.reshape(-1, 1).transpose()
    return np.sqrt(X_sq + Y_sq - 2 * np.dot(X, Y.transpose()))


def cosine_distance(X, Y):
    X_div = np.linalg.norm(X, axis=1).reshape(-1, 1)
    X_div = np.repeat(X_div, Y.shape[0], axis=1)
    Y_div = np.linalg.norm(Y, axis=1).reshape(-1, 1).transpose()
    return 1 - np.dot(X, Y.transpose()) / X_div / Y_div
