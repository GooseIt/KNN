import sys

import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    elements = np.arange(0, n)
    slices = np.array_split(elements, n_folds)
    boundaries = [x[i] for x in slices for i in [0, -1]]
    boundary_idx = np.searchsorted(elements, boundaries).reshape(-1, 2)
    ans = [(np.concatenate([elements[:x[0]],
                            np.setdiff1d(elements[x[0]:x[1] + 1], b, assume_unique=True),
                            elements[x[1] + 1:]]), b)
           for b, x in zip(slices, boundary_idx)]
    return ans


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    if kwargs.get('k', -1) != -1:
        del kwargs['k']
    if score == 'accuracy':
        ans = []
        if cv is None:
            cv = kfold(X.shape[0], 3)
        k_dict = {}
        maxk = sys.float_info.min
        for k in k_list:
            if k > maxk:
                maxk = k
        knn_classifier = KNNClassifier(k=maxk, **kwargs)
        ans = []
        for pair in cv:
            knn_classifier.fit(X[pair[0]], y[pair[0]])
            X_pred = knn_classifier.predict(X[pair[1]])
            ans.append(accuracy(X_pred, y[pair[1]]))
            k_dict[k] = np.array(ans)
        return k_dict
    return None


def accuracy(a, b):
    return a[a == b].shape[0] / a.shape[0]
