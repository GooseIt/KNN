import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import cosine_distance, euclidean_distance
import sys
EPS = 10 ** (-5)


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.initilializedSample = False
        self.sample = None
        self.sample_ans = None
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.params = np.ones(k)

    def fit(self, X, y):
        if not self.initilializedSample:
            self.initilializedSample = True
            self.sample = X
            self.sample_ans = y
        else:
            self.sample = np.concatenate((self.sample, X), axis=0)
            self.sample_ans = np.concatenate((self.sample_ans, y))

    def find_kneighbors(self, X, return_distance=True):
        if self.strategy in ['brute', 'kd_tree', 'ball_tree']:
            NNfinder = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric)
            NNfinder.fit(self.sample)
            distances, neighbors = NNfinder.kneighbors(X)
        if self.strategy == 'my_own':
            if self.metric == 'euclidean':
                dist_matrix = euclidean_distance(X, self.sample)
            else:
                dist_matrix = cosine_distance(X, self.sample)
            neighbors = np.apply_along_axis(lambda xc: np.argpartition(xc, range(self.k))[:self.k], 1, dist_matrix)
            distances = np.apply_along_axis(lambda xc: xc[np.argpartition(xc, range(self.k))[:self.k]],
                                            1, dist_matrix)
        if return_distance:
            return distances, neighbors
        else:
            return neighbors

    def predict(self, X):
        knn_dist, knn_list = self.find_kneighbors(X, True)
        # print(knn_dist, knn_list)
        # print(knn_list)
        ans = []
        for j in range(len(knn_list)):
            newbie_neighbors = knn_list[j]
            newbie_dists = knn_dist[j]
            class_scores = {}
            max_score = sys.float_info.min
            max_score_class = 0
            for i in range(len(newbie_neighbors)):
                if not self.weights:
                    class_scores[self.sample_ans[newbie_neighbors[i]]] = \
                        class_scores.get(self.sample_ans[newbie_neighbors[i]], 0) + self.params[i]
                else:
                    class_scores[self.sample_ans[newbie_neighbors[i]]] = \
                        class_scores.get(self.sample_ans[newbie_neighbors[i]], 0) + \
                        self.params[i] * 1 / (newbie_dists[i] + EPS)
                if class_scores[self.sample_ans[newbie_neighbors[i]]] > max_score:
                    max_score = class_scores[self.sample_ans[newbie_neighbors[i]]]
                    max_score_class = self.sample_ans[newbie_neighbors[i]]
            ans.append(max_score_class)
        return ans
