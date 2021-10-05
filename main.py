import numpy as np
from nearest_neighbors import KNNClassifier


def test_KNN():
    a = np.array([[1, 2], [2, 3], [4, 5], [9, 11], [4, 16], [7, 30]])
    b = np.array([2, 2, 1, 1, 0, 0])
    c = np.array([[4, 5], [8, 8], [7, 27], [1, 2]])
    knn = KNNClassifier(k=2, strategy='kd_tree', metric='euclidean', weights=True, test_block_size=10)
    knn.fit(a, b)
    print(knn.predict(c))
    knn = KNNClassifier(k=2, strategy='my_own', metric='euclidean', weights=True, test_block_size=10)
    knn.fit(a, b)
    print(knn.predict(c))

if __name__ == '__main__':
    test_KNN()
