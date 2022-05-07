import numpy as np


def normalize(A: np.matrix) -> np.matrix:
    if A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square matrix")

    n = A.shape[0]
    A = np.asarray(A)

    # normalize sink nodes
    is_sink_node = np.sum(A, axis=1) == 0
    B = (np.ones_like(A) - np.identity(n)) / (n-1)
    A[:, is_sink_node] += B[:, is_sink_node]

    # normalize other nodes
    D_inv = np.diag(1/np.sum(A, axis=1))
    A = np.dot(D_inv, A)

    return np.matrix(A)
