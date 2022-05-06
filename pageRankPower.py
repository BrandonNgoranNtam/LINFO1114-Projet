import numpy as np


# sources :
# Slides
# Book: A. Langville & C. Meyer (2006) Google’s PageRank and Beyond
# https://hippocampus-garden.com/pagerank/
def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """
    Input: Une matrice d'adjacence A d'un graphe dirigé, pondéré et régulier G, un vecteur
           de personnalisation v, ainsi qu'un paramètre de téléportation alpha compris entre
           0 et 1, alpha ∈ ]0, 1[ (0.9 par défaut et pour les résultats à présenter).
    Output: Un vecteur x contenant les scores d'importance des noeuds ordonnés dans
            le même ordre que la matrice d'adjacence
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A should be a square matrix")

    n = A.shape[0]
    A = np.asarray(A.T)

    # normalize sink nodes
    is_sink_node = np.sum(A, axis=0) == 0
    B = (np.ones_like(A) - np.identity(n)) / (n-1)
    A[:, is_sink_node] += B[:, is_sink_node]

    # normalize other nodes
    D_inv = np.diag(1/np.sum(A, axis=0))
    A = np.dot(A, D_inv)

    # Transition probabilities matrix
    P = A

    # Google matrix
    e = np.matrix(np.ones(n))
    # eT = np.matrix(np.ones(n)).T
    vT = np.matrix(v).T
    G = alpha*P + (1-alpha) * np.dot(vT, e)
    G = np.asarray(G)

    # Power iteration
    pi = np.ones(n)/n
    max_iteration = 100
    epsilon = 1e-9

    for _ in range(max_iteration):
        previous_pi = pi
        # pi = np.dot(A, pi)*alpha + (1-alpha) * np.ones(n)/n
        pi = np.dot(G, pi)
        residual = np.sum(np.abs(pi-previous_pi))
        if residual < epsilon:
            return pi
    return pi
