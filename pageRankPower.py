import numpy as np

from normalize import normalize


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
    # Transition probabilities matrix
    P = normalize(A)
    n = A.shape[0]

    # Following algorithm is build with horizontal personalization vector
    # so we work with the transpose of P
    P = P.T

    # Google matrix cf. slides 142 of chapter 10
    e = np.matrix(np.ones(n))
    vT = np.matrix(v).T
    G = alpha*P + (1-alpha) * np.dot(vT, e)
    G = np.asarray(G)

    # Power iteration
    pi = np.ones(n)/n
    max_iteration = 100
    epsilon = 1e-9

    for _ in range(max_iteration):
        previous_pi = pi
        pi = np.dot(G, pi)
        residual = np.sum(np.abs(pi-previous_pi))
        if residual < epsilon:
            return pi
    return pi
