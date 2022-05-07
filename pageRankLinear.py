import numpy as np

from normalize import normalize


def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """
    Input: Une matrice d'adjacence A d'un graphe dirigé, pondéré et régulier G, un vecteur
           de personnalisation v, ainsi qu'un paramètre de téléportation alpha compris entre
           0 et 1 (0.9 par défaut et pour les résultats à présenter). Toutes ces valeurs sont
           non-négatives.
    Output: Un vecteur x contenant les scores d'importance des noeuds ordonnés dans
            le même ordre que la matrice d'adjacence.
    """
    A = normalize(A)
    n = A.shape[0]

    # Formula cf. slides 148 of chapter 10
    I = np.identity(n)
    A = (1-alpha) * np.linalg.matrix_power((I - (alpha*A.T)), -1)
    x = np.dot(A, v)

    return np.array(x).flatten()  # convert matrix to np.array
