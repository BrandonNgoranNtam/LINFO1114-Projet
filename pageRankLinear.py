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

    P = normalize(A)
    leftside = np.transpose(np.identity(len(A)) - alpha*P)
    rightside = (1-alpha)*v
    return np.linalg.solve(leftside,rightside)
