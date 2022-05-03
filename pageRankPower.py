import numpy as np


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """
    Input: Une matrice d'adjacence A d'un graphe dirigé, pondéré et régulier G, un vecteur
           de personnalisation v, ainsi qu'un paramètre de téléportation alpha compris entre
           0 et 1, alpha ∈ ]0, 1[ (0.9 par défaut et pour les résultats à présenter).
    Output: Un vecteur x contenant les scores d'importance des noeuds ordonnés dans
            le même ordre que la matrice d'adjacence
    """

    return np.array([])