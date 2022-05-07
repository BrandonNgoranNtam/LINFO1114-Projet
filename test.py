import networkx as nx
import numpy as np

from main import read_csv
from pageRankLinear import pageRankLinear
from pageRankPower import pageRankPower


def pageRankNetworkx(A: np.matrix, alpha: float, v: np.array) -> np.array:
    graph = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    pr = nx.pagerank(graph,
                     alpha=alpha,
                     personalization=dict(enumerate(v)),
                     max_iter=100,
                     tol=1e-9)

    return np.array(list(pr.values()))


def test_pageRankLinear(A: np.matrix, alpha: float, v: np.array):
    page_rank_linear = pageRankLinear(A, alpha, v)
    page_rank_networkx = pageRankNetworkx(A, alpha, v)

    assert np.allclose(page_rank_linear,
                       page_rank_networkx,
                       rtol=1e-9,
                       atol=1e-9), \
        "Page rank linear returned wrong values"


def test_pageRankPower(A: np.matrix, alpha: float, v: np.array):
    page_rank_power = pageRankPower(A, alpha, v)
    page_rank_networkx = pageRankNetworkx(A, alpha, v)

    assert np.allclose(page_rank_power,
                       page_rank_networkx,
                       rtol=1e-9,
                       atol=1e-9), \
        "Page rank power returned wrong values"


def tests():
    vector = read_csv('VecteurPersonnalisation_Groupe32.csv')
    matrix = np.matrix(read_csv('matrix.csv'))

    test_pageRankLinear(matrix, 0.9, vector)
    test_pageRankPower(matrix, 0.9, vector)

    print("Everything passed")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    tests()
