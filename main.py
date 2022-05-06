import numpy as np
import networkx as nx

from pageRankLinear import pageRankLinear
from pageRankPower import pageRankPower


def read_csv(file) -> np.ndarray:
    return np.loadtxt(file, delimiter=',')


def print_adj_matrix_template():
    for i in range(ord('A'), ord('J')+1):
        for j in range(ord('A'), ord('J')+1):
            print(f'{chr(i)}->{chr(j)}', end=',' if j != ord('J') else '')
        print()


def main():
    np.set_printoptions(precision=5)
    # print_adj_matrix_template()

    vector = read_csv('VecteurPersonnalisation_Groupe32.csv')
    matrix = np.matrix(read_csv('matrix.csv'))

    print('vector:', vector)
    print('matrix:', matrix)
    print('number of edges in the graph:', np.count_nonzero(matrix))

    print('pageRankLinear:', pageRankLinear(matrix, 0.9, vector))
    print('pageRankPower:', pageRankPower(matrix, 0.9, vector))

    G = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph)
    pr = nx.pagerank(G, alpha=0.9, personalization=dict(enumerate(vector)))
    print('pageRank with networkx:', np.array(list(pr.values())))


if __name__ == "__main__":
    main()
