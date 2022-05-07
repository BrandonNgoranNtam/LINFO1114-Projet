import numpy as np

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
    # print_adj_matrix_template()

    vector = read_csv('VecteurPersonnalisation_Groupe32.csv')
    matrix = np.matrix(read_csv('matrix.csv'))

    print('vector:', vector)
    print('matrix:', matrix)
    print('number of edges in the graph:', np.count_nonzero(matrix))

    print('pageRankLinear:', pageRankLinear(matrix, 0.9, vector))
    print('pageRankPower:', pageRankPower(matrix, 0.9, vector))


if __name__ == "__main__":
    main()
