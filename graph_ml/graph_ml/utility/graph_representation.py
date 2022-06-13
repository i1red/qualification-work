import numpy as np


def normalize_adj_matrix(adj_matrix: np.ndarray, add_self_connection: bool = False) -> None:
    def inverse_sqrt(vec):
        return np.divide(1, vec ** 0.5, out=np.zeros_like(vec), where=vec != 0)

    d_tilda_inv_sqrt_left = np.diag(inverse_sqrt(adj_matrix.sum(axis=1)))
    d_tilda_inv_sqrt_right = np.diag(inverse_sqrt(adj_matrix.sum(axis=0)))

    if add_self_connection:
        adj_matrix = adj_matrix + np.identity(adj_matrix.shape[0])

    return d_tilda_inv_sqrt_left @ adj_matrix @ d_tilda_inv_sqrt_right


def adj_matrix_to_edges(adj_matrix: np.ndarray) -> list[tuple[int, int]]:
    m, n = adj_matrix.shape
    return [(i, j) for i in range(m) for j in range(n) if adj_matrix[i, j] > 0]


def adj_matrix_to_adj_list(adj_matrix):
    m, n = adj_matrix.shape
    return [[i for i in range(n) if adj_matrix[i, j] > 0] for j in range(m)]
