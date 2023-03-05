import numpy as np


def matrix_main_diagonal(size: int) -> np.ndarray:
    return np.diag(np.ones(size))


def matrix_sub_diagonal(size: int) -> np.ndarray:
    return np.flip(np.diag(np.ones(size)), axis=0)


def matrix_horizontal_bar(size: int) -> np.ndarray:
    return np.concatenate(
        (
            np.zeros((size // 2, size)),
            np.ones((1, size)),
            np.zeros((size // 2, size)),
        ),
        axis=0,
    )


def matrix_vertical_bar(size: int) -> np.ndarray:
    return np.concatenate(
        (
            np.zeros((size, size // 2)),
            np.ones((size, 1)),
            np.zeros((size, size // 2)),
        ),
        axis=1,
    )
