import numpy as np
import pytest

from predictive_coding.image_pipeline import (
    process_image,
    restore_image,
    ProcessConfig,
    patch_iterator,
)
from tests.common import (
    matrix_main_diagonal,
    matrix_sub_diagonal,
    matrix_horizontal_bar,
    matrix_vertical_bar,
)


def test_process_restore_image() -> None:
    matrices = (
        matrix_main_diagonal(5) + 1,
        matrix_sub_diagonal(5) + 2,
        matrix_horizontal_bar(5) + 3,
        matrix_vertical_bar(5) + 4,
    )
    image = np.concatenate(matrices, axis=1)

    patched_image = process_image(
        image,
        config=ProcessConfig(
            patch_width=5,
            patch_height=5,
            scale=2,
            apply_dog=False,
        ),
    )
    expected = np.stack(matrices, axis=0)[None, :]
    expected *= 2
    assert expected.shape == patched_image.shape
    assert (expected == patched_image).all()

    restored_image = restore_image(patched_image=patched_image, scale=2)

    assert image.shape == restored_image.shape
    assert (image == restored_image).all()


@pytest.fixture
def patched_image() -> np.ndarray:
    patch = np.concatenate(
        (
            np.concatenate(
                (
                    matrix_main_diagonal(3) + 1,
                    matrix_sub_diagonal(3) + 2,
                ),
                axis=0,
            ),
            np.concatenate(
                (
                    matrix_horizontal_bar(3) + 3,
                    matrix_vertical_bar(3) + 4,
                ),
                axis=0,
            ),
        ),
        axis=1,
    )
    return np.stack((patch, patch * 10), axis=0)[None, :]


def test_patch_iterator_even_window(patched_image) -> None:
    window_size = (3, 3)
    window_step = (1, 1)

    expected_patch_inputs = np.array(
        [
            # row 1, col 1
            matrix_main_diagonal(3) + 1,
            # row 1, col 2
            [
                [1, 1, 3],
                [2, 1, 4],
                [1, 2, 3],
            ],
            # row 1, col 3
            [
                [1, 3, 3],
                [1, 4, 4],
                [2, 3, 3],
            ],
            # row 1, col 4
            matrix_horizontal_bar(3) + 3,
            # row 2, col 1
            [
                [1, 2, 1],
                [1, 1, 2],
                [2, 2, 3],
            ],
            # row 2, col 2
            [
                [2, 1, 4],
                [1, 2, 3],
                [2, 3, 4],
            ],
            # row 2, col 3
            [
                [1, 4, 4],
                [2, 3, 3],
                [3, 4, 5],
            ],
            # row 2, col 4
            [
                [4, 4, 4],
                [3, 3, 3],
                [4, 5, 4],
            ],
            # row 3, col 1
            [
                [1, 1, 2],
                [2, 2, 3],
                [2, 3, 2],
            ],
            # row 3, col 2
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 2, 4],
            ],
            # row 3, col 3
            [
                [2, 3, 3],
                [3, 4, 5],
                [2, 4, 5],
            ],
            # row 3, col 4
            [
                [3, 3, 3],
                [4, 5, 4],
                [4, 5, 4],
            ],
            # row 4, col 1
            matrix_sub_diagonal(3) + 2,
            # row 4, col 2
            [
                [2, 3, 4],
                [3, 2, 4],
                [2, 2, 4],
            ],
            # row 4, col 3
            [
                [3, 4, 5],
                [2, 4, 5],
                [2, 4, 5],
            ],
            # row 4, col 4
            matrix_vertical_bar(3) + 4,
        ]
    )
    expected_inputs = np.array(
        [
            expected_patch_inputs,
            expected_patch_inputs * 10,
        ],
        dtype=np.float32,
    )

    for expected, actual in zip(
        expected_inputs,
        patch_iterator(
            patched_image=patched_image,
            window_size=window_size,
            window_step=window_step,
        ),
    ):
        assert expected.shape == actual.shape
        assert (expected == actual).all()


def test_patch_iterator_uneven_window(patched_image) -> None:
    window_size = (3, 3)
    window_step = (2, 2)

    expected_patch_inputs = np.array(
        [
            matrix_main_diagonal(3) + 1,
            [
                [1, 3, 3],
                [1, 4, 4],
                [2, 3, 3],
            ],
            [
                [1, 1, 2],
                [2, 2, 3],
                [2, 3, 2],
            ],
            [
                [2, 3, 3],
                [3, 4, 5],
                [2, 4, 5],
            ],
        ],
        dtype=np.float32,
    )

    expected_inputs = np.array(
        [
            expected_patch_inputs,
            expected_patch_inputs * 10,
        ],
        dtype=np.float32,
    )

    for expected, actual in zip(
        expected_inputs,
        patch_iterator(
            patched_image=patched_image,
            window_size=window_size,
            window_step=window_step,
        ),
    ):
        assert expected.shape == actual.shape
        assert (expected == actual).all()
