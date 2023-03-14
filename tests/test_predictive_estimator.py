import numpy as np
from scipy.special import softmax
import pytest

from predictive_coding.pe_params import PEParams
from predictive_coding.predictive_estimator import PredictiveEstimator
from tests.common import (
    matrix_main_diagonal,
    matrix_sub_diagonal,
    matrix_horizontal_bar,
    matrix_vertical_bar,
)


def test_construction(basic_pe) -> None:
    pe = PredictiveEstimator(
        input_size=(4, 50),
        repr_size=15,
        params=PEParams(
            memory_uniform_low=0.2,
            memory_uniform_high=0.3,
            k1=0.0005,
            k2=0.005,
            sigma_sq=1.5,
            alpha=2.0,
            lambd=3.02,
        ),
        sigma_sq_hl=10.0,
    )
    assert pe.input_size == (4, 50)
    assert pe.repr_size == 15
    assert pe.k1 == 0.0005
    assert pe.k2 == 0.005
    assert pe.sigma_sq == 1.5
    assert pe.sigma_hl_sq == 10.0
    assert pe.alpha == 2.0
    assert pe.lambd == 3.02
    assert pe.repr.shape == (15,)
    assert np.allclose(pe.repr, 0)
    assert pe.memory.shape == (4, 50, 15)


@pytest.fixture
def basic_pe() -> PredictiveEstimator:
    return PredictiveEstimator(
        input_size=(2, 10),
        repr_size=3,
        params=PEParams(
            memory_uniform_low=0.5,
            memory_uniform_high=0.5,
            k1=0.0005,
            k2=0.005,
            sigma_sq=1.0,
            alpha=1.0,
            lambd=0.02,
        ),
        sigma_sq_hl=10.0,
    )


def test_zeros(basic_pe: PredictiveEstimator) -> None:
    inputs = np.zeros((2, 10))

    preds = basic_pe.predict()
    assert preds.shape == (2, 10)
    assert (preds == 0).all()
    errors = inputs - preds
    errors_hl = basic_pe.perceive(training=True, errors=errors)
    assert errors_hl.shape == (1, *basic_pe.repr.shape)
    assert (errors_hl == 0).all()
    assert basic_pe.repr.shape == (3,)
    assert (basic_pe.repr == 0).all()
    assert basic_pe.memory.shape == (2, 10, 3)
    assert (basic_pe.memory == 0.49995).all()
    assert (basic_pe.predict() == 0).all()


def test_ones(basic_pe: PredictiveEstimator) -> None:
    inputs = np.ones((2, 10))

    preds = basic_pe.predict()
    assert preds.shape == (2, 10)
    assert (preds == 0).all()
    errors = inputs - preds
    errors_hl = basic_pe.perceive(training=True, errors=errors)
    assert errors_hl.shape == (1, *basic_pe.repr.shape)
    assert (errors_hl == 0).all()
    assert basic_pe.repr.shape == (3,)
    assert np.allclose(basic_pe.repr, 0.0025)
    assert basic_pe.memory.shape == (2, 10, 3)
    assert np.allclose(basic_pe.memory, 0.49995)

    preds = basic_pe.predict()
    assert preds.shape == (2, 10)
    assert np.allclose(preds, 0.003749625)
    errors = inputs - preds
    assert errors.shape == (2, 10)
    assert np.allclose(errors, 0.996250375)

    errors_hl = basic_pe.perceive(training=True, errors=errors)
    assert errors_hl.shape == (1, *basic_pe.repr.shape)
    assert (errors_hl == 0).all()
    assert basic_pe.repr.shape == (3,)
    assert np.allclose(basic_pe.repr, 0.00498912687490625)
    assert basic_pe.memory.shape == (2, 10, 3)
    assert np.allclose(basic_pe.memory, 0.4999124581296875)


def test_four_bars() -> None:
    pe = PredictiveEstimator(
        input_size=(4, 25),
        repr_size=10,
        params=PEParams(
            memory_uniform_low=1.0,
            memory_uniform_high=1.0,
            k1=0.0005,
            k2=0.05,
            sigma_sq=1.0,
            alpha=1.0,
            lambd=1.0,
        ),
        sigma_sq_hl=1.0,
    )

    inputs = np.stack(
        (
            matrix_horizontal_bar(5),
            matrix_main_diagonal(5),
            matrix_vertical_bar(5),
            matrix_sub_diagonal(5),
        ),
        axis=0,
    ).reshape(4, 25)

    preds = pe.predict()
    assert preds.shape == (4, 25)
    assert (preds == 0).all()
    errors = inputs - preds
    assert errors.shape == (4, 25)
    assert (errors == inputs).all()

    errors_hl = pe.perceive(training=True, errors=errors)
    assert errors_hl.shape == (1, *pe.repr.shape)
    assert (errors_hl == 0).all()
    assert pe.repr.shape == (10,)
    assert np.allclose(pe.repr, 0.0025)
    assert pe.memory.shape == (4, 25, 10)
    assert np.allclose(pe.memory, 0.95)

    preds = pe.predict()
    assert preds.shape == (4, 25)
    assert np.allclose(preds, 0.02375)
    errors = inputs - preds
    assert errors.shape == (4, 25)

    errors_hl = pe.perceive(training=True, errors=errors)
    assert errors_hl.shape == (1, *pe.repr.shape)
    assert (errors_hl == 0).all()
    assert pe.repr.shape == (10,)
    assert np.allclose(pe.repr, 0.00459172)
    assert pe.memory.shape == (4, 25, 10)
    assert np.allclose(
        pe.memory,
        np.array(
            [
                [
                    *([[0.902497025] * 10] * 10),
                    *([[0.902622025] * 10] * 5),
                    *([[0.902497025] * 10] * 10),
                ],
                [
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 5),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 5),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 5),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 5),
                    [0.902622025] * 10,
                ],
                [
                    *([[0.902497025] * 10] * 2),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 4),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 4),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 4),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 4),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 2),
                ],
                [
                    *([[0.902497025] * 10] * 4),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 3),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 3),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 3),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 3),
                    [0.902622025] * 10,
                    *([[0.902497025] * 10] * 4),
                ],
            ]
        ),
    )


def test_learn_diagonals() -> None:
    pe = PredictiveEstimator(
        input_size=(1, 25),
        repr_size=10,
        params=PEParams(
            memory_uniform_low=1.0,
            memory_uniform_high=1.0,
            k1=0.0005,
            k2=0.05,
            sigma_sq=1.0,
            alpha=1.0,
            lambd=1.0,
        ),
        sigma_sq_hl=1.0,
    )

    inputs = np.stack(
        (
            matrix_main_diagonal(5),
            matrix_sub_diagonal(5),
        ),
        axis=0,
    )

    for _ in range(100):
        input_example: np.ndarray
        for input_example in inputs:
            errors = input_example.reshape(1, 25) - pe.predict()
            pe.perceive(training=True, errors=errors)

    preds = pe.predict()
    assert preds.shape == (1, 25)
    preds = softmax(preds)
    indicators = np.where(preds > preds.mean(), 1.0, 0.0)
    assert (
        indicators.reshape(5, 5)
        == np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
            ]
        )
    ).all()


def test_with_pred_hl() -> None:
    pe = PredictiveEstimator(
        input_size=(1, 25),
        repr_size=10,
        params=PEParams(
            memory_uniform_low=1.0,
            memory_uniform_high=1.0,
            k1=0.0005,
            k2=0.05,
            sigma_sq=1.0,
            alpha=1.0,
            lambd=1.0,
        ),
        sigma_sq_hl=1.0,
    )

    inputs = matrix_main_diagonal(5).reshape(1, 25)

    preds_hl = np.full((1, 10), 1)

    errors = inputs - pe.predict()
    errors_hl = pe.perceive(training=True, errors=errors, preds_hl=preds_hl)
    assert errors_hl.shape == preds_hl.shape
    assert np.allclose(errors_hl, -1)
    assert pe.repr.shape == (10,)
    assert np.allclose(pe.repr, 0.003)
    assert pe.memory.shape == (1, 25, 10)
    assert np.allclose(pe.memory, 0.95)

    errors = inputs - pe.predict()
    errors_hl = pe.perceive(training=True, errors=errors, preds_hl=preds_hl)
    assert errors_hl.shape == preds_hl.shape
    assert np.allclose(errors_hl, -0.997)
    assert pe.repr.shape == (10,)
    assert np.allclose(pe.repr, 0.00553356)
    assert pe.memory.shape == (1, 25, 10)
    assert np.allclose(
        pe.memory,
        np.array(
            [
                [
                    [0.902645725] * 10,
                    *([[0.902495725] * 10] * 5),
                    [0.902645725] * 10,
                    *([[0.902495725] * 10] * 5),
                    [0.902645725] * 10,
                    *([[0.902495725] * 10] * 5),
                    [0.902645725] * 10,
                    *([[0.902495725] * 10] * 5),
                    [0.902645725] * 10,
                ],
            ]
        ),
    )
