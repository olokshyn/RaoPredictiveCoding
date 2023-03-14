import numpy as np
from scipy.special import softmax
import pytest

from predictive_coding.pe_layer_config import PELayerConfig, PEParams
from predictive_coding.predictive_coding_network import PredictiveCodingNetwork
from tests.common import (
    matrix_main_diagonal,
    matrix_sub_diagonal,
)


@pytest.fixture
def network() -> PredictiveCodingNetwork:
    return PredictiveCodingNetwork(
        input_size=25,
        layer_configs=[
            PELayerConfig(
                num_nodes=2,
                repr_size=10,
                params=PEParams(
                    memory_uniform_low=-1.0,
                    memory_uniform_high=1.0,
                    k1=0.0005,
                    k2=0.005,
                    sigma_sq=1.0,
                    alpha=0.8,
                    lambd=0.75,
                ),
            ),
            PELayerConfig(
                num_nodes=2,
                repr_size=5,
                params=PEParams(
                    memory_uniform_low=-0.5,
                    memory_uniform_high=0.5,
                    k1=0.001,
                    k2=0.01,
                    sigma_sq=2.0,
                    alpha=1.0,
                    lambd=1.0,
                ),
            ),
            PELayerConfig(
                num_nodes=1,
                repr_size=10,
                params=PEParams(
                    memory_uniform_low=0.1,
                    memory_uniform_high=0.3,
                    k1=0.002,
                    k2=0.02,
                    sigma_sq=3.0,
                    alpha=1.0,
                    lambd=1.0,
                ),
            ),
        ],
    )


def test_construction(network: PredictiveCodingNetwork) -> None:
    assert network.input_size == (2, 1, 25)

    l0 = network.layers[0]
    pe00, pe01 = l0
    assert pe00.input_size == pe01.input_size == (1, 25)
    assert pe00.repr_size == pe01.repr_size == 10
    assert pe00.k1 == pe01.k1 == 0.0005
    assert pe00.k2 == pe01.k2 == 0.005
    assert pe00.sigma_sq == pe01.sigma_sq == 1.0
    assert pe00.sigma_hl_sq == pe01.sigma_hl_sq == 2.0
    assert pe00.alpha == pe01.alpha == 0.8
    assert pe00.lambd == pe01.lambd == 0.75
    assert pe00.memory.shape == pe01.memory.shape == (1, 25, 10)
    assert pe00.repr.shape == pe01.repr.shape == (10,)

    l1 = network.layers[1]
    pe10, pe11 = l1
    assert pe10.input_size == pe11.input_size == (2, 10)
    assert pe10.repr_size == pe11.repr_size == 5
    assert pe10.k1 == pe11.k1 == 0.001
    assert pe10.k2 == pe11.k2 == 0.01
    assert pe10.sigma_sq == pe11.sigma_sq == 2.0
    assert pe10.sigma_hl_sq == pe11.sigma_hl_sq == 3.0
    assert pe10.alpha == pe11.alpha == 1.0
    assert pe10.lambd == pe11.lambd == 1.0
    assert pe10.memory.shape == pe11.memory.shape == (2, 10, 5)
    assert pe10.repr.shape == pe11.repr.shape == (5,)

    l2 = network.layers[2]
    (pe20,) = l2
    assert pe20.input_size == (2, 5)
    assert pe20.repr_size == 10
    assert pe20.k1 == 0.002
    assert pe20.k2 == 0.02
    assert pe20.sigma_sq == 3.0
    assert pe20.sigma_hl_sq == 1.0
    assert pe20.alpha == 1.0
    assert pe20.lambd == 1.0
    assert pe20.memory.shape == (2, 5, 10)
    assert pe20.repr.shape == (10,)


def test_learn_diagonals(network: PredictiveCodingNetwork) -> None:
    inputs = np.stack(
        (
            matrix_main_diagonal(5),
            matrix_sub_diagonal(5),
        ),
        axis=0,
    ).reshape(2, 25)

    for _ in range(1000):
        network.perceive(training=True, inputs=inputs)
    preds = [softmax(x.predict()) for x in network.layers[0]]
    recall = np.array([np.where(p > p.mean(), 1, 0) for p in preds])
    assert (recall == inputs[:, None]).all()
