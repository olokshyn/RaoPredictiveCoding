import numpy as np
from scipy.special import softmax
import pytest

from predictive_coding.pe_layer_config import PELayerConfig, PEParams
from predictive_coding.predictive_coding_network import PredictiveCodingNetwork


@pytest.fixture
def network() -> PredictiveCodingNetwork:
    return PredictiveCodingNetwork(
        input_size=25,
        layer_configs=[
            PELayerConfig(
                num_nodes=2,
                repr_size=10,
                params=PEParams(
                    memory_loc=4,
                    memory_scale=0.001,
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
                    memory_loc=2,
                    memory_scale=0.001,
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
                    memory_loc=1,
                    memory_scale=0.001,
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
    assert pe00.memory.mean() == pytest.approx(4, abs=1e-3)
    assert pe01.memory.mean() == pytest.approx(4, abs=1e-3)
    assert pe00.memory.std() == pytest.approx(0.001, abs=1e-3)
    assert pe01.memory.std() == pytest.approx(0.001, abs=1e-3)

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
    assert pe10.memory.mean() == pytest.approx(2, abs=1e-3)
    assert pe11.memory.mean() == pytest.approx(2, abs=1e-3)
    assert pe10.memory.std() == pytest.approx(0.001, abs=1e-3)
    assert pe11.memory.std() == pytest.approx(0.001, abs=1e-3)

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
    assert pe20.memory.mean() == pytest.approx(1, abs=1e-3)
    assert pe20.memory.std() == pytest.approx(0.001, abs=1e-3)


def test_learn_diagonals(network: PredictiveCodingNetwork) -> None:
    inputs = np.array(
        [
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
        ]
    ).reshape(2, 25)

    for _ in range(100):
        network.perceive(training=True, inputs=inputs)
    preds = [softmax(x.predict()) for x in network.layers[0]]
    recall = np.array([np.where(p > p.mean(), 1, 0) for p in preds])
    assert (recall == inputs[:, None]).all()
    assert np.std(network.layers[0][0].repr) < 1e-3
    assert np.std(network.layers[0][1].repr) < 1e-3
    assert np.std(network.layers[1][0].repr) < 1e-3
    assert np.std(network.layers[1][1].repr) < 1e-3
    assert np.std(network.layers[2][0].repr) < 1e-3
