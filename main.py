import numpy as np

from predictive_coding import PredictiveCodingNetwork, PELayerConfig, PEParams


def main():
    network = PredictiveCodingNetwork(
        input_size=16 * 16,
        layer_configs=[
            PELayerConfig(
                num_nodes=3,
                repr_size=32,
                params=PEParams(
                    memory_loc=0.5,
                    memory_scale=0.3,
                    k1=0.0005,
                    k2=0.005,
                    sigma_sq=1.0,
                    alpha=1.0,
                    lambd=0.02,
                ),
            ),
            PELayerConfig(
                num_nodes=1,
                repr_size=128,
                params=PEParams(
                    memory_loc=0.5,
                    memory_scale=0.3,
                    k1=0.005,
                    k2=0.05,
                    sigma_sq=10.0,
                    alpha=0.05,
                    lambd=0.00001,
                ),
            ),
        ],
    )
    network.perceive(
        training=True,
        inputs=np.array([np.ones(256), np.ones(256) * 2, np.ones(256) * 3]),
    )
    print(f"Final repr: {network.layers[-1][0].repr}")


if __name__ == "__main__":
    main()
