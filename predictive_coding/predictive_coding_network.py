from itertools import zip_longest

import numpy as np

from predictive_coding.pe_layer_config import PELayerConfig
from predictive_coding.predictive_estimator import PredictiveEstimator


class PredictiveCodingNetwork:
    def __init__(
        self, *, input_size: int | tuple, layer_configs: list[PELayerConfig]
    ) -> None:
        if not layer_configs or any(x.num_nodes <= 0 for x in layer_configs):
            raise ValueError("Layers must not be empty")
        if isinstance(input_size, int):
            input_size = (1, input_size)
        self.input_size = (layer_configs[0].num_nodes, *input_size)
        self.layer_configs = layer_configs
        self.layers: list[list[PredictiveEstimator]] = []
        current_layer: PELayerConfig
        next_layer: PELayerConfig | None
        for current_layer, next_layer in zip_longest(layer_configs, layer_configs[1:]):
            self.layers.append(
                [
                    PredictiveEstimator(
                        input_size=input_size,
                        repr_size=current_layer.repr_size,
                        params=current_layer.params,
                        sigma_sq_hl=next_layer.params.sigma_sq
                        if next_layer is not None
                        else 1.0,
                    )
                    for _ in range(current_layer.num_nodes)
                ]
            )
            input_size = (current_layer.num_nodes, current_layer.repr_size)

    def perceive(self, *, training: bool, inputs: np.ndarray) -> list[np.ndarray]:
        if len(inputs.shape) < len(self.input_size):
            inputs = inputs[:, None]
        assert inputs.shape == self.input_size
        preds = np.array([x.predict() for x in self.layers[0]])
        assert inputs.shape == preds.shape
        layer_errors = inputs - preds
        network_errors = [layer_errors]

        current_layer: list[PredictiveEstimator]
        next_layer: list[PredictiveEstimator] | None
        for current_layer, next_layer in zip_longest(self.layers, self.layers[1:]):
            # TODO: implement weighted sum of higher-level predictions with learnable weights
            preds_next_layer = None
            if next_layer:
                preds_next_layer = np.array([x.predict() for x in next_layer])

            next_layer_errors = []
            for index, node in enumerate(current_layer):
                next_layer_errors.append(
                    node.perceive(
                        training=training,
                        errors=layer_errors[index],
                        preds_hl=preds_next_layer[:, index]
                        if preds_next_layer is not None
                        else None,
                    )
                )
            # FIXME: next layer errors are invalid: this is an error of the mean prediction, not individual predictions.
            layer_errors = np.array(next_layer_errors).transpose((1, 0, -1))
            network_errors.append(layer_errors)
        return network_errors
