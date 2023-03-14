import os

import numpy as np
import click
import cv2
from lovely_numpy import lo

from predictive_coding import PredictiveCodingNetwork, PELayerConfig, PEParams
from predictive_coding.image_pipeline import (
    load_and_process_batch,
    gaussian_mask,
    patch_iterator,
    ProcessConfig,
)


def array_to_greyscale(
    array: np.ndarray, scale: int = 4, atol: float = 1e-8
) -> np.ndarray:
    max_value = array.max()
    min_value = array.min()
    if max_value - min_value > atol:
        array = (array - min_value) / (max_value - min_value)
    else:
        array = np.maximum(array, 0.0) / max_value
    array *= 255
    return cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


@click.command()
@click.option("--image-dir-path", default="data/images/rao")
@click.option("--out-dir-path", default="out")
@click.option("--num-epochs", default=1)
@click.option("--num-train-iters", default=30)
def train_rao(
    image_dir_path: str, out_dir_path: str, num_epochs: int, num_train_iters: int
):
    np.random.seed(42)
    images = load_and_process_batch(
        image_dir_path,
        config=ProcessConfig(
            apply_dog=True,
        ),
    )
    mask = gaussian_mask(width=16)
    network = PredictiveCodingNetwork(
        input_size=16 * 16,
        layer_configs=[
            PELayerConfig(
                num_nodes=3,
                repr_size=32,
                params=PEParams(
                    memory_uniform_low=-3.5,
                    memory_uniform_high=3.5,
                    k1=0.0005,
                    k2=0.005,
                    k2_decay=1.015,
                    sigma_sq=1.0,
                    alpha=1.0,
                    lambd=0.02,
                ),
            ),
            PELayerConfig(
                num_nodes=1,
                repr_size=128,
                params=PEParams(
                    memory_uniform_low=-3.5,
                    memory_uniform_high=3.5,
                    k1=0.005,
                    k2=0.05,
                    k2_decay=1.015,
                    sigma_sq=10.0,
                    alpha=0.05,
                    lambd=0.00001,
                ),
            ),
        ],
    )
    for _ in range(num_epochs):
        iteration = 0
        for image_index, image in enumerate(images):
            for patch_index, patch in enumerate(
                patch_iterator(image, window_size=(16, 16), window_step=(5, 5))
            ):
                patch *= mask
                patch = patch.reshape(patch.shape[0], -1)
                network.train(inputs=patch, iterations=num_train_iters)

                if iteration % 40 == 0:
                    network.decay_learning_rate()
                if patch_index > 0 and patch_index % 25 == 0:
                    print(
                        f"Iter {iteration}, patch {patch_index}\n{lo(network.Us)}\n{lo(network.Us)}\n\n"
                    )

                iteration += 1

    os.makedirs(out_dir_path, exist_ok=True)
    for i in range(32):  # first layer repr size
        node_0_1 = network.layers[0][1].memory[0, :, i].reshape(16, 16)
        node_0_1 = array_to_greyscale(node_0_1)
        cv2.imwrite(os.path.join(out_dir_path, f"node_0_1_{i:0>2}.png"), node_0_1)

    for i in range(128):  # second layer repr size
        pred = np.zeros((16, 26), dtype=np.float32)
        for k in range(3):
            pred[:, 5 * k : 5 * k + 16] += np.dot(
                network.layers[0][k].memory[0], network.layers[1][0].memory[k, :, i]
            ).reshape((16, 16))
        pred = array_to_greyscale(pred)
        cv2.imwrite(os.path.join(out_dir_path, f"l2_pred_{i:0>2}.png"), pred)


if __name__ == "__main__":
    train_rao()
