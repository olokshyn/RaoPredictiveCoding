import os

import numpy as np
import click
import cv2

from predictive_coding import PredictiveCodingNetwork, PELayerConfig, PEParams
from predictive_coding.image_pipeline import (
    load_and_process_batch,
    gaussian_mask,
    patch_iterator,
    ProcessConfig,
)


@click.command()
@click.option("--image-dir-path", default="../predictive_coding/data/test")
@click.option("--out-dir-path", default="out")
@click.option("--num-epochs", default=1)
@click.option("--num-train-iters", default=30)
def train_rao(
    image_dir_path: str, out_dir_path: str, num_epochs: int, num_train_iters: int
):
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
                    memory_loc=0.5,
                    memory_scale=0.3,
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
                    memory_loc=0.5,
                    memory_scale=0.3,
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
        for image_index, image in enumerate(images):
            for patch_index, patch in enumerate(
                patch_iterator(image, window_size=(16, 16), window_step=(5, 5))
            ):
                patch *= mask
                patch = patch.reshape(patch.shape[0], -1)
                network.train(inputs=patch, iterations=num_train_iters)

                if (image_index * patch_index) % 40 == 0:
                    network.decay_learning_rate()

    os.makedirs(out_dir_path, exist_ok=True)
    for i in range(32):  # first layer repr size
        node_0_1 = network.layers[0][1].memory[0, :, i].reshape(16, 16)
        node_0_1 = cv2.resize(
            node_0_1, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(os.path.join(out_dir_path, f"node_0_1_{i:0>2}.png"), node_0_1)

    # for i in range(128):  # second layer  repr size
    # node_1_0 = network.layers[1][0].memory[0, :, i].reshape()


if __name__ == "__main__":
    train_rao()
