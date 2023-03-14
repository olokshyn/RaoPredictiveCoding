import os
from collections.abc import Iterator

import numpy as np
import cv2
from pydantic import BaseModel


class ProcessConfig(BaseModel):
    patch_width: int = 26
    patch_height: int = 16
    scale: float = 1.0
    apply_dog: bool = True
    dog_kernel_size: tuple[int, int] = (5, 5)
    dog_sigma1: float = 1.3
    dog_sigma2: float = 2.6


def load_image(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return image


def load_and_process_batch(
    dir_path: str, config: ProcessConfig = ProcessConfig()
) -> np.ndarray:
    images = []
    for image_name in sorted(os.listdir(dir_path)):
        image_path = os.path.join(dir_path, image_name)
        image = load_image(image_path)
        image = process_image(image, config)
        images.append(image)
    return np.array(images)


def process_image(
    image: np.ndarray, config: ProcessConfig = ProcessConfig()
) -> np.ndarray:
    if config.apply_dog:
        image = difference_of_gaussian_blurs_filter(
            image,
            kernel_size=config.dog_kernel_size,
            sigma1=config.dog_sigma1,
            sigma2=config.dog_sigma2,
        )
    image = image * config.scale

    patches_horizontal = image.shape[1] // config.patch_width
    patches_vertical = image.shape[0] // config.patch_height
    patched_image = np.zeros(
        (patches_vertical, patches_horizontal, config.patch_height, config.patch_width),
        dtype=np.float32,
    )
    for patch_vertical in range(patches_vertical):
        for patch_horizontal in range(patches_horizontal):
            y = (
                patch_vertical * config.patch_height,
                (patch_vertical + 1) * config.patch_height,
            )
            x = (
                patch_horizontal * config.patch_width,
                (patch_horizontal + 1) * config.patch_width,
            )
            patch = image[y[0] : y[1], x[0] : x[1]]
            patched_image[patch_vertical, patch_horizontal] = patch
    return patched_image


def restore_image(
    patched_image: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    patches_horizontal = patched_image.shape[1]
    patches_vertical = patched_image.shape[0]
    patch_width = patched_image.shape[3]
    patch_height = patched_image.shape[2]
    image_width = patches_horizontal * patch_width
    image_height = patches_vertical * patch_height
    image = np.zeros((image_height, image_width), dtype=np.float32)
    for patch_vertical in range(patches_vertical):
        for patch_horizontal in range(patches_horizontal):
            image[
                patch_vertical * patch_height : (patch_vertical + 1) * patch_height,
                patch_horizontal * patch_width : (patch_horizontal + 1) * patch_width,
            ] = patched_image[patch_vertical, patch_horizontal]
    image /= scale
    return image


def difference_of_gaussian_blurs_filter(
    image: np.ndarray,
    kernel_size: tuple[int, int],
    sigma1: float,
    sigma2: float,
) -> np.ndarray:
    image1 = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=sigma1)
    image2 = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=sigma2)
    return image1 - image2


def patch_iterator(
    patched_image: np.ndarray,
    window_size: tuple[int, int],
    window_step: tuple[int, int],
) -> Iterator[np.ndarray]:
    patches_vertical = patched_image.shape[0]
    patches_horizontal = patched_image.shape[1]
    for patch_vertical in range(patches_vertical):
        for patch_horizontal in range(patches_horizontal):
            patch = patched_image[patch_vertical][patch_horizontal]
            if window_step[0] > 0:
                # fmt: off
                horizontal_shifts = (
                    (patch.shape[1] - window_size[0])
                    // window_step[0] + 1
                )
                # fmt: on
            else:
                horizontal_shifts = 1
            if window_step[1] > 0:
                # fmt: off
                vertical_shifts = (
                    (patch.shape[0] - window_size[1])
                    // window_step[1] + 1
                )
                # fmt: on
            else:
                vertical_shifts = 1
            inputs = np.zeros(
                (horizontal_shifts * vertical_shifts, window_size[1], window_size[0]),
                dtype=np.float32,
            )
            for v_shift in range(vertical_shifts):
                for h_shift in range(horizontal_shifts):
                    x = h_shift * window_step[0]
                    y = v_shift * window_step[1]
                    inputs[v_shift * horizontal_shifts + h_shift] = patch[
                        y : y + window_size[1], x : x + window_size[0]
                    ]
            yield inputs


def gaussian_mask(width: int, sigma: float = 0.4) -> np.ndarray:
    mask = np.zeros((width, width), dtype=np.float32)
    c = width // 2
    for i in range(width):
        x = (i - c) / c
        for j in range(width):
            y = (j - c) / c
            r = np.sqrt(x * x + y * y)
            mask[j, i] = gauss(r, sigma)
    mask = mask / np.max(mask)
    return mask


def gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    sigma_sq = sigma * sigma
    return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x * x / (2 * sigma_sq))
