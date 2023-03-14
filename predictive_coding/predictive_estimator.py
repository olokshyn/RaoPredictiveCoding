import numpy as np

from predictive_coding.pe_params import PEParams


class PredictiveEstimator:
    def __init__(
        self,
        *,
        input_size: tuple,
        repr_size: int,
        params: PEParams,
        sigma_sq_hl: float,
    ) -> None:
        self.input_size = input_size
        self.repr_size = repr_size
        self.params = params
        self.k1 = params.k1
        self.k2 = params.k2
        self.sigma_sq = params.sigma_sq
        self.sigma_hl_sq = sigma_sq_hl
        self.alpha = params.alpha
        self.lambd = params.lambd

        self.memory = np.random.uniform(
            low=params.memory_uniform_low,
            high=params.memory_uniform_high,
            size=(*self.input_size, self.repr_size)
        )
        self.repr = np.zeros(self.repr_size)

    def perceive(
        self, *, training: bool, errors: np.ndarray, preds_hl: np.ndarray | None = None
    ) -> np.ndarray:
        assert errors.shape == self.memory.shape[:-1]
        assert preds_hl is None or preds_hl.shape[1:] == self.repr.shape

        if preds_hl is None:
            errors_hl = np.zeros(shape=(1, self.repr_size))
        else:
            errors_hl = self.repr - preds_hl

        # TODO: Weight higher-layer errors
        mean_error_hl = np.mean(errors_hl, axis=0)
        mean_product = np.mean(
            [np.dot(memory.T, error) for memory, error in zip(self.memory, errors)],
            axis=0,
        )

        # fmt: off
        update_repr = (
            (self.k1 / self.sigma_sq) * mean_product
            - (self.k1 / self.sigma_hl_sq) * mean_error_hl
            - self.k1 * self.alpha * self.repr
        )
        # fmt: on

        if training:
            # fmt: off
            update_memory = (
                (self.k2 / self.sigma_sq) * np.outer(errors, self.repr).reshape(self.memory.shape)
                - self.k2 * self.lambd * self.memory
            )
            # fmt: on
            self.memory += update_memory
        self.repr += update_repr

        return errors_hl

    def predict(self) -> np.ndarray:
        return np.dot(self.memory, self.repr)

    def reset_representation(self) -> None:
        self.repr = np.zeros_like(self.repr)

    def decay_learning_rate(self):
        self.k1 /= self.params.k1_decay
        self.k2 /= self.params.k2_decay

    def reset_learning_rate(self):
        self.k1 = self.params.k1
        self.k2 = self.params.k2
