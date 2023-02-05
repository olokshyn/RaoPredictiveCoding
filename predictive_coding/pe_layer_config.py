from predictive_coding.pe_params import PEParams


class PELayerConfig:
    def __init__(self, *, num_nodes: int, repr_size: int, params: PEParams) -> None:
        self.num_nodes = num_nodes
        self.repr_size = repr_size
        self.params = params
