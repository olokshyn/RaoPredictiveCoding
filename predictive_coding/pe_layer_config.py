from pydantic import BaseModel

from predictive_coding.pe_params import PEParams


class PELayerConfig(BaseModel):
    num_nodes: int
    repr_size: int
    params: PEParams
