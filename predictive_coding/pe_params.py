from pydantic import BaseModel


class PEParams(BaseModel):
    memory_uniform_low: float
    memory_uniform_high: float
    k1: float
    k1_decay: float = 1.0
    k2: float
    k2_decay: float = 1.0
    sigma_sq: float
    alpha: float
    lambd: float
