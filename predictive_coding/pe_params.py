from pydantic import BaseModel


class PEParams(BaseModel):
    memory_loc: float
    memory_scale: float
    k1: float
    k2: float
    sigma_sq: float
    alpha: float
    lambd: float
