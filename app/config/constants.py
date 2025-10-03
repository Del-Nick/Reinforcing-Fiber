from dataclasses import dataclass, field
from numpy import pi

@dataclass(frozen=True, slots=True)
class Constants:
    C: float = 299_792_458.0
    LAMBDA_0: float = 1.058e-6
    T_P: float = 1.5e-13

    OMEGA_0: float = field(init=False, repr=False)
    F_0: float = field(init=False, repr=False)

    def __post_init__(self):
        OMEGA_0 = 2 * pi * self.C / self.LAMBDA_0
        F_0 = self.C / self.LAMBDA_0
        object.__setattr__(self, "OMEGA_0", OMEGA_0)
        object.__setattr__(self, "F_0", F_0)


consts = Constants()