from dataclasses import dataclass, field

from app.backend.base import XP, Array
from app.config.constants import consts


@dataclass(frozen=True, slots=True)
class Grid:
    Nt: int
    time_window: float
    xp: XP         # np or cp
    dt: float = None
    t: Array = field(init=False, repr=False)
    freqs: Array = field(init=False, repr=False)
    w: Array = field(init=False, repr=False)
    lam: Array = field(init=False, repr=False)

    def __post_init__(self):
        xp = self.xp
        dt = self.time_window / self.Nt
        t = (xp.arange(self.Nt) - self.Nt // 2) * dt
        freqs = xp.fft.fftfreq(self.Nt, dt)
        w = 2 * xp.pi * freqs
        lam = consts.C / (consts.F_0 + freqs)

        object.__setattr__(self, 'dt', dt)
        object.__setattr__(self, 't', t)
        object.__setattr__(self, 'freqs', freqs)
        object.__setattr__(self, 'w', w)
        object.__setattr__(self, 'lam', lam)


# class GridType(Enum):
#