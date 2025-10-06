from multiprocessing import cpu_count
import numpy as np
import pyfftw

from app.backend.base import XP


class CPUBackend:
    def __init__(self):
        self.xp: XP = np
        pyfftw.interfaces.cache.enable()
        self._plans: dict[tuple[tuple, np.dtype], tuple] = {}

    def _get_plans(self, a: np.ndarray):
        key = (a.shape, a.dtype)

        if key not in self._plans:
            a_in = pyfftw.empty_aligned(shape=a.shape, dtype=a.dtype)
            a_out = pyfftw.empty_aligned(shape=a.shape, dtype=a.dtype)
            forward = pyfftw.FFTW(a_in, a_out, flags=('FFTW_MEASURE',), threads=cpu_count())
            backward = pyfftw.FFTW(a_in, a_out, direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',), threads=cpu_count())

            self._plans[key] = (forward, backward)

        return self._plans[key]

    def fft(self, a: np.ndarray) -> np.ndarray:
        fwd, _ = self._get_plans(a)
        fwd.input_array[...] = a
        return fwd()

    def ifft(self, a: np.ndarray) -> np.ndarray:
        _, inv = self._get_plans(a)
        inv.input_array[...] = a
        y = inv()
        return y

    def fftfreq(self, n: int, d: float):
        """
        Parameters
        ----------
        n : int
            Количество точек в массиве
        d : int
            Расстояние между точками

        Returns
        -------

        """
        return self.xp.fft.fftfreq(n, d)

    def fftshift(self, x):
        return self.xp.fft.fftshift(x)

    def ifftshift(self, x):
        return self.xp.fft.ifftshift(x)
