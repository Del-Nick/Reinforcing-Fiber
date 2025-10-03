import cupy as cp
import cupyx.scipy.fft as cufft

from app.backend.base import XP


class GPUBackend:
    def __init__(self):
        self.xp: XP = cp

    def fft(self, a):
        # Планируем один раз под форму a и переиспользуем
        with cufft.get_fft_plan(a):
            return cufft.fft(a)

    def ifft(self, a):
        with cufft.get_fft_plan(a):
            return cufft.ifft(a)

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