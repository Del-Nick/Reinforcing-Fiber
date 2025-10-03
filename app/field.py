from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from app.backend.base import Array
from app.backend.cpu import CPUBackend
from app.backend.gpu import GPUBackend
from app.backend.methods import Methods
from app.config.constants import consts
from app.config.grid import Grid
from app.graphs.cpu_builder import LiveFourGraphPlotter
from app.optical_elements.fiber import Fiber


class Field:
    def __init__(self,
                 backend: Literal['cpu', 'gpu'] = 'cpu',
                 Nt: int = 2 ** 18,
                 time_window: float = 50e-12,
                 repeat_frequency: float = 71e6,
                 mean_power: float = 108e-3,
                 osc_filepath: Optional[Path] = None,
                 fiber: Fiber = None,):

        self.backend = GPUBackend() if backend == "gpu" else CPUBackend()
        self.xp = self.backend.xp
        self.methods = Methods(self.backend)

        self.consts = consts
        self.grid = Grid(Nt=Nt, time_window=time_window, xp=self.xp)
        self.plot: Optional[LiveFourGraphPlotter] = None

        self.repeat_frequency = repeat_frequency
        self.mean_power = mean_power
        self.initial_energy = self.mean_power / self.repeat_frequency
        self.experimental_graphs = self._load_all_experimental_graphs()

        self.fiber = fiber if fiber else Fiber(name='1. PLMA 25_250 4m', diameter=25,
                                               length=4.75, xp=self.xp, grid=self.grid, gain=False)

        self.E_0 = np.sqrt(self.initial_energy / (self.consts.T_P * self.fiber.A_eff))

        self._from_experiment(path=osc_filepath) if osc_filepath is not None else self._generate_field()
        self.Ez0 = self.Ez.copy()

        self.saver = None


    @property
    def intensity(self):
        return self.xp.abs(self.Ez) ** 2

    @property
    def intensity_prev(self):
        return self.xp.abs(self.Ez0) ** 2

    @property
    def power_spectral_density(self):
        return self.xp.abs(self.methods.fft(self.Ez)) ** 2

    @property
    def power_spectral_density_prev(self):
        return self.xp.abs(self.methods.fft(self.Ez0)) ** 2

    @property
    def energy(self) -> float:
        """
        Returns
        -------
        Энергия в нДж
        """
        return self.xp.sum(self.intensity).item() * self.grid.dt * self.fiber.A_eff * 1e9


    @property
    def energy_Ez0(self) -> float:
        """

        Returns
        -------
        Энергия в нДж
        """
        return self.xp.sum(self.xp.abs(self.Ez0) ** 2) * self.grid.dt * self.fiber.A_eff * 1e9

    @property
    def time_borders_FWHN(self) -> tuple[int, int]:
        intensity = self.intensity
        half = self.xp.max(intensity) * .5
        values = self.xp.where(intensity > half)[0]
        return values[0], values[-1] + 1

    @property
    def duration(self) -> float:
        level = .5
        intensity = self.intensity
        half = self.xp.max(intensity) * level
        left, right = self.time_borders_FWHN

        left = self._interp_by_idxs(left, half, self.grid.t, intensity)
        right = self._interp_by_idxs(right, half, self.grid.t, intensity)
        return float(right - left)

    @property
    def spectrum_borders_FWHM(self) -> tuple[int, int]:
        """

        Returns
        -------
        Возвращает левый и правый индексы после fftshift
        """
        psd = self.methods.fftshift(self.power_spectral_density)
        half = self.xp.max(psd) * .5
        values = self.xp.argwhere(psd > half)
        values = values.reshape((values.size,))
        return values[0], values[-1] + 1

    @property
    def spectrum_width_FWHM_nm(self) -> float:
        left, right = self.spectrum_borders_FWHM
        lam = self.methods.fftshift(self.grid.lam)
        psd = self.methods.fftshift(self.power_spectral_density)
        half = self.xp.max(psd) * 0.5

        left = self._interp_by_idxs(left, half, lam, psd)
        right = self._interp_by_idxs(right, half, lam, psd)
        return abs(float((right - left) * 1e9))

    def _interp_by_idxs(self, idx: int, half: float, x: Array, y: Array) -> float:
        """
        Интерполирует поле для более точного определения длительности импульса и ширины спектра
        Parameters
        ----------
        idx : int
                Индекс элемента, возле которого нужна аппроксимация
        half : float
                Значение на полувысоте
        x : np.ndarray | cp.ndarray
                Ось времени или длин волн
        y : np.ndarray | cp.ndarray
                Интенсивность или спектральная плотность мощности

        Returns
        -------
        Возвращает время или длину волны вблизи заданного индекса на полувысоте
        """
        y0, y1 = y[idx], y[idx + 1]
        x0, x1 = x[idx], x[idx + 1]
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)


    def change_grid(self, Nt: int, time_window: float):
        """
        Пересчитывает все сетки
        Parameters
        ----------
        Nt : int
                Количество точек в сетке
        time_window : float
                Устанавливает, какой временной диапазон в секундах должен помещаться на сетке

        Returns
        -------

        """
        old_t_cpu = self._asnumpy(self.grid.t)
        Ez_cpu = self._asnumpy(self.Ez)

        f = interp1d(x=old_t_cpu, y=Ez_cpu, kind="cubic", bounds_error=False, fill_value=0.0)
        self.grid = Grid(Nt=Nt, time_window=time_window, xp=self.xp)

        new_Ez_cpu = f(self._asnumpy(self.grid.t))
        self.Ez = self.xp.asarray(new_Ez_cpu, dtype=self.xp.complex128)
        self.Ez0 = self.Ez.copy()

    def _asnumpy(self, a: Array) -> np.ndarray:
        if getattr(self.xp, '__name__', None) == 'cupy':
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def _from_experiment(self, path: Path):
        data = np.loadtxt(path)
        lam_exp, I_exp = data[:, 0] * 1e-9, data[:, 1]
        I_exp[I_exp < 0] = 0

        I_exp *= np.exp(-((lam_exp - 1.058e-6) / 0.015e-6) ** 8)
        I_exp = I_exp / I_exp.max() * self.E_0

        w = self.grid.w
        lam = 2 * self.xp.pi * self.consts.C / (self.consts.OMEGA_0 + w)
        f = interp1d(x=lam_exp, y=I_exp, kind="cubic", bounds_error=False, fill_value=0.0)

        self.Ez = self.xp.asarray(f(self._asnumpy(lam)), dtype=self.xp.complex128)
        self.Ez = self.backend.fftshift(self.methods.ifft(self.Ez))

        energy = self.xp.sum(self.xp.abs(self.Ez) ** 2) * self.grid.dt * self.fiber.A_eff
        self.Ez *= self.xp.sqrt(self.initial_energy / energy)

    def _generate_field(self):
        tau = 215e-15
        sigma = tau / (2.0 * (2.0 * self.xp.log(2.0))**0.5)
        I0 = self.initial_energy / ((2.0 * self.xp.pi)**0.5 * sigma) / self.fiber.A_eff
        self.Ez = self.xp.sqrt(I0 * self.xp.exp(-self.grid.t**2 / (2.0 * sigma**2)))
        self.Ez = self.xp.asarray(self.Ez, dtype=self.xp.complex128)

    def _load_all_experimental_graphs(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        filespath = Path(__file__).parent.parent / 'NotCode' / 'Эксперимент'
        files = glob(f'{filespath}/*.dat')

        arrays = {}
        for file in files:
            num = int(file.split('/')[-1].split('.')[0])
            lam, ampl = np.loadtxt(file, dtype=np.float64).T
            arrays[num] = (lam * 1e-9, ampl)

        return arrays
