import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.interpolate import interp1d

from app.backend.base import XP, Array
from app.config.constants import consts
from app.config.grid import Grid


class RefractiveIndex:
    def __init__(self):
        self._precompute_symbolic_equation()

    def get_quartz_refractive_index(self, lam, xp):
        """
        Считает массив коэффициентов преломления для массива длин волн по Сельмееру в квартце
        Parameters
        ----------
        lam : cp.ndarray | np.ndarray
                массив длин волн в метрах
        xp : cp | np
                backend для массива

        Returns
        -------
        Массив n(λ)
        """
        if getattr(xp, "__name__", str(xp)) == 'cupy':
            lam = xp.asnumpy(lam)
        lam_um = lam * 1e6  # Формула Сельмейера в мкм

        n = np.sqrt(1 + 0.6961663 * lam_um ** 2 / (lam_um ** 2 - 0.0684043 ** 2) +
                       0.4079426 * lam_um ** 2 / (lam_um ** 2 - 0.1162414 ** 2) +
                       0.8974794 * lam_um ** 2 / (lam_um ** 2 - 9.896161 ** 2) + 0j)
        return xp.asarray(n)

    def _get_sym_refractive_index(self, lam_um: sp.Symbol):
        return sp.sqrt(1 + 0.6961663 * lam_um ** 2 / (lam_um ** 2 - 0.0684043 ** 2) +
                       0.4079426 * lam_um ** 2 / (lam_um ** 2 - 0.1162414 ** 2) +
                       0.8974794 * lam_um ** 2 / (lam_um ** 2 - 9.896161 ** 2) + 0j)

    def _precompute_symbolic_equation(self):
        C = consts.C
        OMEGA_0 = consts.OMEGA_0

        w_sym = sp.symbols("w", real=True)
        lam_um = 2 * sp.pi * C / (OMEGA_0 + w_sym) * 1e6

        self.n = self._get_sym_refractive_index(lam_um)
        self.k = self.n * (w_sym + OMEGA_0) / C

        self.beta_1_sym = sp.diff(self.k, w_sym)
        self.beta_2_sym = sp.diff(self.beta_1_sym, w_sym)
        self.beta_3_sym = sp.diff(self.beta_2_sym, w_sym)
        self.beta_4_sym = sp.diff(self.beta_3_sym, w_sym)
        self.beta_5_sym = sp.diff(self.beta_4_sym, w_sym)

        self.b2 = float(self.beta_2_sym.subs({w_sym: 0}))
        self.b3 = float(self.beta_3_sym.subs({w_sym: 0}))
        self.b4 = float(self.beta_4_sym.subs({w_sym: 0}))
        self.b5 = float(self.beta_5_sym.subs({w_sym: 0}))

    def get_k_and_betas(self, w, xp):
        """
        Считает показатель преломления n(ω) в волокне и дисперсионные зависимости k(ω), β₂(ω) - β₅(ω)
        Parameters
        ----------
        w : cp.ndarray | np.ndarray
                Массив циклических частот
        xp : cp | np
                Бэкенд для расчёта
        key : str
                ключ для сохранения массива в кэш по правилу size_dt_first_last для сетки

        Returns
        -------

        """
        if getattr(xp, "__name__", str(xp)) == 'cupy':
            w = xp.asnumpy(w)
        w_sym = sp.symbols("w", real=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            n = sp.lambdify((w_sym,), self.n, "numpy")
            n = np.array(n(w), dtype=np.float64)
        n[np.isinf(n)] = 0

        k = n * (w + consts.OMEGA_0) / consts.C
        return xp.asarray(n), xp.asarray(k), self.b2, self.b3, self.b4, self.b5


refractive_index = RefractiveIndex()


@dataclass
class Fiber:
    name: str
    diameter: int           # Мкм
    length: float           # м
    grid: Grid
    xp: XP
    gain: bool = False
    g_0: float = 35         # Дб
    W_s: float = 1e-6       # Дж
    gain_norm_coef: float = 0.5

    n_nl: float = 2.5e-20       # n = n_0 + n_nl * |E|^2
    A_eff: float = None
    gamma_without_A_eff: Array = None
    alpha: float = 8.0          # Коэффициент потерь, Дб
    delta: float = 10           # Спектральный профиль усиления, ТГц
    gain_profile_path: Path = None

    n: Array = None
    k: Array = None
    beta2: Array = None
    beta3: Array = None
    beta4: Array = None
    beta5: Array = None

    L_disp: float = None
    phase: Array = None

    def __post_init__(self):
        xp = self.xp

        self.A_eff = xp.pi * (self.diameter * 1e-6) ** 2 / 4
        self.gamma_without_A_eff = self.n_nl * (consts.OMEGA_0 + self.grid.w) / consts.C

        if self.gain:
            self.g_0 = (self.g_0 / 10.0) * xp.log(10.0).item() / self.length
            self._build_gain_profile(self.grid, self.gain_norm_coef, xp)

        self.alpha = 10 ** (self.alpha / 20)
        self.delta *= 1e12 * xp.pi

        self.n, self.k, self.beta2, self.beta3, self.beta4, self.beta5 = (
            refractive_index.get_k_and_betas(w=self.grid.w, xp=xp)
        )

        self.L_disp = (consts.T_P * .5) ** 2 / xp.abs(self.beta2)
        w = self.grid.w
        self.phase = (
            self.beta2 * w**2 / 2.0
            + self.beta3 * w**3 / 6.0
            + self.beta4 * w**4 / 24.0
            + self.beta5 * w**5 / 120.0
        )

    def _build_gain_profile(self, grid: Grid, norm_coef: float, xp: XP):
        if not os.path.exists(self.gain_profile_path):
            self.gain_profile = xp.ones_like(grid.w)
            return

        lam, ampl_db = np.loadtxt(self.gain_profile_path, dtype=np.float64).T

        ampl_lin = 10 ** (ampl_db / 10)
        lam *= 1e-9

        grid_lam = xp.asnumpy(grid.lam) if getattr(xp, '__name__', None) == 'cupy' else grid.lam
        f = interp1d(x=lam, y=ampl_lin, kind="cubic", bounds_error=False, fill_value=0.0)
        gain_profile = f(grid_lam)
        gain_profile = np.sign(gain_profile) * np.abs(gain_profile) ** norm_coef
        gain_profile /= gain_profile.max()
        self.gain_profile = xp.asarray(gain_profile) ** 0
