import typing
from typing import Optional

import numpy as np
from numpy import pi
from numpy.polynomial import Chebyshev, Legendre
import sympy as sp
from sympy.utilities.autowrap import ufuncify, autowrap
from scipy.optimize import differential_evolution, minimize
from scipy.special import factorial
from tqdm import tqdm

from app.config.constants import consts

import matplotlib.pyplot as plt

if typing.TYPE_CHECKING:
    from app.field import Field


class Compressor:
    def __init__(self, gamma: Optional[float] = 51, l_g: Optional[float] = .392):
        self.d = 1 / 1_200_000
        self.gamma = gamma
        self.l_g = l_g
        self._symbolic_derivation()

    def _symbolic_derivation(self):
        """
                  λ³       1
        β₂ = - ——————— ————————— L
                πс²d²  cos[θ(λ)]

                  3λ         λ   tan[θ(λ)]
        β₃ = β₂ —————— {1 + ——— ———————————}
                 2πс         d   cos[θ(λ)]

                               λ
        cos[θ(λ)] = sqrt(1 - (——— - sin(γ))²)
                               d
        :return:
        """
        wavelengths, w, l_g, gamma, beta2, beta3, beta4, beta5 = sp.symbols('λ w l_g gamma β_2 β_3 β_4 β_5')
        wavelengths = 2 * pi * consts.C / w
        cos_theta = sp.sqrt(1 - (wavelengths / self.d - sp.sin(gamma)) ** 2)
        L = l_g / cos_theta
        self.beta2 = - wavelengths ** 3 / (pi * consts.C ** 2 * self.d ** 2 * cos_theta ** 2) * L
        self.beta3 = sp.diff(self.beta2, w)
        self.beta4 = sp.diff(self.beta3, w)
        self.beta5 = sp.diff(self.beta4, w)

        self.beta2 = np.vectorize(autowrap(self.beta2, args=(w, l_g, gamma), backend='cython'))
        self.beta3 = np.vectorize(autowrap(self.beta3, args=(w, l_g, gamma), backend='cython'))
        self.beta4 = np.vectorize(autowrap(self.beta4, args=(w, l_g, gamma), backend='cython'))
        self.beta5 = np.vectorize(autowrap(self.beta5, args=(w, l_g, gamma), backend='cython'))

    def _prepare_arrays(self, E: 'Field'):
        fftshift = E.methods.fftshift
        self.xp = E.xp
        self.w = fftshift(E.grid.w)
        self.lam = fftshift(E.grid.lam)
        self.PSD = fftshift(E.power_spectral_density)
        self.PSD /= self.PSD.max()
        self.range_spectrum = self.xp.argwhere(self.PSD > 1e-2).reshape((-1,))

        # всё считаем в «сдвинутом» мире, чтобы индексы совпадали
        S = fftshift(E.methods.fft(fftshift(E.Ez)))
        self.amp = self.xp.abs(S)

    def _get_pulse_betas(self, E: 'Field'):
        fftshift = E.methods.fftshift
        phase = E.xp.unwrap(np.angle(fftshift(E.methods.fft(fftshift(E.Ez)))))
        coeffs = E.xp.polyfit(self.w[self.range_spectrum], phase[self.range_spectrum], 5)[:-2]
        for i in range(len(coeffs)-1, -1, -1):
            coeffs[i] *= factorial(len(coeffs) - i + 1).item()
        return coeffs[::-1]
    
    def _get_spectral_phase(self, E: 'Field'):
        coeffs = self._get_pulse_betas(E)
        phi = E.xp.zeros(E.grid.w.shape)
        for i, beta in enumerate(coeffs):
            phi += beta / factorial(i + 2).item() * self.w ** (i + 2)
        phi = E.xp.where(E.xp.isfinite(phi), phi, 0.0)
        return phi

    def _get_compressor_phase(self, E: 'Field', gamma: float = None, l_g: float = None) -> np.ndarray:
        gamma = gamma if gamma is not None else self.gamma
        l_g = l_g if l_g is not None else self.l_g

        beta2 = self.beta2(E.consts.OMEGA_0, l_g, gamma)
        beta3 = self.beta3(E.consts.OMEGA_0, l_g, gamma)
        beta4 = self.beta4(E.consts.OMEGA_0, l_g, gamma)
        beta5 = self.beta5(E.consts.OMEGA_0, l_g, gamma)

        phi = (beta2 / 2 * self.w ** 2 +
                beta3 / 6 * self.w ** 3 +
                beta4 / 24 * self.w ** 4 +
                beta5 / 120 * self.w ** 5)
        phi = E.xp.where(E.xp.isfinite(phi), phi, 0.0)
        return phi

    def _search_optimal_parameters(self, params: tuple[float, float], E: 'Field') -> float:
        gamma_deg, L = params
        gamma = np.deg2rad(gamma_deg)
        xp = E.methods.xp

        # фаза входа
        phi_in = self._get_spectral_phase(E)

        # фаза компрессора (возвращает np-массив → переведём в xp, если нужно)
        phi_c = self._get_compressor_phase(E, gamma=gamma, l_g=L)

        # остаточная фаза
        phi_res = phi_in[self.range_spectrum] - phi_c[self.range_spectrum]
        if xp.any(~xp.isfinite(phi_res)).item():
            return xp.inf

        return xp.sum(phi_res ** 2).item()

    def _polish_local(self, x0, E, bounds):
        fun = lambda x: self._search_optimal_parameters((x[0], x[1]), E)
        res = minimize(fun, x0=np.array(x0), bounds=bounds, method='L-BFGS-B',
                       options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 2000})
        return res.x, res.fun

    def _search_parameters(self, E: 'Field', bounds=None, tol: float = 1e-10, maxiter: int = 2000) -> tuple[float, float]:
        bounds = [(20, 30), (0.4, 0.5)] if bounds is None else bounds

        self._de_iter = 0
        self.pbar = tqdm()

        def _cb(xk, convergence):
            self._de_iter += 1
            self.pbar.set_description(f"[DE] it={self._de_iter:03d}  gamma={xk[0]:.4f}°  L={xk[1]:.6f}")

        res = differential_evolution(
            func=self._search_optimal_parameters,
            bounds=bounds,
            init='latinhypercube',
            strategy='currenttobest1bin',
            args=(E,),
            tol=tol,
            atol=1e-12,
            maxiter=maxiter,
            polish=False,
            workers=1,
            updating="deferred",
            callback=_cb,
        )
        self.pbar.close()
        x_best = res.x
        # локальный полиш с сужением границ
        b_polish = [(x_best[0] - 0.3, x_best[0] + 0.3), (x_best[1] - 0.01, x_best[1] + 0.01)]
        x_loc, _ = self._polish_local(x_best, E, b_polish)
        return x_loc

    def compress_it(self, E: 'Field', gamma: Optional[float] = None, l_g: Optional[float] = None):
        """
        Сжимает импульс
        Parameters
        ----------
        E : Field
                Объект поля
        gamma : float | None
                Угол между решётками в градусах
        l_g : float | None
                Расстояние между решётками в метрах

        Returns
        -------
        Огибающую поля
        """
        self._prepare_arrays(E)
        if gamma is None:
            gamma, l_g = self._search_parameters(E)
            print(f'gamma={gamma}, l_g={l_g}')

        gamma = E.xp.deg2rad(gamma).item()
        compressed = E.methods.ifft(E.methods.fft(E.Ez) *
                                    E.xp.exp(-1j * E.methods.fftshift(self._get_compressor_phase(E, gamma=gamma, l_g=l_g))))

        return compressed

