import typing
from typing import Optional

import numpy as np
from numpy import pi
from numpy.polynomial import Chebyshev, Legendre
import sympy as sp
from sympy.utilities.autowrap import ufuncify, autowrap
from scipy.optimize import differential_evolution
from scipy.special import factorial
from app.config.constants import consts

import matplotlib.pyplot as plt

if typing.TYPE_CHECKING:
    from app.field import Field


class Compressor:
    def __init__(self, gamma: Optional[float] = 51, l_g: Optional[float] = .392):
        self.d = 1 / 1_200_000
        self.gamma = np.deg2rad(gamma).item()
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

        # всё считаем в «сдвинутом» мире, чтобы индексы совпадали
        S = fftshift(E.methods.fft(fftshift(E.Ez)))
        self.amp = self.xp.abs(S)

    def _get_pulse_betas(self, E: 'Field'):
        fftshift = E.methods.fftshift
        idxs = E.xp.argwhere(self.PSD > self.PSD.max() * 1e-1).reshape((-1,))
        phase = E.xp.unwrap(np.angle(fftshift(E.methods.fft(fftshift(E.Ez)))))
        # coeffs = E.xp.polyfit(self.w[idxs], phase[idxs], 5)[:-2]
        # for i in range(len(coeffs)-1, -1, -1):
        #     coeffs[i] *= factorial(len(coeffs) - i + 1).item()
        # return coeffs[::-1]
        cheb = Chebyshev.fit(E._asnumpy(self.w), E._asnumpy(phase), deg=6, w=E._asnumpy(self.PSD))
        coeffs = cheb.convert(kind=np.polynomial.Polynomial).coef[2:]
        for i in range(len(coeffs)):
            coeffs[i] *= factorial(i + 2).item()
        return coeffs
    
    def _get_spectral_phase(self, E: 'Field'):
        coeffs = self._get_pulse_betas(E)
        phi = E.xp.zeros(E.grid.w.shape)
        for i, beta in enumerate(coeffs):
            phi += beta / factorial(i + 2).item() * self.w ** (i + 2)
        phi = E.xp.where(E.xp.isfinite(phi), phi, 0.0)
        return phi

        # xp = E.xp
        # if not hasattr(self, "S"):
        #     self._prepare_arrays(E)
        # S = self.S
        # phi = xp.unwrap(xp.angle(S))
        # # полоса по мощности
        # m = self.PSD > (self.PSD.max() * 1e-2)
        # # убрать piston+tilt (взвешенно по PSD⋅mask)
        # coeffs = self._get_pulse_betas(E)
        # phi_clean = phi - coeffs[0] - coeffs[1] * self.w
        # return phi_clean

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

    def _remove_piston_tilt(self, phi_res, weights, xp):
        """
            phi_res(ω): остаточная фаза (xp-массив 1D)
            weight: w(ω) = |E(ω)|^2 (xp-массив 1D)

            Возвращает: resid (xp), a0 (float), a1 (float)
            """
        W = weights
        w = self.w

        A0 = xp.ones_like(w)

        # Нормальные уравнения (взвешенные)
        WA0 = W * A0
        WA1 = W * w
        M00 = xp.sum(A0 * WA0)
        M01 = xp.sum(A0 * WA1)
        M11 = xp.sum(w * WA1)
        b0 = xp.sum(phi_res * WA0)
        b1 = xp.sum(phi_res * WA1)

        det = M00 * M11 - M01 * M01 + 1e-30
        a0 = (b0 * M11 - b1 * M01) / det
        a1 = (-b0 * M01 + b1 * M00) / det

        fit = a0 + a1 * w
        resid = phi_res - fit
        return resid, a0, a1

    def _get_weighted_rms(self, resid, weight, xp):
        num = xp.sum(weight * resid ** 2)
        den = xp.sum(weight) + 1e-300
        val = xp.sqrt(num / den)
        return float(val.item() if hasattr(val, "item") else val)

    def _side_energy_penalty(self, amp, resid, E, xp, k=3.0):
        # amp=|E(ω)|, resid=остаточная фаза после удаления a0,a1
        Et = E.methods.ifft(amp * xp.exp(1j * resid))
        It = xp.abs(Et)**2
        t = E.grid.t

        imax = xp.max(It)
        imax = float(imax.item() if hasattr(imax, "item") else imax)
        if imax <= 0:
            return 0.0

        mask = It > 0.5 * imax
        has_main = bool(xp.any(mask))
        if has_main:
            t_main = t[mask]
            width = xp.max(t_main) - xp.min(t_main)
            width = float(width.item() if hasattr(width, "item") else width)
            t0 = t[xp.argmax(It)]
            t0 = float(t0.item() if hasattr(t0, "item") else t0)
            core = xp.abs(t - t0) <= max(3.0 * width, 1e-15)
        else:
            core = xp.ones_like(It, dtype=bool)

        side = xp.sum(It[~core]) / (xp.sum(It) + 1e-300)
        return float(side.item() if hasattr(side, "item") else side)

    def _search_optimal_parameters(self, params: tuple[float, float], E: 'Field') -> float:
        gamma_deg, L = params
        gamma = np.deg2rad(gamma_deg)
        xp = E.methods.xp

        # фаза входа
        phi_in = self._get_spectral_phase(E)

        # фаза компрессора (возвращает np-массив → переведём в xp, если нужно)
        phi_c = self._get_compressor_phase(E, gamma=gamma, l_g=L)

        # остаточная фаза
        phi_res = phi_in + phi_c
        if not xp.all(xp.isfinite(phi_res)):
            return 1e9

        return xp.sum(self.PSD * phi_res ** 2).item()

        # веса
        W = self.PSD

        # убираем пистон + наклон
        resid, *_ = self._remove_piston_tilt(phi_res, W, xp)

        # метрика: RMS резидуальной фазы + небольшой штраф по хвостам
        num = xp.sum(W * resid ** 2)
        den = xp.sum(W) + 1e-30
        return float(num / den)

    def _search_parameters(self, E: 'Field', bounds=None, tol: float = 1e-4, maxiter: int = 200) -> tuple[float, float]:
        bounds = [(35, 55), (0.34, 0.54)] if bounds is None else bounds

        self._de_iter = 0

        def _cb(xk, convergence):
            self._de_iter += 1
            print(f"[DE] it={self._de_iter:03d}  gamma={xk[0]:.4f}°  L={xk[1]:.6f}  conv={convergence:.3e}")

        res = differential_evolution(
            func=self._search_optimal_parameters,
            bounds=bounds,
            args=(E,),
            tol=tol,
            maxiter=maxiter,
            seed=42,
            polish=True,
            workers=1,
            updating="deferred",
            disp=True,
            callback=_cb,
        )
        return res.x

    def compress_it(self, E: 'Field', gamma: Optional[float] = None, l_g: Optional[float] = None):
        if gamma is None:
            self._prepare_arrays(E)
            gamma, l_g = self.gamma, self.l_g
        #     gamma, l_g = self._search_parameters(E)
        #
        xp = E.xp
        # phi_in = self._get_spectral_phase(E)[::1000]
        #
        # # фаза компрессора (возвращает np-массив → переведём в xp, если нужно)
        # phi_c = self._get_compressor_phase(E, gamma=xp.deg2rad(gamma).item(), l_g=l_g)[::1000]
        #
        # # остаточная фаза
        # phi_res = (phi_in + phi_c)
        # lam = E._asnumpy(self.lam[::1000])
        #
        # print(f'{gamma = }      {l_g = }')
        # print(f'Входной:        {self._get_pulse_betas(E)}')
        # beta2 = self.beta2(E.consts.OMEGA_0, l_g, gamma)
        # beta3 = self.beta3(E.consts.OMEGA_0, l_g, gamma)
        # beta4 = self.beta4(E.consts.OMEGA_0, l_g, gamma)
        # beta5 = self.beta5(E.consts.OMEGA_0, l_g, gamma)
        # print(f'Компрессор:     {[beta2, beta3, beta4, beta5]}')
        #
        # print(f'{consts.LAMBDA_0 = }')
        # idx = E.xp.argmax(self.PSD)
        # print(f'{self.lam[idx] = }')
        #
        # idx = np.argsort(lam)
        # fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True, num="Спектральная фаза")
        # ax.plot(lam[idx], E._asnumpy(phi_in)[idx], label='Входная')
        # ax.plot(lam[idx], E._asnumpy(phi_c)[idx], label='Компрессор', ls='--')
        # ax.plot(lam[idx], E._asnumpy(phi_res)[idx], label='Сумма', ls=(0, (1, 5)))
        # ax.set_xlim(1.04e-6, 1.07e-6)
        # ax.legend()
        # ax.grid()
        # plt.show(block=True)

        # compressed = E.methods.ifft(E.methods.fft(E.Ez) *
        #                             E.xp.exp(-1j * self._get_compressor_phase(E=E, gamma=gamma, l_g=l_g)))
        phi_in = self._get_spectral_phase(E)
        w = E.grid.w
        # phi_in = -(w ** 2 * 19.28126e-24 / 2 -
        #        w ** 3 * .16907e-36 / 6 +
        #        w ** 4 * .0023141e-48 / 24 -
        #        w ** 5 * .000045379e-60 / 120)

        compressed = E.methods.ifft(E.methods.fft(E.Ez) *
                                    E.xp.exp(-1j * E.methods.fftshift(phi_in)))

        return compressed, gamma, l_g

