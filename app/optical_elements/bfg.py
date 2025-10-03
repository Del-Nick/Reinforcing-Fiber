from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from scipy.optimize import least_squares

from app.backend.base import Array, XP
from app.config.constants import consts

if typing.TYPE_CHECKING:
    from app.field import Field


BASE = Path(__file__).parent.parent.parent


def supergauss(w: np.ndarray, A: float, w0: float, d: float, p: float,) -> np.ndarray:
    """Модель супергаусса: A * exp(-abs((x - x0)/w)**p) + B"""
    return A * np.exp(-((w-w0) / d) ** p)

@dataclass
class SuperGaussFit:
    A: float
    w0: float
    d: float
    p: float

    def model(self) -> Callable[[np.ndarray], np.ndarray]:
        return lambda w: supergauss(w, self.A, self.w0, self.d, self.p)

def fit_supergauss(
    w: np.ndarray,
    y: np.ndarray,
    p_fixed: Optional[int] = None,
    robust: bool = True,
) -> SuperGaussFit:
    """
    Аппроксимирует (w, y) супергауссом:
        y ≈ A * exp(-((w - w0)/d)^(2p))

    Параметры
    ---------
    w, y : массивы одной длины
    p_fixed : если задано, порядок p фиксируется (целое).
    robust : использовать робастную потерю 'soft_l1'.

    Возвращает
    ----------
    SuperGaussFit: параметры A, w0, d, p.
    """
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)

    # Начальные оценки
    A0 = np.max(y)
    w0_0 = w[np.argmax(y)]
    d0 = (w.max() - w.min()) / 5
    p0 = 2 if p_fixed is None else float(p_fixed)

    if p_fixed is None:
        # Оптимизируем A, w0, d, p
        x0_params = np.array([A0, w0_0, d0, p0])
        lb = [0, w.min(), (np.ptp(w))/1000, 1.0]
        ub = [np.inf, w.max(), np.ptp(w)*10, 20.0]
        def resid(theta):
            A, w0, d, p = theta
            return supergauss(w, A, w0, d, p) - y
    else:
        # Оптимизируем A, w0, d при фиксированном p
        x0_params = np.array([A0, w0_0, d0])
        lb = [0, w.min(), (np.ptp(w))/1000]
        ub = [np.inf, w.max(), np.ptp(w)*10]
        def resid(theta):
            A, w0, d = theta
            return supergauss(w, A, w0, d, float(p_fixed)) - y

    loss = 'soft_l1' if robust else 'linear'
    lsq = least_squares(resid, x0_params, bounds=(lb, ub), loss=loss)

    if p_fixed is None:
        A, w0, d, p = lsq.x
    else:
        A, w0, d = lsq.x
        p = float(p_fixed)

    return SuperGaussFit(A=A, w0=w0, d=d, p=p)


@dataclass(slots=True)
class BFG:
    @staticmethod
    def _rho_poly(w: Array, xp: XP, E) -> Array:
        R = xp.sqrt(0.8722055018389234 * exp(-((w + 158803885884.2738) / 17238621280048.354) ** 20))
        phi = (w ** 2 * 19.28126e-24 / 2 -
               w ** 3 * .16907e-36 / 6 +
               w ** 4 * .0023141e-48 / 24 -
               w ** 5 * .000045379e-60 / 120)

        return R * xp.exp(-1j * phi)

    def _plot_reflectivity(self, E: 'Field', R):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(E._asnumpy(E.grid.lam), E._asnumpy(R))
        ax.grid()
        plt.show(block=True)

    def stretch_it(self, E: 'Field'):
        S = E.methods.fft(E.methods.ifftshift(E.Ez))
        S_out = S * self._rho_poly(E.grid.w, E.xp, E)
        return E.methods.fftshift(E.methods.ifft(S_out))


if __name__ == '__main__':
    lam, amp = np.loadtxt(fname=BASE / 'Параметры элементов' / 'BFG Real.dat', dtype=np.float64).T
    lam *= 1e-9
    amp *= 1e-2

    w = 2 * np.pi * consts.C / lam - consts.OMEGA_0

    best_fit = None
    best_resid = np.inf
    for p in range(1, 41):
        fit = fit_supergauss(w, amp, p_fixed=p * 2)
        resid = np.sum((fit.model()(w) - amp) ** 2)
        if resid < best_resid:
            best_resid = resid
            best_fit = fit
            best_p = p * 2

    print(best_fit)
    print(best_p)