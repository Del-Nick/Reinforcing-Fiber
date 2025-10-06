import typing
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from app.backend.base import FFTBackend, XP

if typing.TYPE_CHECKING:
    from app.field import Field


@dataclass
class Methods:
    backend: FFTBackend
    # Для динамического переключения между E.xp и cp
    xp: XP = None

    def __post_init__(self):
        self.xp = self.backend.xp

    def fft(self, a):  return self.backend.fft(a)
    def ifft(self, a): return self.backend.ifft(a)
    def fftshift(self, a): return self.backend.fftshift(a)
    def ifftshift(self, a): return self.backend.ifftshift(a)

    def dispersion(self, Ez, phase, dz: float):
        return self.ifft(self.fft(Ez) * self.xp.exp(-1j * phase * dz))

    def spm(self, Ez, gamma):
        return self.ifft(gamma * self.fft(-1j * self.xp.abs(Ez) ** 2 * Ez))

    def rk4(self, Ez, gamma, dz: float):
        k1 = dz * self.spm(Ez,           gamma)
        k2 = dz * self.spm(Ez + .5 * k1, gamma)
        k3 = dz * self.spm(Ez + .5 * k2, gamma)
        k4 = dz * self.spm(Ez + k3,      gamma)
        return Ez + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def gain(self, E: 'Field', dz: float):
        g_eff = 0.5 * E.fiber.g_0 / (1 + E.energy * 1e-9 / E.fiber.W_s) * E.fiber.gain_profile
        return self.ifft(self.fft(E.Ez) * self.xp.exp(g_eff * dz))

    def beam_evo(self, E: 'Field', num: int, gain: bool = False):
        E.plot.update(E=E, num=num)

        B_integral = 0

        # Максимально допустимое изменение фазы на шаге интегрирования
        d_phi, dz_max = 1e-3, .01

        z = 0
        pbar = tqdm(range(1_000_000), desc=f'name = {E.fiber.name} | '
                                             f'length = {E.fiber.length} | '
                                             f'z = {z:.3f} ({z / E.fiber.length:.2f}%) | '
                                             f'dz = {dz_max}')

        # z_array, E_array = [0], [E.energy]

        for _ in pbar:
            dz = min(d_phi / (E.fiber.gamma_without_A_eff[0] * self.xp.max(self.xp.abs(E.Ez)) ** 2).item(), dz_max,
                         E.fiber.length - z)

            if gain:
                E.Ez = self.gain(E=E, dz=dz)

            E.Ez = self.rk4(Ez=E.Ez, gamma=E.fiber.gamma_without_A_eff, dz=dz)
            E.Ez = self.dispersion(Ez=E.Ez, phase=E.fiber.phase, dz=dz)

            z += dz
            B_integral += float((E.fiber.gamma_without_A_eff[0] * self.xp.max(self.xp.abs(E.Ez)) ** 2) * dz)

            pbar.set_description(f'name = {E.fiber.name} | '
                                 f'length = {E.fiber.length} | '
                                 f'z = {z:.3f} ({z / E.fiber.length * 100:.2f}%) | '
                                 f'dz = {dz}')

            # if gain:
            #     E_array.append(E.energy)
            #     z_array.append(z)

            if _ % 1000 == 0:
                E.plot.update(E=E, num=num)

            if z >= E.fiber.length:
                break

        pbar.close()
        E.plot.update(E=E, num=num)

        # if gain:
        #     E.plot.plot_gain_history(z=z_array,
        #                              energy_nJ=E_array,
        #                              filename=f"gain_{num:02d}.png",
        #                              title=f"{E.fiber.name}: E(z)")

        return E.Ez
