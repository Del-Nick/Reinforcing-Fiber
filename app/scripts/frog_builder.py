from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from app.field import Field
from app.optical_elements.compressor import Compressor
from app.sim_steps import StateSaver

BASE = Path(__file__).parent.parent.parent
exp_folder = BASE / 'NotCode' / 'Эксперимент' / 'FROG-спектрограммы для сравнения'

MODE: Literal['Gauss Pulse v2', 'Experimental Pulse v2'] = 'Gauss Pulse v2'
state_saver = StateSaver(BASE / 'NotCode' / 'Graphs' / MODE / 'states')


@dataclass
class ExperimentalFROG:
    filename: Path

    M: int = None
    N: int = None
    dt: float = None
    d_lam: float = None
    lam_0: float = None

    amps: np.ndarray = None
    delays: np.ndarray = None
    lams: np.ndarray = None

    def __post_init__(self):
        self._get_config()
        self._load_amps()
        self._generate_axes()

    def _get_config(self):
        with open(self.filename, 'r') as f:
            self.M = int(f.readline().rstrip('\r\n'))
            self.N = int(f.readline().rstrip('\r\n'))
            self.dt = float(f.readline().rstrip('\r\n'))
            self.d_lam = float(f.readline().rstrip('\r\n'))
            self.lam_0 = float(f.readline().rstrip('\r\n'))

    def _load_amps(self):
        exp_data = np.loadtxt(self.filename, skiprows=5, delimiter='	')
        exp_data /= exp_data.max()
        exp_data[exp_data <= 0] = exp_data[exp_data > 0].min()
        self.amps = exp_data

    def _generate_axes(self):
        self.delays = (np.arange(self.M) - self.M // 2) * self.dt * 1e-15
        self.lams = (np.arange(self.N) - self.N // 2) * self.d_lam + self.lam_0


class FROG:
    amps: np.ndarray

    def __init__(self, exp: ExperimentalFROG, filename: str):
        self._build_frog(exp=exp, filename=filename)

    def _build_frog(self, exp: ExperimentalFROG, filename: str):
        self.E: Field = state_saver.restore_to(Field(), filename=filename)

        compressor = Compressor()
        self.E.Ez = compressor.compress_it(self.E, gamma=24.349385552998392, l_g=0.442092062038187)
        self.E.change_grid(Nt=2 ** 16, time_window=5e-10)

        self.amps = np.empty((exp.M, self.E.Ez.size), dtype=np.float64)

        for i, tau in enumerate(tqdm(exp.delays)):
            E_delayed = self.E.methods.ifft(self.E.methods.fft(self.E.Ez) * self.E.xp.exp(-1j * self.E.grid.w * tau))
            S_w = self.E.methods.fft(self.E.Ez * E_delayed)
            self.amps[i, :] = np.abs(S_w) ** 2

        self.amps /= self.amps.max()


# pipeline = (
#     ('1 MHz passive AMP3 CFBG = -3_10_0', '22 E=34.37 nJ'),
#     ('1 MHz 2.1A 1.05 W CFBG = -3_10_0', '22 E=1019.86 nJ'),
#     ('1 MHz 2.9A 1.9 W CFBG = -2_-15_0', '22 E=1900.74 nJ')
# )
pipeline = (
    ('phi = 1.5 theta =-0.5 I_pa2 = 1.1A FBF38 1.5m PM980', '14'),
)
for exp_file, state in pipeline:
    # exp = ExperimentalFROG(filename=exp_folder / 'AMP3' / '1 MHz' / f'{exp_file}.txt')
    exp = ExperimentalFROG(filename=exp_folder / 'AMP2' / f'{exp_file}.txt')
    frog = FROG(exp, filename=state)
    exp.delays *= 1e15

    plt.figure(figsize=(12, 6), num=f'{frog.E.energy:.2f} nJ')
    plt.suptitle(f'{frog.E.energy:.2f} nJ')
    # Эксперимент
    plt.subplot(1, 2, 1)
    plt.pcolormesh(exp.delays, exp.lams, exp.amps.T, cmap='inferno', shading='auto',
                   norm=LogNorm(vmin=1e-4, vmax=1.0)) # лучше задать vmin явно
    plt.xlabel("Задержка τ, фс"); plt.ylabel("Длина волны, нм"); plt.title('Эксперимент')
    plt.colorbar()

    idx = np.argsort(frog.E.grid.lam)
    lam = frog.E.grid.lam[idx] * 1e9
    amps = frog.amps[:, idx]

    # Расчёт
    plt.subplot(1, 2, 2)
    plt.pcolormesh(exp.delays, lam[::10] / 2, amps[:, ::10].T, cmap='inferno', shading='auto',
                   norm=LogNorm(vmin=1e-5, vmax=1.0)) # лучше задать vmin явно
    plt.xlabel("Задержка τ, фс"); plt.ylabel("Длина волны, нм"); plt.title('Расчёт')

    plt.ylim(exp.lams.min(), exp.lams.max())
    plt.xlim(exp.delays.min(), exp.delays.max())
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(exp_folder / exp.filename.with_name(exp.filename.stem + f' моделирование {frog.E.energy:.2f} nJ' + '.png'), dpi=300)
    plt.close()
