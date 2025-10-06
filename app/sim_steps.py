import typing
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import numpy as np

from app.backend.base import XP
from app.config.grid import Grid
from app.graphs.cpu_builder import LiveFourGraphPlotter
from app.optical_elements.bfg import BFG
from app.optical_elements.compressor import Compressor
from app.optical_elements.fiber import Fiber

if typing.TYPE_CHECKING:
    from app.field import Field


class StateSaver:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_state(self, E: 'Field', filename: str):
        Ez = E.Ez
        Ez0 = E.Ez0
        if getattr(E.xp, "__name__", "") == "cupy":
            import cupy as cp
            Ez, Ez0 = cp.asnumpy(Ez), cp.asnumpy(Ez0)

        meta = {
            'Nt': E.grid.Nt,
            'time_window': E.grid.time_window,
            'repeat_frequency': E.repeat_frequency,
            'fiber_name': E.fiber.name,
            'fiber_length': E.fiber.length,
            'fiber_diameter': E.fiber.diameter,
        }
        np.savez(self.base_path / filename, Ez=Ez, Ez0=Ez0, **meta)

    def load_state(self, filename: str) -> dict:
        data = np.load(self.base_path / f'{filename}.npz')
        return {key: data[key] for key in data}

    def restore_to(self, E: 'Field', filename: str):
        data = np.load(self.base_path / f'{filename}.npz')
        Nt = int(data['Nt'])
        time_window = float(data['time_window'])
        repeat_frequency = float(data['repeat_frequency'])

        # перестраиваем сетку поля под сохранённую
        E.change_grid(Nt=Nt, time_window=time_window)   # есть в Field.change_grid
        E.repeat_frequency = repeat_frequency

        Ez_np = data['Ez']
        Ez0_np = data['Ez0']

        if getattr(E.xp, "__name__", "") == "cupy":
            import cupy as cp
            E.Ez = cp.asarray(Ez_np)
            E.Ez0 = cp.asarray(Ez0_np)
        else:
            E.Ez = np.asarray(Ez_np)
            E.Ez0 = np.asarray(Ez0_np)

        # пересобираем fiber в соответствии с мета-данными и текущей сеткой E.grid
        name = str(data['fiber_name'])
        length = float(data['fiber_length'])
        diameter = int(data['fiber_diameter'])

        E.fiber = Fiber(name=name, diameter=diameter, length=length,
                        xp=E.xp, grid=E.grid)

        return E


class Step(ABC):
    def __init__(self, num: int, name: str):
        self.num = num
        self.name = name

    @abstractmethod
    def process(self, E: 'Field') -> 'Field':
        pass

    def __str__(self):
        return f'{self.num}. {self.name}'


class ChangeGrid(Step):
    def __init__(self, num: int, Nt: int, time_window: float):
        super().__init__(num=num, name=f'{num}. ChangeGrid(Nt=2**{int(np.log2(Nt))}, TW={time_window * 1e9} нс)')
        self.Nt = Nt
        self.time_window = time_window

    def process(self, E: 'Field') -> 'Field':
        E.change_grid(Nt=self.Nt, time_window=self.time_window)
        return E


class FiberProcess(Step):
    def __init__(self,
                 num: int,
                 length: float,
                 diameter: int,
                 name: str = '',
                 gain: bool = False,
                 field_scale_factor: Optional[float] = None,
                 g_0: Optional[float] = 35,
                 W_s: Optional[float] = 1e-6,
                 gain_norm_coef: Optional[float] = .5,
                 gain_profile_path: Path = None):

        super().__init__(num=num, name=f'{num}. {name}' if name else f'{num}')
        self.length = length
        self.diameter = diameter
        self.gain = gain
        self.field_scale_factor = field_scale_factor
        self.g_0 = g_0
        self.W_s = W_s
        self.gain_profile_path = gain_profile_path
        self.gain_norm_coef = gain_norm_coef
        self.fiber = None

    def process(self, E: 'Field') -> 'Field':
        self.fiber = Fiber(name=self.name,
                           diameter=self.diameter,
                           length=self.length,
                           gain=self.gain,
                           g_0=self.g_0,
                           W_s=self.W_s,
                           gain_norm_coef=self.gain_norm_coef,
                           xp=E.xp,
                           grid=E.grid,
                           gain_profile_path=self.gain_profile_path)

        if self.field_scale_factor:
            E.Ez *= self.field_scale_factor

        E.fiber = self.fiber
        E.Ez = E.methods.beam_evo(E=E, num=self.num, gain=self.gain)
        return E


class CompressorProcess(Step):
    def __init__(self, num: int, name: str = 'compressed', gamma: Optional[float] = None, l_g: Optional[float] = None):
        super().__init__(num=num, name=f'{num}. {name}' if name else f'{num}')
        self.compressor = Compressor()
        self.gamma = gamma
        self.l_g = l_g

    def process(self, E: 'Field') -> 'Field':
        E.Ez = self.compressor.compress_it(E, gamma=self.gamma, l_g=self.l_g)
        return E


class BFGProcess(Step):
    def __init__(self, num: int, name: str = ''):
        super().__init__(num=num, name=f'{num}. {name}' if name else f'{num}')
        self.bfg = BFG()

    def process(self, E: 'Field') -> 'Field':
        E.Ez = self.bfg.stretch_it(E=E)
        return E


class AOMProcess(Step):
    def __init__(self, num: int, name: str = '', repeat_frequency: float = 1e6, field_scale_factor: Optional[float] = None):
        super().__init__(num=num, name=f'{num}. {name}' if name else f'{num}')
        self.repeat_frequency = repeat_frequency
        self.field_scale_factor = field_scale_factor

    def process(self, E: 'Field') -> 'Field':
        if self.field_scale_factor is not None:
            E.Ez *= self.field_scale_factor
        E.repeat_frequency = self.repeat_frequency
        return E


class Pipeline:
    def __init__(self, steps: list[Step], saver: StateSaver | None = None, plotter: LiveFourGraphPlotter = None):
        self.steps = sorted(steps, key=lambda s: s.num)
        self.saver = saver
        self.plotter = plotter

    def run(self, E: 'Field', start: int | None = None, stop: int | None = None,
            resume: str | None = None, autosave: bool = True) -> 'Field':
        if resume:
            if self.saver is None:
                raise RuntimeError("StateSaver is not set for Pipeline, cannot resume.")
            name = resume if resume.endswith('.npz') else f'{resume}.npz'
            E = self.saver.restore_to(E, name[:-4] if name.endswith('.npz') else name)

        # 2) Авто-resume: если хотим начать с k, грузим (k-1).npz
        elif start and start > 1 and self.saver:
            ckpt = f'{start - 1:02d}'
            E = self.saver.restore_to(E, ckpt)

        for step in self.steps:
            if start is not None and step.num < start:
                continue
            if stop is not None and step.num > stop:
                break

            if self.plotter:
                self.plotter.update(E=E, num=step.num, title=step.name)

            if type(step) is CompressorProcess and step.num < 22:
                Ez_old = E.Ez.copy()

            E = step.process(E)
            if self.plotter:
                self.plotter.update(E, step.num)
                self.plotter.finalize(filename=step.name, num=step.num, energy=E.energy if step.num > 20 else None)

            if type(step) is CompressorProcess and step.num < 22:
                E.Ez = Ez_old.copy()

            E.Ez0 = E.Ez.copy()

            if autosave and self.saver:
                if step.num > 20:
                    self.saver.save_state(E, filename=f'{step.num:02d} E={E.energy:.2f} nJ')
                else:
                    self.saver.save_state(E, filename=f'{step.num:02d}')

        return E