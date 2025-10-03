from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

from app.config.grid import Grid
from app.graphs.cpu_builder import LiveFourGraphPlotter
from app.sim_steps import FiberProcess, CompressorPrecess, BFGProcess, StateSaver, ChangeGrid, Pipeline, AOMProcess
from app.field import Field

BASE = Path(__file__).parent.parent
BASE_SAVE_FOLDER = Path(__file__).parent.parent / 'NotCode'
MODE: Literal['Gauss Pulse v2', 'Experimental Pulse v2'] = 'Experimental Pulse v2'

state_saver = StateSaver(BASE / 'NotCode' / 'Graphs' / MODE / 'states')
plotter = LiveFourGraphPlotter(BASE / 'NotCode' / 'Graphs' / MODE / 'graphs')

first_element = 1
last_element = 24


E = Field(Nt=2**16, time_window=50e-11,
          osc_filepath=BASE / 'Параметры элементов' / 'oscillator.dat' if 'Experimental' in MODE else None,
          backend='gpu')

E.plot = plotter
E.saver = state_saver

pipeline = [
    FiberProcess(num=1, name='PLMA 25_250 4m', length=4.75, diameter=25, gain=False),
    FiberProcess(num=2, length=1.2, diameter=6, gain=False,
                 field_scale_factor=25 / 6 * E.xp.sqrt(47 / 108)),
    FiberProcess(num=3, length=0.9, diameter=6, gain=False),

    # Брэгговская решётка
    ChangeGrid(num=4, Nt=2**22, time_window=1e-8),
    BFGProcess(num=4, name='BFG'),
    CompressorPrecess(num=4),
    ChangeGrid(num=4, Nt=2**18, time_window=6e-9),

    FiberProcess(num=5, length=0.9, diameter=6, gain=False),
    FiberProcess(num=6, length=1.35, diameter=6, gain=False, field_scale_factor=E.xp.sqrt(20/35.2)),
    FiberProcess(num=7, length=0.95, diameter=6, gain=False, field_scale_factor=E.xp.sqrt(13/20)),

    # ПЕРВЫЙ КАСКАД
    FiberProcess(num=8, name='Yb3+ 6_125', length=.9, diameter=6, gain=True, g_0=30, W_s=8.2e-10,
                 gain_profile_path=BASE / 'Параметры элементов' / 'Спектр усиления Yb3+.dat'),
    CompressorPrecess(num=8),
    FiberProcess(num=9, length=0.9, diameter=6, gain=False),
    FiberProcess(num=10, length=1.9, diameter=6, gain=False),

    # АОМ
    AOMProcess(num=11, name='AOM', repeat_frequency=1e6, field_scale_factor=E.xp.sqrt(1.12 / 2.7)),
    FiberProcess(num=12, length=1.8, diameter=6, gain=False),
    FiberProcess(num=13, length=1.7, diameter=6, gain=False),

    # ВТОРОЙ КАСКАД
    FiberProcess(num=14, name='Yb3+ 6_125', length=0.8, diameter=6, gain=True, g_0=46, W_s=1e-8, gain_norm_coef=1,
                 gain_profile_path=BASE / 'Параметры элементов' / 'Спектр усиления Yb3+.dat'),
    CompressorPrecess(num=14),
    FiberProcess(num=15, length=0.25, diameter=6, gain=False),
    FiberProcess(num=16, length=0.45, diameter=6, gain=False, field_scale_factor=E.xp.sqrt(20/22)),
    FiberProcess(num=17, length=0.35, diameter=6, gain=False, field_scale_factor=E.xp.sqrt(17/20)),
    FiberProcess(num=18, length=0.6, diameter=25, gain=False,
                 field_scale_factor=6 / 25),
    FiberProcess(num=19, length=0.64, diameter=25, gain=False),
    FiberProcess(num=20, length=0.32, diameter=25, gain=False),

    # ТРЕТИЙ КАСКАД
    FiberProcess(num=21, name='Yb3+ 25_250', length=2, diameter=25, gain=True, g_0=45, W_s=1.7e-7,
                 gain_profile_path=BASE / 'Параметры элементов' / 'Спектр усиления Yb3+.dat'),
    ChangeGrid(num=22, Nt=2**22, time_window=5e-9),
    CompressorPrecess(num=22),
]

runner = Pipeline(steps=pipeline, saver=state_saver, plotter=plotter)


if __name__ == '__main__':
    runner.run(E, start=1)


