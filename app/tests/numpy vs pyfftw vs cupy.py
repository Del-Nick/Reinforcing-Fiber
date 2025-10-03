from pathlib import Path

from app.backend.cpu import CPUBackend
from app.backend.gpu import GPUBackend
from app.backend.methods import Methods
import cupy as cp

import numpy as np
import timeit

from app.field import Field

cpu_backend = CPUBackend()
gpu_backend = GPUBackend()



def benchmark(func, repeat: int = 5, number: int = 5):
    timer = timeit.Timer(func)
    times = timer.repeat(repeat=repeat, number=number)
    per_iter = [t / number for t in times]
    mean = np.mean(per_iter)
    stdev = np.std(per_iter, ddof=1)
    print(f"{func.__name__}: {mean:.3f} s ± {stdev:.3f} s (per iter, n={number})")
    return mean, stdev


def test_numpy_fft():
    E.methods.beam_evo(E, 1)

def test_cpu_fft():
    E.methods.beam_evo(E, 1)

def test_gpu_fft():
    E.methods.beam_evo(E, 1)
    cp.cuda.Stream.null.synchronize()


E = Field(backend='gpu', Nt=2**12, osc_filepath=Path(__file__).parent.parent.parent / 'Параметры элементов' / 'oscillator.dat')
benchmark(test_gpu_fft)

E = Field(backend='cpu', Nt=2**12, osc_filepath=Path(__file__).parent.parent.parent / 'Параметры элементов' / 'oscillator.dat')
benchmark(test_cpu_fft)

# test_cpu_fft() # Прогрев
# benchmark(test_cpu_fft)


