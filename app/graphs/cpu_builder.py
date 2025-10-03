from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

if typing.TYPE_CHECKING:
    from app.field import Field

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 0.5
mpl.rcParams["agg.path.chunksize"] = 20000

DEFAULT_MAX_POINTS = 10_000


@dataclass
class CPUArrays:
    count_t_points: int
    count_lam_points: int
    t: np.ndarray
    lam: np.ndarray
    I: np.ndarray
    I_prev: np.ndarray
    PSD: np.ndarray
    PSD_prev: np.ndarray
    freq_shift: np.ndarray
    group_delay: np.ndarray
    _gpu: bool

    gain_profile: np.ndarray = None
    experiment_x: np.ndarray = None
    experiment_y: np.ndarray = None

    def __post_init__(self):
        if self._gpu:
            import cupy as cp
            self.t = cp.asnumpy(self.t)
            self.lam = cp.asnumpy(self.lam)
            self.I = cp.asnumpy(self.I)
            self.I_prev = cp.asnumpy(self.I_prev)
            self.PSD = cp.asnumpy(self.PSD)
            self.PSD_prev = cp.asnumpy(self.PSD_prev)
            self.freq_shift = cp.asnumpy(self.freq_shift)
            self.group_delay = cp.asnumpy(self.group_delay)
            if self.gain_profile is not None:
                self.gain_profile = cp.asnumpy(self.gain_profile)


        if self.experiment_x is not None:
            idxs = np.argwhere((self.experiment_x > self.lam[-1]) & (self.experiment_x < self.lam[0]))
            self.experiment_x = self.experiment_x[idxs]
            self.experiment_y = self.experiment_y[idxs]
            self.experiment_y = self.experiment_y / self.experiment_y.max() * self.PSD.max()


def prepare_arrays(E: 'Field', num: int):
    def _get_slice(left: int, right: int, range_k: int, step: int) -> slice:
        shift = range_k * (right - left)
        left -= min(shift, left)
        right += min(shift, E.Ez.size - right - 1)
        return slice(left, right, step)

    time_left, time_right = E.time_borders_FWHN
    step = max((time_right - time_left) // DEFAULT_MAX_POINTS, 1)
    time_sl = _get_slice(left=time_left, right=time_right, range_k=2, step=step)

    lam_left, lam_right = E.spectrum_borders_FWHM
    step = max(abs(time_right - time_left) // DEFAULT_MAX_POINTS, 1)
    lam_sl = _get_slice(left=lam_left, right=lam_right, range_k=2, step=step)

    t = E.grid.t[time_sl]
    lam = E.methods.fftshift(E.grid.lam)[lam_sl]

    Ez = E.Ez
    I = E.intensity[time_sl]
    I_prev = E.intensity_prev[time_sl]

    PSD = E.methods.fftshift(E.power_spectral_density)[lam_sl]
    PSD_prev = E.methods.fftshift(E.power_spectral_density_prev)[lam_sl]

    phi_t = E.xp.unwrap(np.angle(Ez))
    freq_shift = E.xp.gradient(phi_t, E.grid.t)
    freq_shift -= freq_shift[freq_shift.size // 2]
    freq_shift = freq_shift[time_sl]

    gain_profile = E.methods.fftshift(E.fiber.gain_profile)[lam_sl] if E.fiber.gain else None

    phi_w = E.xp.unwrap(E.xp.angle(E.methods.fft(E.methods.fftshift(Ez))))
    group_delay = -E.xp.gradient(phi_w, E.grid.w)
    group_delay = E.methods.fftshift(group_delay)[lam_sl]

    experiment = E.experimental_graphs.get(num, None)
    experiment_x = experiment[0] if experiment is not None else None
    experiment_y = experiment[1] if experiment is not None else None

    return CPUArrays(count_t_points=time_right - time_left, 
                     count_lam_points=abs(lam_right - lam_left),
                     t=t, lam=lam,
                     I=I, I_prev=I_prev , PSD=PSD, PSD_prev=PSD_prev,
                     freq_shift=freq_shift, group_delay=group_delay,
                     _gpu=getattr(E.xp, "__name__", "") == "cupy",
                     gain_profile=gain_profile,
                     experiment_x=experiment_x, experiment_y=experiment_y)


class LiveFourGraphPlotter:
    """
        Окно 2×2 с быстрым обновлением (blitting).
        Сюжеты:
          (0,0) pulse: |E(t)|²
          (0,1) spectrum: PSD(λ)
          (1,0) shift frequency: IF(t)
          (1,1) group delay: GD(λ)
        """
    def __init__(self, base_path: Path = None, num: int | str = None):
        plt.ion()

        self.base_path = base_path
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.num = num
        self._bg = {}
        self._lims = {}
        self._cid_resize = None
        self._cid_draw = None
        self._build_figure()

    def _build_figure(self):
        if hasattr(self, "fig") and self.fig is not None:
            try:
                if plt.fignum_exists(self.fig.number):
                    return
            except Exception:
                pass

        self.fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, num=self.num)
        self.ax_pulse, self.ax_psd, self.ax_shift_frec, self.ax_gd = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

        self.lines_pulse = self._init_ax(self.ax_pulse, xlabel='t, s', ylabel='I(t)', prev=True)
        self.lines_psd = self._init_ax(self.ax_psd, xlabel='λ, м', ylabel='SPD', prev=True, exp=True, gain=True)
        self.lines_shift_frec = self._init_ax(self.ax_shift_frec, xlabel='t, s', ylabel='Гц')
        self.lines_gd = self._init_ax(self.ax_gd, xlabel='λ, м', ylabel='Group Delay')

        self.fig.canvas.draw()
        self._snapshot_backgrounds()

        self.txt_pulse = AnchoredText("", loc="upper left", prop=dict(size=9), frameon=True)
        self.ax_pulse.add_artist(self.txt_pulse)

        self.txt_psd = AnchoredText("", loc="upper left", prop=dict(size=9), frameon=True)
        self.ax_psd.add_artist(self.txt_psd)

        if self._cid_resize is not None:
            self.fig.canvas.mpl_disconnect(self._cid_resize)
        if self._cid_draw is not None:
            self.fig.canvas.mpl_disconnect(self._cid_draw)

        self._cid_resize = self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self._cid_draw = self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("close_event", lambda evt: setattr(self, "fig", None))

        self.fig.show()
        plt.pause(0.001)

    def _snapshot_backgrounds(self):
        self._bg = {ax: self.fig.canvas.copy_from_bbox(ax.bbox) for ax in
                    (self.ax_pulse, self.ax_psd, self.ax_shift_frec, self.ax_gd)}
        self._lims = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in self._bg}

    def _on_resize(self, _evt=None):
        self.fig.canvas.draw()
        self._snapshot_backgrounds()

    def _on_draw(self, _evt=None):
        # любое «серьёзное» действие тулбара (zoom/pan) вызывает полный draw — пере-«сфоткаем» фон
        self._snapshot_backgrounds()

    def _ensure_alive(self):
        # если окно закрыто — пересобрать
        if not hasattr(self, "fig") or self.fig is None or not plt.fignum_exists(self.fig.number):
            self._build_figure()

    @staticmethod
    def _init_ax(ax: plt.Axes, xlabel: str, ylabel: str, prev: bool = False, exp: bool = False, gain: bool = False) -> list[Line2D]:
        """
        Инициализирует график
        Parameters
        ----------
        ax : plt.Axes
                Ось
        xlabel : str
                Подпись по оси x
        ylabel : str
                Подпись по оси y
        prev : bool
                Будет ли график предыдущего состояния
        exp : bool
                Будет ли экспериментальный график

        Returns
        -------
        Возвращает кортеж линий или объект, если линия одна
        """
        lines = [ax.plot([], [], label='now')[0]]
        if prev:
            lines.append(ax.plot([], [], ls='--', label='prev')[0])
        if exp:
            lines.append(ax.plot([], [], label='exp')[0])
        if gain:
            lines.append(ax.plot([], [], ls='--', label='gain')[0])

        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return lines

    @staticmethod
    def _set_line(line, x: np.ndarray, y: Optional[np.ndarray] = None):
        if y is None:
            line.set_data([], [])
        else:
            line.set_data(x, y)

    def _on_resize(self, _evt=None):
        self.fig.canvas.draw()
        for ax in self._bg:
            self._bg[ax] = self.fig.canvas.copy_from_bbox(ax.bbox)
            self._lims[ax] = (ax.get_xlim(), ax.get_ylim())

    def update(self, E: 'Field', num: int, title: Optional[str] = None):
        arrays = prepare_arrays(E=E, num=num)

        self._ensure_alive()

        if title is not None:
            self.fig.canvas.manager.set_window_title(title)

        self._set_line(self.lines_pulse[0], arrays.t, arrays.I)
        self._set_line(self.lines_pulse[1], arrays.t, arrays.I_prev)
        self.ax_pulse.set_yscale('log')
        self.ax_pulse.relim()
        self.ax_pulse.autoscale_view()
        self.ax_pulse.set_ylim(0.9 * arrays.I.min(), arrays.I.max() * 1.1)
        energy = E.energy
        duration = E.duration
        if duration < 1e-12:
            duration *= 1e15
            unit = 'фс'
        elif duration < 1e-9:
            duration *= 1e12
            unit = 'пс'
        else:
            duration *= 1e9
            unit = 'нс'

        self.txt_pulse.txt.set_text(f'dots: {arrays.count_t_points} ({arrays.count_t_points / E.grid.t.size * 100:.1f}%)\n'
                                    f'$\\tau$ = {duration:.2f} {unit}\n'
                                    f'E = {energy:.2f} нДж\n'
                                    f'$\\bar{{P}}$ = {energy * E.repeat_frequency * 1e-6:.2f} мВт')

        self._set_line(self.lines_psd[0], arrays.lam, arrays.PSD)
        self._set_line(self.lines_psd[1], arrays.lam, arrays.PSD_prev)
        self._set_line(self.lines_psd[2], arrays.experiment_x, arrays.experiment_y)
        if E.fiber.gain and arrays.gain_profile.size > 0:
            self._set_line(self.lines_psd[3], arrays.lam, arrays.gain_profile * arrays.PSD.max(),)
        self.ax_psd.set_yscale('log')
        self.ax_psd.relim()
        self.ax_psd.autoscale_view(scaley=False)
        self.ax_psd.set_ylim(0.9 * arrays.PSD.min(), arrays.PSD.max() * 1.1)

        self.txt_psd.txt.set_text(f'dots: {arrays.count_lam_points} ({arrays.count_lam_points / E.grid.lam.size * 100:.1f}%)\n'
                                  f'$\\delta\\lambda$ = {E.spectrum_width_FWHM_nm:.2f} нм')

        self._set_line(self.lines_shift_frec[0], arrays.t, arrays.freq_shift)
        self.ax_shift_frec.relim()
        self.ax_shift_frec.autoscale_view()

        self._set_line(self.lines_gd[0], arrays.lam, arrays.group_delay)
        self.ax_gd.relim()
        self.ax_gd.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def finalize(self, filename: str):
        """Завершить live и дать полноценное интерактивное окно (zoom/pan)."""
        self._ensure_alive()
        plt.ioff()
        self.fig.canvas.draw()
        plt.savefig(self.base_path / f'{filename}.png', dpi=300)
        # plt.show(block=True)
        plt.close(self.fig)
        plt.ion()

    def plot_gain_history(self, z, energy_nJ, filename: str = "gain.png",
                          title: str | None = None):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True, num=f"gain_{self.num}")
        ax.plot(z, energy_nJ, label="E (нДж)")
        ax.set_xlabel("z, м")
        ax.set_ylabel(r'E')
        ax.legend()
        ax.grid()

        if title:
            try:
                fig.canvas.manager.set_window_title(title)
            except Exception:
                pass
            fig.suptitle(title)

        # сохранить рядом с основными графиками
        out = self.base_path / filename if self.base_path else filename
        fig.savefig(out, dpi=200)

        plt.show(block=True)
        plt.close(fig)