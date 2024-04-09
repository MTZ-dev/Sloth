# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.synchronize import Event
from numpy import ndarray, zeros, int64
from pandas import DataFrame
import matplotlib.pyplot as plt

from slothpy.core._config import settings
from slothpy.core._slt_file import SltGroup
from slothpy.core._drivers import SingleProcessed, MulitProcessed, ensure_ready
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET, H_CM_1
from slothpy._general_utilities._system import _from_shared_memory_to_array
from slothpy._magnetism._zeeman import _zeeman_splitting, _zeeman_splitting_average

class SltStatesEnergiesCm1(SingleProcessed):

    __slots__ = SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group: SltGroup, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._start_state = start_state
        self._stop_state = stop_state

    def __repr__(self) -> str:
        return f"<{RED}SltStatesEnergiesCm1{RESET} object from {BLUE}Group{RESET} '{self._group_name}' {GREEN}File{RESET} '{self._hdf5}'.>"

    def _executor(self):
        return self._slt_group.energies[self._start_state:self._stop_state] * H_CM_1
    
    @ensure_ready
    def save(self, slt_save = None):
        new_group = SltGroup(self._hdf5, self._slt_save if slt_save is None else slt_save)
        new_group["STATES_ENERGIES_CM_1"] = self._result
        new_group["STATES_ENERGIES_CM_1"].attributes["Description"] = "States' energies in cm-1."
        new_group.attributes["Type"] = "ENERGIES"
        new_group.attributes["Kind"] = "CM_1"
        new_group.attributes["States"] = self._result.shape[0]
        new_group.attributes["Precision"] = settings.precision.upper()
        new_group.attributes["Description"] = f"States' energies in cm-1 from Group '{self._group_name}'."

    def _load(self):
        self._result = self._slt_group["STATES_ENERGIES_CM_1"][:]

    @ensure_ready
    def plot(self):
        fig, ax = plt.subplots()
        x_min = 0
        x_max = 1
        for energy in self._result:
            ax.hlines(y=energy, xmin=x_min, xmax=x_max, colors='skyblue', linestyles='solid', linewidth=2)
            ax.text(x_max + 0.1, energy, f'{energy:.1f}', va='center', ha='left')
        ax.set_ylabel('Energy (cm$^{-1}$)')
        ax.set_title('Energy Levels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticks([])
        plt.tight_layout()
        plt.show()
    
    @ensure_ready
    def to_data_frame(self):
        self._df = DataFrame({'Energy (cm^-1)': self._result})
        self._df.index.name = 'State Number'
        return self._df


class SltZeemanSplitting(MulitProcessed):

    __slots__ = SingleProcessed.__slots__ + ["_magnetic_fields", "_orientations"]
     
    def __init__(self, slt_group: SltGroup,
        number_of_states: int,
        magnetic_fields: ndarray,
        orientations: ndarray,
        states_cutoff: int,
        number_cpu: int,
        number_threads: int,
        autotune: bool,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0] , number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        ######### Load only when autotune or running to save memory
        self._args_arrays = (slt_group.energies[:states_cutoff], slt_group.magnetic_dipole_momenta[:, :states_cutoff, :states_cutoff], self._magnetic_fields, self._orientations)
        if orientations.shape[1] == 4:
            self._returns = True
        elif orientations.shape[1] == 3:
            self._result = zeros((magnetic_fields.shape[0] * orientations.shape[0], number_of_states), dtype=magnetic_fields.dtype, order="C")
            self._result_shape = (orientations.shape[0], magnetic_fields.shape[0], number_of_states)
        self._args = (number_of_states,)
        self._method_no_return = _zeeman_splitting
        self._method_return = _zeeman_splitting_average
        
    def __repr__(self) -> str:
        return f"<{RED}SltZeemanSplitting{RESET} object from {BLUE}Group{RESET} '{self._group_name}' {GREEN}File{RESET} '{self._hdf5}'.>"
    
    def _gather_results(self, results):
        return super()._gather_results()
    
    @ensure_ready
    def save(self, slt_save = None):
        if self._orientations.shape[1] == 4 and self._result.ndim == 2:
            average = True
        else:
            average = False
        new_group = SltGroup(self._hdf5, self._slt_save if slt_save is None else slt_save)
        new_group["ZEEMAN_SPLITTING"] = self._result
        new_group["MAGNETIC_FIELDS"] = self._magnetic_fields
        new_group["ORIENTATIONS"] = self._orientations
        new_group.attributes["Type"] = "ZEEMAN_SPLITTING"
        new_group.attributes["Kind"] = "AVERAGE" if average else "DIRECTIONAL"
        new_group.attributes["Precision"] = settings.precision.upper()
        new_group.attributes["Description"] = f"Group containing Zeeman splitting calculated from Group '{self._group_name}'."
        new_group["ZEEMAN_SPLITTING"].attributes["Description"] = f"Dataset containing Zeeman splitting in the form {'[fields, energies]' if average else '[orientations, fields, energies]'}."
        new_group["MAGNETIC_FIELDS"].attributes["Description"] = "Dataset containing magnetic field (T) values used in the simulation."
        new_group["ORIENTATIONS"].attributes["Description"] = "Dataset containing magnetic fields' orientation grid used in the simulation."

    def _load(self):
        self._result = self._slt_group["ZEEMAN_SPLITTING"][:]
        self._magnetic_fields = self._slt_group["MAGNETIC_FIELDS"][:]
        self._orientations = self._slt_group["ORIENTATIONS"][:]

    @ensure_ready
    def plot(self):
        fig, ax = plt.subplots()
        x_min = 0
        x_max = 1
        for energy in self._result:
            ax.hlines(y=energy, xmin=x_min, xmax=x_max, colors='skyblue', linestyles='solid', linewidth=2)
            ax.text(x_max + 0.1, energy, f'{energy:.1f}', va='center', ha='left')
        ax.set_ylabel('Energy (cm$^{-1}$)')
        ax.set_title('Energy Levels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticks([])
        plt.tight_layout()
        plt.show()
    
    @ensure_ready
    def to_data_frame(self):
        self._df = DataFrame({'Energy (cm^-1)': self._result})
        self._df.index.name = 'State Number'
        return self._df