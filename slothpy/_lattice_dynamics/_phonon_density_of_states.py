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

from numpy import asarray

from slothpy.core._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy.core._hessian_object import Hessian
from slothpy._general_utilities._constants import AU_BOHR_CM_1

def _phonon_density_of_states_proxy(sm_arrays_info_list: list[SharedMemoryArrayInfo], args_list, process_index, start: int, end: int):
    hessian_object = Hessian(sm_arrays_info_list[:2], start_frequency=args_list[0], stop_frequency=args_list[1], eigen_range="V")
    sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list[2:])
    kpoints_grid, progress_array = arrays
    au_bohr_cm_1 = asarray(AU_BOHR_CM_1, dtype=kpoints_grid.dtype)
    
    frequencies_list = []

    for i in range(start, end):
        hessian_object._kpoint = kpoints_grid[i]
        frequencies_list.extend(hessian_object.frequencies * au_bohr_cm_1)
        progress_array[process_index] += 1

    return frequencies_list