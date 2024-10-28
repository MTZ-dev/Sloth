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

from numpy import array, where, sqrt, abs

from slothpy.core._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy.core._hessian_object import Hessian
from slothpy._general_utilities._constants import AU_BOHR_CM_1

def _phonon_dispersion_proxy(sm_arrays_info_list: list[SharedMemoryArrayInfo], args_list, process_index, start: int, end: int):
    hessian = Hessian(sm_arrays_info_list[:2], args_list[0])
    sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list[2:])
    kpoints, progress_array, dispersion_array = arrays
    au_bohr_cm_1 = array(AU_BOHR_CM_1, dtype=kpoints.dtype)
    
    for i in range(start, end):
        hessian._kpoint = kpoints[i]
        frequencies_squared = hessian.frequencies_squared()
        frequencies = where(frequencies_squared >= 0, sqrt(abs(frequencies_squared)), -sqrt(abs(frequencies_squared))) * au_bohr_cm_1
        dispersion_array[i, :] = frequencies
        progress_array[process_index] += 1
