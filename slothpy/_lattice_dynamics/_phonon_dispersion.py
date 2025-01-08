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

from numpy import ndarray, asarray

from slothpy.core._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy.core._hessian_object import Hessian
from slothpy._general_utilities._constants import AU_BOHR_CM_1
from slothpy.core._system import shared_memory_proxy

@shared_memory_proxy
def _phonon_dispersion_proxy(hessian: ndarray, masses_inv_sqrt: ndarray, kpoints: ndarray, start_mode: int, stop_mode: int, result_array: ndarray, progress_array: ndarray, process_index: int, start: int, end: int):
    hessian_object = Hessian(hessian, masses_inv_sqrt, start_mode=start_mode, stop_mode=stop_mode, eigen_range="I")
    au_bohr_cm_1 = asarray(AU_BOHR_CM_1, dtype=kpoints.dtype)
    
    for i in range(start, end):
        hessian_object._kpoint = kpoints[i]
        frequencies = hessian_object.frequencies * au_bohr_cm_1
        result_array[i, :] = frequencies
        progress_array[process_index] += 1
