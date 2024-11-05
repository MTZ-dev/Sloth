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

from numpy import ndarray, finfo, empty, float64, float32, complex64, complex128

from slothpy.core._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy._general_utilities._numba_methods import _dynamical_matrix

class Hessian():
    
    __slots__ = ["_sm", "_hessian", "_masses_inv_sqrt", "_kpoint", "_heevr_lwork", "_heevr", "_lwork", "_il", "_iu", "_vl", "_vu", "_dtype_array"]

    def __init__(self, sm_arrays_info_list: list[SharedMemoryArrayInfo], kpoint: ndarray = None, start_mode: int = 0, stop_mode: int = 0, start_frequency: float = None, stop_frequency: float = None, single_process: bool = False):
        if single_process:
            arrays = sm_arrays_info_list
        else:
            sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list)
            self._sm = sm
        self._hessian = arrays[0]
        self._masses_inv_sqrt = arrays[1]
        self._kpoint = kpoint
        if self._hessian.dtype == float64:
            from slothpy._general_utilities._lapack import _zheevr_lwork as _heevr_lwork, _zheevr as _heevr
        else:
            from slothpy._general_utilities._lapack import _cheevr_lwork as _heevr_lwork, _cheevr as _heevr
        self._heevr_lwork, self._heevr = _heevr_lwork, _heevr
        self._lwork = None
        self._il = start_mode + 1
        self._iu = stop_mode
        self._vl = finfo(self._hessian.dtype).min if start_frequency is None else start_frequency
        self._vu = finfo(self._hessian.dtype).max if stop_frequency is None else stop_frequency
        if self._hessian.dtype == float32:
            self._dtype_array = empty(1, dtype=complex64)
        else:
            self._dtype_array = empty(1, dtype=complex128)

    @property
    def build_dynamical_matrix(self):
        return _dynamical_matrix(self._hessian, self._masses_inv_sqrt, self._kpoint, self._dtype_array).T

    @property
    def frequencies_squared(self):
        if self._lwork is None:
            self._lwork = self._heevr_lwork(self._masses_inv_sqrt.shape[0], jobz='N', range='I', il=self._il, iu=self._iu)
        return self._heevr(self.build_dynamical_matrix, *self._lwork, jobz='N', range='I', il=self._il, iu=self._iu)
    
    @property
    def frequencies_squared_eigenvectors(self):
        if self._lwork is None:
            self._lwork = self._heevr_lwork(self._masses_inv_sqrt.shape[0], jobz='V', range='V', vl=self._vl, vu=self._vu)
        return self._heevr(self.build_dynamical_matrix, *self._lwork, jobz='V', range='V', vl=self._vl, vu=self._vu)

    





    




        
