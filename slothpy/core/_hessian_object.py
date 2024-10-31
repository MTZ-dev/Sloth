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

from numpy import empty, float64, float32, complex64, complex128

from slothpy.core._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy._general_utilities._numba_methods import _dynamical_matrix


class Hessian():
    
    __slots__ = ["_sm", "_hessian", "_masses_inv_sqrt", "_kpoint", "_modes_cutoff", "_heevr_lwork", "_heevr", "_lwork", "_dtype_array"]

    def __init__(self, sm_arrays_info_list: list[SharedMemoryArrayInfo], modes_cutoff: int = 0, single_process: bool = False):
        if single_process:
            arrays = sm_arrays_info_list
        else:
            sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list)
            self._sm = sm
        self._hessian = arrays[0]
        self._masses_inv_sqrt = arrays[1]
        self._kpoint = None
        self._modes_cutoff = modes_cutoff
        if self._hessian.dtype == float64:
            from slothpy._general_utilities._lapack import _zheevr_lwork as _heevr_lwork, _zheevr as _heevr
        else:
            from slothpy._general_utilities._lapack import _cheevr_lwork as _heevr_lwork, _cheevr as _heevr
        self._heevr_lwork, self._heevr = _heevr_lwork, _heevr
        self._lwork = None
        if self._hessian.dtype == float32:
            self._dtype_array = empty(1, dtype=complex64)
        else:
            self._dtype_array = empty(1, dtype=complex128)

    def build_dynamical_matrix(self):
        return _dynamical_matrix(self._hessian, self._masses_inv_sqrt, self._kpoint, self._dtype_array)

    def frequencies_squared(self):
        if self._lwork is None:
            self._lwork = self._heevr_lwork(self._masses_inv_sqrt.shape[0], jobz='N', range='I', il=1, iu=self._modes_cutoff)
        return self._heevr(self.build_dynamical_matrix().T, *self._lwork, jobz='N', range='I', il=1, iu=self._modes_cutoff)
    
    def frequencies_squared_eigenvectors(self, frequency_start: float = 0, frequency_stop: float = 0):
        if self._lwork is None:
            self._lwork = self._heevr_lwork(self._masses_inv_sqrt.shape[0], jobz='V', range='I' if (self._modes_cutoff != 0) else 'V', il=1, iu=self._modes_cutoff, vl=frequency_start, vu=frequency_stop)
        return self._heevr(self.build_dynamical_matrix().T, *self._lwork, jobz='V', range='I' if (self._modes_cutoff != 0) else 'V', il=1, iu=self._modes_cutoff, vl=frequency_start, vu=frequency_stop)

    # def e_under_fields(self):
    #     if self._number_of_centers > 1:
    #         energies_list = []
    #         hamiltonian = self._interaction_matrix.copy()
    #         for i in range(self._number_of_centers):
    #             zeeman_matrix = self.build_hamiltonian(i)
    #             ops = [ascontiguousarray(zeeman_matrix[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if k == i else self._cutoff_info_list[k][1] for k in range(self._number_of_centers)]
    #             hamiltonian += _kron_mult(ops)
    #             if self._cutoff_info_list[i][2] < self._cutoff_info_list[i][0] and self._local_states:
    #                 start_state = self._cutoff_info_list[i][2] + 1
    #                 _lwork = self._heevr_lwork(zeeman_matrix.shape[1], jobz='N', range='I', il=start_state, iu=zeeman_matrix.shape[1])
    #                 energies = self._heevr(zeeman_matrix.T, *_lwork, jobz='N', range='I', il=start_state, iu=zeeman_matrix.shape[1])
    #                 energies_list.append(energies)
    #         if self._lwork is None:
    #             self._lwork = self._heevr_lwork(hamiltonian.shape[1], jobz='N', range='I', il=1, iu=self._cutoff)
    #         energies =  self._heevr(hamiltonian.T, *self._lwork, jobz='N', range='I', il=1, iu=self._cutoff)
    #         energies_list.append(energies)

    #         return energies_list
    #     else:
    #         if self._lwork is None:
    #             self._lwork = self._heevr_lwork(self.m[0].shape[1], jobz='N', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
    #         energies =  self._heevr(self.build_hamiltonian(0).T, *self._lwork, jobz='N', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
    #         if len(energies) < 1:
    #             raise ValueError("Auto-cutoff failed! Please set it manually and run the calculation using the reviewed settings once again.")

    #     return energies

    # def e_slpjm_under_fields(self, mode, orientation, return_energies=True):
    #     if self._number_of_centers > 1:
    #         result_list = []
    #         energies_list = []
    #         hamiltonian = self._interaction_matrix.copy()
    #         result_matrix = zeros_like(hamiltonian) if isinstance(orientation, ndarray) else zeros((3, hamiltonian.shape[0], hamiltonian.shape[1]))
    #         for i in range(self._number_of_centers):
    #             local_states_to_add = False
    #             zeeman_matrix = self.build_hamiltonian(i)
    #             ops = [ascontiguousarray(zeeman_matrix[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if k == i else self._cutoff_info_list[k][1] for k in range(self._number_of_centers)]
    #             hamiltonian += _kron_mult(ops)
    #             if self._cutoff_info_list[i][2] < self._cutoff_info_list[i][0] and self._local_states:
    #                 local_states_to_add = True
    #                 start_state = self._cutoff_info_list[i][2] + 1
    #                 _lwork = self._heevr_lwork(zeeman_matrix.shape[1], jobz='V', range='I', il=start_state, iu=zeeman_matrix.shape[1])
    #                 energies, eigenvectors = self._heevr(zeeman_matrix.T, *_lwork, jobz='V', range='I', il=start_state, iu=zeeman_matrix.shape[1])
    #                 energies_list.append(energies)
    #             if not isinstance(orientation, ndarray):
    #                 if local_states_to_add:
    #                     result_local = empty((3, eigenvectors.shape[1]), dtype=energies.dtype, order='C')
    #                 for k in range(3):
    #                     matrix_local = getattr(self, mode)[i][k]
    #                     ops_result = [ascontiguousarray(matrix_local[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if l == i else self._cutoff_info_list[l][1] for l in range(self._number_of_centers)]
    #                     result_matrix[k,:,:] += _kron_mult(ops_result)
    #                     if local_states_to_add:
    #                         result_local[k,:] = self._utmud(eigenvectors, matrix_local.T)
    #             else:
    #                 matrix_local = tensordot(getattr(self, mode)[i], orientation, axes=(0,0))
    #                 ops_result = [ascontiguousarray(matrix_local[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if l == i else self._cutoff_info_list[l][1] for l in range(self._number_of_centers)]
    #                 result_matrix += _kron_mult(ops_result)
    #                 if local_states_to_add:
    #                     result_local = self._utmud(eigenvectors, matrix_local.T)
    #             if local_states_to_add:
    #                 result_list.append(result_local)
    #         if self._lwork is None:
    #             self._lwork = self._heevr_lwork(hamiltonian.shape[1], jobz='V', range='I', il=1, iu=self._cutoff)
    #         energies, eigenvectors =  self._heevr(hamiltonian.T, *self._lwork, jobz='V', range='I', il=1, iu=self._cutoff)
    #         energies_list.append(energies)
    #         if not isinstance(orientation, ndarray):
    #             result = empty((3, eigenvectors.shape[1]), dtype=energies.dtype, order='C')
    #             for i in range(3):
    #                result[i,:] = self._utmud(eigenvectors, result_matrix[i].T) 
    #         else:
    #             result = self._utmud(eigenvectors, result_matrix.T)
    #         result_list.append(result)

    #         return (energies_list, result_list) if return_energies else result
    #     else:
    #         if self._lwork is None:
    #             self._lwork = self._heevr_lwork(self.m[0].shape[1], jobz='V', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
    #         energies, eigenvectors =  self._heevr(self.build_hamiltonian(0).T, *self._lwork, jobz='V', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
    #         if len(energies) < 1:
    #             raise ValueError("Auto-cutoff failed! Please set it manually and run the calculation using the reviewed settings once again.")
    #         if not isinstance(orientation, ndarray):
    #             result = empty((3, eigenvectors.shape[1]), dtype=energies.dtype, order='C')
    #             for i in range(3):
    #                 result[i,:] = self._utmud(eigenvectors, getattr(self, mode)[0][i].T)
    #         else:
    #             result = self._utmud(eigenvectors, tensordot(getattr(self, mode)[0], orientation, axes=(0,0)).T)

    #     return (energies, result) if return_energies else result

    # def e_slpjm_matrix_under_fields(self, mode, orientation, return_energies=True):
    #     if self._number_of_centers > 1:
    #         hamiltonian = self._interaction_matrix.copy()
    #         result_matrix = zeros_like(hamiltonian) if isinstance(orientation, ndarray) else zeros((3, hamiltonian.shape[0], hamiltonian.shape[1]))
    #         for i in range(self._number_of_centers):
    #             zeeman_matrix = self.build_hamiltonian(i)
    #             ops = [ascontiguousarray(zeeman_matrix[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if k == i else self._cutoff_info_list[k][1] for k in range(self._number_of_centers)]
    #             hamiltonian += _kron_mult(ops)
    #             if not isinstance(orientation, ndarray):
    #                 for k in range(3):
    #                     matrix_local = getattr(self, mode)[i][k]
    #                     ops_result = [ascontiguousarray(matrix_local[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if l == i else self._cutoff_info_list[l][1] for l in range(self._number_of_centers)]
    #                     result_matrix[k,:,:] += _kron_mult(ops_result)
    #             else: 
    #                 matrix_local = tensordot(getattr(self, mode)[i], orientation, axes=(0,0))
    #                 ops_result = [ascontiguousarray(matrix_local[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if l == i else self._cutoff_info_list[l][1] for l in range(self._number_of_centers)]
    #                 result_matrix += _kron_mult(ops_result)
    #         if self._lwork is None:
    #             self._lwork = self._heevr_lwork(hamiltonian.shape[1], jobz='V', range='I', il=1, iu=self._cutoff)
    #         energies, eigenvectors =  self._heevr(hamiltonian.T, *self._lwork, jobz='V', range='I', il=1, iu=self._cutoff)
    #         if not isinstance(orientation, ndarray):
    #             result = empty((3, eigenvectors.shape[1], eigenvectors.shape[1]), dtype=energies.dtype, order='C')
    #             for i in range(3):
    #                result[i,:,:] = self._utmu(eigenvectors, result_matrix[i].T) 
    #         else:
    #             result = self._utmu(eigenvectors, result_matrix.T)
    #     else:
    #         if self._lwork is None:
    #             self._lwork = self._heevr_lwork(self.m[0].shape[1], jobz='V', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
    #         energies, eigenvectors =  self._heevr(self.build_hamiltonian(0).T, *self._lwork, jobz='V', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
    #         if len(energies) < 1:
    #             raise ValueError("Auto-cutoff failed! Please set it manually and run the calculation using the reviewed settings once again.")
    #         if not isinstance(orientation, ndarray):
    #             result = empty((3, eigenvectors.shape[1], eigenvectors.shape[1]), dtype=energies.dtype, order='C')
    #             for i in range(3):
    #                 result[i,:] = self._utmu(eigenvectors, getattr(self, mode)[0][i].T)
    #         else:
    #             result = self._utmu(eigenvectors, tensordot(getattr(self, mode)[0], orientation, axes=(0,0)).T)

    #     return (energies, result) if return_energies else result

    





    




        
