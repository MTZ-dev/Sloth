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

from numpy import ndarray, zeros, array, exp, pi
from numba import jit, prange, types, int64, float32, float64, complex64, complex128

@jit(
    [
        types.Array(complex64, 2, 'C')(types.Array(float32, 5, 'C', True), types.Array(float32, 2, 'C', True), types.Array(float32, 1, 'C', True), types.Array(complex64, 1, 'C', True)),
        types.Array(complex128, 2, 'C')(types.Array(float64, 5, 'C', True), types.Array(float64, 2, 'C', True), types.Array(float64, 1, 'C', True), types.Array(complex128, 1, 'C', True)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _dynamical_matrix(hessian: ndarray, masses_inv_sqrt: ndarray, kpoint: ndarray, array_dtype: ndarray):

    dyn_mat = zeros(masses_inv_sqrt.shape, dtype=array_dtype.dtype)

    for nx in prange(hessian.shape[0]):
        for ny in range(hessian.shape[1]):
            for nz in range(hessian.shape[2]):
                dyn_mat += hessian[nx, ny, nz, :, :] * exp(-2j * pi * (kpoint[0] * nx + kpoint[1] * ny + kpoint[2] * nz)) # conj for fortran

    dyn_mat *= masses_inv_sqrt
    dyn_mat += dyn_mat.conj().T
    
    return array(-0.5, dtype=array_dtype.dtype) * dyn_mat