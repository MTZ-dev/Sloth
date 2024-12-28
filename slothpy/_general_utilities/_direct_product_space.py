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

from numpy import zeros, kron, eye
from numpy import int64
from numba import jit, types, prange, complex128, complex64, float32, float64, int64 as nb_int64
from slothpy.core._config import settings

@jit([types.Array(complex64, 2, 'C')(nb_int64, types.Array(complex64, 2, 'C', True)), types.Array(complex128, 2, 'C')(nb_int64, types.Array(complex128, 2, 'C', True)), types.Array(float32, 2, 'C')(nb_int64, types.Array(float32, 2, 'C', True)), types.Array(float64, 2, 'C')(nb_int64, types.Array(float64, 2, 'C', True))],
nopython=True,
nogil=True,
cache=True,
fastmath=True,
parallel=True,
)
def _kron_eye_N_A(N, A):
    n = A.shape[0]
    out = zeros((n * N, n * N), dtype=A.dtype)
    for i in prange(N):
        out[i * n:(i + 1) * n, i * n:(i + 1) * n] = A
    return out

def _kron_mult(ops):
    if isinstance(ops[0], (int, int64)):
        result = eye(ops[0], dtype=settings.complex)
    else:
        result = ops[0]
    
    for op in ops[1:]:
        if isinstance(op, (int, int64)):
            result = _kron_eye_N_A(op, result)
        else:
            result = kron(op, result)
    
    return result
