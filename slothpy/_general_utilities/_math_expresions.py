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

from math import factorial
from numpy import ndarray, array, zeros, ascontiguousarray, arange, tile, ones, argsort, take_along_axis, abs, mod, sqrt, min, max, power, float64 as np_float64, int64 as np_int64, min as np_min, exp, pi
from numpy.linalg import eigh, inv
from numba import jit, prange, types, int64, float32, float64, complex64, complex128
from slothpy._general_utilities._constants import GE, MU_B_AU
from slothpy.core._slothpy_exceptions import SltInputError

def _lorentzian(x, mu, gamma, area):
    return area * (gamma) / (pi) / ((x - mu)**2 + gamma**2)


def _gaussian(x, mu, sigma, area):
    normalization = area / (sigma * sqrt(2 * pi))
    return normalization * exp(-0.5 * ((x - mu) / sigma) ** 2)


@jit(
    [
        types.Array(complex64, 2, 'C')(types.Array(complex64, 3, 'C', True), types.Array(float32, 1, 'C', True)),
        types.Array(complex128, 2, 'C')(types.Array(complex128, 3, 'C', True), types.Array(float64, 1, 'C', True)),
        types.Array(complex64, 1, 'C')(types.Array(complex64, 2, 'C', True), types.Array(float32, 1, 'C', True)),
        types.Array(complex128, 1, 'C')(types.Array(complex128, 2, 'C', True), types.Array(float64, 1, 'C', True)),
        types.Array(float32, 1, 'C')(types.Array(float32, 2, 'C', True), types.Array(float32, 1, 'C', True)),
        types.Array(float64, 1, 'C')(types.Array(float64, 2, 'C', True), types.Array(float64, 1, 'C', True)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _3d_dot(m, xyz):
    return m[0] * xyz[0] + m[1] * xyz[1] + m[2] * xyz[2]


@jit(
    [
        (types.Array(complex64, 2, 'C', False), types.Array(float32, 1, 'C', True)),
        (types.Array(complex128, 2, 'C', False), types.Array(float64, 1, 'C', True)),
    ],
nopython=True,
nogil=True,
cache=True,
fastmath=True,
inline="always",
parallel=True,
)
def _add_diagonal(matrix, diagonal):

    for k in prange(matrix.shape[0]):
        matrix[k, k] += diagonal[k]


@jit(
    [
        (types.Array(complex64, 2, 'C', False), float32),
        (types.Array(complex128, 2, 'C', False), float64)
    ],
nopython=True,
nogil=True,
cache=True,
fastmath=True,
inline="always",
parallel=True,
)
def _subtract_const_diagonal(matrix, const):

    for k in prange(matrix.shape[0]):
        matrix[k, k] -= const


@jit("float64(float64, float64)", nopython=True, cache=True, nogil=True)
def _binom(n, k):
    if k > n - k:
        k = n - k
    res = 1
    for i in range(k):
        res *= n - i
        res /= i + 1
    return res


@jit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nopython=True,
    cache=True,
    nogil=True,
)
def Clebsh_Gordan(j1, m1, j2, m2, j3, m3):
    cg_coeff = 0

    if (
        (m1 + m2 != m3)
        or (j1 < 0.0)
        or (j2 < 0.0)
        or (j3 < 0.0)
        or abs(m1) > j1
        or abs(m2) > j2
        or abs(m3) > j3
        or (abs(j1 - j2) > j3)
        or ((j1 + j2) < j3)
        or (abs(j2 - j3) > j1)
        or ((j2 + j3) < j1)
        or (abs(j3 - j1) > j2)
        or ((j3 + j1) < j2)
        or (mod(int(2.0 * j1), 2) != mod(int(2.0 * abs(m1)), 2))
        or (mod(int(2.0 * j2), 2) != mod(int(2.0 * abs(m2)), 2))
        or (mod(int(2.0 * j3), 2) != mod(int(2.0 * abs(m3)), 2))
    ):
        return cg_coeff

    J = j1 + j2 + j3
    C = sqrt(
        _binom(2 * j1, J - 2 * j2)
        * _binom(2 * j2, J - 2 * j3)
        / (
            _binom(J + 1, J - 2 * j3)
            * _binom(2 * j1, j1 - m1)
            * _binom(2 * j2, j2 - m2)
            * _binom(2 * j3, j3 - m3)
        )
    )
    z_min = max(array([0, j1 - m1 - J + 2 * j2, j2 + m2 - J + 2 * j1]))
    z_max = min(array([J - 2 * j3, j1 - m1, j2 + m2]))
    for z in range(z_min, z_max + 1):
        cg_coeff += (
            (-1) ** z
            * _binom(J - 2 * j3, z)
            * _binom(J - 2 * j2, j1 - m1 - z)
            * _binom(J - 2 * j1, j2 + m2 - z)
        )

    return cg_coeff * C


@jit(
    "float64(float64, float64, float64, float64, float64, float64)",
    nopython=True,
    cache=True,
    nogil=True,
)
def _Wigner_3j(j1, j2, j3, m1, m2, m3):
    return (
        (-1) ** (j1 - j2 - m3)
        / sqrt(2 * j3 + 1)
        * Clebsh_Gordan(j1, m1, j2, m2, j3, -m3)
    )


def _central_finite_difference_stencil(diff_order: int, num_of_points: int, step: float):
    stencil_len = 2 * num_of_points + 1

    if diff_order >= stencil_len:
        raise SltInputError(ValueError(f"Insufficient number of points to evaluate coefficients. Provide a number of points greater than (derivative order - 1) / 2."))

    return inv(tile(arange(-num_of_points, num_of_points + 1).astype(np_int64),(stencil_len, 1)) ** arange(0, stencil_len).reshape(-1, 1))[:, diff_order] * factorial(diff_order) / power(step, diff_order)


@jit(
    "complex128[:,:](complex128[:,:], complex128[:,:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _hermitian_x_in_basis_of_hermitian_y(x_matrix, y_matrix): ## That can be used for the ORCA import or lapack
    x_matrix = ascontiguousarray(x_matrix)
    _, eigenvectors = eigh(y_matrix)

    return eigenvectors.conj().T @ x_matrix @ eigenvectors


@jit(
    "float64[:,:](complex128[:,:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _decomposition_of_hermitian_matrix(matrix):
    _, eigenvectors = eigh(matrix)

    return (eigenvectors * eigenvectors.conj()).real.T * 100


@jit([
        types.Array(float32, 2, 'C')(types.Array(float32, 2, 'C', False)),
        types.Array(float64, 2, 'C')(types.Array(float64, 2, 'C', False)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _normalize_grid_vectors(grid: ndarray):

    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError(
                "A custom grid has to be a (n,4) array in the format:"
                " [[direction_x, direction_y, direction_z, weight],...]."
        )

    norm = zeros(1, dtype=grid.dtype)

    for vector_index in range(grid.shape[0]):
        length = sqrt(
            grid[vector_index][0] ** 2
            + grid[vector_index][1] ** 2
            + grid[vector_index][2] ** 2
        )
        norm[0] += grid[vector_index][3]
        if length == 0:
            raise ValueError("Vector of length zero detected in the input grid.")

        grid[vector_index][:3] = grid[vector_index][:3] / length

    if norm[0] != 0:
        grid[:, 3] = grid[:, 3] / norm[0]

    return grid


@jit([
        types.Array(float32, 2, 'C')(types.Array(float32, 2, 'C', False)),
        types.Array(float64, 2, 'C')(types.Array(float64, 2, 'C', False)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _normalize_orientations_comp(orientations: ndarray):
    for vector_index in range(orientations.shape[0]):
        length = orientations[vector_index][0] ** 2 + orientations[vector_index][1] ** 2 + orientations[vector_index][2] ** 2
        if length == 0:
            raise ValueError("Vector of length zero detected in the input orientations.")
        length = array(1/sqrt(length), dtype=orientations.dtype)
        orientations[vector_index] = orientations[vector_index] * length
    return orientations


def _normalize_orientations(orientations: ndarray):
    if orientations.ndim != 2 or orientations.shape[1] != 3:
        raise ValueError("Orientations has to be (n,3) array in the format: [[direction_x, direction_y, direction_z],...].")      
    return _normalize_orientations_comp(orientations)


@jit([
        types.Array(float64, 1, 'C')(types.Array(float64, 1, 'C', True)),
        types.Array(float32, 1, 'C')(types.Array(float32, 1, 'C', True)),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _normalize_orientation_comp(orientation: ndarray):

    length = orientation[0] ** 2 + orientation[1] ** 2 + orientation[2] ** 2
    if length == 0:
        raise ValueError("Vector of length zero detected in the input orientation.")
    length = array(1/sqrt(length), dtype=orientation.dtype)

    return orientation * length


def _normalize_orientation(orientation: ndarray):
    if orientation.ndim != 1 or orientation.shape[0] != 3:
        raise ValueError("Orientation has to be a 1D array in the format: [direction_x, direction_y, direction_z].")
    return _normalize_orientation_comp(orientation)


@jit([
        types.Array(float32, 1, 'C')(types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True)),
        types.Array(float32, 2, 'C')(types.Array(float32, 2, 'C', True), types.Array(float32, 2, 'C', True)),
        types.Array(complex64, 3, 'C')(types.Array(complex64, 3, 'C', True), types.Array(complex64, 3, 'C', True)),
        types.Array(complex64, 2, 'C')(types.Array(complex64, 2, 'C', True), types.Array(complex64, 2, 'C', True)),
        types.Array(float64, 1, 'C')(types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True)),
        types.Array(float64, 2, 'C')(types.Array(float64, 2, 'C', True), types.Array(float64, 2, 'C', True)),
        types.Array(complex128, 2, 'C')(types.Array(complex128, 2, 'C', True), types.Array(complex128, 2, 'C', True)),
        types.Array(complex128, 3, 'C')(types.Array(complex128, 3, 'C', True), types.Array(complex128, 3, 'C', True)),
    ],
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
    parallel=True,
)
def _magnetic_dipole_momenta_from_spins_angular_momenta(spins: ndarray, angular_momenta: ndarray):
    ge = array(GE, dtype=spins.dtype)
    mu_b = array(MU_B_AU, dtype=spins.dtype)
    return -mu_b*(ge * spins + angular_momenta)


@jit([
        types.Array(float32, 1, 'C')(types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True)),
        types.Array(float32, 2, 'C')(types.Array(float32, 2, 'C', True), types.Array(float32, 2, 'C', True)),
        types.Array(complex64, 2, 'C')(types.Array(complex64, 2, 'C', True), types.Array(complex64, 2, 'C', True)),
        types.Array(complex64, 3, 'C')(types.Array(complex64, 3, 'C', True), types.Array(complex64, 3, 'C', True)),
        types.Array(float64, 1, 'C')(types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True)),
        types.Array(float64, 2, 'C')(types.Array(float64, 2, 'C', True), types.Array(float64, 2, 'C', True)),
        types.Array(complex128, 2, 'C')(types.Array(complex128, 2, 'C', True), types.Array(complex128, 2, 'C', True)),
        types.Array(complex128, 3, 'C')(types.Array(complex128, 3, 'C', True), types.Array(complex128, 3, 'C', True)),
    ],
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
    parallel=True,
)
def _total_angular_momenta_from_spins_angular_momenta(spins: ndarray, angular_momenta: ndarray):
    return spins + angular_momenta


def _subtract_min_from_arrays_list(array_list):
    min_value = min([np_min(arr) for arr in array_list])
    for arr in array_list:
        arr -= min_value


@jit([
    (types.Array(types.float32, 2, 'C', True), types.Array(types.float32, 2, 'C', True), types.Array(types.float32, 1, 'C', False), types.Array(types.float32, 1, 'C', False)),
    (types.Array(types.float64, 2, 'C', True), types.Array(types.float64, 2, 'C', True), types.Array(types.float64, 1, 'C', False), types.Array(types.float64, 1, 'C', False))
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _compute_products_and_sums(sorted_partition_array, sorted_momenta_array, partition_product, magnetisation_sum):
    for col in prange(sorted_partition_array.shape[1]):
        for row in range(sorted_partition_array.shape[0]):
            if partition_product[col] * sorted_partition_array[row, col] > 1e-300:
                partition_product[col] *= sorted_partition_array[row, col]
                magnetisation_sum[col] += sorted_momenta_array[row, col] / sorted_partition_array[row, col]
            else:
                partition_product[col] = 0
                magnetisation_sum[col] = 0
                break


def _validate_and_compute_partition_product_and_magnetisation_sum(partition_array, momenta_array):
    sorted_indices = argsort(partition_array, axis=0)[::-1]
    sorted_partition_array = take_along_axis(partition_array, sorted_indices, axis=0)
    sorted_momenta_array = take_along_axis(momenta_array, sorted_indices, axis=0)
    partition_product = ones(partition_array.shape[1], dtype=sorted_momenta_array.dtype)
    magnetisation_sum = ones(momenta_array.shape[1], dtype=sorted_momenta_array.dtype)

    _compute_products_and_sums(sorted_partition_array, sorted_momenta_array, partition_product, magnetisation_sum)

    return partition_product, magnetisation_sum

