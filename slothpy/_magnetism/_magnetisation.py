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

from typing import Literal
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
from numpy import (
    ndarray,
    dtype,
    array,
    vdot,
    sum,
    zeros,
    ascontiguousarray,
    diag,
    newaxis,
    exp,
    float64,
    complex128,
    int64,
    array_equal,
    concatenate,
    prod,
    divide,
)
from numpy.linalg import eigh
from numba import jit, set_num_threads, prange, float32, float64
from numba import types
from slothpy._general_utilities._constants import KB, MU_B
from slothpy._general_utilities._system import (
    _get_num_of_processes,
    _distribute_chunks,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._grids_over_sphere import (
    _fibonacci_over_sphere,
    _meshgrid_over_sphere_flatten,
)
import time
from datetime import datetime
import psutil
from slothpy.core._config import settings

from slothpy._general_utilities._math_expresions import _validate_and_compute_partition_product_and_magnetisation_sum
from slothpy._general_utilities._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy.core._hamiltonian_object import Hamiltonian


@jit(
    "float64[:](complex128[:,:], complex128[:,:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _diag_utmu(m, u):
    u = ascontiguousarray(u)
    m = ascontiguousarray(m)
    res = zeros((u.shape[0],), dtype=complex128)
    m = u.T.conjugate() @ m
    for i in prange(u.shape[0]): ######## tutaj sprawdz racing conditions anulując prange i parallel
        for k in range(u.shape[0]):
            res[i] += m[i, k] * u[k, i]

    return res.real


@jit(
    "complex128[:,:](complex128[:,:,:], complex128[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _3d_dot(u, m):
    return u[0] * m[0] + u[1] * m[1] + u[2] * m[2]


@jit(
    "complex128[:,:](complex128[:,:,:], complex128[:,:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _3d_dot_nie_czaje(u, m):
    return u[0] * m[0] + u[1] * m[1] + u[2] * m[2]

@jit(
    "types.UniTuple(complex128[:,:],2)(complex128[:,:,:], float64[:], float64,"
    " complex128[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _calculate_zeeman_matrix2(
    magnetic_momenta, soc_energies, field, orientation
):
    magnetic_momenta_orient = _3d_dot_nie_czaje(magnetic_momenta, orientation) #################################### Wrong signature after modifications
    magnetic_momenta = -field * MU_B * magnetic_momenta_orient

    # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
    for k in prange(magnetic_momenta.shape[0]):
        magnetic_momenta[k, k] += soc_energies[k]

    return magnetic_momenta_orient, magnetic_momenta


@jit([types.Tuple((float32, float32))(types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True), float32),
      types.Tuple((float64, float64))(types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True), float64),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
    parallel=True,
)
def _magnetic_moment_partition_function(energies, states_momenta, temperature):
    factor = KB * temperature
    z = 0
    m = 0
    for i in prange(energies.shape[0]):
        e = exp(-energies[i] / factor)
        z += e
        m += e * states_momenta[i]

    return m, z

@jit([(types.Array(float32, 1, 'C', False), types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True), float32),
      (types.Array(float64, 1, 'C', False), types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True), float64),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
)
def _magnetisation_temperature_single_center_av(magnetisation_array, energies, states_momenta, temperatures, weight):
    for t in range(temperatures.shape[0]):
        m, z = _magnetic_moment_partition_function(energies, states_momenta, temperatures[t])
        magnetisation_array[t] += m / z * weight


@jit([(types.Array(float32, 1, 'C', False), types.Array(float32, 1, 'C', False), types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True), types.Array(float32, 1, 'C', True), float32),
      (types.Array(float64, 1, 'C', False), types.Array(float64, 1, 'C', False), types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True), types.Array(float64, 1, 'C', True), float64),
    ],
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
)
def _momenta_partition_functions_multi_center_av(momenta_array, partition_functions_array, energies, states_momenta, temperatures, weight):
    for t in range(temperatures.shape[0]):
        m, z = _magnetic_moment_partition_function(energies, states_momenta, temperatures[t])
        momenta_array[t] += m * weight
        partition_functions_array[t] += z


def _magnetisation_proxy(slt_hamiltonian_info, sm_arrays_info_list: list[SharedMemoryArrayInfo], args_list, process_index, start: int, end: int, returns: bool = False):
    hamiltonian = Hamiltonian(sm_arrays_info_list, slt_hamiltonian_info[0], slt_hamiltonian_info[1])
    sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list[hamiltonian._shared_memory_index:])
    if returns:
        return _magnetisation_average(hamiltonian, *arrays, process_index, start, end) #no *args because the list is empty
    else:
        _magnetisation(hamiltonian, *arrays, process_index, start, end)


def _magnetisation(
    hamiltonian: Hamiltonian,
    magnetic_fields: ndarray,
    orientations: ndarray,
    temperatures: ndarray,
    progress_array: ndarray,
    magnetisation_array: ndarray,
    number_of_states: int,
    process_index: int,
    start: int,
    end: int,
):  
    pass
    # magnetic_fields_shape_0 = magnetic_fields.shape[0]
    
    # for i in range(start, end):
    #     hamiltonian._magnetic_field = orientations[i//magnetic_fields_shape_0, :3] * magnetic_fields[i%magnetic_fields_shape_0]
    #     energies = hamiltonian.zeeman_energies(number_of_states)
    #     zeeman_array[i] = energies[:number_of_states] * h_cm_1
    #     progress_array[process_index] += 1


def _magnetisation_average(
    hamiltonian: Hamiltonian,
    magnetic_fields: ndarray,
    orientations: ndarray,
    temperatures: ndarray,
    progress_array: ndarray,
    process_index: int,
    start: int,
    end: int,
):
    orientations_shape_0 = orientations.shape[0]
    start_field_index = start // orientations_shape_0
    end_field_index = (end - 1) // orientations_shape_0 + 1
    previous_field_index = start_field_index
    magnetisation_array = zeros((end_field_index - start_field_index, temperatures.shape[0]), dtype=temperatures.dtype)
    magnetisation_index = 0
    for i in range(start, end):
        current_field_index = i // orientations_shape_0
        orientation_index = i % orientations_shape_0
        if current_field_index != previous_field_index:
            magnetisation_index += 1
            previous_field_index = current_field_index
        hamiltonian._magnetic_field = magnetic_fields[current_field_index] * orientations[orientation_index, :3]
        energies, states_momenta = hamiltonian.slpjm_under_magnetic_field("m", orientations[orientation_index, :3])
        if isinstance(energies, list):
            momenta = zeros((len(energies), temperatures.shape[0]), dtype=temperatures.dtype, order='C')
            partition_functions = zeros((len(energies), temperatures.shape[0]), dtype=temperatures.dtype, order='C')
            for index, (energy, moment) in enumerate(zip(energies, states_momenta)):
                _momenta_partition_functions_multi_center_av(momenta[index, :], partition_functions[index, :], energy, moment, temperatures, orientations[orientation_index, 3])
            partition_product, magnetisation_sum = _validate_and_compute_partition_product_and_magnetisation_sum(partition_functions[:-1, :], momenta[:-1, :])
            print(partition_product, magnetisation_sum)
            magnetisation_array[magnetisation_index, :] += (momenta[-1, :] + magnetisation_sum * partition_product) / (partition_functions[-1, :] + partition_product)
        else:
            _magnetisation_temperature_single_center_av(magnetisation_array[magnetisation_index, :], energies, states_momenta, temperatures, orientations[orientation_index, 3])
        progress_array[process_index] += 1

    return start_field_index, end_field_index, magnetisation_array


@jit(
    "float64(float64[:], float64[:], float64)",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
)
def _calculate_magnetization(
    energies: ndarray, states_momenta: ndarray, temperature: float64
) -> float64:
    energies = ascontiguousarray(energies)
    states_momenta = ascontiguousarray(states_momenta)
    # Boltzman weights
    energies = exp(-(energies - energies[0]) / (KB * temperature))

    # Partition function
    z = sum(energies)

    # Weighted magnetic moments of microstates
    m = vdot(states_momenta, energies)

    return m / z


@jit(
    "(float64[:,:], complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:], int64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
    inline="always",
)
def _mt_over_fields_grid(
    mht_array,
    magnetic_momenta,
    soc_energies,
    fields,
    grid,
    temperatures,
    progress,
):
    magnetic_momenta = ascontiguousarray(magnetic_momenta)

    for i in range(fields.shape[0]):
        for j in range(grid.shape[0]):
            orientation = ascontiguousarray(
                grid[j, :3].copy().astype(complex128)
            )
            oriented_momenta, zeeman_matrix = _calculate_zeeman_matrix2(
                magnetic_momenta, soc_energies, fields[i], orientation
            )
            energies, zeeman_matrix = eigh(zeeman_matrix)
            states_momenta = _diag_utmu(oriented_momenta, zeeman_matrix)
            energies = energies - energies[0]
            for t in range(temperatures.shape[0]):
                mht_array[i, t] += (
                    _calculate_magnetization(
                        energies, states_momenta, temperatures[t]
                    )
                    * grid[j, 3]
                )
            progress += 1


@jit(
    "float64[:,:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _mt_over_fields_tensor(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    temperatures: ndarray,
):
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]

    mht_tensor_array = ascontiguousarray(
        zeros((fields_shape_0, temperatures_shape_0, 3, 3), dtype=float64)
    )

    for f in range(fields_shape_0):
        # Initialize arrays
        mt_tensor_array = ascontiguousarray(
            zeros((temperatures_shape_0, 3, 3), dtype=float64)
        )

        # Perform calculations for each tensor component
        for i in range(3):
            for j in range(3):
                # Construct Zeeman matrix
                zeeman_matrix = -fields[f] * MU_B * magnetic_momenta[j]
                for k in range(zeeman_matrix.shape[0]):
                    zeeman_matrix[k, k] += soc_energies[k]

                # Diagonalize full Hamiltonian matrix
                eigenvalues, eigenvectors = eigh(zeeman_matrix)
                magnetic_momenta = ascontiguousarray(magnetic_momenta)

                # Transform momentum according to the new eigenvectors
                states_momenta = (
                    eigenvectors.conj().T @ magnetic_momenta[i] @ eigenvectors
                )

                # Get diagonal momenta of the new states
                states_momenta = diag(states_momenta).real.astype(float64)

                # Compute partition function and magnetization for each T
                for t in range(temperatures.shape[0]):
                    mt_tensor_array[t, i, j] = _calculate_magnetization(
                        eigenvalues, states_momenta, temperatures[t]
                    )
        mht_tensor_array[f, :, :, :] = mt_tensor_array[:, :, :]

    return mht_tensor_array


@jit(
    "float64[:,:,:](complex128[:,:,:], float64[:], float64[:], float64[:,:],"
    " float64[:])",
    nopython=True,
    nogil=True,
    cache=True,
    fastmath=True,
)
def _mt_over_grid_fields(
    magnetic_momenta: ndarray,
    soc_energies: ndarray,
    fields: ndarray,
    grid: ndarray,
    temperatures: ndarray,
):
    fields_shape_0 = fields.shape[0]
    temperatures_shape_0 = temperatures.shape[0]
    grid_shape_0 = grid.shape[0]

    mght_array = zeros(
        (grid_shape_0, fields_shape_0, temperatures_shape_0), dtype=float64
    )

    for g in range(grid_shape_0):
        mht_array = zeros(
            (fields_shape_0, temperatures_shape_0), dtype=float64
        )
        grid_momenta = (
            grid[g, 0] * magnetic_momenta[0]
            + grid[g, 1] * magnetic_momenta[1]
            + grid[g, 2] * magnetic_momenta[2]
        )

        for f in range(fields_shape_0):
            zeeman_matrix = magnetic_momenta[0] # _calculate_zeeman_matrix(
            #     magnetic_momenta, soc_energies, fields[f], grid[g]
            # )
            eigenvalues, eigenvectors = eigh(zeeman_matrix)

            states_momenta = (
                eigenvectors.conj().T @ grid_momenta @ eigenvectors
            )

            states_momenta_diag = diag(states_momenta).real.astype(float64)

            for t in range(temperatures_shape_0):
                mht_array[f, t] = _calculate_magnetization(
                    eigenvalues, states_momenta_diag, temperatures[t]
                )

        mght_array[g, :, :] = mht_array[:, :]

    return mght_array


def _calculate_mht(
    mht_array_name: str,
    magnetic_momenta_name: str,
    soc_energies_name: str,
    fields_name: str,
    grid_name: str,
    temperatures_name: str,
    mht_array_shape: str,
    magnetic_momenta_shape: tuple,
    soc_energies_shape: tuple,
    grid_shape: tuple,
    temperatures_shape: tuple,
    field_chunk: tuple,
    progress_name,
    index,
) -> ndarray:
    magnetic_momenta_shared = SharedMemory(magnetic_momenta_name)
    magnetic_momenta_array = ndarray(
        magnetic_momenta_shape,
        complex128,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float64, soc_energies_shared.buf
    )
    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray(grid_shape, float64, grid_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float64,
        temperatures_shared.buf,
    )
    progress_shared = SharedMemory(progress_name)
    progress_array = ndarray(
        (1,), int64, progress_shared.buf, index * dtype(int64).itemsize
    )

    offset = dtype(float64).itemsize * field_chunk[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray((chunk_length,), float64, fields_shared.buf, offset)

    offset = dtype(float64).itemsize * field_chunk[0] * temperatures_shape[0]
    chunk_length = field_chunk[1] - field_chunk[0]

    mht_array_shared = SharedMemory(mht_array_name)
    mht_array_array = ndarray(
        (chunk_length, temperatures_shape[0]),
        float64,
        mht_array_shared.buf,
        offset,
    )

    # Switch to tensor calculation
    # if array_equal(grid_array, array([1.0])):
    #     return _mt_over_fields_tensor(
    #         magnetic_momenta_array,
    #         soc_energies_array,
    #         fields_array,
    #         temperatures_array,
    #     )

    return _mt_over_fields_grid(
        mht_array_array,
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        temperatures_array,
        progress_array,
    )


def _calculate_mht_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mht(*args)

    return mt


def _arg_iter_mht(
    mht_name,
    magnetic_momenta_name,
    soc_energies_name,
    fields_name,
    grid_name,
    temperatures_name,
    mht_shape,
    magnetic_momenta_shape,
    soc_energies_shape,
    grid_shape,
    temperatures_shape,
    fields_chunks,
    progress_name,
):
    # Iterator generator for arguments with different field values to be
    # distributed along num_process processes
    for index, field_chunk in enumerate(fields_chunks):
        yield (
            mht_name,
            magnetic_momenta_name,
            soc_energies_name,
            fields_name,
            grid_name,
            temperatures_name,
            mht_shape,
            magnetic_momenta_shape,
            soc_energies_shape,
            grid_shape,
            temperatures_shape,
            field_chunk,
            progress_name,
            index,
        )


def _mth(
    filename: str,
    group: str,
    fields: ndarray[float64],
    grid: ndarray[float64],
    temperatures: ndarray[float64],
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    rotation: ndarray[float64] = None,
) -> ndarray:
    # Flag for tensor calculation
    tensor_calc = False
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    # Get number of parallel proceses to be used
    num_processes, num_threads = _get_num_of_processes(
        num_cpu, num_threads, fields.shape[0]
    )

    #  Allocate arrays as contiguous
    fields = ascontiguousarray(fields, dtype=float64)
    temperatures = ascontiguousarray(temperatures, dtype=float64)
    grid = ascontiguousarray(grid, dtype=float64)
    mht_array = zeros((fields.shape[0], temperatures.shape[0]), dtype=float64)
    progress = zeros((num_processes,), dtype=int64)

    if array_equal(grid, array([1.0])):
        tensor_calc = True

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)
        mht_array_shared = smm.SharedMemory(size=mht_array.nbytes)
        progress_shared = smm.SharedMemory(size=progress.nbytes)

        # Copy data to shared memory
        magnetic_momenta_shared_arr = ndarray(
            magnetic_momenta.shape,
            dtype=magnetic_momenta.dtype,
            buffer=magnetic_momenta_shared.buf,
        )
        soc_energies_shared_arr = ndarray(
            soc_energies.shape,
            dtype=soc_energies.dtype,
            buffer=soc_energies_shared.buf,
        )
        fields_shared_arr = ndarray(
            fields.shape, dtype=fields.dtype, buffer=fields_shared.buf
        )
        grid_shared_arr = ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )
        temperatures_shared_arr = ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )
        mht_array_shared_arr = ndarray(
            mht_array.shape,
            dtype=mht_array.dtype,
            buffer=mht_array_shared.buf,
        )
        progress_shared_arr = ndarray(
            progress.shape,
            dtype=int64,
            buffer=progress_shared.buf,
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]
        mht_array_shared_arr[:] = mht_array[:]
        progress_shared_arr[:] = progress[:]

        del magnetic_momenta
        del soc_energies
        del fields
        del grid
        del temperatures
        del mht_array
        del progress

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                set_num_threads(num_threads)
                with Pool(num_processes) as p:
                    p.map(
                        _calculate_mht_wrapper,
                        _arg_iter_mht(
                            mht_array_shared.name,
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            temperatures_shared.name,
                            mht_array_shared_arr.shape,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            grid_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            _distribute_chunks(
                                fields_shared_arr.shape[0], num_processes
                            ),
                            progress_shared.name,
                        ),
                    )
        mht_array_shared_arr = array(mht_array_shared_arr).T

    # # Hidden option for susceptibility tensor calculation.
    # if tensor_calc:
    #     return concatenate(mht)

    # # Collecting results in plotting-friendly convention for M(H)
    # mth_array = concatenate(mht).T

    return mht_array_shared_arr  # Returning values in Bohr magnetons


def _update_progress_bar(current, total, start_time):
    percentage = 100 * (current / total)
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    current_time = time.perf_counter() - start_time
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    print(
        f"\rProgress: |{bar}| {percentage:.2f}% | Time: {current_time:.1f} s |"
        f" CPU Usage: {cpu_usage:.1f}% | Memory Usage: {memory_usage}% |",
        end="\r",
    )


def _monitor_progress(shared_counter, N, num_processes, num_threads):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"Calculate Magnetisation started: {current_time} | Processes:"
        f" {num_processes} | Threads per Process: {num_threads} |"
    )
    start_time = time.perf_counter()
    while True:
        current_value = sum(shared_counter)
        _update_progress_bar(
            current_value,
            N,
            start_time,
        )
        if current_value >= N:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\nCompleated: {current_time}")
            print("Collecting and processing results...")
            break
        time.sleep(0.19)


def _arg_iter_mght(
    magnetic_momenta_name,
    soc_energies_name,
    fields_name,
    grid_name,
    temperatures_name,
    magnetic_momenta_shape,
    soc_energies_shape,
    fields_shape,
    temperatures_shape,
    grids_chunks,
):
    # Iterator generator for arguments with different grid values to be
    # distributed along num_process processes
    for grid_chunk in grids_chunks:
        yield (
            magnetic_momenta_name,
            soc_energies_name,
            fields_name,
            grid_name,
            temperatures_name,
            magnetic_momenta_shape,
            soc_energies_shape,
            fields_shape,
            temperatures_shape,
            grid_chunk,
        )


def _calculate_mght(
    magnetic_momenta_name: str,
    soc_energies_name: str,
    fields_name: str,
    grid_name: str,
    temperatures_name: str,
    magnetic_momenta_shape: tuple,
    soc_energies_shape: tuple,
    fields_shape: tuple,
    temperatures_shape: tuple,
    grid_chunk: tuple,
) -> ndarray:
    magnetic_momenta_shared = SharedMemory(magnetic_momenta_name)
    magnetic_momenta_array = ndarray(
        magnetic_momenta_shape,
        complex128,
        magnetic_momenta_shared.buf,
    )
    soc_energies_shared = SharedMemory(soc_energies_name)
    soc_energies_array = ndarray(
        soc_energies_shape, float64, soc_energies_shared.buf
    )
    fields_shared = SharedMemory(fields_name)
    fields_array = ndarray(fields_shape, float64, fields_shared.buf)
    temperatures_shared = SharedMemory(temperatures_name)
    temperatures_array = ndarray(
        temperatures_shape,
        float64,
        temperatures_shared.buf,
    )

    offset = dtype(float64).itemsize * grid_chunk[0] * 3
    chunk_length = grid_chunk[1] - grid_chunk[0]

    grid_shared = SharedMemory(grid_name)
    grid_array = ndarray((chunk_length, 3), float64, grid_shared.buf, offset)

    return _mt_over_grid_fields(
        magnetic_momenta_array,
        soc_energies_array,
        fields_array,
        grid_array,
        temperatures_array,
    )


def _calculate_mght_wrapper(args):
    # Unpack arguments and call the function
    mt = _calculate_mght(*args)

    return mt


def _mag_3d(
    filename: str,
    group: str,
    fields: ndarray,
    grid_type: Literal["mesh", "fibonacci"],
    grid_number: int,
    temperatures: ndarray,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    rotation: ndarray = None,
    sus_3d_num: bool = False,
) -> ndarray:
    # Read data from HDF5 file
    (
        magnetic_momenta,
        soc_energies,
    ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    if grid_type != "mesh" and grid_type != "fibonacci":
        raise ValueError(
            'The only allowed grid types are "mesh" or "fibonacci".'
        ) from None
    if grid_type == "mesh":
        grid = _meshgrid_over_sphere_flatten(grid_number)
        num_points = 2 * grid_number**2
    elif grid_type == "fibonacci":
        grid = _fibonacci_over_sphere(grid_number)
        num_points = grid_number
    else:
        raise (
            ValueError('Grid type can only be set to "mesh" or "fibonacci".')
        )
    # Get number of parallel proceses to be used
    num_process, num_threads = _get_num_of_processes(
        num_cpu, num_threads, num_points
    )

    #  Allocate arrays as contiguous
    fields = ascontiguousarray(fields, dtype=float64)
    temperatures = ascontiguousarray(temperatures, dtype=float64)
    grid = ascontiguousarray(grid, dtype=float64)

    with SharedMemoryManager() as smm:
        # Create shared memory for arrays
        magnetic_momenta_shared = smm.SharedMemory(
            size=magnetic_momenta.nbytes
        )
        soc_energies_shared = smm.SharedMemory(size=soc_energies.nbytes)
        fields_shared = smm.SharedMemory(size=fields.nbytes)
        grid_shared = smm.SharedMemory(size=grid.nbytes)
        temperatures_shared = smm.SharedMemory(size=temperatures.nbytes)

        # Copy data to shared memory
        magnetic_momenta_shared_arr = ndarray(
            magnetic_momenta.shape,
            dtype=magnetic_momenta.dtype,
            buffer=magnetic_momenta_shared.buf,
        )
        soc_energies_shared_arr = ndarray(
            soc_energies.shape,
            dtype=soc_energies.dtype,
            buffer=soc_energies_shared.buf,
        )
        fields_shared_arr = ndarray(
            fields.shape, dtype=fields.dtype, buffer=fields_shared.buf
        )
        grid_shared_arr = ndarray(
            grid.shape, dtype=grid.dtype, buffer=grid_shared.buf
        )
        temperatures_shared_arr = ndarray(
            temperatures.shape,
            dtype=temperatures.dtype,
            buffer=temperatures_shared.buf,
        )

        magnetic_momenta_shared_arr[:] = magnetic_momenta[:]
        soc_energies_shared_arr[:] = soc_energies[:]
        fields_shared_arr[:] = fields[:]
        grid_shared_arr[:] = grid[:]
        temperatures_shared_arr[:] = temperatures[:]

        del magnetic_momenta
        del soc_energies
        del fields
        del grid
        del temperatures

        with threadpool_limits(limits=num_threads, user_api="blas"):
            with threadpool_limits(limits=num_threads, user_api="openmp"):
                set_num_threads(num_threads)
                with Pool(num_process) as p:
                    mght = p.map(
                        _calculate_mght_wrapper,
                        _arg_iter_mght(
                            magnetic_momenta_shared.name,
                            soc_energies_shared.name,
                            fields_shared.name,
                            grid_shared.name,
                            temperatures_shared.name,
                            magnetic_momenta_shared_arr.shape,
                            soc_energies_shared_arr.shape,
                            fields_shared_arr.shape,
                            temperatures_shared_arr.shape,
                            _distribute_chunks(
                                grid_shared_arr.shape[0], num_process
                            ),
                        ),
                    )

        mght = concatenate(mght)
        grid = grid_shared_arr[:].copy()
        fields_shape = fields_shared_arr.shape[0]
        temperatures_shape = temperatures_shared_arr.shape[0]

    if grid_type == "mesh":
        mag_3d = mght.reshape(
            (
                grid_number,
                grid_number * 2,
                fields_shape,
                temperatures_shape,
            )
        )
        mag_3d = mag_3d.transpose((2, 3, 0, 1))
        grid = grid.reshape(grid_number, grid_number * 2, 3)
        grid = grid.transpose((2, 0, 1))
        if sus_3d_num:
            return mag_3d, grid
        mag_3d_array = grid[:, newaxis, newaxis, :, :] * mag_3d
    elif grid_type == "fibonacci":
        if sus_3d_num:
            return mght, grid
        mag_3d_array = grid[:, :, newaxis, newaxis] * mght[:, newaxis, :, :]
        mag_3d_array = mag_3d_array.transpose((2, 3, 0, 1))

    return mag_3d_array
