import time
import multiprocessing
import threadpoolctl
import numpy as np
from numba import jit
from slothpy.general_utilities.system import _get_num_of_processes
from slothpy.general_utilities.io import (
    _get_soc_mag_mom_and_ener_from_hdf5,
)
from slothpy.magnetism.zeeman import calculate_zeeman_matrix


@jit(
    "float64(float64[:], float64[:], float64)",
    nopython=True,
    cache=True,
    nogil=True,
)
def calculate_magnetization(
    energies: np.ndarray, states_momenta: np.ndarray, temperature: np.float64
) -> np.float64:
    """
    Calculates the magnetization for a given array of states energies, momenta, and temperature.

    Args:
        energies (np.ndarray[np.float64]): Array of energies.
        states_momenta (np.ndarray[np.float64]): Array of states momenta.
        temperature (np.float64): Temperature value.

    Returns:
        np.float64: Magnetization value.

    """
    kB = 3.166811563e-6  # Boltzmann constant a.u./K

    # Boltzman weights
    exp_diff = np.exp(-(energies - energies[0]) / (kB * temperature))

    # Partition function
    z = np.sum(exp_diff)

    # Weighted magnetic moments of microstates
    m = np.sum(states_momenta * exp_diff)

    return m / z


@jit(
    (
        "float64[:](complex128[:,:,:], float64[:], float64, float64[:,:],"
        " float64[:])"
    ),
    nopython=True,
    cache=True,
    nogil=True,
)
def calculate_mt(
    magnetic_momenta: np.ndarray,
    soc_energies: np.ndarray,
    field: np.float64,
    grid: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Calculates the M(T) array for a given array of magnetic moments, SOC energies, directional grid for powder averaging,
    and temperatures for a particular value of magnetic field.

    Args:
        magnetic_moment (np.ndarray[np.complex128]): Array of magnetic moments.
        soc_energies (np.ndarray[np.float64]): Array of SOC energies.
        field (np.float64): Value of magnetic field.
        grid (np.ndarray[np.float64]): Grid array.
        temperatures (np.ndarray[np.float64]): Array of temperatures.

    Returns:
        np.ndarray[np.float64]: M(T) array.

    """

    # Initialize arrays
    mt_array = np.ascontiguousarray(
        np.zeros((temperatures.shape[0]), dtype=np.float64)
    )
    magnetic_momenta = np.ascontiguousarray(magnetic_momenta)
    soc_energies = np.ascontiguousarray(soc_energies)

    # Perform calculations for each magnetic field orientation
    for j in range(grid.shape[0]):
        # Construct Zeeman matrix
        orientation = grid[j, :3]

        zeeman_matrix = calculate_zeeman_matrix(
            magnetic_momenta, soc_energies, field, orientation
        )

        # Diagonalize full Hamiltonian matrix
        eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
        eigenvalues = np.ascontiguousarray(eigenvalues)
        eigenvectors = np.ascontiguousarray(eigenvectors)

        # Transform momenta according to the new eigenvectors
        states_momenta = (
            eigenvectors.conj().T
            @ (
                grid[j, 0] * magnetic_momenta[0]
                + grid[j, 1] * magnetic_momenta[1]
                + grid[j, 2] * magnetic_momenta[2]
            )
            @ eigenvectors
        )

        # Get diagonal momenta of the new states
        states_momenta = np.diag(states_momenta).real.astype(np.float64)

        # Compute partition function and magnetization for each T
        for t in range(temperatures.shape[0]):
            mt_array[t] += (
                calculate_magnetization(
                    eigenvalues, states_momenta, temperatures[t]
                )
                * grid[j, 3]
            )

    return mt_array


def calculate_mt_wrapper(args):
    """Wrapper function for parallel use of M(T) calulations

    Args:
        args (tuple): Tuple of arguments for calculate_mt function

    Returns:
        np.ndarray[np.float64]: M(T) array.
    """
    # Unpack arguments and call the function
    mt = calculate_mt(*args)

    return mt


def arg_iter_mth(magnetic_momenta, soc_energies, fields, grid, temperatures):
    # Iterator generator for arguments with different field values to be distributed along num_process processes
    for i in range(fields.shape[0]):
        yield (magnetic_momenta, soc_energies, fields[i], grid, temperatures)


def _mth(
    filename: str,
    group: str,
    fields: np.ndarray,
    grid: np.ndarray,
    temperatures: np.ndarray,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
) -> np.ndarray:
    """
    Calculates the M(T,H) array using magnetic moments and SOC energies for given fields, grid (for powder direction averaging), and temperatures.
    The function is parallelized across num_cpu CPUs for simultaneous calculations over different magnetic field values.

    Args:
        path (str): Path to the file.
        hdf5_file (str): Name of the HDF5 file.
        states_cutoff (int): States cutoff value.
        fields (np.ndarray[np.float64]): Array of fields.
        grid (np.ndarray[np.float64]): Grid array of directions for powder averaging
        temperatures (np.ndarray[np.float64]): Array of temperatures
        num_cpu (int): Number of CPU used for calculations. (work will be distributed across num_cpu//num_threads processes)

    Raises:
        ValueError: If insufficient number of CPUs were assigned considering number of the desired threads.

    Returns:
        np.ndarray[np.float64]: M(T,H) array.

    """

    # Get number of parallel proceses to be used
    num_process = _get_num_of_processes(num_cpu, num_threads)

    # Initialize the result array
    mth_array = np.zeros((temperatures.size, fields.size), dtype=np.float64)

    # Get magnetic field in a.u. and allocate arrays as contiguous
    fields = np.ascontiguousarray(fields)
    grid = np.ascontiguousarray(grid)
    temperatures = np.ascontiguousarray(temperatures)

    # Read data from HDF5 file
    (
        mag_mom,
        soc_ener,
    ) = _get_soc_mag_mom_and_ener_from_hdf5(filename, group, states_cutoff)

    with threadpoolctl.threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpoolctl.threadpool_limits(
            limits=num_threads, user_api="openmp"
        ):
            with multiprocessing.Pool(num_process) as p:
                mt = p.map(
                    calculate_mt_wrapper,
                    arg_iter_mth(
                        mag_mom,
                        soc_ener,
                        fields,
                        grid,
                        temperatures,
                    ),
                )

    # Collecting results in plotting-friendly convention for M(H)
    for i in range(fields.shape[0]):
        mth_array[:, i] = mt[i]

    return mth_array  # Returning values in Bohr magnetons


def arg_iter_mag_3d(
    magnetic_moment, soc_energies, fields, theta, phi, temperatures
):
    for k in range(fields.shape[0]):
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                yield (
                    magnetic_moment,
                    soc_energies,
                    fields[k],
                    np.array(
                        [
                            [
                                np.sin(phi[i, j]) * np.cos(theta[i, j]),
                                np.sin(phi[i, j]) * np.sin(theta[i, j]),
                                np.cos(phi[i, j]),
                                1.0,
                            ]
                        ]
                    ),
                    temperatures,
                )


def mag_3d(
    filename: str,
    group: str,
    states_cutoff: int,
    fields: np.ndarray,
    spherical_grid: int,
    temperatures: np.ndarray,
    num_cpu: int,
    num_threads: int,
) -> np.ndarray:
    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu, num_threads)

    # Create a gird
    theta = np.linspace(0, 2 * np.pi, 2 * spherical_grid)
    phi = np.linspace(0, np.pi, spherical_grid)
    theta, phi = np.meshgrid(theta, phi)

    # Initialize the result array
    mag_3d_array = np.zeros(
        (fields.shape[0], temperatures.shape[0], phi.shape[0], phi.shape[1]),
        dtype=np.float64,
    )

    # Read data from HDF5 file
    (
        magnetic_moment,
        soc_energies,
    ) = get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff
    )

    with threadpoolctl.threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpoolctl.threadpool_limits(
            limits=num_threads, user_api="openmp"
        ):
            # Parallel M(T,H) calculation over different grid points
            with multiprocessing.Pool(num_process) as p:
                mth = p.map(
                    calculate_mt_wrapper,
                    arg_iter_mag_3d(
                        magnetic_moment,
                        soc_energies,
                        fields,
                        theta,
                        phi,
                        temperatures,
                    ),
                )

            pool_index = 0
            for k in range(fields.shape[0]):
                for i in range(phi.shape[0]):
                    for j in range(phi.shape[1]):
                        mag_3d_array[k, :, i, j] = mth[pool_index][:]
                        pool_index += 1

    x = (np.sin(phi) * np.cos(theta))[
        np.newaxis, np.newaxis, :, :
    ] * mag_3d_array
    y = (np.sin(phi) * np.sin(theta))[
        np.newaxis, np.newaxis, :, :
    ] * mag_3d_array
    z = (np.cos(phi))[np.newaxis, np.newaxis, :, :] * mag_3d_array

    return x, y, z
