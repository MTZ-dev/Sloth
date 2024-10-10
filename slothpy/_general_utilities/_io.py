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

from typing import Tuple, Literal
from os import remove
from os.path import join
from re import compile, search, findall

from h5py import File, string_dtype
from numpy import ndarray, dtype, array, ascontiguousarray, zeros, empty, any, diagonal, loadtxt, sum, reshape, mean, arange, transpose, newaxis, min, int64, float64, complex128
from scipy.linalg import eigh, eigvalsh

from slothpy.core._slothpy_exceptions import SltFileError, SltReadError
from slothpy._angular_momentum._rotation import _rotate_vector_operator
from slothpy.core._config import settings
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil
from slothpy._general_utilities._constants import A_BOHR

##### I get regex warinngs with invalid escape sequences!
# /home/mikolaj/Sloth/Sloth/slothpy/_general_utilities/_io.py:715: SyntaxWarning: invalid escape sequence '\('
#   regex = compile("SOC MATRIX \(A\.U\.\)\n")
# /home/mikolaj/Sloth/Sloth/slothpy/_general_utilities/_io.py:778: SyntaxWarning: invalid escape sequence '\('
#   regex = compile("SOC and SSC MATRIX \(A\.U\.\)\n")
# /home/mikolaj/Sloth/Sloth/slothpy/_general_utilities/_io.py:822: SyntaxWarning: invalid escape sequence '\('
#   regex = compile("SOC MATRIX \(A\.U\.\)\n")
##### Use asarray no copy (or here it is unnecessary at all)

def _grep_to_file(
    path_inp: str,
    inp_file: str,
    pattern: str,
    path_out: str,
    out_file: str,
    lines: int = 1,
) -> None:
    """
    Extracts lines from an input file that match a specified pattern and saves
    them to an output file.

    Args:
        path_inp (str): Path to the input file.
        inp_file (str): Name of the input file.
        pattern (str): Pattern to search for in the input file.
        path_out (str): Path to the output file.
        out_file (str): Name of the output file.
        lines (int, optional): Number of lines to save after a matching pattern
        is found. Defaults to 1 (save only the line with pattern occurrence).

    Raises:
        ValueError: If the pattern is not found in the input file.
    """
    regex = compile(pattern)

    input_file = join(path_inp, inp_file)
    output_file = join(path_out, out_file)

    with open(input_file, "r") as file, open(output_file, "w") as output:
        save_lines = False
        pattern_found = False

        for line in file:
            if regex.search(line):
                save_lines = True
                pattern_found = True
                remaining_lines = lines

            if save_lines:
                output.write(line)
                remaining_lines -= 1

                if remaining_lines <= 0:
                    save_lines = False

        if not pattern_found:
            raise ValueError("Pattern not found in the input file.")


def _group_exists(hdf5_file, group_name: str):
    with File(hdf5_file, "r") as file:
        return group_name in file


def _dataset_exists(hdf5_file, group_name, dataset_name):
    with File(hdf5_file, "r") as file:
        if group_name in file:
            group = file[group_name]
            return dataset_name in group

        return False


def _get_orca_so_blocks_size(
    path: str, orca_file: str
) -> tuple[int, int, int, int]:
    """
    Retrieves the dimensions and block sizes for spin-orbit calculations from
    an ORCA file.

    Args:
        path (str): Path to the ORCA file.
        orca_file (str): Name of the ORCA file.

    Returns:
        tuple[int, int, int, int]: A tuple containing the spin-orbit dimension,
        block size, number of whole blocks, and remaining columns.

    Raises:
        ValueError: If the spin-orbit dimension is not found in the ORCA file.
    """
    orca_file_path = join(path, orca_file)

    with open(orca_file_path, "r") as file:
        content = file.read()

    so_dim_match = search(r"Dim\(SO\)\s+=\s+(\d+)", content)
    if so_dim_match:
        so_dim = int(so_dim_match.group(1))
    else:
        raise ValueError("Dim(SO) not found in the ORCA file.")

    num_blocks = so_dim // 6

    if so_dim % 6 != 0:
        num_blocks += 1

    block_size = ((so_dim + 1) * num_blocks) + 1
    num_of_whole_blocks = so_dim // 6
    remaining_columns = so_dim % 6

    return so_dim, block_size, num_of_whole_blocks, remaining_columns


def _orca_spin_orbit_to_slt(
    path_orca: str,
    inp_orca: str,
    path_out: str,
    hdf5_output: str,
    name: str,
    pt2: bool = False,
) -> None:
    """
    Converts spin-orbit calculations from an ORCA .out file to
    a HDF5 file format.

    Args:
        path_orca (str): Path to the ORCA file.
        inp_orca (str): Name of the ORCA file.
        path_out (str): Path for the output files.
        hdf5_output (str): Name of the HDF5 output file.
        pt2 (bool): Get results from the second-order perturbation-corrected
                    states.

    Raises:
        ValueError: If the spin-orbit dimension is not found in the ORCA file.
        ValueError: If the pattern is not found in the input file.
    """
    hdf5_file = join(path_out, hdf5_output)

    # Retrieve dimensions and block sizes for spin-orbit calculations
    (
        so_dim,
        block_size,
        num_of_whole_blocks,
        remaining_columns,
    ) = _get_orca_so_blocks_size(path_orca, inp_orca)

    # Create HDF5 file and ORCA group
    output = File(f"{hdf5_file}.slt", "a")
    orca = output.create_group(str(name))
    orca.attrs["Description"] = (
        f"Group({name}) containing results of relativistic SOC ORCA"
        " calculations - angular momenta and SOC matrix in CI basis"
    )

    # Extract and process matrices (SX, SY, SZ, LX, LY, LZ)
    matrices = ["SX", "SY", "SZ", "LX", "LY", "LZ"]
    descriptions = [
        "real part",
        "imaginary part",
        "real part",
        "real part",
        "real part",
        "real part",
    ]

    for matrix_name, description in zip(matrices, descriptions):
        out_file_name = f"{matrix_name}.tmp"
        out_file = join(path_out, out_file_name)
        pattern = f"{matrix_name} MATRIX IN CI BASIS\n"

        # Extract lines matching the pattern to a temporary file
        _grep_to_file(
            path_orca,
            inp_orca,
            pattern,
            path_out,
            out_file_name,
            block_size + 4,
        )  # The first 4 lines are titles

        with open(out_file, "r") as file:
            if pt2:
                for _ in range(block_size + 4):
                    file.readline()  # Skip non-pt2 block
            for _ in range(4):
                file.readline()  # Skip the first 4 lines
            matrix = empty((so_dim, so_dim), dtype=float64)
            l = 0
            for _ in range(num_of_whole_blocks):
                file.readline()  # Skip a line before each block of 6 columns
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(6):
                        matrix[i, l + j] = float64(line[j + 1])
                l += 6

            if remaining_columns > 0:
                file.readline()  # Skip a line before the remaining columns
                for i in range(so_dim):
                    line = file.readline().split()
                    for j in range(remaining_columns):
                        matrix[i, l + j] = float64(line[j + 1])

            # Create dataset in HDF5 file and assign the matrix
            dataset = orca.create_dataset(
                f"SF_{matrix_name}", shape=(so_dim, so_dim), dtype=float64
            )
            dataset[:, :] = matrix[:, :]
            dataset.attrs["Description"] = (
                f"Dataset containing {description} of SF_{matrix_name} matrix"
                " in CI basis (spin-free)"
            )

        # Remove the temporary file
        remove(out_file)

    # Extract and process SOC matrix
    _grep_to_file(
        path_orca,
        inp_orca,
        r"SOC MATRIX \(A\.U\.\)\n",
        path_out,
        "SOC.tmp",
        2 * block_size + 4 + 2,
    )  # The first 4 lines are titles, two lines in the middle for Im part

    soc_file = join(path_out, "SOC.tmp")

    with open(soc_file, "r") as file:
        content = file.read()

    # Replace "-" with " -"
    modified_content = content.replace("-", " -")

    # Write the modified content to the same file
    with open(soc_file, "w") as file:
        file.write(modified_content)

    with open(soc_file, "r") as file:
        if pt2:
            for _ in range(2 * block_size + 4 + 2):
                file.readline()  # Skip non-pt2 block
        for _ in range(4):
            file.readline()  # Skip the first 4 lines
        matrix_real = empty((so_dim, so_dim), dtype=float64)
        l = 0
        for _ in range(num_of_whole_blocks):
            file.readline()  # Skip a line before each block of 6 columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_real[i, l + j] = float64(line[j + 1])
            l += 6

        if remaining_columns > 0:
            file.readline()  # Skip a line before the remaining columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(remaining_columns):
                    matrix_real[i, l + j] = float64(line[j + 1])

        for _ in range(2):
            file.readline()  # Skip 2 lines separating real and imaginary part

        matrix_imag = empty((so_dim, so_dim), dtype=float64)
        l = 0
        for _ in range(num_of_whole_blocks):
            file.readline()  # Skip a line before each block of 6 columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(6):
                    matrix_imag[i, l + j] = float64(line[j + 1])
            l += 6

        if remaining_columns > 0:
            file.readline()  # Skip a line before the remaining columns
            for i in range(so_dim):
                line = file.readline().split()
                for j in range(remaining_columns):
                    matrix_imag[i, l + j] = float64(line[j + 1])

    # Create a dataset in HDF5 file for SOC matrix
    dataset = orca.create_dataset(
        "SOC", shape=(so_dim, so_dim), dtype=complex128
    )
    complex_matrix = array(matrix_real + 1j * matrix_imag, dtype=complex128)
    dataset[:, :] = complex_matrix[:, :]
    dataset.attrs[
        "Description"
    ] = "Dataset containing complex SOC matrix in CI basis (spin-free)"

    # Remove the temporary file
    remove(soc_file)

    # Close the HDF5 file
    output.close()


def _molcas_to_slt(molcas_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool = False) -> None:
    if not molcas_filepath.endswith(".rassi.h5"):
        slt_filepath += ".rassi.h5"

    with File(f"{molcas_filepath}", "r") as rassi:
        with File(f"{slt_filepath}", "a") as slt:
            group = slt.create_group(group_name)
            group.attrs["Type"] = "HAMILTONIAN"
            group.attrs["Kind"] = "MOLCAS"
            group.attrs["Precision"] = settings.precision.upper()
            if electric_dipole_momenta:
                group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
            group.attrs["Description"] = "Relativistic MOLCAS results."

            dataset_rassi = rassi["SOS_ENERGIES"][:] - min(rassi["SOS_ENERGIES"][:])
            group.attrs["States"] = dataset_rassi.shape[0]
            dataset_out = group.create_dataset("STATES_ENERGIES", shape=dataset_rassi.shape, dtype=settings.float, data=dataset_rassi.astype(settings.float), chunks=True)
            dataset_out.attrs["Description"] = "SOC energies."

            dataset_rassi = rassi["SOS_SPIN_REAL"][:, :, :] + 1j * rassi["SOS_SPIN_IMAG"][:, :, :]
            dataset_out = group.create_dataset("SPINS", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Sx, Sy, and Sz spin matrices in the SOC basis [(x-0, y-1, z-2), :, :]."

            dataset_rassi = 1j * rassi["SOS_ANGMOM_REAL"][:, :, :] - rassi["SOS_ANGMOM_IMAG"][:, :, :]
            dataset_out = group.create_dataset("ANGULAR_MOMENTA", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Lx, Ly, and, Lz angular momentum matrices in the SOC basis [(x-0, y-1, z-2), :, :]."

            if electric_dipole_momenta:
                dataset_rassi = rassi["SOS_EDIPMOM_REAL"][:, :, :] + 1j * rassi["SOS_EDIPMOM_REAL"][:, :, :]
                dataset_out = group.create_dataset("ELECTRIC_DIPOLE_MOMENTA", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
                dataset_out.attrs["Description"] = "Px, Py, and Pz electric dipole momentum matrices in the SOC basis [(x-0, y-1, z-2), :, :]."


def _xyz_to_slt(slt_filepath, group_name, elements, positions, charge, multiplicity, group_type = "XYZ", description = "XYZ file."):
    with File(slt_filepath, 'a') as slt:
        group = slt.create_group(group_name)
        group.attrs["Type"] = group_type
        group.attrs["Number_of_atoms"] = len(elements)
        group.attrs["Precision"] = settings.precision.upper()
        group.attrs["Description"] = description
        dt = string_dtype(encoding='utf-8')
        dataset = group.create_dataset('ELEMENTS', data=array(elements, dtype='S'), dtype=dt, chunks=True)
        dataset.attrs["Description"] = "List of elements from the XYZ file."
        dataset = group.create_dataset('COORDINATES', data=positions, dtype=settings.float, chunks=True)
        dataset.attrs["Description"] = "List of elements coordinates from the XYZ file."
        if charge is not None:
            group.attrs["Charge"] = charge
        if multiplicity is not None:
            group.attrs["Multiplicity"] = multiplicity


def _unit_cell_to_slt(slt_filepath, group_name, elements, positions, cell, multiplicity, group_type = "UNIT_CELL", description = "Unit cell group containing xyz coordinates and unit cell vectors.", description_cell = "Unit cell vectors as 3x3 matrix (with vectors in rows)."):
    _xyz_to_slt(slt_filepath, group_name, elements, positions, None, multiplicity, group_type, description)
    with File(slt_filepath, 'a') as slt:
        group = slt[group_name]
        dataset = group.create_dataset('CELL', data=cell, dtype=settings.float, chunks=True)
        dataset.attrs["Description"] = description_cell


def _supercell_to_slt(slt_filepath, group_name, elements, positions, cell, nx, ny, nz, multiplicity, group_type = "SUPERCELL", description = "Supercell group containing xyz coordinates and supercell vectors.", description_cell = "Supercell vectors as 3x3 matrix (with vectors in rows)."):
    _unit_cell_to_slt(slt_filepath, group_name, elements, positions, cell, multiplicity, group_type, description, description_cell)
    with File(slt_filepath, 'a') as slt:
        group = slt[group_name]
        group.attrs["Supercell_Repetitions"] = [nx, ny, nz]


def _hessian_to_slt(slt_filepath, group_name, elements, positions, cell, nx, ny, nz, multiplicity, hessian, born_charges = None):
    _supercell_to_slt(slt_filepath, group_name, elements, positions, cell, nx, ny, nz, multiplicity, "HESSIAN", "Hessian group containing force constants and supercell parameters.")
    with File(slt_filepath, 'a') as slt:
        group = slt[group_name]
        dataset = group.create_dataset('HESSIAN', data=hessian, dtype=settings.float, chunks=True)
        dataset.attrs["Description"] = "Hessian matrix (2nd order force constants) a.u. / Bohr**2 in the form [nx, ny, nz, dof_number, dof_number] where the last index is for the unit cell at the origin."
        if born_charges is not None:
            dataset = group.create_dataset('BORN_CHARGES', data=born_charges, dtype=settings.float, chunks=True)
            dataset.attrs["Description"] = "Born charges in the form [dof_number, 3] where the last index is for xyz polarization."


def _read_forces_cp2k(filepath, dof_number):
    return loadtxt(filepath, dtype = settings.float, skiprows=4, usecols=(3,4,5), max_rows=dof_number)


def _read_dipole_momenta_cp2k(file_path):
    with open(file_path, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        block_size = 512
        if file_size < block_size:
            f.seek(0)
        else:
            f.seek(-block_size, 2)
        data = f.read()
    data = data.decode('utf-8', errors='ignore')
    lines = data.split('\n')
    lines = lines[::-1]
    dipole_pattern = r'X=\s*([^\s]+)\s+Y=\s*([^\s]+)\s+Z=\s*([^\s]+)'

    for line in lines:
        if 'X=' in line:
            match = search(dipole_pattern, line)
            if match:
                x = settings.float(match.group(1))
                y = settings.float(match.group(2))
                z = settings.float(match.group(3))
                return array([x, y, z], dtype=settings.float)

    raise ValueError('Dipole moment line not found in file')


def _read_hessian_born_charges_from_dir(dirpath, format, dof_number, nx, ny, nz, displacement_number, step, accoustic_sum_rule: Literal["symmetric", "self_term", "without"] = "symmetric", dipole_momenta = True, force_files_suffix = None, dipole_momenta_files_suffix = None):
    atoms_in_file = nx * ny * nz * dof_number // 3
    hessian = zeros(shape=(dof_number, atoms_in_file, 3), order="C", dtype = settings.float)
    finite_difference_stencil = _central_finite_difference_stencil(1, displacement_number, step * A_BOHR)

    if dipole_momenta:
        dipole_momenta_array = zeros(shape=(dof_number, 3), order="C", dtype = settings.float)

    try:
        if format == "CP2K":
            read_forces = _read_forces_cp2k
            read_dipole_momenta = _read_dipole_momenta_cp2k
            default_force_suffix = "-1_0.xyz"
            default_dipole_momenta_suffix = "-moments-1_0.dat"
        else:
            raise ValueError("The only suported format is 'CP2K'.")

        for dof in range(dof_number):
            stencil_index = -1
            for disp in range(-displacement_number, displacement_number + 1, 1):
                stencil_index += 1
                if disp == 0:
                    continue
                file_preffix = join(dirpath, f"dof_{dof}_disp_{disp}")
                force_filepath = file_preffix + (force_files_suffix if force_files_suffix else default_force_suffix)
                hessian[dof, :, :] += read_forces(force_filepath, atoms_in_file) * finite_difference_stencil[stencil_index]
                if dipole_momenta:
                    dipole_momenta_filepath = file_preffix + (dipole_momenta_files_suffix if dipole_momenta_files_suffix else default_dipole_momenta_suffix)
                    dipole_momenta_array[dof, :] += read_dipole_momenta(dipole_momenta_filepath) * finite_difference_stencil[stencil_index]
    except Exception as exc:
        raise SltReadError(file_preffix, exc, f"Failed to load Hessian from directory '{dirpath}'")

    if dipole_momenta:
        dipole_momenta_array = reshape(dipole_momenta_array, (-1, 3, 3), order="C")
    
    if accoustic_sum_rule == "symmetric":
        hessian -= mean(hessian, axis=1, keepdims=True)
        if dipole_momenta:
            dipole_momenta_array -= mean(dipole_momenta_array, axis=0, keepdims=True)
    elif accoustic_sum_rule == "self_term":
        self_term = sum(hessian, axis=1)
        indices = arange(dof_number)
        hessian[indices, indices, :] -= self_term[indices, :]
        if dipole_momenta:
            dipole_self_term = sum(dipole_momenta_array, axis=0)
            dipole_momenta_array[0] -= dipole_self_term

    hessian = reshape(hessian, (dof_number, nx, ny, nz, dof_number))
    hessian = transpose(hessian, (1, 2, 3, 4, 0))
    if dipole_momenta:
        dipole_momenta_array = reshape(dipole_momenta_array, (-1,3), order="C")
        return hessian, dipole_momenta_array
    else:
        return hessian, None


def _load_orca_hdf5(filename, group, rotation):
    try:
        # Read data from HDF5 file
        with File(filename, "r") as file:
            shape = file[str(group)]["SOC"][:].shape[0]
            angular_momenta = zeros((6, shape, shape), dtype=complex128)
            soc_mat = file[str(group)]["SOC"][:]
            angular_momenta[0][:] = 0.5 * file[str(group)]["SF_SX"][:]
            angular_momenta[1][:] = 0.5j * file[str(group)]["SF_SY"][:]
            angular_momenta[2][:] = 0.5 * file[str(group)]["SF_SZ"][:]
            angular_momenta[3][:] = 1j * file[str(group)]["SF_LX"][:]
            angular_momenta[4][:] = 1j * file[str(group)]["SF_LY"][:]
            angular_momenta[5][:] = 1j * file[str(group)]["SF_LZ"][:]

        # Perform diagonalization of SOC matrix
        soc_energies, eigenvectors = eigh(soc_mat)

        angular_momenta = ascontiguousarray(angular_momenta)
        soc_energies = ascontiguousarray(soc_energies.astype(float64))
        eigenvectors = ascontiguousarray(eigenvectors.astype(complex128))

        # Apply transformations to spin and orbital operators
        angular_momenta = (
            eigenvectors.conj().T @ angular_momenta @ eigenvectors
        )

        if (rotation is not None) or (rotation != None):
            angular_momenta[0:3, :, :] = _rotate_vector_operator(
                angular_momenta[0:3, :, :], rotation
            )
            angular_momenta[3:6, :, :] = _rotate_vector_operator(
                angular_momenta[3:6, :, :], rotation
            )

        # Return operators in SOC basis
        return soc_energies, angular_momenta

    except Exception as exc:
        raise SltFileError(
            filename,
            exc,
            message=(
                "Failed to load SOC, spin, and angular momenta data in"
                " the ORCA format from the .slt file."
            ),
        ) from None


def _load_molcas_hdf5(filename, group, rotation, edipmom = False):
    try:
        with File(filename, "r") as file:
            shape = file[group]["SOC_ENERGIES"][:].shape[0]
            angular_momenta = zeros((6, shape, shape), dtype=settings.complex)
            soc_energies = file[group]["SOC_ENERGIES"][:]
            angular_momenta[0][:] = file[group]["SOC_SX"][:]
            angular_momenta[1][:] = file[group]["SOC_SY"][:]
            angular_momenta[2][:] = file[group]["SOC_SZ"][:]
            angular_momenta[3][:] = file[group]["SOC_LX"][:]
            angular_momenta[4][:] = file[group]["SOC_LY"][:]
            angular_momenta[5][:] = file[group]["SOC_LZ"][:]

            angular_momenta = ascontiguousarray(angular_momenta)
            soc_energies = ascontiguousarray(soc_energies.astype(settings.float))

            if (rotation is not None) or (rotation != None):
                angular_momenta[0:3, :, :] = _rotate_vector_operator(
                    angular_momenta[0:3, :, :], rotation
                )
                angular_momenta[3:6, :, :] = _rotate_vector_operator(
                    angular_momenta[3:6, :, :], rotation
                )
            
            if edipmom:
                if file[group].attrs["Additional"] == "Edipmom":
                    shape = file[group]["SOC_EDIPMOMX"][:].shape[0]
                    electric_dipolar_momenta = zeros((3, shape, shape), dtype=settings.complex)
                    electric_dipolar_momenta[0][:] = file[group]["SOC_EDIPMOMX"][:]
                    electric_dipolar_momenta[1][:] = file[group]["SOC_EDIPMOMY"][:]
                    electric_dipolar_momenta[2][:] = file[group]["SOC_EDIPMOMZ"][:]

                    if (rotation is not None) or (rotation != None):
                        electric_dipolar_momenta = _rotate_vector_operator(
                            electric_dipolar_momenta, rotation
                        )

                    electric_dipolar_momenta = ascontiguousarray(electric_dipolar_momenta)

                    return soc_energies, angular_momenta, electric_dipolar_momenta

                else:
                    raise KeyError(f"Group '{group}' does not contain electric dipolar momenta in a valid format.")

            return soc_energies, angular_momenta

    except Exception as exc:
        raise SltFileError(
            filename,
            exc,
            message=(
                "Failed to load SOC, spin, angular momenta or electric dipole momenta data in"
                " the MOLCAS format from the .slt file."
            ),
        ) from None


def _get_soc_energies_and_soc_angular_momenta_from_hdf5(
    filename: str, group: str, rotation: ndarray = None
) -> Tuple[ndarray, ndarray]:
    if _dataset_exists(filename, group, "SOC"):
        return _load_orca_hdf5(filename, group, rotation)

    if _dataset_exists(filename, group, "SOC_ENERGIES"):
        return _load_molcas_hdf5(filename, group, rotation)

    else:
        raise SltReadError(
            filename,
            ValueError(""),
            message=(
                "Incorrect group name or data format. The program was unable"
                " to read SOC, spin, and angular momenta."
            ),
        )


def _get_soc_magnetic_momenta_and_energies_from_hdf5(
    filename: str, group: str, states_cutoff: int, rotation: ndarray = None
) -> Tuple[ndarray, ndarray]:
    # (
    #     soc_energies,
    #     angular_momenta,
    # ) = _get_soc_energies_and_soc_angular_momenta_from_hdf5(
    #     filename, group, rotation
    # )

    # soc_energies = soc_energies[:states_cutoff] - soc_energies[0]
    # magnetic_momenta = _magnetic_momenta_from_angular_momenta(
    #     angular_momenta, 0, states_cutoff
    # )

    # return magnetic_momenta, soc_energies

    pass


def _get_soc_total_angular_momenta_and_energies_from_hdf5(
    filename: str, group: str, states_cutoff: int, rotation=None
) -> Tuple[ndarray, ndarray]:
    # (
    #     soc_energies,
    #     angular_momenta,
    # ) = _get_soc_energies_and_soc_angular_momenta_from_hdf5(
    #     filename, group, rotation
    # )

    # soc_energies = soc_energies[:states_cutoff] - soc_energies[0]
    # total_angular_momenta = _total_angular_momenta_from_angular_momenta(
    #     angular_momenta, 0, states_cutoff
    # )

    # return total_angular_momenta, soc_energies
    pass


def _get_soc_energies_cm_1(
    filename: str, group: str, num_of_states: int = 0
) -> ndarray:
    # if num_of_states < 0 or (not isinstance(num_of_states, int)):
    #     raise ValueError(
    #         "Invalid number of states. Set it to positive integer or 0 for"
    #         " all states."
    #     )
    # with File(filename, "r") as file:
    #     try:
    #         soc_matrix = file[str(group)]["SOC"][:]
    #         soc_energies = eigvalsh(soc_matrix)
    #     except Exception as exc1:
    #         error_message_1 = str(exc1)
    #         try:
    #             soc_energies = file[str(group)]["SOC_energies"][:]
    #         except Exception as exc2:
    #             error_message_2 = str(exc2)
    #             if error_message_1 == error_message_2:
    #                 error_message_2 = ""
    #             raise RuntimeError(
    #                 f"{error_message_1}. {error_message_2}."
    #             ) from None

    #     if num_of_states > soc_energies.shape[0]:
    #         raise ValueError(
    #             "Invalid number of states. Set it less or equal to the overal"
    #             f" number of SOC states: {soc_energies.shape[0]}"
    #         )

    #     if num_of_states != 0:
    #         soc_energies = soc_energies[:num_of_states]

    #     # Return operators in SOC basis
    #     return (soc_energies - soc_energies[0]) * H_CM_1
    pass


def _get_states_magnetic_momenta(
    filename: str, group: str, states: ndarray = None, rotation: ndarray = None
):
    # if any(states < 0):
    #     raise ValueError("States list contains negative values.")

    # if isinstance(states, int):
    #     states_cutoff = states
    #     (
    #         magnetic_momenta,
    #         _,
    #     ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
    #         filename, group, states_cutoff, rotation
    #     )
    #     magnetic_momenta = diagonal(magnetic_momenta, axis1=1, axis2=2)
    # else:
    #     states = array(states, dtype=int64)
    #     if states.ndim == 1:
    #         states_cutoff = max(states)
    #         (
    #             magnetic_momenta,
    #             _,
    #         ) = _get_soc_magnetic_momenta_and_energies_from_hdf5(
    #             filename, group, states_cutoff, rotation
    #         )
    #         magnetic_momenta = magnetic_momenta[:, states, states]
    #     else:
    #         raise ValueError("The list of states has to be a 1D array.")

    # return magnetic_momenta.real
    pass


def _get_states_total_angular_momenta(#######################
    filename: str, group: str, states: ndarray = None, rotation: ndarray = None
):
    if any(states < 0):
        raise ValueError("States list contains negative values.")

    if isinstance(states, int):
        states_cutoff = states
        (
            total_angular_momenta,
            _,
        ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
            filename, group, states_cutoff, rotation
        )
        total_angular_momenta = diagonal(
            total_angular_momenta, axis1=1, axis2=2
        )
    else:
        states = array(states, dtype=int64)
        if states.ndim == 1:
            states_cutoff = max(states)
            (
                total_angular_momenta,
                _,
            ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
                filename, group, states_cutoff, rotation
            )
            total_angular_momenta = total_angular_momenta[:, states, states]
        else:
            raise ValueError("The list of states has to be a 1D array.")

    return total_angular_momenta.real


def _get_magnetic_momenta_matrix(#######################
    filename: str, group: str, states_cutoff: int, rotation: ndarray = None
):
    magnetic_momenta, _ = _get_soc_magnetic_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    return magnetic_momenta


def _get_total_angular_momneta_matrix(##############
    filename: str, group: str, states_cutoff: int, rotation: ndarray = None
):
    (
        total_angular_momenta,
        _,
    ) = _get_soc_total_angular_momenta_and_energies_from_hdf5(
        filename, group, states_cutoff, rotation
    )

    return total_angular_momenta


def _get_dataset_slt_dtype(file_path, dataset_path):
    with File(file_path, 'r') as file:
            dataset = file[dataset_path]
            _dtype = dataset.dtype
            match str(_dtype)[0]:
                case "c":
                    return dtype(settings.complex)
                case "f":
                    return dtype(settings.float)
                case "i":
                    return dtype(settings.int)
                case _:
                    return _dtype
                

def _save_data_to_slt(file_path, group_name, data_dict, metadata_dict):
    with File(file_path, 'a') as file:
        group = file.create_group(group_name)
        for key, value in data_dict.items():
            dataset = group.create_dataset(key, shape=value[0].shape, dtype=value[0].dtype, chunks=True)
            dataset[:] = value[0]
            dataset.attrs['Description'] = value[1]
        for key, value in metadata_dict.items():
            group.attrs[key] = value


def _get_orca_blocks_size_new(orca_filepath: str) -> tuple[int, int, int]:
    
    with open(orca_filepath, "r") as file:
        regex = compile("SOC MATRIX \(A\.U\.\)\n")
        so_dim = -1
        for line in file:
            if regex.search(line):
                for _ in range(4):
                    file.readline()
                for line in file:
                    elements = findall(r'[-+]?\d*\.\d+|[+-]?\d+', line)
                    so_dim += 1
                    if len(elements) == 6:
                        break
                break

    num_blocks = so_dim // 6
    num_of_whole_blocks = num_blocks
    remaining_columns = so_dim % 6

    if remaining_columns != 0:
        num_blocks += 1 

    return so_dim, num_of_whole_blocks, remaining_columns


def _orca_reader(so_dim: int, num_of_whole_blocks: int, remaining_columns: int, file) -> ndarray:
    matrix = empty((so_dim, so_dim), dtype=settings.float)
    l = 0
    for _ in range(num_of_whole_blocks):
        file.readline()  # Skip a line before each block of 6 columns
        for i in range(so_dim):
            line = file.readline()
            elements = findall(r'[-+]?\d*\.\d+', line)
            for j in range(6):       
                matrix[i, l + j] = settings.float((elements[j]))
        l += 6
        
    if remaining_columns > 0:
        file.readline()  # Skip a line before the remaining columns
        for i in range(so_dim):
            line = file.readline()
            elements = findall(r'[-+]?\d*\.\d+', line)
            for j in range(remaining_columns):
                matrix[i, l + j] = float64(elements[j])

    return matrix
    
def _orca_to_slt(orca_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool, ssc: bool, pt2: bool) -> None:
    
    # Retrieve dimensions and block sizes for spin-orbit calculations
    (so_dim, num_of_whole_blocks, remaining_columns) = _get_orca_blocks_size_new(orca_filepath)
    
    # Create HDF5 file and ORCA group
    with File(f"{slt_filepath}", "a") as slt:
        group = slt.create_group(group_name)
        group.attrs["Type"] = "HAMILTONIAN"
        group.attrs["Kind"] = "ORCA"
        if ssc:
            group.attrs["Energies"] = "SSC_ENERGIES"
        else:
            group.attrs["Energies"] = "SOC_ENERGIES" 
        if electric_dipole_momenta:
            group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
        group.attrs["Description"] = "Relativistic ORCA results."

        # Extract and process matrices
        pattern_type = [["SOC and SSC MATRIX\s*\(A\.U\.\)\s*"]] if ssc else [["SOC MATRIX\s*\(A\.U\.\)\s*"]]
        pattern_type += [["SX MATRIX IN CI BASIS\n", "SY MATRIX IN CI BASIS\n", "SZ MATRIX IN CI BASIS\n"], ["LX MATRIX IN CI BASIS\n", "LY MATRIX IN CI BASIS\n", "LZ MATRIX IN CI BASIS\n"]]
        matrix_type = ["STATES_ENERGIES", "SPINS", "ANGULAR_MOMENTA"]
        descriptions = ["States energies.", "Sx, Sy, and Sz spin matrices in the SOC basis [(x-0, y-1, z-2), :, :].", "Lx, Ly, and Lz angular momentum matrices in the SOC basis [(x-0, y-1, z-2), :, :]."]

        if electric_dipole_momenta:
            pattern_type.append(["Matrix EDX in CI Basis\n", "Matrix EDY in CI Basis\n", "Matrix EDZ in CI Basis\n"])
            matrix_type.append("ELECTRIC_DIPOLE_MOMENTA")
            descriptions.append("Px, Py, and Pz electric dipole momentum")
            
        final_number = (3 if ssc else 1) + (1 if pt2 else 0)
        energy_final_number = 2 if pt2 else 1

        for matrix, patterns, description in zip(matrix_type, pattern_type, descriptions):
            if matrix == "STATES_ENERGIES":
                dataset = group.create_dataset(f"{matrix}", shape = so_dim, dtype = settings.float, chunks = True)
            else:
                dataset = group.create_dataset(f"{matrix}", shape = (3, so_dim, so_dim), dtype = settings.complex, chunks = True)
            
            dataset.attrs["Description"] = description

            for index, pattern in enumerate(patterns):
                regex = compile(pattern)
          
                with open(f"{orca_filepath}", "r") as file:
                    matrix_number = 0
                    for line in file:
                        if regex.search(line):
                            matrix_number += 1
                            if matrix == "STATES_ENERGIES":
                                final_number =  energy_final_number  
                            if matrix_number == final_number:
                                if matrix != "ELECTRIC_DIPOLE_MOMENTA":
                                    for _ in range(3):
                                        file.readline()  # Skip the first 3 lines if not electric dipole momenta
                                data = _orca_reader(so_dim, num_of_whole_blocks, remaining_columns, file)
                                if matrix == "STATES_ENERGIES":
                                    for _ in range(2):
                                        file.readline()  # Skip 2 lines separating real and imaginary part
                                    data_imag = _orca_reader(so_dim, num_of_whole_blocks, remaining_columns, file)
                                    energy_matrix = array(data + 1j * data_imag, dtype=settings.complex)
                                    energies, eigenvectors = eigh(energy_matrix)
                                    dataset[:] = energies - min(energies)  # Assign energies to the dataset
                                else:
                                    transformed_data = (eigenvectors.conj().T @ data @ eigenvectors)
                                    dataset[index] = transformed_data[:]# Assign transformed matrix

                                break

            
