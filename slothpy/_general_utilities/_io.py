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
from os.path import join
from re import compile, search, findall

from h5py import File, string_dtype
from numpy import ndarray, dtype, bytes_, asarray, zeros, empty, loadtxt, sum, reshape, mean, arange, transpose, min
from scipy.linalg import eigh

from slothpy.core._slothpy_exceptions import SltReadError
from slothpy.core._config import settings
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil
from slothpy._general_utilities._constants import A_BOHR

#####################
# Helpers for SltFile
#####################


def _group_exists(hdf5_file, group_name: str):
    with File(hdf5_file, "r") as file:
        return group_name in file


def _dataset_exists(hdf5_file, group_name, dataset_name):
    with File(hdf5_file, "r") as file:
        if group_name in file:
            group = file[group_name]
            return dataset_name in group

        return False


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
            if isinstance(value[0], list) and isinstance(value[0], str):
                data = asarray(value[0], dtype='S')
            else:
                data = value[0]
            dataset = group.create_dataset(key, data=data)
            dataset.attrs['Description'] = value[1]
        for key, value in metadata_dict.items():
            group.attrs[key] = value


################################
# Helpers for creation functions
################################


def _xyz_to_slt(slt_filepath, group_name, elements, positions, charge, multiplicity, group_type = "XYZ", description = "XYZ file."):
    with File(slt_filepath, 'a') as slt:
        group = slt.create_group(group_name)
        group.attrs["Type"] = group_type
        group.attrs["Number_of_atoms"] = len(elements)
        group.attrs["Precision"] = settings.precision.upper()
        group.attrs["Description"] = description
        dt = string_dtype(encoding='utf-8')
        dataset = group.create_dataset('ELEMENTS', data=asarray(elements, dtype='S'), dtype=dt, chunks=True)
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


def _orca_to_slt(orca_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool, ssc: bool, pt2: bool) -> None:
    
    # Retrieve dimensions and block sizes for spin-orbit calculations
    (so_dim, num_of_whole_blocks, remaining_columns) = _get_orca_blocks_size(orca_filepath)
    
    # Create HDF5 file and ORCA group
    with File(f"{slt_filepath}", "a") as slt:
        group = slt.create_group(group_name)
        group.attrs["Type"] = "HAMILTONIAN"
        group.attrs["Kind"] = "ORCA"
        group.attrs["Precision"] = settings.precision.upper()
        group.attrs["States"] = so_dim
        if ssc:
            group.attrs["Energies_type"] = "SSC"
        else:
            group.attrs["Energies_type"] = "SOC" 
        if electric_dipole_momenta:
            group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
        group.attrs["Description"] = "Relativistic ORCA results."

        # Extract and process matrices
        pattern_type = [[r"SOC and SSC MATRIX \(A\.U\.\)\n"]] if ssc else [[r"SOC MATRIX \(A\.U\.\)\n"]]
        pattern_type += [[r"SX MATRIX IN CI BASIS\n", r"SY MATRIX IN CI BASIS\n", r"SZ MATRIX IN CI BASIS\n"], [r"LX MATRIX IN CI BASIS\n", r"LY MATRIX IN CI BASIS\n", r"LZ MATRIX IN CI BASIS\n"]]
        matrix_type = ["STATES_ENERGIES", "SPINS", "ANGULAR_MOMENTA"]
        descriptions = ["States energies.", "Sx, Sy, and Sz spin matrices [(x-0, y-1, z-2), :, :].", "Lx, Ly, and Lz angular momentum matrices [(x-0, y-1, z-2), :, :]."]

        if electric_dipole_momenta:
            pattern_type.append([r"Matrix EDX in CI Basis\n", r"Matrix EDY in CI Basis\n", r"Matrix EDZ in CI Basis\n"])
            matrix_type.append("ELECTRIC_DIPOLE_MOMENTA")
            descriptions.append("Px, Py, and Pz electric dipole momentum [(x-0, y-1, z-2), :, :].")
            
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
                                        file.readline() # Skip the first 3 lines if not electric dipole momenta
                                data = _orca_matrix_reader(so_dim, num_of_whole_blocks, remaining_columns, file)
                                if matrix == "STATES_ENERGIES":
                                    for _ in range(2):
                                        file.readline() # Skip 2 lines separating real and imaginary part
                                    data_imag = _orca_matrix_reader(so_dim, num_of_whole_blocks, remaining_columns, file)
                                    energy_matrix = asarray(data + 1j * data_imag, dtype=settings.complex, order="C")
                                    energies, eigenvectors = eigh(energy_matrix, driver="evr", check_finite=False, overwrite_a=True, overwrite_b=True)
                                    dataset[:] = energies - min(energies)  # Assign energies to the dataset
                                else:
                                    transformed_data = (eigenvectors.conj().T @ data @ eigenvectors)
                                    dataset[index] = transformed_data[:] # Assign transformed matrix
                                break


def _molcas_to_slt(molcas_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool = False) -> None:
    if not molcas_filepath.endswith(".rassi.h5"):
        slt_filepath += ".rassi.h5"

    with File(f"{molcas_filepath}", "r") as rassi:
        with File(f"{slt_filepath}", "a") as slt:
            group = slt.create_group(group_name)
            group.attrs["Type"] = "HAMILTONIAN"
            group.attrs["Kind"] = "MOLCAS"
            group.attrs["Precision"] = settings.precision.upper()
            group.attrs["Energies_type"] = "SOC"
            if electric_dipole_momenta:
                group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
            group.attrs["Description"] = "Relativistic MOLCAS results."

            dataset_rassi = rassi["SOS_ENERGIES"][:] - min(rassi["SOS_ENERGIES"][:])
            group.attrs["States"] = dataset_rassi.shape[0]
            dataset_out = group.create_dataset("STATES_ENERGIES", shape=dataset_rassi.shape, dtype=settings.float, data=dataset_rassi.astype(settings.float), chunks=True)
            dataset_out.attrs["Description"] = "SOC energies."

            dataset_rassi = rassi["SOS_SPIN_REAL"][:, :, :] + 1j * rassi["SOS_SPIN_IMAG"][:, :, :]
            dataset_out = group.create_dataset("SPINS", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Sx, Sy, and Sz spin matrices [(x-0, y-1, z-2), :, :]."

            dataset_rassi = 1j * rassi["SOS_ANGMOM_REAL"][:, :, :] - rassi["SOS_ANGMOM_IMAG"][:, :, :]
            dataset_out = group.create_dataset("ANGULAR_MOMENTA", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Lx, Ly, and, Lz angular momentum matrices [(x-0, y-1, z-2), :, :]."

            if electric_dipole_momenta:
                dataset_rassi = rassi["SOS_EDIPMOM_REAL"][:, :, :] + 1j * rassi["SOS_EDIPMOM_REAL"][:, :, :]
                dataset_out = group.create_dataset("ELECTRIC_DIPOLE_MOMENTA", shape=dataset_rassi.shape, dtype=settings.complex, data=dataset_rassi.astype(settings.complex), chunks=True)
                dataset_out.attrs["Description"] = "Px, Py, and Pz electric dipole momentum matrices [(x-0, y-1, z-2), :, :]."


#########################################
# Helpers for internal creation functions
#########################################


def _hessian_to_slt(slt_filepath, group_name, elements, positions, cell, nx, ny, nz, multiplicity, hessian, born_charges = None):
    _supercell_to_slt(slt_filepath, group_name, elements, positions, cell, nx, ny, nz, multiplicity, "HESSIAN", "Hessian group containing force constants and supercell parameters.")
    with File(slt_filepath, 'a') as slt:
        group = slt[group_name]
        group.attrs["Modes"] = hessian.shape[3]
        dataset = group.create_dataset('HESSIAN', data=hessian, dtype=settings.float, chunks=True)
        dataset.attrs["Description"] = "Hessian matrix (2nd order force constants) a.u. / Bohr**2 in the form [nx, ny, nz, dof_number, dof_number] where the last index is for the unit cell at the origin."
        if born_charges is not None:
            dataset = group.create_dataset('BORN_CHARGES', data=born_charges, dtype=settings.float, chunks=True)
            dataset.attrs["Description"] = "Born charges in the form [dof_number, 3] where the last index is for xyz polarization."


def _exchange_hamiltonian_to_slt(slt_filepath: str, group_name: str, states: int, magnetic_centers: dict, exchange_interactions: dict, contains_electric_dipole_momenta: bool, electric_dipole_interactions: bool, magnetic_dipole_interactions: bool):
    with File(slt_filepath, 'a') as file:
        group = file.create_group(group_name)
        group.attrs["Type"] = "EXCHANGE_HAMILTONIAN"
        group.attrs["Kind"] = "SLOTHPY"
        group.attrs["States"] = states
        group.attrs["Description"] = f"A custom exchange Hamiltonian created by the user."
        if contains_electric_dipole_momenta:
            group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
        interactions = "mp" if electric_dipole_interactions and magnetic_dipole_interactions else "m" if magnetic_dipole_interactions else "p" if electric_dipole_interactions else None
        if interactions is not None:
            group.attrs["Interactions"] = interactions
        
        save_dict_to_group(group, magnetic_centers, "MAGNETIC_CENTERS")
        save_dict_to_group(group, exchange_interactions, "EXCHANGE_INTERACTIONS")


def _create_dataset(group, name, data):
    if data is None:
        data=bytes_('None')
    elif isinstance(data, (ndarray, list)):
        data = asarray(data)
    elif isinstance(data, (int, float)):
        data = asarray(data)
    elif isinstance(data, str):
        data = bytes_(data)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    group.create_dataset(name, data=data)


def save_dict_to_group(group, data_dict, subgroup_name):
    subgroup = group.create_group(subgroup_name)
    for key, value in data_dict.items():
        key_str = str(key) if not isinstance(key, str) else f"'{key}'"
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                item_name = f"{key_str}_{i}"
                _create_dataset(subgroup, item_name, item)
        else:
            _create_dataset(subgroup, key_str, value)


##############################
# Helpers for import functions
##############################


def _get_orca_blocks_size(orca_filepath: str) -> tuple[int, int, int]:
    
    with open(orca_filepath, "r") as file:
        regex = compile(r"SOC MATRIX \(A\.U\.\)\n")
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


def _orca_matrix_reader(so_dim: int, num_of_whole_blocks: int, remaining_columns: int, file) -> ndarray:
    matrix = empty((so_dim, so_dim), dtype=settings.float, order="C")
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
                matrix[i, l + j] = settings.float(elements[j])

    return matrix


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
                return asarray([x, y, z], dtype=settings.float)

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
            
