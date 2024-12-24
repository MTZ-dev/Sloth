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

from typing import Union, Literal, Iterator
from os import remove
from os.path import join, exists
from glob import glob
from re import compile, search, findall, MULTILINE, IGNORECASE
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from subprocess import Popen, PIPE

from h5py import File, string_dtype
from numpy import ndarray, dtype, bytes_, asarray, zeros, eye, empty, loadtxt, sum, reshape, mean, arange, transpose, fromstring, asfortranarray, min, int64, float64
from scipy.linalg import eigh, kron, block_diag

from slothpy.core._slothpy_exceptions import SltReadError
from slothpy.core._config import settings
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil, _calculate_wavefunction_overlap_phase_correction
from slothpy._general_utilities._constants import A_BOHR
from slothpy.core._process_pool import _worker_wrapper

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


def _orca_to_slt(orca_source: Union[str, Iterator], slt_filepath: str, group_name: str, pt2: bool, electric_dipole_momenta: bool, ssc: bool, ci_basis: bool) -> None:
    should_close = False
    if isinstance(orca_source, str):
        file = open(orca_source, "r")
        should_close = True
    else:
        file = orca_source

    try:
        dtype = settings.complex
        with File(f"{slt_filepath}", "a") as slt:
            group = slt.create_group(group_name)
            group.attrs["Type"] = "HAMILTONIAN"
            group.attrs["Kind"] = "ORCA"
            group.attrs["Precision"] = settings.precision.upper()
            if ssc:
                group.attrs["Hamiltonian_type"] = "SOC_SSC"
            else:
                group.attrs["Hamiltonian_type"] = "SOC"
            if ci_basis:
                group.attrs["Basis"] = "CI"
            else:
                group.attrs["Basis"] = "DIAGONAL"
            if electric_dipole_momenta:
                group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
            group.attrs["Description"] = "Relativistic ORCA results."

            pattern_type = [[r"SOC and SSC MATRIX \(A\.U\.\)\n"]] if ssc else [[r"SOC MATRIX \(A\.U\.\)\n"]]
            pattern_type += [[r"SX MATRIX IN CI BASIS\n", r"SY MATRIX IN CI BASIS\n", r"SZ MATRIX IN CI BASIS\n"], [r"LX MATRIX IN CI BASIS\n", r"LY MATRIX IN CI BASIS\n", r"LZ MATRIX IN CI BASIS\n"]]
                
            if ci_basis:
                matrix_types = ["SOC_SSC_MATRIX"] if ssc else ["SOC_MATRIX"]
                descriptions = [f"SOC {"and SSC " if ssc else ""}matrix in CI basis."]
                descriptions += ["Sx, Sy, and Sz spin matrices [(x-0, y-1, z-2), :, :] in CI basis.", "Lx, Ly, and Lz angular momentum matrices [(x-0, y-1, z-2), :, :] in CI basis."]
            else:
                matrix_types = ["STATES_ENERGIES"]
                descriptions = ["States energies.", "Sx, Sy, and Sz spin matrices [(x-0, y-1, z-2), :, :].", "Lx, Ly, and Lz angular momentum matrices [(x-0, y-1, z-2), :, :]."]
            
            matrix_types += ["SPINS", "ANGULAR_MOMENTA"]
            
            if electric_dipole_momenta:
                pattern_type.append([r"Matrix EDX in CI Basis\n", r"Matrix EDY in CI Basis\n", r"Matrix EDZ in CI Basis\n"])
                matrix_types.append("ELECTRIC_DIPOLE_MOMENTA")
                descriptions.append(f"Px, Py, and Pz electric dipole momentum [(x-0, y-1, z-2), :, :]{" in CI basis" if ci_basis else ""}.")

            energy_number = 2 if pt2 else 1

            multiplicities, nroots, active_orbitals, total_orbitals, inactive_orbitals  = _get_orca_dimension_info(file)
            dim = sum(multiplicities * nroots)
            group.attrs["States"] = dim
            number_of_whole_blocks = dim // 6
            remaining_columns = dim % 6
            energy_block_flag = True
            matrix_number = 0

            if ci_basis:
                determinant_info = _parse_orca_spin_determinant_ci(file, multiplicities, nroots)
                group.attrs["Inactive_orbitals"] = inactive_orbitals
                group.attrs["Active_orbitals"] = active_orbitals
                group.attrs["Total_orbitals"] = total_orbitals
                group.attrs["Multiplicities"] = asarray(list(determinant_info.keys()), dtype=int64, order='C')
                for mult, info in determinant_info.items():
                    mult_group = group.create_group(f"MULTIPLICITY_{mult}")
                    mult_group.attrs["Description"] = "Determinants type and root composition."
                    mult_group.create_dataset("ALPHA_ORBITALS", data=info[0], chunks=True)
                    mult_group.create_dataset("BETA_ORBITALS", data=info[1], chunks=True)
                    mult_group.create_dataset("ROOTS_CI_COEFFICIENTS", data=info[2], chunks=True)

            for matrix_type, patterns, description in zip(matrix_types, pattern_type, descriptions):
                if not energy_block_flag:
                    data = empty((3, dim, dim), dtype=settings.complex, order='C')
                for index, pattern in enumerate(patterns):
                    regex = compile(pattern)
                    while True:
                        line = next(file)
                        if regex.search(line):
                            if energy_block_flag:
                                matrix_number += 1
                            if matrix_number == energy_number:
                                if energy_block_flag:
                                    for _ in range(3):
                                        next(file) # Skip 3 the first 3 lines if not electric dipole momenta
                                    data_real = _orca_matrix_reader(dim, number_of_whole_blocks, remaining_columns, file, dtype, True)
                                    for _ in range(2):
                                        next(file) # Skip 2 lines separating real and imaginary part
                                    data_imag = _orca_matrix_reader(dim, number_of_whole_blocks, remaining_columns, file, dtype, True)
                                    data = data_real + 1j * data_imag
                                    if not ci_basis:
                                        energies, eigenvectors = eigh(data, driver="evr", check_finite=False, overwrite_a=True, overwrite_b=True)
                                        data = energies - min(energies)  # Assign energies to the dataset
                                    energy_block_flag = False
                                    break
                                if matrix_type != "ELECTRIC_DIPOLE_MOMENTA":
                                    for _ in range(3):
                                        next(file) # Skip the first 3 lines if not electric dipole momenta
                                matrix = _orca_matrix_reader(dim, number_of_whole_blocks, remaining_columns, file, dtype)
                                if pattern not in [r"SX MATRIX IN CI BASIS\n", r"SZ MATRIX IN CI BASIS\n"]:
                                    matrix = 1j*matrix
                                if pattern in [r"SX MATRIX IN CI BASIS\n", r"SY MATRIX IN CI BASIS\n", r"SZ MATRIX IN CI BASIS\n"]:
                                    matrix = matrix*(0.5 + 0j)
                                if ci_basis:
                                    data[index] = matrix
                                else:
                                    transformed_data = (eigenvectors.conj().T @ matrix @ eigenvectors)
                                    data[index] = transformed_data # Assign transformed matrix
                                break

                dataset = group.create_dataset(f"{matrix_type}", data=data, chunks = True)
                dataset.attrs["Description"] = description

    finally:
        if should_close:
            file.close()


def _molcas_to_slt(molcas_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool = False) -> None:
    if not molcas_filepath.endswith(".rassi.h5"):
        slt_filepath += ".rassi.h5"

    with File(f"{molcas_filepath}", "r") as rassi:
        with File(f"{slt_filepath}", "a") as slt:
            group = slt.create_group(group_name)
            group.attrs["Type"] = "HAMILTONIAN"
            group.attrs["Kind"] = "MOLCAS"
            group.attrs["Precision"] = settings.precision.upper()
            group.attrs["Hamiltonian_type"] = "SOC"
            if electric_dipole_momenta:
                group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
            group.attrs["Description"] = "Relativistic MOLCAS results."

            dataset_rassi = rassi["SOS_ENERGIES"][:] - min(rassi["SOS_ENERGIES"][:])
            group.attrs["States"] = dataset_rassi.shape[0]
            dataset_out = group.create_dataset("STATES_ENERGIES", data=dataset_rassi.astype(settings.float), chunks=True)
            dataset_out.attrs["Description"] = "SOC energies."

            dataset_rassi = rassi["SOS_SPIN_REAL"][:, :, :] + 1j * rassi["SOS_SPIN_IMAG"][:, :, :]
            dataset_out = group.create_dataset("SPINS", data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Sx, Sy, and Sz spin matrices [(x-0, y-1, z-2), :, :]."

            dataset_rassi = 1j * rassi["SOS_ANGMOM_REAL"][:, :, :] - rassi["SOS_ANGMOM_IMAG"][:, :, :]
            dataset_out = group.create_dataset("ANGULAR_MOMENTA", data=dataset_rassi.astype(settings.complex), chunks=True)
            dataset_out.attrs["Description"] = "Lx, Ly, and, Lz angular momentum matrices [(x-0, y-1, z-2), :, :]."

            if electric_dipole_momenta:
                dataset_rassi = rassi["SOS_EDIPMOM_REAL"][:, :, :] + 1j * rassi["SOS_EDIPMOM_REAL"][:, :, :]
                dataset_out = group.create_dataset("ELECTRIC_DIPOLE_MOMENTA", data=dataset_rassi.astype(settings.complex), chunks=True)
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


def _hamiltonian_derivatives_from_dir_to_slt(dirpath: str, slt_filepath: str, group_name: str, displacement_number: int, step: float, number_processes: int, number_threads: int, format: Literal["ORCA"] = "ORCA", pt2: bool = False, electric_dipole_momenta: bool = False, ssc: bool = False, _orca_fragovl_path: str = "."):
    try:
        with File(f"{slt_filepath}", "a") as slt:
            group = slt.create_group(group_name)
            group.attrs["Type"] = "HAMILTONIAN_DERIVATIVES"
            group.attrs["Kind"] = format
            group.attrs["Precision"] = settings.precision.upper()
            group.attrs["Displacement_number"] = displacement_number
            if ssc:
                group.attrs["Hamiltonian_type"] = "SOC_SSC"
            else:
                group.attrs["Hamiltonian_type"] = "SOC"
            if electric_dipole_momenta:
                group.attrs["Additional"] = "ELECTRIC_DIPOLE_MOMENTA"
            group.attrs["Description"] = "Relativistic ORCA hamiltonian derivatives."

        if format == "ORCA":
            read_hamiltonian = _orca_to_slt
            read_overlaps_hamiltonians = _orca_process_overlap_mch_basis_hamiltonian
        else:
            raise ValueError("Currently the only suported format is 'ORCA'.")

        energy_matrix_type = "SOC_SSC_MATRIX" if ssc else "SOC_MATRIX"

        dof_disp_out_files = glob(join(dirpath, "dof_*.out"))
        pattern_simple = compile(join(dirpath, r'dof_(-?\d+)_disp_(-?\d+)\.out'))
        pattern_extended = compile(join(dirpath, r'dof_(-?\d+)_nx_(-?\d+)_ny_(-?\d+)_nz_(-?\d+)_disp_(-?\d+)\.out'))

        displacement_data = defaultdict(set)

        for filename in dof_disp_out_files:
            match = pattern_simple.match(filename)
            if match:
                dof_number = int(match.group(1))
                disp_number = int(match.group(2))
                nx = ny = nz = 0
            else:
                match = pattern_extended.match(filename)
                if match:
                    dof_number = int(match.group(1))
                    nx = int(match.group(2))
                    ny = int(match.group(3))
                    nz = int(match.group(4))
                    disp_number = int(match.group(5))
                else:
                    continue

            displacement_data[(dof_number, nx, ny, nz)].add(disp_number)

        required_disps_0 = set(range(-displacement_number, displacement_number+1))
        required_disps = required_disps_0 - {0}
        found_dof_0 = False

        for (dof_number, nx, ny, nz), disps in displacement_data.items():
            if dof_number == 0 and nx == 0 and ny == 0 and nz == 0:
                if disps != required_disps_0:
                    raise ValueError(f"Output files for dof=0 have incorect set of displacements. They should be in the set: {required_disps_0}.")
                else:
                    found_dof_0 = True
            else:
                if disps != required_disps:
                    missing = required_disps - disps
                    extra = disps - required_disps
                    message_parts = []
                    if missing:
                        message_parts.append(f"missing: {sorted(missing)}")
                    if extra:
                        message_parts.append(f"extra: {sorted(extra)}")

                    error_message = (f"Incorrect displacement set for dof={dof_number}, nx={nx}, ny={ny}, nz={nz}. Discrepancies: {', '.join(message_parts)}.")
                    raise ValueError(error_message)
            
            gbw_exists = False

            for disp in disps:
                if nx == 0 and ny == 0 and nz == 0:
                    gbw_exists = exists(join(dirpath, f"dof_{dof_number}_disp_{disp}.gbw"))
                if not gbw_exists:
                    gbw_exists = exists(join(dirpath, f"dof_{dof_number}_nx_{nx}_ny_{ny}_nz_{nz}_disp_{disp}.gbw"))
                if not gbw_exists:
                    raise ValueError(f"GBW file for dof={dof_number}, disp={disp}, nx={nx}, ny={ny}, nz={nz} not found in the directory.")
        
        if not found_dof_0:
            raise ValueError("Output files for dof=0 not found in the directory.")
        
        sorted_displacement_data = {key: displacement_data[key] for key in sorted(displacement_data)}
        
        try:
            tmp_files = []
            with ProcessPoolExecutor(number_processes-1) as executor:
                futures = []
                for (dof_number, nx, ny, nz), disps in sorted_displacement_data.items():
                    for disp in disps:
                        out_exists = False
                        if nx == 0 and ny == 0 and nz == 0:
                            out_file = f"dof_{dof_number}_disp_{disp}.out"
                            out_filepath = join(dirpath, out_file)
                            out_exists = exists(out_filepath)
                        if not out_exists:
                            out_file = f"dof_{dof_number}_nx_{nx}_ny_{ny}_nz_{nz}_disp_{disp}.out"
                            out_filepath = join(dirpath, out_file)
                        if disp == 0:
                            tmp_0 = join(dirpath, "0.tmp")
                            tmp_files.append(tmp_0)
                            gbw_0 = out_filepath[:-3] + "gbw"
                            read_hamiltonian(out_filepath, tmp_0, "tmp", pt2, electric_dipole_momenta, ssc, True)
                            with File(tmp_0, 'r') as file:
                                tmp_group = file["tmp"]
                                dim = tmp_group.attrs["Total_orbitals"]
                                multiplicities_zero = tmp_group.attrs["Multiplicities"]
                                roots = []
                                spin_roots = 0
                                for mult in multiplicities_zero:
                                    root_number = tmp_group[f"MULTIPLICITY_{mult}"]["ROOTS_CI_COEFFICIENTS"].shape[1]
                                    roots.append(root_number)
                                    spin_roots += mult * root_number
                        else:
                            gbw_tmp = out_filepath[:-3] + "gbw"
                            tmp_files.append(join(dirpath, f"{dof_number}_{nx}_{ny}_{nz}_{disp}.tmp"))
                            args = [dirpath, out_filepath, pt2, electric_dipole_momenta, ssc, gbw_0, gbw_tmp, dim, tmp_0, dof_number, nx, ny, nz, disp, _orca_fragovl_path]
                            futures.append(executor.submit(_worker_wrapper, read_overlaps_hamiltonians, args, number_threads))

                displacements_list = list(sorted_displacement_data.keys())
                number_of_displacements = len(displacements_list)
                mch_overlap_matrix_derivative_list = [zeros((number_of_displacements, root, root), dtype=settings.float, order='C') for root in roots]
                difference_stencil = _central_finite_difference_stencil(1, displacement_number, step)

                from slothpy.core._slt_file import SltGroup, SltHamiltonian

                with File(f"{slt_filepath}", "a") as slt:
                    group = slt[group_name]
                    for completed_dof_disp in as_completed(futures):
                        completed_file, multiplicities, dof_number, nx, ny, nz, displacement = completed_dof_disp.result()
                        displacement_shift = displacement + displacement_number
                        stencil_index = displacement_shift if displacement < 0 else displacement_shift - 1
                        slt_group_hamiltonian = SltGroup(completed_file, "tmp")
                        with File(completed_file, 'r') as tmp_file:
                            tmp_group = tmp_file["tmp"]
                            for index, mult in enumerate(multiplicities):
                                mch_overlap_matrix_derivative_list[index][displacements_list.index((dof_number, nx, ny, nz))] += tmp_group[str(mult)]["OVERLAP"][:] * difference_stencil[stencil_index]
                            slt_group = group.create_group(f"{dof_number}_{nx}_{ny}_{nz}_{displacement}")
                            slt_group.create_dataset("HAMILTONIAN_MATRIX", data=tmp_group[energy_matrix_type][:], chunks=True)
                            slt_group.create_dataset("MAGNETIC_DIPOLE_MOMENTA", data=slt_group_hamiltonian.magnetic_dipole_momentum_matrices().eval(), chunks=True)
                            #####################################################################
                            ################# PHASES CORRECTION HERE ############################
                            #####################################################################

                            # remove files instantly, time it or at leas tqdm and tqdm to slothpy requirements
                    mch_overlap_matrix_derivative = zeros((number_of_displacements, spin_roots, spin_roots), dtype=settings.float, order='C')

                    for d in range(number_of_displacements):
                        mch_overlap_matrix_derivative[d] += block_diag(*[kron(eye(mult), overlap[d]) for mult, overlap in zip(multiplicities, mch_overlap_matrix_derivative_list)])
                    
                    for index, displacement_info in enumerate(displacements_list):
                        slt_group = group.create_dataset(f"{displacement_info[0]}_{displacement_info[1]}_{displacement_info[2]}_{displacement_info[3]}", data=mch_overlap_matrix_derivative[index], chunks=True)

        finally:
            for tmp_file in tmp_files:
                if exists(tmp_file):
                    try:
                        remove(tmp_file)
                    except Exception as e:
                        print(f"Error removing {file} file. You must do it manually.")
                        raise

    except Exception as exc:
        raise SltReadError(slt_filepath, exc, f"Failed to load Hamiltonian derivatives from directory '{dirpath}'")


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


def _get_orca_dimension_info(file: Iterator) -> tuple[int, int, int, int, int]:
    input_file_start_re = compile(r'^\s*INPUT FILE')
    input_file_end_re = compile(r'^\s*\*{4}END OF INPUT\*{4}')
    casscf_start_re = compile(r'^\s*\|\s*\d+>\s*%casscf', IGNORECASE)
    casscf_end_re = compile(r'^\s*\|\s*\d+>\s*end', IGNORECASE)
    mult_re = compile(r'^\s*\|\s*\d+>\s*mult\s+(.*)', IGNORECASE)
    nroots_re = compile(r'^\s*\|\s*\d+>\s*nroots\s+(.*)', IGNORECASE)

    active_re = compile(r'Number of active orbitals\s+\.\.\.\s+(\d+)')
    total_re = compile(r'Total number of orbitals\s+\.\.\.\s+(\d+)')
    internal_re = compile(r'Internal\s+\d+\s*-\s*\d+\s*\(\s*(\d+)\s+orbitals\)')

    input_section = False
    casscf_section = False
    found_mult = False
    found_nroots = False
    multiplicities = None
    nroots = None

    while True:
        line = next(file)
        if input_file_start_re.match(line):
            input_section = True
            continue
        elif input_file_end_re.match(line):
            input_section = False
            if not (found_mult and found_nroots):
                raise ValueError("Could not find multiplicities or nroots in the input section.")
            break

        if input_section:
            if casscf_start_re.match(line):
                casscf_section = True
                continue
            elif casscf_end_re.match(line) and casscf_section:
                casscf_section = False
                continue

            if casscf_section:
                if not found_mult:
                    mult_match = mult_re.match(line)
                    if mult_match:
                        mult_values = mult_match.group(1)
                        multiplicities = asarray(list(map(int, findall(r'\d+', mult_values))), dtype=int64)
                        found_mult = True
                        if found_nroots:
                            break

                if not found_nroots:
                    nroots_match = nroots_re.match(line)
                    if nroots_match:
                        nroots_values = nroots_match.group(1)
                        nroots = asarray(list(map(int, findall(r'\d+', nroots_values))), dtype=int64)
                        found_nroots = True
                        if found_mult:
                            break

    if multiplicities is None or nroots is None:
        raise ValueError("Failed to find multiplicities or nroots.")

    active_orbitals = None
    total_orbitals = None
    inactive_orbitals = None
    determined_ranges_found = False

    while True:
        line = next(file)

        if active_orbitals is None:
            active_match = active_re.search(line)
            if active_match:
                active_orbitals = int(active_match.group(1))
        
        if total_orbitals is None:
            total_match = total_re.search(line)
            if total_match:
                total_orbitals = int(total_match.group(1))

        if "Determined orbital ranges:" in line:
            determined_ranges_found = True
            continue

        if determined_ranges_found and inactive_orbitals is None:
            internal_match = internal_re.search(line)
            if internal_match:
                inactive_orbitals = int(internal_match.group(1))
                break

    if active_orbitals is None:
        raise ValueError("Could not find the number of active orbitals.")
    if total_orbitals is None:
        raise ValueError("Could not find the total number of orbitals.")
    if inactive_orbitals is None:
        raise ValueError("Could not find the number of inactive (internal) orbitals.")

    return multiplicities, nroots, active_orbitals, total_orbitals, inactive_orbitals


def _decode_orca_determinant(det_str: str) -> tuple[list[int], list[int]]:
    alpha_orbs = []
    beta_orbs = []
    for i, ch in enumerate(det_str):
        orb_num = i
        if ch == 'u':
            alpha_orbs.append(orb_num)
        elif ch == 'd':
            beta_orbs.append(orb_num)
        elif ch == '2':
            alpha_orbs.append(orb_num)
            beta_orbs.append(orb_num)
        elif ch == '0':
            pass
        else:
            raise ValueError(f"Unknown determinant character '{ch}' in {det_str}")
    return alpha_orbs, beta_orbs


def _parse_orca_spin_determinant_ci(file: Iterator, multiplicities: int, nroots: int):
    dtype = settings.float

    ci_start_re = compile(r'^\s*Spin-Determinant CI Printing\s*$')
    root_start_re = compile(r'^ROOT\s+(\d+):\s+E=')
    det_line_re = compile(r'^\s*\[([ud20]+)\]\s+([-\d.]+)')

    results = {}

    for mult, nr in zip(multiplicities, nroots):
        while True:
            line = next(file)
            if ci_start_re.search(line):
                break

        determinant_patterns = []

        def parse_root_block(parse_determinants=False):
            next(file) # Skip line after ROOT info
            coeffs = []
            count = 0
            while True:
                line = next(file)
                det_match = det_line_re.match(line)
                if det_match:
                    det_str, coeff_str = det_match.groups()
                    coeff_value = dtype(coeff_str)
                    coeffs.append(coeff_value)
                    if parse_determinants:
                        determinant_patterns.append(det_str)
                    count += 1
                else:
                    break
            
            coeffs = asarray(coeffs, dtype=dtype, order='C')
            
            return coeffs, count

        while True:
            line = next(file)
            rmatch = root_start_re.match(line)
            if rmatch:
                root_idx = int(rmatch.group(1))
                if root_idx == 0:
                    first_root_coeffs, M = parse_root_block(parse_determinants=True)
                    determinant_alpha_info = []
                    determinant_beta_info = []
                    for det_str in determinant_patterns:
                        alpha_list, beta_list = _decode_orca_determinant(det_str)
                        determinant_alpha_info.append(alpha_list)
                        determinant_beta_info.append(beta_list)
                    determinant_alpha_info = asarray(determinant_alpha_info, dtype=int64, order='C')
                    determinant_beta_info = asarray(determinant_beta_info, dtype=int64, order='C')
                    ci_coeffs = zeros((M, nr), dtype=float)
                    ci_coeffs[:, 0] = first_root_coeffs
                    break
                else:
                    raise RuntimeError(f"Could not find 'ROOT 0' for the multiplicity {mult}.")

        for root_index in range(1, nr):
            line = next(file)
            rmatch = root_start_re.match(line)
            if not rmatch or int(rmatch.group(1)) != root_index:
                while not (rmatch and int(rmatch.group(1)) == root_index):
                    line = next(file)
                    rmatch = root_start_re.match(line)

            root_coeffs, count = parse_root_block(parse_determinants=False)
            if count != ci_coeffs.shape[0]:
                raise ValueError(f"Inconsistent number of determinants for ROOT {root_index}. Expected {ci_coeffs.shape[0]}, got {count}.")
            ci_coeffs[:, root_index] = root_coeffs

        results[mult] = (determinant_alpha_info, determinant_beta_info, ci_coeffs)

    return results


def _orca_matrix_reader(dim: int, number_of_whole_blocks: int, remaining_columns: int, file: Iterator, dtype: dtype, fix: bool = False) -> ndarray:
    matrix = empty((dim, dim), dtype=dtype, order="C")
    l = 0

    remove_indices_pattern = compile(r'^\s*\d+\s*', MULTILINE)
    fix_negative_overlap_pattern = compile(r'(\d)(-)')

    for _ in range(number_of_whole_blocks):
        next(file)  # Skip a line before each block

        # Read all lines for the current block at once
        data_str = ''.join([next(file) for _ in range(dim)])

        # Remove leading row indices
        data_str = remove_indices_pattern.sub('', data_str)

        if fix:
            # Insert spaces where negative signs overlap previous numbers
            data_str = fix_negative_overlap_pattern.sub(r'\1 -', data_str)

        data = fromstring(data_str, sep=' ', dtype=dtype)
        data = data.reshape(dim, 6)
        matrix[:, l:l+6] = data
        l += 6

    if remaining_columns > 0:
        next(file)
        data_str = ''.join([next(file) for _ in range(dim)])
        data_str = remove_indices_pattern.sub('', data_str)
        if fix:
            data_str = fix_negative_overlap_pattern.sub(r'\1 -', data_str)
        data = fromstring(data_str, sep=' ', dtype=dtype)
        data = data.reshape(dim, remaining_columns)
        matrix[:, l:l+remaining_columns] = data

    return matrix


def _orca_fragovl_reader(orca_fragovl_source: Union[str, Iterator], dim: int) -> ndarray:
    should_close = False
    if isinstance(orca_fragovl_source, str):
        file = open(orca_fragovl_source, "r")
        should_close = True
    else:
        file = orca_fragovl_source

    try:
        dtype = settings.float
        number_of_whole_blocks = dim // 6
        remaining_columns = dim % 6

        for _ in range(9):
            next(file)
        fragment_fragment_matrix = _orca_matrix_reader(dim, number_of_whole_blocks, remaining_columns, file, dtype, True)
        for _ in range(5):
            next(file)
        fragment_A_MO_matrix = _orca_matrix_reader(dim, number_of_whole_blocks, remaining_columns, file, dtype, True)
        for _ in range(5):
            next(file)
        fragment_B_MO_matrix = _orca_matrix_reader(dim, number_of_whole_blocks, remaining_columns, file, dtype, True)

        fragment_fragment_MO_overlap_matrix = fragment_A_MO_matrix.T @ fragment_fragment_matrix @ fragment_B_MO_matrix
    
    finally:
        if should_close:
            file.close()
    
    return fragment_fragment_MO_overlap_matrix


def _orca_process_overlap_mch_basis_hamiltonian(out_dir, out_filepath: str, pt2: bool, electric_dipole_momenta: bool, ssc: bool, gbw_zero_filepath:str, gbw_tmp_filepath: str, dim: int, zero_filepath: str, dof_number: int, nx: int, ny: int, nz: int, displacement: int, _orca_fragovl_filepath: str):
    tmp_filepath = join(out_dir, f"{dof_number}_{nx}_{ny}_{nz}_{displacement}.tmp")
    _orca_to_slt(out_filepath, tmp_filepath, "tmp", pt2, electric_dipole_momenta, ssc, True)
    
    process = Popen([_orca_fragovl_filepath, gbw_zero_filepath, gbw_tmp_filepath], stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True)
    fragment_fragment_MO_overlap_matrix = _orca_fragovl_reader(process.stdout, dim)

    dtype=settings.float

    if dtype == float64:
        from slothpy._general_utilities._lapack import _zdetinv as _detinv
    else:
        from slothpy._general_utilities._lapack import _sdetinv as _detinv

    with File(tmp_filepath, 'r+') as tmp_file:
        tmp_group = tmp_file["tmp"]
        inactive_orbitals = tmp_group.attrs["Inactive_orbitals"]
        active_orbitals = tmp_group.attrs["Active_orbitals"]
        multiplicities = tmp_group.attrs["Multiplicities"]

        inner_matrix = asfortranarray(fragment_fragment_MO_overlap_matrix[:inactive_orbitals, :inactive_orbitals], dtype=dtype)
        active_right_matrix = asfortranarray(fragment_fragment_MO_overlap_matrix[:inactive_orbitals, inactive_orbitals:inactive_orbitals+active_orbitals], dtype=dtype)
        active_left_matrix = asfortranarray(fragment_fragment_MO_overlap_matrix[inactive_orbitals:inactive_orbitals+active_orbitals, :inactive_orbitals], dtype=dtype)
        active_active_matrix = asfortranarray(fragment_fragment_MO_overlap_matrix[inactive_orbitals:inactive_orbitals+active_orbitals, inactive_orbitals:inactive_orbitals+active_orbitals], dtype=dtype)
        
        del fragment_fragment_MO_overlap_matrix
        
        det_inner_matrix, inv_inner_matrix = _detinv(inner_matrix)
        inv_active_right = inv_inner_matrix @ active_right_matrix

        del inv_inner_matrix
        del active_right_matrix

        det_inner_matrix_sqr = det_inner_matrix**2

        for mult in multiplicities:
            with File(zero_filepath, 'r') as zero_file:
                zero_tmp_group = zero_file["tmp"]
                mult_group = zero_tmp_group[f"MULTIPLICITY_{mult}"]
                zero_alpha_orbitals = mult_group["ALPHA_ORBITALS"][:]
                zero_beta_orbitals = mult_group["BETA_ORBITALS"][:]
                zero_ci_coefficients = mult_group["ROOTS_CI_COEFFICIENTS"].astype(dtype)[:]

            mult_group = tmp_group[f"MULTIPLICITY_{mult}"]
            alpha_orbitals = mult_group["ALPHA_ORBITALS"][:]
            beta_orbitals = mult_group["BETA_ORBITALS"][:]
            ci_coefficients = mult_group["ROOTS_CI_COEFFICIENTS"].astype(dtype)[:]

            overlap, phases = _calculate_wavefunction_overlap_phase_correction(det_inner_matrix_sqr, inv_active_right, active_left_matrix, active_active_matrix, zero_alpha_orbitals, zero_beta_orbitals, alpha_orbitals, beta_orbitals, zero_ci_coefficients, ci_coefficients)

            mult_group = tmp_group.create_group(str(mult))
            mult_group.create_dataset("OVERLAP", data=overlap, chunks=True)
            mult_group.create_dataset("PHASES", data=phases, chunks=True)
        
    return tmp_filepath, multiplicities, dof_number, nx, ny, nz, displacement


def _hamiltonian_derivatives_matrix_in_ci_basis(slt_filepath: str, gbw_path: str, dof: int, nx: int, ny: int, nz: int, displacement_number: int):
    hamiltonian_derivative_matrix = None
    displacements_phase_corrections = None
    return hamiltonian_derivative_matrix, displacements_phase_corrections


def _read_forces_cp2k(filepath: str, dof_number: int):
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


def _read_hessian_born_charges_from_dir(dirpath: str, format: Literal["CP2K"], dof_number: int, nx: int, ny: int, nz: int, displacement_number: int , step: float, accoustic_sum_rule: Literal["symmetric", "self_term", "without"] = "symmetric", dipole_momenta: bool = True, force_files_suffix: str = None, dipole_momenta_files_suffix: str = None):
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
            raise ValueError("Currently the only suported format is 'CP2K'.")

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
            
