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

import inspect
from functools import wraps
from os import cpu_count
from warnings import warn
from numpy import ndarray, asarray, ascontiguousarray, allclose, identity, linspace, meshgrid, vstack, log, int64, int32, max, min
from slothpy.core._slothpy_exceptions import SltInputError, SltFileError, SltSaveError, SltWarning, slothpy_exc
from slothpy._general_utilities._grids_over_hemisphere import lebedev_laikov_grid_over_hemisphere, fibonacci_over_hemisphere, meshgrid_over_hemisphere
from slothpy._general_utilities._math_expresions import _normalize_grid_vectors, _normalize_orientations, _normalize_orientation
from slothpy._general_utilities._constants import GREEN, BLUE, RESET, KB, H_CM_1, B_AU_T, F_AU_VM, AU_BOHR_CM_1
from slothpy._general_utilities._io import _group_exists
from slothpy.core._config import settings


def validate_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        from slothpy.core._slt_file import SltGroup
        slt_group: SltGroup = bound_args.arguments["self"] if "self" in bound_args.arguments.keys() else bound_args.arguments["slt_group"]

        if not slt_group._exists:
            raise SltFileError(slt_group._hdf5, None, f"{BLUE}Group{RESET}: '{slt_group._group_name}' does not exist in the .slt file.")
        
        if "slt_save" in bound_args.arguments.keys() and bound_args.arguments["slt_save"] is not None:
            if _group_exists(slt_group._hdf5, bound_args.arguments["slt_save"]):
                raise SltSaveError(
                    slt_group._hdf5,
                    NameError(""),
                    message=f"Unable to save the results. {BLUE}Group{RESET} '{bound_args.arguments['slt_save']}' already exists in the {GREEN}File{RESET}: '{slt_group._hdf5}'. Delete it manually.",
                ) from None

        try:
            for name, value in bound_args.arguments.items():
                match name:
                    case "number_cpu":
                        if value is None:
                            value = settings.number_cpu
                        if value == 0:
                            value = int(cpu_count())    
                        elif not (isinstance(value, (int, int32, int64)) and value > 0 and value <= int(cpu_count())):
                            raise ValueError(f"The number of CPUs must be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
                    case "number_threads":
                        max_threads = int(cpu_count()) if bound_args.arguments['number_cpu'] == 0 else bound_args.arguments['number_cpu']
                        if max_threads is None:
                            max_threads = settings.number_cpu
                        if value is None:
                            value = settings.number_threads
                        if value == 0:
                            value = max_threads
                        elif not (isinstance(value, (int, int32, int64)) and value > 0 and value <= max_threads):
                            raise ValueError(f"The number of threads must be a nonnegative integer less than or equal to the number of available logical CPUs: {int(cpu_count())} (0 for all the CPUs).")
                    case "magnetic_fields":
                        try:
                            value = asarray(value, order='C', dtype=settings.float) / asarray(B_AU_T, order='C', dtype=settings.numba_float)
                        except Exception:
                            raise ValueError("Magnetic fields must be an arraylike object of floats.")
                        if value.ndim != 1:
                            raise ValueError("The list of fields must be a 1D array.")
                    case "temperatures":
                        try:
                            value = asarray(value, order='C', dtype=settings.float)
                        except Exception:
                            raise ValueError("Temperatures must be an arraylike object of floats.")
                        if value.ndim != 1:
                            raise ValueError("The list of temperatures must be a 1D array.")
                        if (value <= 0).any():
                            raise ValueError("Zero or negative temperatures were detected in the input.")
                    case "grid":
                        if isinstance(value, (int, int32, int64)):
                            value = lebedev_laikov_grid_over_hemisphere(value, settings.precision)
                        else:
                            try:
                                value = asarray(value, order='C', dtype=settings.float)
                            except Exception:
                                raise ValueError("Grid must be an arraylike object of floats.")
                            if value.ndim != 2:
                                raise ValueError("The grid array must be a 2D array in the form [[direction_x, direction_y, direction_z, weight],...].")
                            if value.shape[1] == 3:
                                value = _normalize_grid_vectors(value)
                            else:
                                raise ValueError("The grid must be set to an integer from 0-11, or a custom one must be in the form [[direction_x, direction_y, direction_z, weight],...].")
                    case "orientations":
                        if isinstance(value, (int, int32, int64)):
                            value = lebedev_laikov_grid_over_hemisphere(value, settings.precision)
                        elif isinstance(value, (tuple, list)) and len(value) == 2 and not (isinstance(value[0], (tuple, list, ndarray)) and isinstance(value[1], (tuple, list, ndarray))):
                            if not isinstance(value[1], (int, int32, int64)):
                                raise ValueError("The second entry in the orientation list/tuple must contain an integer controlling the number of the grid points.")
                            if value[0] == "fibonacci":
                                value = ["fibonacci", fibonacci_over_hemisphere(value[1], settings.precision)]
                            if value[0] == "mesh":
                                value = ["mesh", meshgrid_over_hemisphere(value[1], settings.precision)]
                            if value[0] == "lebedev_laikov":
                                value = ["lebedev_laikov", ascontiguousarray(lebedev_laikov_grid_over_hemisphere(value[1], settings.precision)[:, :3])]
                            else:
                                raise ValueError("The only orientation grids available are: 'fibonacci', 'mesh', and 'lebedev_laikov'.")
                        else:
                            try:
                                value = asarray(value, order='C', dtype=settings.float)
                            except Exception:
                                raise ValueError("Orientations must be an arraylike object of floats.")
                            if value.ndim != 2:
                                raise ValueError("The array of orientations must be a 2D array in the form: [[direction_x, direction_y, direction_z],...] or [[direction_x, direction_y, direction_z, weight],...] for powder-averaging (or integer from 0-11).")
                            if value.shape[1] == 4:
                                value = _normalize_grid_vectors(value)
                            elif value.shape[1] == 3:
                                value = _normalize_orientations(value)
                            else:
                                raise ValueError("The orientations' array must be (n,3) in the form: [[direction_x, direction_y, direction_z],...] or (n,4) array in the form: [[direction_x, direction_y, direction_z, weight],...] for powder-averaging (or integer from 0-11).")
                    case "states_cutoff":
                        if slt_group.attributes["Kind"] == "SLOTHPY": ################################################ slt_group.type == "EXCHANGE_HAMILTONIAN"
                            if value != [0, "auto"]:
                                warn("State cutoff was modified, but it has no impact on the SlothPy user-defined Hamiltonian.", SltWarning)
                            continue
                        if not isinstance(value, list) or len(value) != 2:
                            raise ValueError("The states' cutoff must be a Python's list of length 2.")
                        if value[0] == 0:
                            value[0] = int(slt_group.attributes["States"])
                        elif not isinstance(value[0], (int, int32, int64)) or value[0] < 0:
                            raise ValueError(f"The states' cutoff must be a nonnegative integer less than or equal to the overall number of available states: {slt_group.attributes['States']} (or 0 for all the states).")
                        elif value[0] > slt_group.attributes["States"]:
                            raise ValueError(f"Set the states' cutoff to a nonnegative integer less than or equal to the overall number of available states: {slt_group.attributes['States']} (or 0 for all the states).")
                        if value[1] == 0:
                            value[1] = value[0]
                        if value[1] == "auto":
                            if "number_of_states" in bound_args.arguments.keys() and bound_args.arguments["number_of_states"] == 0:
                                value[1] = value[0]
                            elif "number_of_states" in bound_args.arguments.keys() and isinstance(bound_args.arguments["number_of_states"], (int, int32, int64)) and bound_args.arguments["number_of_states"] <= value[0]:
                                value[1] = bound_args.arguments["number_of_states"]
                            if "temperatures" in bound_args.arguments.keys():
                                value[1] = settings.float(max(bound_args.arguments["temperatures"]) * KB * -log(1e-16 if settings.precision == "double" else 1e-8))
                        elif not isinstance(value[1], (int, int32, int64)) or value[1] < 0 or value[1] > value[0]:
                            raise ValueError("Set the second entry of states' cutoff to a nonnegative integer less or equal to the first entry or 0 for all the states from the first entry or 'auto' to let the SlothPy decide on a suitable cutoff.")
                    case "number_of_states":
                        if not isinstance(value, (int, int32, int64)) or value < 0:
                            raise ValueError("The number of states must be a positive integer or 0 for all of the calculated states.")
                        if not isinstance(bound_args.arguments["states_cutoff"], list) or len(bound_args.arguments["states_cutoff"]) != 2:
                            raise ValueError("The states' cutoff must be a Python's list of length 2.")
                        max_states = int(slt_group.attributes["States"]) if bound_args.arguments["states_cutoff"][0] == 0 or slt_group.attributes["Kind"] == "SLOTHPY" else bound_args.arguments["states_cutoff"][0] ################################################# slt_group.type == "EXCHANGE_HAMILTONIAN"
                        if isinstance(bound_args.arguments["states_cutoff"][1], (int, int32, int64)) and (bound_args.arguments["states_cutoff"][1] > 0) and (bound_args.arguments["states_cutoff"][1] <= bound_args.arguments["states_cutoff"][0]) if isinstance(bound_args.arguments["states_cutoff"][0], (int, int32, int64)) else False:
                            if value == 0:
                                value = bound_args.arguments["states_cutoff"][1]
                            max_states = bound_args.arguments["states_cutoff"][1]
                        elif bound_args.arguments["states_cutoff"][1] in [0, "auto"] and value == 0:
                            value = max_states
                        if value > max_states:
                            print(max_states)
                            raise ValueError("The number of states must be less or equal to the states' cutoff or overall number of states.")
                    case "start_state":
                        if not (isinstance(value, (int, int32, int64)) and value >= 0 and value < slt_group.attributes["States"]):
                            raise ValueError(f"The first (start) state's number must be a nonnegative integer less than or equal to the overall number of states - 1: {slt_group.attributes['States'] - 1}.")
                    case "stop_state":
                        if value == 0:
                            value = int(slt_group.attributes["States"])
                        if not isinstance(value, (int, int32, int64)) or value < 0 or value > slt_group.attributes["States"]:
                            raise ValueError(f"The last (stop) state's number must be a nonnegative integer less than or equal to the overall number of states: {slt_group.attributes['States']}.")
                        if "start_state" in bound_args.arguments.keys():
                            if isinstance(bound_args.arguments["start_state"], (int, int32, int64)) and (bound_args.arguments["start_state"] >= 0) and bound_args.arguments["start_state"] <= slt_group.attributes["States"] and value < bound_args.arguments["start_state"]:
                                raise ValueError(f"The last (stop) state's number must be equal or greater than the first (start) state's number: {bound_args.arguments['start_state']}.")
                    case "xyz":
                        if value not in ["xyz", "x", "y", "z"]:
                            try:
                                value = asarray(value, order='C', dtype=settings.float)
                                if value.ndim != 1 or value.size != 3:
                                    raise Exception
                            except Exception:
                                raise ValueError(f"The xyz argument must be one of 'xyz', 'x', 'y', 'z' or it can be an orientation arraylike object of floats in the form [x,y,z].")
                            value = _normalize_orientation(value)
                    case "rotation":
                        value = _parse_rotation(value)
                    case "electric_field_vector":
                        if value is not None:
                            try:
                                if slt_group.attributes["Additional"] != "ELECTRIC_DIPOLE_MOMENTA":
                                    raise ValueError("The provided Hamiltonian gorup does not contain electric dipole momenta integrals.")
                                try:
                                    value = asarray(value, order='C', dtype=settings.float) / settings.float(F_AU_VM)
                                except Exception:
                                    raise ValueError("Electric field vector must be an arraylike object of floats.")
                                if value.ndim != 1 or value.shape[0] != 3:
                                    raise ValueError("The electric field vector must be an array-like object in the form of [x, y, z] components.")
                            except SltFileError:
                                raise ValueError(f"The provided Hamiltonian {BLUE}Group{RESET}: '{slt_group._group_name}' does not contain electric dipole momenta to include effect of electric field vector.")
                    case "hyperfine":
                        if value != None:
                            raise NotImplementedError("Hyperfine interactions have not been implemented yet. They are scheduled to be released in the 0.4 major release.")
                    case "show":
                        if not isinstance(value, bool):
                            raise ValueError("Show must be a boolean (False/True). It decides if plot is shown in gui or returned as mathplotlib objects (Figure and Axis).")
                    case "energy_unit":
                        if value not in ['kj/mol', 'eh', 'hartree', 'au', 'ev', 'kcal/mol', 'wavenumber']:
                            raise ValueError(f"Energy unit must be one of the following strings: kj/mol, eh, hartree, au, ev, kcal/mol, you entered {value} with type {type(value)}")
                    case "xyz_filepath":
                        try:
                            if value is not None:
                                if not value.endswith(".xyz"):
                                    value += ".xyz"
                        except Exception:
                            raise ValueError("XYZ fielpath must be a string.")
                    case "atom_indices":
                        if len(value) != len(bound_args.arguments["new_symbols"]):
                            raise ValueError("The lists 'atom_indices' and 'new_symbols' must have the same length.")
                        num_atoms = slt_group.attributes["Number_of_atoms"]  
                        for idx in value:
                            if not (0 <= idx < num_atoms):
                                raise IndexError(f"Atom index {idx} is out of bounds for a structure with {num_atoms} atoms.")
                    case "new_symbols":
                        from ase.data import chemical_symbols
                        for symbol in value:
                            if symbol not in chemical_symbols:
                                raise ValueError(f"Invalid element symbol: '{symbol}'. Please provide valid chemical element symbols.")
                    case "displacement_number":
                        if not isinstance(value, (int, int32, int64)) or value < 0:
                            raise ValueError("The displacement_number must be a nonnegative integer.")
                    case "step":
                        try:
                            float(value)
                        except Exception as exc:
                            raise ValueError(exc, "Invalid step provided.")
                    case "output_option":
                        allowed_values = ['xyz', 'slt']
                        xyz_output = "xyz_filepath"
                        if func.__name__ != "supercell":
                            allowed_values.append('iterator')
                            xyz_output = "custom_directory"
                        if value not in allowed_values:
                            raise ValueError("Invalid output_option. Choose from 'xyz', 'iterator', or 'slt'.")
                        if value == 'xyz' and (bound_args.arguments[xyz_output] is None or not isinstance(bound_args.arguments[xyz_output], str)):
                            raise ValueError("The custom_directory must be specified as a string when output_option is 'xyz'.")
                        if value == 'slt' and (bound_args.arguments["slt_group_name"] is None or not isinstance(bound_args.arguments["slt_group_name"], str)):
                            raise SltInputError(ValueError("The slt_group_name must be specified when output_option is 'slt'.")) from None
                    case "output_format":
                        if value not in ["xyz", "pdb"]:
                            raise ValueError("The output format must be 'xyz' or 'pdb'.")
                    case "nx":
                        from slothpy._general_utilities._utils import _check_n
                        _check_n(value, bound_args.arguments["ny"], bound_args.arguments["nz"], True)
                    case "format":
                        if value != "CP2K":
                            raise ValueError("The only suported format is 'CP2K'.")
                    case "accoustic_sum_rule":
                        if not value in ["symmetric", "self_term", "without"]:
                            raise ValueError("Invalid accoustic_sum_rule option. Choose from 'symmetric', 'self_term', 'without'.")
                    case "born_charges":
                        if not isinstance(value, bool):
                            raise ValueError("Set born_charges option to True or False.")
                    case "brillouin_zone_path":
                        try:
                            slt_group.atoms_object.cell.bandpath(path=value, npoints=bound_args.arguments["npoints"], density=bound_args.arguments["density"], special_points=bound_args.arguments["special_points"], eps=bound_args.arguments["symmetry_eps"])
                        except Exception:
                            raise
                    case "modes_cutoff":
                        if value == 0:
                            value = int(slt_group.attributes["Modes"])
                        elif not isinstance(value, (int, int32, int64)) or value < 0:
                            raise ValueError(f"The modes' cutoff must be a nonnegative integer less than or equal to the overall number of available modes: {slt_group.attributes['Modes']} (or 0 for all the states).")
                        if value > slt_group.attributes["Modes"]:
                            raise ValueError(f"Set the modes' cutoff to a nonnegative integer less than or equal to the overall number of available modes: {slt_group.attributes['Modes']} (or 0 for all the states).")
                    case "start_mode":
                        if not isinstance(value, (int, int32, int64)) or value < 0:
                            raise ValueError(f"The start mode number must be a nonnegative integer less than or equal to the overall number of available modes: {slt_group.attributes['Modes']} (or 0 for all the states).")
                        elif value > slt_group.attributes["Modes"]:
                            raise ValueError(f"Set the starrt mode number to a nonnegative integer less than or equal to the overall number of available modes: {slt_group.attributes['Modes']} (or 0 for all the states).")
                    case "stop_mode":
                        if value == 0:
                            value = int64(slt_group.attributes["Modes"])
                        elif not isinstance(value, (int, int32, int64)) or value < 0:
                            raise ValueError(f"The stop mode number must be a nonnegative integer less than or equal to the overall number of available modes: {slt_group.attributes['Modes']} (or 0 for all the states).")
                        if value > slt_group.attributes["Modes"]:
                            raise ValueError(f"Set the stop mode number to a nonnegative integer less than or equal to the overall number of available modes: {slt_group.attributes['Modes']} (or 0 for all the states).")
                        if value < bound_args.arguments["start_mode"]:
                            raise ValueError("The stop mode number must be greater or equal to the start mode number.")
                    case "kpoint":
                        try:
                            value = asarray(value, order='C', dtype=settings.float)
                        except Exception:
                            raise ValueError("K-point must be an arraylike object of floats.")
                        if value.shape != (3,):
                            raise ValueError("The k-point must have shape (3,).")
                    case "convolution":
                        if value not in [None, "lorentzian", "gaussian"]:
                            raise ValueError("The only valid options for the convolution are 'lorentzian', 'gaussian' or None.")
                    case "fwhm":
                        value = settings.float(value)
                        if value <= 0:
                            raise ValueError("FWHM must be greater than zero.")
                    case "resolution":
                        if (not isinstance(value, (int, int32, int64)) or value <= 0) and value is not None:
                            raise ValueError("Resolution of the spectra must be set to an integer greater than zero.")
                    case "start_wavenumber":
                        if value is not None:
                            value = settings.float(value)
                            au_bohr_cm_1 = asarray(AU_BOHR_CM_1, dtype=settings.float)
                            value = value / au_bohr_cm_1
                            value = value * value if value >= 0 else -value * value
                    case "stop_wavenumber":
                        if value is not None:
                            value = settings.float(value)
                            au_bohr_cm_1 = asarray(AU_BOHR_CM_1, dtype=settings.float)
                            value = value / au_bohr_cm_1
                            value = value * value if value >= 0 else -value * value
                            if bound_args.arguments["start_wavenumber"] is not None and value <= bound_args.arguments["start_wavenumber"]:
                                raise ValueError("The stop_wavenumber must be strictly greater than the start_wavenumber.")
                    case "kpoints_grid":
                        if isinstance(value, (int, int32, int64)):
                            if value <= 0:
                                raise ValueError("The k-points grid must be greater than 0.")
                            else:
                                q = linspace(0, 1, value, endpoint=False)
                                q_grid = meshgrid(q, q, q, indexing='ij')
                                value = vstack([grid.ravel() for grid in q_grid]).T
                        else:
                            try:
                                value = asarray(value, order='C', dtype=settings.float)
                            except Exception:
                                raise ValueError("The k-points grid must be an arraylike object of floats.")
                            if value.ndim != 2 or value.shape[1] != 3:
                                raise ValueError("The k-points grid array must be a 2D array in the form [[k_x, k_y, k_z],...] in the fractional coordinate system of the reciprocal lattice.")
                    case "modes_list":
                        try:
                            value = asarray(value, order='C', dtype=int64)
                            max_mode = int64(slt_group.attributes["Modes"])
                            if max(value) >= max_mode or min(value) < 0:
                                raise Exception
                        except Exception:
                                raise ValueError(f"The modes list must be an arraylike object of nonnegative integers less than the number of available modes {max_mode}.")
                    case "frames":
                        if not isinstance(value, (int, int32, int64)) or value <= 0:
                            raise ValueError("Frames must be set to an integer greater than zero.")
                    case "amplitude":
                        value = settings.float(value)
                        if value <= 0:
                            raise ValueError("Amplitude must be greater than zero.")
                    case "unit_cell_group_name":
                        if SltGroup(slt_group._hdf5, value).attributes["Type"] != "UNIT_CELL":
                            raise ValueError(f"The given name of a unit cell {BLUE}Group{RESET}: '{value}' does not correspond to a valid {GREEN}UNIT_CELL{RESET} group in the .slt file.")
                    case "central_atom":
                        try:
                            value = asarray(value, order='C', dtype=settings.float)
                        except Exception:
                            raise ValueError("The central_atom parameter must be an arraylike object of floats.")
                        if value.ndim != 1 or len(value) != 3:
                            raise ValueError("The central_atom parameter must be a 1D array in the form [x, y, z] in Å.")


                bound_args.arguments[name] = value
                
        except Exception as exc:
            raise SltInputError(exc) from None

        return func(**bound_args.arguments)
    
    return wrapper


@slothpy_exc("SltInputError")
def _parse_hamiltonian_dicts(slt_file, magnetic_centers: dict, exchange_interactions: dict):

    if not isinstance(magnetic_centers, dict) or not isinstance(exchange_interactions, dict):
        raise ValueError("Magnetic centers and exchange interactions parameters must be dictionaries.")
    
    if not all(isinstance(key, (int, int32, int64)) for key in magnetic_centers.keys()):
        raise ValueError("Magnetic centers in the dictionary must be enumerated by integers.")
    
    n = len(magnetic_centers)
    expected_keys = set(range(n))
    actual_keys = set(magnetic_centers.keys())

    if expected_keys != actual_keys:
        raise ValueError("Magnetic centers in the dictionary must be enumerated by integers from 0 to the number of centers - 1 without repetitions.")
    
    states = 1
    exchange_states = 1
    contains_electric_dipole_momenta = True

    for value in magnetic_centers.values():
        if not isinstance(value, list):
            raise ValueError("The values of the magnetic centers dictionary must be Python's lists.")
        if slt_file[value[0]].attributes["Type"] != "HAMILTONIAN" or slt_file[value[0]].attributes["Kind"] == "SLOTHPY": #############################################################################
            raise ValueError(f"Group {value[0]} either does not exist or has a wrong type: expected HAMILTONIAN (it cannot be a custom SlothPy Hamiltonian).")
        try:
            if slt_file[value[0]].attributes["Additional"] != "ELECTRIC_DIPOLE_MOMENTA":
                contains_electric_dipole_momenta = False
        except SltFileError:
            contains_electric_dipole_momenta = False
        if (len(value[1]) != 3 or not all(isinstance(x, (int, int32, int64)) for x in value[1])):
            raise ValueError("States cutoff must be an interable of length 3 [local_cutoff, mixing_cutoff, exchange_cutoff] with integer values.")
        if value[1][0] < value[1][1] or value[1][1] < value[1][2]:
            raise ValueError("The cutoff parameters must satisfy local_cutoff >= mixing_cutoff >= exchange_cutoff.")
        states *= value[1][1]
        exchange_states *= value[1][2]
        if value[2] != None:
            value[2] = _parse_rotation(value[2])
        if value[3] != None and (len(value[3]) != 3 or not all(isinstance(x, (int, int32, int64, float)) for x in value[3])):
            raise ValueError("Coordinates must be None or iterable of length 3 with numerical values in Angstrom [x, y, z].")
       #TODO: value[4] hyperfine add checks when implemented

    if states >= 8000 or (states >= 6000 and exchange_states >= 100):
        print(f"You created a custom Hamiltonian with {states}x{states} exchange space, and computations will require to find {exchange_states} eigenvalues/eigenvectors, which is considered very expensive. You must know what you are doing. All computations will be very lengthy, if possible at all.")

    for key, value in exchange_interactions.items():
        if len(key) != 2 or not all(isinstance(x, (int, int32, int64)) and 0 <= x < n for x in key):
            raise ValueError("Two-center exchange interactions must be enumerated with iterables of length 2 [centerA_number, centerB_number] containing integers from 0 to the number of centers - 1.")
        value = asarray(value, order='C', dtype=settings.float) / H_CM_1
        if value.ndim != 2 or value.shape != (3,3):
            raise ValueError("The J exchange interaction parameters must be real arrays with shape(3,3).")
        exchange_interactions[key] = value

    return states, contains_electric_dipole_momenta


def _parse_rotation(value):
    from slothpy._angular_momentum._rotation import SltRotation
    from scipy.spatial.transform import Rotation
    if isinstance(value, (SltRotation, Rotation)):
        if not value.single:
            raise ValueError("SlothPy only supports (Slt)Rotation class instances with a single rotation.")
        value = value.as_matrix().astype(settings.float)
    elif value is not None:
        value = asarray(value, order='C', dtype=settings.float)
        if value.shape != (3, 3):
            raise ValueError("The rotation matrix must be a 3x3 array.")
        product = value.T @ value
        if not allclose(product, identity(3), atol=1e-2, rtol=0):
            raise ValueError("Input rotation matrix must be orthogonal.")
    return value
      





