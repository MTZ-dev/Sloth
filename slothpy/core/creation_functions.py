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

from typing import Optional
from os.path import splitext

from numpy import ndarray, array, int64, int32
from ase.io import read

from slothpy.core._config import settings
from slothpy.core._slothpy_exceptions import SltFileError, SltInputError
from slothpy.core.slt_file_object import SltFile
from slothpy._general_utilities._io import _xyz_to_slt, _unit_cell_to_slt, _supercell_to_slt, _orca_to_slt, _molcas_to_slt
from slothpy._general_utilities._utils import _check_n

###########################
# SltFile creation function
###########################


def slt_file(slt_filepath: str) -> SltFile:
    """
    Create an SltFile object from the existing .slt file.

    Parameters
    ----------
    slt_filepath : str
        Path to the existing .slt file to be loaded.

    Returns
    -------
    SltFile
        An instance of SltFile class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a SltFile from a given file.
    """

    if not slt_filepath.endswith(".slt"):
        raise SltInputError(ValueError("The file to be loaded must have a .slt extension."))
    try:
        return SltFile._new(slt_filepath)
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message="Failed to load SltFIle from the .slt file.") from None


#############################
# Topology creation functions
#############################


def xyz(input_filepath: str, slt_filepath: str, group_name: str, charge: Optional[int] = None, multiplicity: Optional[int] = None) -> SltFile:
    """
    Reads a .xyz or .cif file and saves the data as a xyz group in a .slt file.

    Parameters:
    ----------
    input_filepath : str
        Path to the input .xyz or .cif file.
    slt_filepath : str
        Path of the existing or new .slt file to which the XYZ will
        be saved.
    group_name : str
        Name of a group to which the XYZ will be saved.
    charge : int, optional
        Charge of the chemical species., by default None
    multiplicity : int, optional
        Multiplicity of the chemical species given as 2S+1., by default None
    Raises:
    ------
    SltInputError:
        If input_file has an unsupported extension
    SltFileError:
        If there are issues reading the input file or writing to the .slt file.
    """

    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    try:
        if charge is not None and not isinstance(charge, (int, int64, int32)):
            raise ValueError("Charge must be an integer or None.")
        if multiplicity is not None and (not isinstance(multiplicity, (int, int64, int32)) or multiplicity <= 0):
            raise ValueError("Multiplicity must be an integer greater than 0 or None.")
        _, ext = splitext(input_filepath)
        ext = ext.lower()
        if ext not in ['.xyz', '.cif']:
            raise ValueError("Unsupported file extension. Only .cif and .xyz files are supported.")
        try:
            atoms = read(input_filepath)
            elements = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
        except Exception as exc:
            raise IOError(f"Error reading the input file '{input_filepath}': {exc}")
        
    except Exception as exc:
        raise SltInputError(exc, message="Failed to save xyz to .slt file.") from None
    
    try:
        _xyz_to_slt(slt_filepath, group_name, elements, positions, charge, multiplicity)

        return SltFile._new(slt_filepath)
    
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message="Failed to save unit cell to .slt file.") from None


def unit_cell(input_filepath: str, slt_filepath: str, group_name: str, cell_vectors: Optional[ndarray] = None, cell_params: Optional[ndarray] = None, cell_from_cif_path: Optional[str] = None, multiplicity: Optional[int] = None) -> SltFile:
    """
    Reads a .cif or .xyz file along with cell parameters and saves the data
    as a unit cell group in a .slt file.

    Parameters:
    ----------
    input_filepath : str
        Path to the input .cif or .xyz file.
    slt_filepath : str
        Path of the existing or new .slt file to which the unit cell will
        be saved.
    group_name : str
        Name of a group to which the unit cell will be saved.
    cell_vectors : list or ndarray, optional
        A 3x3 list or NumPy array (ArrayLike structure) in the form [abc, :]
        representing the unit cell vectors. Required if input_file is a .xyz
        file and cell_params and cell_from_cif_path are not provided. If
        provided along with a .cif input file, the cell parameters will be
        overwritten by those from cell_vectors, by default None
    cell_params : list, tuple, or ndarray, optional
        An ArrayLike structure containing [a, b, c, alpha, beta, gamma],
        where a, b, c are the unit cell lengths and alpha, beta, gamma are the
        unit cell angles in degrees. Required if input_file is a .xyz file
        and cell_vectors and cell_from_cif_path are not provided. If provided
        along with a .cif input file, the cell parameters will be overwritten
        by those from cell_params, by default None
    cell_from_cif_path : str, optional
        Path to the input .cif from which the unit cell parameters will be
        extracted. Required if input_file is a .xyz file and cell_params and
        cell_vectors are not provided. If provided along with a .cif input
        file, the cell parameters will be overwritten by those from 
        cell_from_cif_path, by default None
    multiplicity : int, optional
        Multiplicity of the unit cell given as 2S+1., by default None

    Raises:
    ------
    SltInputError:
        If input_file has an unsupported extension or required cell parameters
        are not provided for .xyz files.
    SltFileError:
        If there are issues reading the input file or writing to the .slt file.
    """
    
    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    try:
        if sum(param is None for param in [cell_vectors, cell_params, cell_from_cif_path]) < 2:
            raise ValueError("Only one parameter from: cell_vectors, cell_params, cell_from_cif_path can be provided.")
        _, ext = splitext(input_filepath)
        ext = ext.lower()
        if ext not in ['.xyz', '.cif']:
            raise ValueError("Unsupported file extension. Only .cif and .xyz files are supported.")
        try:
            atoms = read(input_filepath)
            elements = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            cell = None
            if cell_vectors is not None:
                cell = array(cell_vectors, dtype=settings.float)
                if cell.shape != (3, 3):
                    raise ValueError("The cell_params must be a 3x3 matrix.")
            elif cell_params is not None:
                if len(cell_params) != 6:
                    raise ValueError("The cell_params must contain exactly six values: [a, b, c, alpha, beta, gamma].")
                cell = array(cell_params, dtype=settings.float)
            elif cell_from_cif_path is not None:
                if not cell_from_cif_path.endswith(".cif"):
                    raise ValueError("The cell_from_cif_path must have .cif extension.")
                cell = read(cell_from_cif_path).get_cell().array
            if cell is not None:
                atoms.set_cell(cell)
                cell = atoms.get_cell().array
            else:
                cell = atoms.get_cell().array

        except Exception as exc:
            raise IOError(f"Error reading the input file '{input_filepath}': {exc}")
        
    except Exception as exc:
        raise SltInputError(exc, message="Failed to save unit cell to .slt file.") from None
    
    try:
        _unit_cell_to_slt(slt_filepath, group_name, elements, positions, cell, multiplicity)

        return SltFile._new(slt_filepath)
    
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message="Failed to save unit cell to .slt file.") from None
    

def supercell(xyz_filepath: str, slt_filepath: str, group_name: str, nx: int, ny: int, nz: int, supercell_vectors: Optional[ndarray] = None, supercell_params: Optional[ndarray] = None, multiplicity: Optional[int] = None) -> SltFile:
    """
    Reads a .xyz file along with supercell parameters and saves the data
    as a supercell group in a .slt file.

    Parameters:
    ----------
    xyz_filepath : str
        Path to the input .xyz file.
    slt_filepath : str
        Path of the existing or new .slt file to which the supercell will
        be saved.
    group_name : str
        Name of a group to which the supercell will be saved.
    nx, ny, nz : int
        Number of repetitions along the x, y, and z axes of a unit cell within
        the provided supercell. Note that the supercell coordinates must be in
        order of repeating unit cells where .xyz file starts with coordinates
        of cell for nx = ny = nz = 0 and then the slowest varying index is nx
        while the fastest is nz.
    supercell_vectors : list or ndarray, optional
        A 3x3 list or NumPy array (ArrayLike structure) in the form [abc, :]
        representing the supercell vectors. Required if supercell_params are
        not provided., by default None
    cell_params : list, tuple, or ndarray, optional
        An ArrayLike structure containing [a, b, c, alpha, beta, gamma],
        where a, b, c are the supercell lengths and alpha, beta, gamma are the
        supercell angles in degrees. Required if supercell_vectors are not
        provided., by default None
    multiplicity : int, optional
        Multiplicity of the supercell given as 2S+1., by default None

    Raises:
    ------
    SltInputError:
        If input_file has an unsupported extension or required cell parameters
        are not provided for .xyz files.
    SltFileError:
        If there are issues reading the input file or writing to the .slt file.

    Note
    -----
    To create a supercell group from a .cif file, first create a unit_cell and
    then pass it to or use its supercell method directly with
    output_option = 'slt'.
    """
    
    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    _check_n(nx, ny, nz)

    try:
        check_sum = sum(param is None for param in [supercell_vectors, supercell_params])
        if check_sum != 1:
            raise ValueError("Provide one and only one parameter from: supercell_vectors, supercell_params.")
        _, ext = splitext(xyz_filepath)
        ext = ext.lower()
        if ext != '.xyz':
            raise ValueError("Unsupported file extension. Only .xyz files are supported.")
        try:
            atoms = read(xyz_filepath)
            elements = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            if len(atoms) % (nx * ny * nz) != 0:
                raise ValueError("Inconsistent number of atoms and unit cell repetitions for the provided nx, ny, nz, and the .xyz file.")
            supercell = None
            if supercell_vectors is not None:
                supercell = array(supercell_vectors, dtype=settings.float)
                if supercell.shape != (3, 3):
                    raise ValueError("The supercell_params must be a 3x3 matrix.")
            elif supercell_params is not None:
                if len(supercell_params) != 6:
                    raise ValueError("The supercell_params must contain exactly six values: [a, b, c, alpha, beta, gamma].")
                supercell = array(supercell_params, dtype=settings.float)
            if supercell is not None:
                atoms.set_cell(supercell)
                supercell = atoms.get_cell().array
            else:
                raise RuntimeError("Failed to obtain and set supercell parameters.")

        except Exception as exc:
            raise IOError(f"Error reading the input file '{xyz_filepath}': {exc}")
        
    except Exception as exc:
        raise SltInputError(exc, message="Failed to save supercell to .slt file.") from None
    
    try:
        _supercell_to_slt(slt_filepath, group_name, elements, positions, supercell, nx, ny, nz, multiplicity)

        return SltFile._new(slt_filepath)
    
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message="Failed to save unit cell to .slt file.") from None


################################
# Hamiltonian creation functions
################################


def hamiltonian_from_orca(orca_filepath: str, slt_filepath: str, group_name: str, pt2: bool = False, electric_dipole_momenta: bool = False, ssc: bool = False) -> SltFile:

    """
    Create or append data to an SltFile from the ORCA output file.

    Parameters
    ----------
    orca_filepath : str
        Path to the ORCA output file.
    slt_filepath : str
        Path of the existing or new .slt file to which the results will
        be saved.
    group_name : str
        Name of a group to which results of relativistic ab initio calculations
        will be saved.
    pt2 : bool, optional
        If True the results of CASPT2/NEVPT2 second-order perturbative
        corrections will be loaded to the file., by default False.
    electric_dipole_momenta : bool, optional
        If set to True, electric dipole moment integrals will be read from
        the ORCA file for simulations of spectroscopic properties.,
        by default False.
    ssc : bool, optional
        If set to True, SSC energies will be read from
        the ORCA instead of SOC energies, by default False.        

    Returns
    -------
    SltFile
        An instance of SltFile class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a SltFile from given files.

    Note
    ----
    ORCA calculations have to be done with the "printlevel 4" (or 5 for dipole momenta)
    keyword in the "rel" section for outputs to be readable by SlothPy.
    """

    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    if not isinstance(group_name, str):
        raise SltInputError(f"The group name has to be a string not {type(group_name)}.")
    try:
        _orca_to_slt(orca_filepath, slt_filepath, group_name, pt2, electric_dipole_momenta, ssc, False)

        return SltFile._new(slt_filepath)

    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message=("Failed to create a .slt file from the ORCA output file")) from None


def hamiltonian_from_molcas(molcas_filepath: str, slt_filepath: str, group_name: str, electric_dipole_momenta: bool = False) -> SltFile:
    """
    Create or append data to an SltFile from the MOLCAS rassi.h5 file.

    Parameters
    ----------
    molcas_filepath : str
        Path to the MOLCAS .rassi.h5 file.
    slt_filepath : str
        Path of the existing or new .slt file to which the results will
        be saved.
    group_name : str
        Name of a group to which results of relativistic ab initio calculations
        will be saved.
    electric_dipole_momenta : bool, optional
        If set to True, electric dipole moment integrals will be read from
        the MOLCAS .rassi.h5 file for simulations of spectroscopic properties.

    Returns
    -------
    SltFile
        An instance of SltFile class associated with the given .slt file, that
        serves as an user interface, holding all the available methods.

    Raises
    ------
    SltFileError
        If the program is unable to create a SltFile from given files.

    Note
    ----
    MOLCAS calculations have to be done with the "MESO" keyword within the
    RASSI section and the installation has to support HDF5 files for .rassi.h5
    files to be readable by SlothPy.
    """

    if not slt_filepath.endswith(".slt"):
        slt_filepath += ".slt"

    if not molcas_filepath.endswith(".rassi.h5"):
        raise SltInputError(ValueError("The molcas file to be loaded must have a .rassi.h5 extension."))
    if not isinstance(group_name, str):
        raise SltInputError(ValueError(f"The group name has to be a string not {type(group_name)}."))
    try:
        _molcas_to_slt(molcas_filepath, slt_filepath, group_name, electric_dipole_momenta)

        return SltFile._new(slt_filepath)
    
    except Exception as exc:
        raise SltFileError(slt_filepath, exc, message=("Failed to create a .slt file from the MOLCAS rassi.h5 file")) from None