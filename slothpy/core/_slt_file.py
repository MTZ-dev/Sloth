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

from __future__ import annotations

from typing import Union, Optional, Iterator, List, Mapping, Sequence
import warnings
from ast import literal_eval
from os import makedirs, remove
from os.path import join, exists

from h5py import File, Group, Dataset, string_dtype
from numpy import ndarray, asarray, asfortranarray, empty, real, linspace, outer, repeat, tensordot, diag, meshgrid, stack, arange, tensordot, einsum, all, where, float32, float64,  abs,  conjugate, sqrt, exp, pi, newaxis
from numpy.exceptions import ComplexWarning
from numpy.linalg import norm, inv
warnings.filterwarnings("ignore", category=ComplexWarning)
from scipy.linalg import eigvalsh
from ase import Atoms
from ase.io import write, read
from ase.cell import Cell
from ase.io.trajectory import Trajectory

from slothpy.core._registry import MethodTypeMeta, MethodDelegateMeta
from slothpy.core._config import settings
from slothpy.core._slothpy_exceptions import slothpy_exc, KeyError, SltFileError, SltReadError
from slothpy.core._hessian_object import Hessian
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET
from slothpy._general_utilities._math_expresions import _magnetic_dipole_momenta_from_spins_angular_momenta, _total_angular_momenta_from_spins_angular_momenta, is_approximately_integer
from slothpy._general_utilities._io import _get_dataset_slt_dtype, _group_exists, _dataset_exists, _xyz_to_slt, _supercell_to_slt, _hessian_to_slt, _read_hessian_born_charges_from_dir
from slothpy._general_utilities._constants import U_PI_A_AU, E_PI_A_AU
from slothpy._general_utilities._direct_product_space import _kron_mult
from slothpy._general_utilities._math_expresions import _subtract_const_diagonal
from slothpy._general_utilities._utils import _check_n
from slothpy.core._delayed_methods import *

############
# Attributes
############


class SltAttributes:

    __slots__ = ["_hdf5", "_item_path"]
                 
    def __init__(self, hdf5_file, item_path):
        self._hdf5 = hdf5_file
        self._item_path = item_path

    @slothpy_exc("SltFileError")
    def __getitem__(self, attr_name):
        with File(self._hdf5, 'r') as file:
            item = file[self._item_path]
            return item.attrs[attr_name]

    @slothpy_exc("SltFileError")
    def __setitem__(self, attr_name, value):
        with File(self._hdf5, 'r+') as file:
            item = file[self._item_path]
            item.attrs[attr_name] = value

    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._item_path]
            return f"<{YELLOW}SltAttributes{RESET} for {BLUE}Group{RESET}/{PURPLE}Dataset{RESET} '{self._item_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"

    @slothpy_exc("SltFileError")
    def __str__(self):
        with File(self._hdf5, 'a') as file:
            item = file[self._item_path]
            dict_str = f"{RED}Attributes{RESET}: " + ', '.join([f"{YELLOW}{key}{RESET}: {value}" for key, value in item.attrs.items()])
            formatted_str = f"{{{dict_str}}}".rstrip()
            return formatted_str
    
    @slothpy_exc("SltFileError")
    def __delitem__(self, attr_name):
        with File(self._hdf5, 'r+') as file:
            item = file[self._item_path]
            del item.attrs[attr_name]

    @slothpy_exc("SltFileError")
    def __contains__(self, item):
        with File(self._hdf5, 'r') as file:
            file_item = file[self._item_path]
            return item in file_item.attrs


#########
#Datasets
#########


class SltDataset:
    def __init__(self, hdf5_file_path, dataset_path):
        self._hdf5 = hdf5_file_path
        self._dataset_path = dataset_path
        self.attributes = SltAttributes(hdf5_file_path, dataset_path)

    @slothpy_exc("SltReadError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            dtype = dataset.dtype
            match str(dtype)[0]:
                case "c":
                    return dataset.astype(settings.complex)[slice_]
                case "f":
                    return dataset.astype(settings.float)[slice_]
                case "i":
                    return dataset.astype(settings.int)[slice_]
                case _:
                    if isinstance(dataset[0], bytes):
                        return [element.decode('utf-8') for element in dataset[slice_]]
                    return dataset[slice_]
        
    @slothpy_exc("SltSaveError")
    def __setitem__(self, slice_, value):
        with File(self._hdf5, 'r+') as file:
            dataset = file[self._dataset_path]
            if isinstance(value, list) and isinstance(value[0], str):
                dataset[slice_] = asarray(value, dtype='S')
            else:
                dataset[slice_] = asarray(value)

    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._dataset_path]
            return f"<{PURPLE}SltDataset{RESET} '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"
    
    @slothpy_exc("SltFileError")
    def __str__(self):
        with File(self._hdf5, 'r+') as file:
            item = file[self._dataset_path]
            representation = f"{RED}Dataset{RESET}: {PURPLE}{self._dataset_path}{RESET} from File: {GREEN}{self._hdf5}{RESET}"
            for attribute_name, attribute_text in item.attrs.items():
                representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
            return representation.rstrip()
        
    @property
    def attrs(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes
    
    @property
    def shape(self):
        """
        Property to mimic h5py's shape access convention.
        """
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            return dataset.shape
        
    @property
    def dtype(self):
        """
        Property to mimic h5py's shape access convention.
        """
        return _get_dataset_slt_dtype(self._hdf5, self._dataset_path)


class SltDatasetSLP():

    __slots__ = ["_hdf5", "_dataset_path", "_slp", "_xyz", "_xyz_dict"]

    def __init__(self, slt_group: SltGroup, slp: Literal["S", "L", "P"], xyz: int = None):
        self._hdf5 = slt_group._hdf5
        self._slp = slp
        _slp_dict = {"S": "SPINS", "L": "ANGULAR_MOMENTA", "P": "ELECTRIC_DIPOLE_MOMENTA"}
        self._dataset_path = f"{slt_group._group_name}/{_slp_dict[self._slp]}"
        self._xyz = xyz
        self._xyz_dict = {0: "x", 1: "y", 2: "z"}

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path].astype(settings.complex)
            if self._xyz is None:
                return dataset[*(slice_,) if isinstance(slice_, slice) else slice_]
            else:
                return dataset[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        return f"<{PURPLE}SltDataset{self._slp}{self._xyz_dict[self._xyz] if self._xyz is not None else ''}{RESET} from '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"

    @property
    def shape(self):
        """
        Property to mimic h5py's shape access convention.
        """
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            if self._xyz in [0,1,2]:
                return dataset.shape[1:]
            else:
                return dataset.shape
        
    @property
    def dtype(self):
        """
        Property to mimic h5py's shape access convention.
        """
        return _get_dataset_slt_dtype(self._hdf5, self._dataset_path)
    
    def _get_diagonal(self, start, stop):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            size = stop - start
            if self._xyz is None:
                diag = empty((3,size), dtype = dataset.dtype)
                for i in range(start, stop):
                    diag[:,i] = dataset[:,i,i]
            else:
                diag = empty(size, dtype = dataset.dtype)
                for i in range(start, stop):
                    diag[i] = dataset[self._xyz,i,i]
            return diag.astype(settings.float)


class SltDatasetJM():
    def __init__(self, slt_group: SltGroup, jm: Literal["J", "M"], xyz: int = None):
        self._hdf5 = slt_group._hdf5
        self._group_name = slt_group._group_name
        self._jm = jm
        self._xyz = xyz
        self._xyz_dict = {0: "x", 1: "y", 2: "z"}

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            group = file[self._group_name]
            if self._xyz is not None:
                dataset_s = group["SPINS"].astype(settings.complex)[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
                dataset_l = group["ANGULAR_MOMENTA"].astype(settings.complex)[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
            else:
                dataset_s = group["SPINS"].astype(settings.complex)[slice_]
                dataset_l = group["ANGULAR_MOMENTA"].astype(settings.complex)[slice_]
            if self._jm == "J":
                return _total_angular_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            elif self._jm == "M":
                return  _magnetic_dipole_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic dipole momenta.")
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        return f"<{PURPLE}SltDataset{self._jm}{self._xyz_dict[self._xyz] if self._xyz is not None else ''}{RESET} from {BLUE}Group{RESET} '{self._group_name}' in {GREEN}File{RESET} '{self._hdf5}'.>"
    
    @property
    def shape(self):
        """
        Property to mimic h5py's shape access convention.
        """
        with File(self._hdf5, 'r') as file:
            dataset = file[self._group_name]["SPINS"]
            if self._xyz in [0,1,2]:
                return dataset.shape[1:]
            else:
                return dataset.shape
        
    @property
    def dtype(self):
        """
        Property to mimic h5py's shape access convention.
        """
        return _get_dataset_slt_dtype(self._hdf5, f"{self._group_name}/SPINS")
    
    def _get_diagonal(self, start, stop):
        with File(self._hdf5, 'r') as file:
            group = file[self._group_name]
            size = stop - start
            dataset_s = group["SPINS"]
            dataset_l = group["ANGULAR_MOMENTA"]
            if self._xyz is None:
                diag_s = empty((3,size), dtype = dataset_s.dtype)
                diag_l = empty((3,size), dtype = dataset_l.dtype)
                for i in range(start, stop):
                    diag_s[:,i] = dataset_s[:,i,i]
                    diag_l[:,i] = dataset_l[:,i,i]
            else:
                diag_s = empty(size, dtype = dataset_s.dtype)
                diag_l = empty(size, dtype = dataset_l.dtype)
                for i in range(start, stop):
                    diag_s[i] = dataset_s[self._xyz,i,i]
                    diag_l[i] = dataset_l[self._xyz,i,i]
            if self._jm == "J":
                return _total_angular_momenta_from_spins_angular_momenta(diag_s.astype(settings.float), diag_l.astype(settings.float))
            elif self._jm == "M":
                return  _magnetic_dipole_momenta_from_spins_angular_momenta(diag_s.astype(settings.float), diag_l.astype(settings.float))
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic dipole momenta.")


########
# Groups
########


class SltGroup(metaclass=MethodDelegateMeta):

    __slots__ = ["_hdf5", "_group_name", "_exists"]

    def __init__(self, hdf5_file, group_name):
        self._hdf5 = hdf5_file
        self._group_name = group_name
        self._exists = _group_exists(hdf5_file, group_name)

    @slothpy_exc("SltFileError")
    def __getitem__(self, key):
        full_path = f"{self._group_name}/{key}"
        with File(self._hdf5, 'r') as file:
            if full_path in file:
                item = file[full_path]
                if isinstance(item, Dataset):
                    return SltDataset(self._hdf5, full_path)
                elif isinstance(item, Group):
                    raise KeyError(f"Hierarchy only up to {BLUE}Group{RESET}/{PURPLE}Dataset{RESET} or standalone {PURPLE}Datasets{RESET} are supported in .slt files.")
            else:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' doesn't exist in the {BLUE}Group{RESET} '{self._group_name}'.")

    @slothpy_exc("SltSaveError")        
    def __setitem__(self, key, value):
        with File(self._hdf5, 'r+') as file:
            group = file.require_group(self._group_name)
            self._exists = True
            if key in group:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' already exists within the {BLUE}Group{RESET} '{self._group_name}'. Delete it manually to ensure your data safety.")
            if isinstance(value, list) and isinstance(value[0], str):
                data = asarray(value, dtype='S')
            else:
                data = asarray(value)
            group.create_dataset(key, data=data, chunks=True)

    @slothpy_exc("SltFileError")
    def __repr__(self): 
        if self._exists:
            return f"<{BLUE}SltGroup{RESET} '{self._group_name}' in {GREEN}File{RESET} '{self._hdf5}'.>"
        else:
            raise RuntimeError(f"This is a {BLUE}Proxy Group{RESET} and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")

    @slothpy_exc("SltFileError")
    def __str__(self):
        if self._exists:
            with File(self._hdf5, 'r+') as file:
                item = file[self._group_name]
                representation = f"{RED}Group{RESET}: {BLUE}{self._group_name}{RESET} from File: {GREEN}{self._hdf5}{RESET}"
                for attribute_name, attribute_text in item.attrs.items():
                    representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
                representation += "\nDatasets: \n"
                for dataset_name, dataset in item.items():
                    representation += f"{PURPLE}{dataset_name}{RESET}"
                    for attribute_name, attribute_text in dataset.attrs.items():
                        representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
                    representation += "\n"
                return representation.rstrip()
        else:
            raise RuntimeError("This is a {BLUE}Proxy Group{RESET} and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")

    @slothpy_exc("SltFileError")
    def __delitem__(self, key):
        with File(self._hdf5, 'r+') as file:
            group = file[self._group_name]
            if key not in group:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' does not exist in the {BLUE}Group{RESET} '{self._group_name}'.")
            del group[key]

    @property
    def type(self):
        try:
            return self.attributes["Type"]
        except SltFileError as exc:
            raise SltReadError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_name}' is not a valid SlothPy group and has no type.") from None

    @property
    def attributes(self):
        return SltAttributes(self._hdf5, self._group_name)
    
    @property
    def attrs(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes

    @property
    def atoms_object(self) -> Atoms: pass

    @property
    def cell_object(self) -> Cell: pass
    
    @property
    def charge(self) -> int: pass
    
    @property
    def multiplicity(self) -> int: pass

    @property
    def hessian(self) -> ndarray: pass

    @property
    def born_charges(self) -> ndarray: pass
    
    @property
    def e(self) -> SltDataset: pass
    
    energies = e
    
    @property
    def s(self) -> SltDatasetSLP: pass
    
    spins = s
    
    @property
    def sx(self) -> SltDatasetSLP: pass
    
    @property
    def sy(self) -> SltDatasetSLP: pass
    
    @property
    def sz(self) -> SltDatasetSLP: pass
    
    @property
    def l(self) -> SltDatasetSLP: pass
    
    angular_momenta = l
    
    @property
    def lx(self) -> SltDatasetSLP: pass
    
    @property
    def ly(self) -> SltDatasetSLP: pass
    
    @property
    def lz(self) -> SltDatasetSLP: pass
    
    @property
    def p(self) -> SltDatasetSLP: pass
    
    electric_dipole_momenta = p
    
    @property
    def px(self) -> SltDatasetSLP: pass
    
    @property
    def py(self) -> SltDatasetSLP: pass
    
    @property
    def pz(self) -> SltDatasetSLP: pass
    
    @property
    def j(self) -> SltDatasetJM: pass
    
    total_angular_momenta = j
    
    @property
    def jx(self) -> SltDatasetJM: pass
    
    @property
    def jy(self) -> SltDatasetJM: pass
    
    @property
    def jz(self) -> SltDatasetJM: pass
    
    @property
    def m(self) -> SltDatasetJM: pass
    
    magnetic_dipole_momenta = m
    
    @property
    def mx(self) -> SltDatasetJM: pass
    
    @property
    def my(self) -> SltDatasetJM: pass
    
    @property
    def mz(self) -> SltDatasetJM: pass

    def plot(self, *args, **kwargs): pass
    
    def to_numpy_arrays(self, *args, **kwargs): pass

    def to_data_frame(self, *args, **kwargs): pass

    def to_csv(self, csv_filepath: str, separator: str = ",", *args, **kwargs): pass

    def to_xyz(self, xyz_filepath: str, hese: int, *args, **kwargs): pass

    def show_bandpath(self, brillouin_zone_path: str = None, npoints: int = None, density: float = None, special_points: Mapping[str, Sequence[float]] = None, symmetry_eps: float = 2e-4) -> None: pass

    def replace_atoms(self, atom_indices: List[int], new_symbols: List[str]) -> None:
        """
        Replaces atoms at specified indices with new element symbols and updates
        the HDF5 group.
        
        Parameters:
        ----------
        atom_indices : List[int]
            List of 0-based atom indices to be replaced.
        new_symbols : List[str]
            List of new element symbols corresponding to each atom index.

        """
        pass

    def generate_finite_stencil_displacements(self, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        """
        Generates finite stencil displacements for derivative calculations.

        Displaces each atom along x, y, and z axes in both negative and
        positive directions by a specified number of steps and step size.

        If this is used for supercells it only displaces each atom in the first
        unit cell.

        Parameters:
        ----------
        displacement_number : int
            Number of displacement steps in each direction (negative and positive).
        step : float
            Magnitude of each displacement step in Angstroms.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write displaced structures as `.xyz files`.
            - 'iterator': Return an iterator yielding tuples of displaced ASE `Atoms`
            objects, degree of freedom (DOF) numbers, and displacement numbers.
            - 'slt': Dump all displaced structures into the .slt file.
            Default is 'xyz'.
        custom_directory : str, optional
            Directory path to save `.xyz files`. Required if `output_option` is 'xyz'.
        slt_group_name : str, optional
            Name of the `SltGroup` to store displaced structures. Required if
            `output_option` is 'slt'.

        Returns:
        -------
        Iterator[Atoms] or None
            Returns an iterator of ASE `Atoms` objects if `output_option `is 'iterator'.
            Otherwise, returns None.
        """
        pass
    
    def generate_finite_stencil_displacements_reduced_to_unit_cell(self, unit_cell_group_name: str, central_atom: ndarray[Union[float32, float64]], displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        """
        Generates finite stencil displacements reduced to the unit cell for derivative
        calculations.

        This method identifies unique atoms in a cluster that correspond to atoms in
        the given unit cell, based on periodic boundary conditions and the closeness
        to the central atom. It then displaces only these unique atoms along the
        x, y, and z axes in both negative and positive directions by a specified number
        of steps and step size.

        Parameters:
        ----------
        unit_cell_group_name : str
            Name of the `SltGroup` representing the unit cell to which the cluster
            corresponds.
        central_atom : ndarray[Union[float32, float64]]
            Coordinates of the central atom or point in the cluster used as a reference
            point for uniquely matching the closest atoms.
        displacement_number : int
            Number of displacement steps in each direction (negative and positive).
        step : float
            Magnitude of each displacement step in Angstroms.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write displaced structures as `.xyz` files.
            - 'iterator': Return an iterator yielding tuples of displaced ASE `Atoms`
            objects, degree of freedom (DOF) numbers, and displacement numbers.
            - 'slt': Dump all displaced structures into the `.slt` file.
            Default is 'xyz'.
        custom_directory : str, optional
            Directory path to save `.xyz` files. Required if `output_option` is 'xyz'.
        slt_group_name : str, optional
            Name of the `SltGroup` to store displaced structures. Required if
            `output_option` is 'slt'.

        Returns:
        -------
        Iterator[Atoms] or None
            Returns an iterator of ASE `Atoms` objects if `output_option` is 'iterator'.
            Otherwise, returns `None`.
        """
        pass
    
    def supercell(self, nx: int, ny: int, nz: int, output_option: Literal["xyz", "slt"] = "xyz", xyz_filepath: Optional[str] = None, slt_group_name: Optional[str] = None) -> SltGroup:
        """
        Generates a supercell by repeating the unit cell along x, y, and z axes.

        Repeats the unit cell `nx`, `ny`, and `nz` times along the x, y, and z
        axes respectively to create a supercell where cordinates are such that
        they start with unit cell for nx = ny = nz = 0 and then the slowest
        varying index is nx while the fastest is nz.

        Parameters:
        ----------
        nx, ny, nz : int
            Number of repetitions along the x, y, and z-axis.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write the supercell structure as a `.xyz` file.
            - 'slt': Save the supercell structure in the `.slt` file.
            Default is 'xyz'.
        xyz_filepath : str, optional
            File path to save the `.xyz` file. Required if `output_option` is 'xyz'.
        slt_group_name : str, optional
            Name of the `SltGroup` to store the supercell structure. Required if
            `output_option` is 'slt'.

        Returns:
        -------
        None

        Notes:
        -----
        - If the object is already a supercell, a warning will be issued indicating
        that a mega-cell will be created by multiplying the current parameters.
        """
        pass

    def generate_supercell_finite_stencil_displacements(self, nx: int, ny: int, nz: int, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None, save_supercell_to_slt: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        """
        Generates a new supercell and finite stencil displacements for it by
        displacing atoms in the first unit cell.

        Displaces each atom in the first unit cell of a supercell along x, y,
        and z axes in both negative and positive directions by a specified
        number of steps and step size.

        Parameters:
        ----------
        nx, ny, nz : int
            Number of repetitions along the x, y, and z axes to create the supercell.
        displacement_number : int
            Number of displacement steps in each direction (negative and positive).
        step : float
            Magnitude of each displacement step in Angstroms.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write displaced structures as .xyz files.
            - 'iterator': Return an iterator yielding tuple of dislaced ASE
            Atoms objects, dofs numbers, displacements numbers, nx, ny, and nz.
            - 'slt': Dump all displaced structures into the .slt file.
            Default is 'xyz'.
        custom_directory : str, optional
            Directory path to save .xyz files. Required if output_option is 'xyz'.
        slt_group_name : str, optional
            Name of the SltGroup to store displaced structures. Required if
            output_option is 'slt'.
        save_supercell_to_slt: str, optional
            When provided, the created supercell is saved to the group of this
            name in the .slt file and can be used for further processing, e.g.
            creating Hessian after finite displacement calculations

        Returns:
        -------
        Iterator[Atoms] or None
            Returns an iterator of ASE Atoms objects if output_option is 'iterator'.
            Otherwise, returns None.

        Note:
        -----
        For a supercell, use generate_finite_stencil_displacements if you do
        not wish to repeat it further with new nx, ny, and nz.
        """
        pass

    def hessian_from_finite_displacements(self, dirpath: str, format: Literal["CP2K"], slt_group_name: str, displacement_number: int, step: float, accoustic_sum_rule: Literal["symmetric", "self_term", "without"] = "symmetric", born_charges: bool = False, force_files_suffix: Optional[str] = None, dipole_momenta_files_suffix: Optional[str] = None) -> SltGroup:
        """
        Computes the Hessian (second-order force constants) from finite displacement
        calculations and saves it into the .slt file.

        Reads forces (and optionally dipole moments) from finite displacement
        calculations stored in the specified directory, constructs the Hessian matrix,
        and saves it under the given group name in the .slt file.

        Parameters
        ----------
        dirpath : str
            Path to the directory containing the finite displacement calculation
            results.
        format : str
            Format of the finite displacement calculations. Currently, only 'CP2K'
            is supported.
        slt_group_name : str
            Name of the SltGroup where the computed Hessian will be stored in the
            .slt file.
        displacement_number : int
            Number of displacement steps in each direction used in the finite
            difference stencil.
        step : float
            Magnitude of each displacement step in Angstroms.
        acoustic_sum_rule : str, optional
            Method to enforce the acoustic sum rule on the Hessian matrix. Options
            are:
            - 'symmetric' (default): Enforces the sum of forces to be zero
            symmetrically.
            - 'self_term': Subtracts the sum from the diagonal terms (self terms).
            - 'without': Does not enforce the acoustic sum rule.
        born_charges : bool, optional
            If True, includes Born effective charges (dipole moment derivatives)
            in the calculation and saves them. Default is False.
        force_files_suffix : str, optional
            Suffix of the force files to be read. If None, default suffix is used
            based on the specified format.
        dipole_momenta_files_suffix : str, optional
            Suffix of the dipole moment files to be read. If None, default suffix
            is used based on the specified format.

        Returns
        -------
        SltGroup
            Returns a new SltGroup containing the Hessian and (if applicable) the
            Born effective charges.

        Notes
        -----
        The method reads forces (and optionally dipole moments) from files in the
        specified directory. The files are expected to follow a naming convention
        of `dof_{dof}_disp_{disp}{suffix}`, exactly as produced by the
        generate_*_finite_stencil_displacements methods where `{dof}` is the
        degree of freedom index, `{disp}` is the displacement step number, and 
        `{suffix}` is the file suffix (e.g., '.xyz' or '-1_0.xyz') based on the
        specified format.

        Examples
        --------
        >>> supercell.hessian_from_finite_displacements(
        ...     dirpath='finite_displacements',
        ...     format='CP2K',
        ...     slt_group_name='HessianGroup',
        ...     displacement_number=1,
        ...     step=0.01,
        ...     born_charges=True
        ... )
        """
        pass
            
    def phonon_frequencies(self, kpoint: ndarray[Union[float32, float64]] = [0, 0, 0], start_mode: Optional[int] = 0, stop_mode: Optional[int] = 0, slt_save: str = None) -> SltPhononFrequencies:
        """
        Calculates the phonon frequencies at a specified k-point.

        Computes the phonon frequencies (vibrational modes) at a given k-point
        in the Brillouin zone using the dynamical matrix derived from the
        Hessian.

        Parameters
        ----------
        kpoint : ndarray[Union[float32, float64]], optional
            The k-point in reciprocal space at which to compute the phonon
            frequencies. Given as a 3-element array. Default is [0, 0, 0] (Gamma point).
        start_mode : int, optional
            Starting index of the vibrational modes to compute. If set to zero,
            all modes starting from the lowest frequency will be computed, by default 0.
        stop_mode : int, optional
            Ending index (exclusive) of the vibrational modes to compute.
            If set to zero, all available modes up to the highest frequency will
            be computed, by default 0.
        slt_save : str, optional
            If given, the results will be saved in a group of this name to the
            corresponding .slt file, by default None.

        Returns
        -------
        SltPhononFrequencies
            A delayed computation object for phonon frequencies. This object enables
            further computation, saving, plotting, and reading from the file.
            The actual results are not immediately computed upon object creation. The
            resulting numpy arrays (after invoking `eval` or `to_numpy_arrays` methods)
            are in the form (mode_numbers, frequencies) and give mode numbers
            and frequencies in cm⁻¹ as 1D arrays.

        See Also
        --------
        slothpy.plot.phonon_frequencies : For plotting the phonon frequencies.

        Notes
        -----
        The computation uses the Hessian matrix stored in the .slt file associated with
        the group. The dynamical matrix is constructed from the Hessian and masses.

        """
        pass

    def phonon_dispersion(self, brillouin_zone_path: str = None, npoints: int = None, density: float = None, special_points: Mapping[str, Sequence[float]] = None, symmetry_eps: float = 2e-4, start_mode: int = 0, stop_mode: int = 0, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltPhononDispersion:
        """
        Calculates the phonon dispersion along a specified path in the Brillouin zone.

        Computes the phonon frequencies for a sequence of k-points along a
        specified path in the Brillouin zone, allowing visualization of the
        phonon dispersion relations.

        Parameters
        ----------
        brillouin_zone_path : str, optional
            String specifying the high-symmetry path in the Brillouin zone.
            For example, 'GXL' where 'G' stands for Gamma point.
            If None, a default path is used, by default None.
        npoints : int, optional
            Total number of k-points along the path. If None, it is determined
            automatically based on the path length and density, by default None.
        density : float, optional
            Density of k-points per reciprocal lattice unit distance. Used if
            npoints is None, by default None.
        special_points : Mapping[str, Sequence[float]], optional
            Dictionary mapping special point labels to their fractional coordinates.
            If None, standard special points are used, by default None.
        symmetry_eps : float, optional
            Tolerance for symmetry operations when generating the path.
            Default is 2e-4.
        start_mode : int, optional
            Starting index of the vibrational modes to compute. If set to zero,
            all modes starting from the lowest frequency will be computed, by default 0.
        stop_mode : int, optional
            Ending index (exclusive) of the vibrational modes to compute.
            If set to zero, all available modes up to the highest frequency will
            be computed, by default 0.
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used. If None, the
            default number from the SlothPy settings is used. See:
            slothpy.set_number_cpu(), slothpy.settings.number_cpu., by default None.
        number_threads : int, optional
            Number of threads used in multithreaded linear algebra libraries during
            the calculation. Higher values benefit from larger matrices over CPU
            parallelization. If set to zero, `number_cpu` will be used. If None, the
            default number from the SlothPy settings is used. See:
            slothpy.set_number_threads(), slothpy.settings.number_threads., by default None.
        slt_save : str, optional
            If given, the results will be saved in a group of this name to the
            corresponding .slt file, by default None.
        autotune : bool, optional
            If True, the program will automatically choose the best number of threads
            (and parallel processes) for the given number of CPUs during the calculation.
            Note that this process can take significant time, so use it for
            medium-sized calculations where it becomes necessary, by default False.

        Returns
        -------
        SltPhononDispersion
            A delayed computation object for phonon dispersion. This object enables
            further computation, saving, plotting, and reading from the file.
            The actual results are not immediately computed upon object creation.
            The resulting data (after invoking `eval`, or `to_numpy_arrays` methods)
            contains phonon frequencies along the specified path in the Brillouin zone.
            A delayed computation object for phonon frequencies. This object enables
            further computation, saving, plotting, and reading from the file.
            The actual results are not immediately computed upon object creation. The
            resulting numpy arrays (after invoking `eval` or `to_numpy_arrays` methods)
            are in the form (x, x_coords, x_labels, kpts, dispersion) where x contains
            coordinates for the dispersion plotting, x_coords stores coordinates of the
            special point labels, x_labels contain the special point labels, kpts
            is an array of k-point path in the fractional coordinates of the reciprocal
            lattice and dispersion gives frequencies in the form [kpts, modes] in cm⁻¹.

        See Also
        --------
        slothpy.plot.phonon_dispersion : For plotting the phonon dispersion relations.

        Notes
        -----
        Here, (`number_cpu` // `number_threads`) parallel processes are used to
        distribute the workload over the provided k-points along the path.

        """
        pass

    def phonon_density_of_states(self, kpoints_grid: Union[int, ndarray], start_wavenumber: float, stop_wavenumber: float, resolution: int, convolution: Optional[Literal["lorentzian", "gaussian"]] = None, fwhm: float = 3, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltPhononDensityOfStates:
        """
        Calculates the phonon density of states (DOS).

        Computes the phonon density of states over a range of wavenumbers by
        sampling the Brillouin zone using a specified k-point grid and
        applying broadening techniques.

        Parameters
        ----------
        kpoints_grid : int or ndarray
            If an integer, specifies the size of a Monkhorst-Pack grid in
            each reciprocal lattice direction. If an ndarray, specifies the
            custom k-point grid in the form [[kx, ky, kz]...].
        start_wavenumber : float
            Starting wavenumber (in cm⁻¹) for the DOS calculation.
        stop_wavenumber : float
            Ending wavenumber (in cm⁻¹) for the DOS calculation.
        resolution : int
            Number of points between `start_wavenumber` and `stop_wavenumber`.
        convolution : {'lorentzian', 'gaussian'}, optional
            Type of convolution to apply for broadening the DOS.
            Options are 'lorentzian' or 'gaussian'. Default is None (no broadening).
        fwhm : float, optional
            Full width at half maximum (FWHM) for the convolution function,
            in cm⁻¹. Default is 3 cm⁻¹.
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used. If None, the
            default number from the SlothPy settings is used. See:
            slothpy.set_number_cpu(), slothpy.settings.number_cpu., by default None.
        number_threads : int, optional
            Number of threads used in multithreaded linear algebra libraries during
            the calculation. Higher values benefit from larger matrices over CPU
            parallelization. If set to zero, `number_cpu` will be used. If None, the
            default number from the SlothPy settings is used. See:
            slothpy.set_number_threads(), slothpy.settings.number_threads., by default None.
        slt_save : str, optional
            If given, the results will be saved in a group of this name to the
            corresponding .slt file, by default None.
        autotune : bool, optional
            If True, the program will automatically choose the best number of threads
            (and parallel processes) for the given number of CPUs during the calculation.
            Note that this process can take significant time, so use it for
            medium-sized calculations where it becomes necessary, by default False.

        Returns
        -------
        SltPhononDensityOfStates
            A delayed computation object for phonon density of states. This object
            allows for further computation, saving, plotting, and reading from the file.
            The actual results are not immediately computed upon object creation. The
            resulting numpy arrays (after invoking `eval` or `to_numpy_arrays` methods)
            are in the form (kpoints_grid, bin_edges, histogram) or (kpoints_grid,
            bin_edges, histogram, frequency_range, convolution),  when convolution is
            not None, where kpoints_grid contains the k-point grid used for the
            calculation [[x,y,z], ...] in the fractional coordinates of the reciprocal
            lattice, bin_edges is bin edges in cm⁻¹ for the histogram of phonon DOS,
            histogram stores normalized histogram of the phonon DOS, frequency_range
            contain frequencies (from start, stop wavenumber) in cm⁻¹ for the DOS
            plotting and convolution gives normalized DOS values with given broadening
            and FWHM.

        See Also
        --------
        slothpy.plot.phonon_density_of_states : For plotting the phonon DOS.

        Notes
        -----
        The k-point sampling grid is used to compute the DOS by integrating over the
        Brillouin zone. The convolution parameters allow for broadening of the discrete
        frequencies to simulate experimental conditions.

        Here, (`number_cpu` // `number_threads`) parallel processes are used to
        distribute the workload over the provided k-points.

        """
        pass

    def ir_spectrum(self, start_wavenumber: float, stop_wavenumber: float, convolution: Optional[Literal["lorentzian", "gaussian"]] = None, fwhm: float = 3, resolution: Optional[int] = None, slt_save: Optional[str] = None) -> SltIrSpectrum:
        """
        Calculates the infrared (IR) absorption spectrum.

        Computes the IR spectrum by calculating the frequencies and intensities
        of vibrational modes that are IR-active, using the Born effective charges.

        Parameters
        ----------
        start_wavenumber : float
            Starting wavenumber (in cm⁻¹) for the IR spectrum.
        stop_wavenumber : float
            Ending wavenumber (in cm⁻¹) for the IR spectrum.
        convolution : {'lorentzian', 'gaussian'}, optional
            Type of convolution to apply for broadening the spectrum.
            Options are 'lorentzian' or 'gaussian'. Default is None (no broadening).
        fwhm : float, optional
            Full width at half maximum (FWHM) for the convolution function,
            in cm⁻¹. Default is 3 cm⁻¹.
        resolution : int, optional
            Number of points between `start_wavenumber` and `stop_wavenumber`.
            If None, a default value is used, by default None.
        slt_save : str, optional
            If given, the results will be saved in a group of this name to the
            corresponding .slt file, by default None.

        Returns
        -------
        SltIrSpectrum
            A delayed computation object for the infrared spectrum. This object allows
            for further computation, saving, plotting, and reading from the file.
            The actual results are not immediately computed upon object creation. The
            resulting numpy arrays (after invoking `eval` or `to_numpy_arrays` methods)
            are in the form (frequencies_intensities,) or (frequencies_intensities,
            frequency_range_convolution) when convolution is not None, where
            frequencies_intensities is a 2D array with shape (number_modes, 5) in the
            form [(0-frequencies, 1-x, 2-y, 3-z, 4-average), mode] so the first row
            gives mode frequencies in in cm⁻¹, second to fourth x, y, z, and average
            intensities, while frequency_range_convolution is in the form
            [(0-frequencies, 1-x, 2-y, 3-z, 4-average), wavenumber,] so that the first
            row consists of frequencies range in cm⁻¹ and the rest is x, y, z, and
            average convolution.

        See Also
        --------
        slothpy.plot.ir_spectrum : For plotting the IR spectrum.

        Notes
        -----
        The computation requires the Born effective charges to be present in the
        Hessian group in .slt file. The IR intensities are computed from the transition
        dipole moments associated with the vibrational modes. The returned intensities
        and convolutions are normalizes within group x, y, z considering maximal
        element among all components while the average is normalized separately.

        """
        pass

    def animate_normal_modes(self, modes_list: list[int], output_directory: str, kpoint: ndarray[Union[float32, float64]] = [0, 0, 0], frames: int = 60, amplitude: float = 0.8, output_prefix: str = "", output_format: Literal["xyz", "pdb"] = "pdb") -> None:
        """
        Creates animations of specified normal modes.

        Generates animated trajectories for the specified normal modes by
        displacing atoms according to the eigenvectors of the modes and
        saves the animations in the specified format.

        Parameters
        ----------
        modes_list : list[int]
            List of mode indices to animate. Mode indices start from 0.
        output_directory : str
            Directory where the animation files will be saved.
        kpoint : ndarray[Union[float32, float64]], optional
            The k-point in reciprocal space for which the normal modes are
            calculated. Default is [0, 0, 0] (Gamma point).
        frames : int, optional
            Number of frames in the animation. Default is 60.
        amplitude : float, optional
            Amplitude of atomic displacements in the animation. Default is 0.8 Å.
        output_prefix : str, optional
            Prefix for the output file names. Default is an empty string.
        output_format : {'xyz', 'pdb'}, optional
            Format of the output animation files. Options are 'xyz' or 'pdb'.
            Default is 'pdb'.

        Returns
        -------
        None
            No immediate results are returned. The animation (trajectory) files are
            saved to the specified output directory for further viewing and analysis.

        See Also
        --------

        Notes
        -----
        The animations are generated by displacing the atomic positions along the
        eigenvectors of the specified normal modes. The resulting animations can
        be viewed using compatible molecular visualization software. For 'xyz' we
        recomend VMD while for 'pdb' files topology + animation Blender's module
        MolecularNodes: https://bradyajohnston.github.io/MolecularNodes/.

        """
        pass

    def states_energies_cm_1(self, start_state=0, stop_state=0, slt_save=None) -> SltStatesEnergiesCm1: pass
    
    def states_energies_au(self, start_state=0, stop_state=0, slt_save=None) -> SltStatesEnergiesAu: pass
    
    def spin_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltSpinMatrices: pass

    def states_spins(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesSpins: pass

    def angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltAngularMomentumMatrices: pass
    
    def states_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesAngularMomenta: pass

    def electric_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltElectricDipoleMomentumMatrices: pass

    def states_electric_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesElectricDipoleMomenta: pass

    def total_angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltTotalAngularMomentumMatrices: pass

    def states_total_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesTotalAngularMomenta: pass

    def magnetic_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltMagneticDipoleMomentumMatrices: pass

    def states_magnetic_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesMagneticDipoleMomenta: pass

    def _slt_hamiltonian_from_slt_group(self, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> SltHamiltonian : pass

    def zeeman_splitting(self, magnetic_fields: ndarray[Union[float32, float64]], orientations: ndarray[Union[float32, float64]], number_of_states: int = 0, states_cutoff: int = [0, "auto"], rotation: ndarray = None, electric_field_vector: ndarray = None, hyperfine: dict = None, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltZeemanSplitting:
        """
        Calculates directional or powder-averaged Zeeman splitting for a given
        number of states and a list of magnetic field orientations and values.

        Parameters
        ----------
        hamiltonian_group_name : str
            Name of a Hamiltonian-type group from the corresponding .slt file
            which will be used to compute the Zeeman splitting.
        number_of_states : int
            Number of states whose energy splitting will be given in the result
            array. If set to zero, states_cutoff[0] states will be given.,
            by default 0
        magnetic_fields : ndarray[Union[float32, float64]]
            ArrayLike structure (can be converted to numpy.NDArray) of magnetic
            field values (T) at which Zeeman splitting will be computed.
        orientations : Union[ndarray[Union[float32, float64]], int, list]
            ArrayLike structure (can be converted to numpy.NDArray) of magnetic
            field orientations in the list format: [[direction_x, direction_y,
            direction_z],...]. Users can choose from predefined orientational 
            grids providing a list or tuple in the form [gird_name,
            number_of points] where grid_name can be: 'fibonacci', 'mesh',
            'lebedev_laikov', and number_of_points is an integer controlling
            the grid density. If the orientations is set to an integer from
            0-11, the prescribed Lebedev-Laikov grids over the hemisphere will
            be used (see slothpy.lebedev_laikov_grid_over_hemisphere
            documentation) and powder-averaging will be performed. Otherwise,
            the user can provide an ArrayLike structure with the convention:
            [[direction_x, direction_y, direction_z, weight],...] for
            powder-averaging over the chosen directions using the provided
            weights. Custom grids and orientations will be automatically
            normalized to the unit directional vectors.
        states_cutoff : list, optional
            List of integers of length 2 where the first one represents the
            number of states that will be taken into account for construction
            of the Zeeman Hamiltonian. If it is set to zero, all available
            states from the Hamiltonian group will be used. The second integer
            controls the number of eigenvalues and eigenvectors to be found
            during Hamiltonian diagonalization. For methods taking
            number_of_states parameter, the second entry must be at least equal
            to the desired number of states, while for functions including
            temperature dependencies, it should be large enough to accommodate
            states with energies relevant to a given range of temperatures.
            You can let SlothPy decide automatically on the optimal value of
            the second parameter setting it to 'auto'. Note that when using
            custom SlothPy-type exchange Hamiltonian, this argument has no
            effect since the cutoff scheme is already defined while creating
            one (see the documentation of SltFile.hamiltonian creation method).
            , by default [0, 'auto']
        rotation: Union[ndarray[Union[float32, float64]], SltRotation, Rotation], optional
            A (3,3) orthogonal rotation matrix, instance of SltRotation class
            or scipy.spatial.transform.Rotation (only single rotation) used to
            rotate operators matrices. Note that the inverse matrix has to be
            given to rotate the reference frame instead.,by default None
        electric_field_vector: ndarray[Union[float32, float64]], optional
            ArrayLike structure (can be converted to numpy.NDArray)
            representing a vector [Fx, Fy, Fz] of static uniform electric field
            (V/m) to be included in the Hamiltonian., by default None
        hyperfine: dict, optional
            Hyperfine interactions are not available yet and are scheduled to
            be implemented in the upcoming 0.4 major release, by default None
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used. If None, the 
            default number from the SlothPy settings is used. See:
            slothpy.set_number_cpu(), slothpy.settings.number_cpu.,
            by default None
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries during the calculation. Higher values benefit
            from the increasing size of matrices (states_cutoff) over the
            parallelization over CPUs. If set to zero, a number_cpu will be
            used. If None, the default number from the SlothPy settings is used.
            See: slothpy.set_number_threads, slothpy.settings.number_threads.,
            by default None
        slt_save : str, optional
            If given, the results will be saved in a group of this name to the
            corresponding .slt file., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 300 with
            dense grids or a higher number of field values) where it becomes
            a necessity., by default False

        Returns
        -------
        SltZeemanSplitting
            A delayed computation object for Zeeman splitting. This object enables
            further computation, saving and plotting. The actual results are not
            immediately computed upon object creation.
            The resulting numpy array (after invoking `eval` or
            `to_numpy_arrays` methods) gives Zeeman splitting of number_of_states
            energy levels in cm⁻¹ for each orientation in the form
            [orientations, fields, energies] - the first dimension
            runs over different orientations, the second over field values, and
            the last gives energies of number_of_states states, unless the
            orientations argument is of 'mesh' type, then the retuened array is
            in the form [mesh, mesh, fields, temperatures] - the first two
            dimensions are in the form of meshgrids over theta and phi angles,
            ready to be combined with xyz orientational meshgrids for 3D plots
            (see slothpy.meshgrid_over_hemisphere documentation).
            When the powder-average calculation is performed, the array is
            returned in the form: [fields, energies].

        See Also
        --------
        slothpy.plot.zeeman, sltohpy.SltRotation, 
        slothpy.fibonacci_over_hemisphere, slothpy.meshgrid_over_hemisphere,
        slothpy.lebedev_laikov_grid_over_hemisphere : For the description of
                                        the prescribed orientations grids.

        Note
        -----
        Here, (`number_cpu` // `number_threads`) parallel processes are used to
        distribute the workload over the provided field and orientation values.
        """
        pass

    def magnetisation(self, magnetic_fields: ndarray[Union[float32, float64]], orientations: ndarray[Union[float32, float64]], temperatures: ndarray[Union[float32, float64]], states_cutoff: int = [0, "auto"], rotation: ndarray = None, electric_field_vector: ndarray = None, hyperfine: dict = None, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltMagnetisation:
        pass


####################
# Hamiltonian Groups
####################


class SltHamiltonian(metaclass=MethodTypeMeta): # here you can only leave *args and *kwargs in arguments
    _method_type = "HAMILTONIAN"

    __slots__ = ["_slt_group", "_states"]

    def __init__(self, slt_group: SltGroup, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> None: ### w sumie tutaj tez moze byc modyfikowanie parametrow jak z delayed methods bedziesz robil SltHamiltonian() z init i lepiej bo nie idzie przez delegowanie i input parser
        self._slt_group: str = slt_group
    
    def _slt_hamiltonian_from_slt_group(self, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> SltHamiltonian : pass
    ####################### Implement THIS!!!!!!!!!!!!!! ################### to include generation for methods and rotation (without local states or keep them to have the same calling convention in methods)
    # then also the following methods must be implmented to return packed data for methods to shared memory

    # @property
    # def arrays_to_shared_memory(self):
    #     arrays = [item for property in self._mode for item in getattr(self, property)]
    #     if len(self._magnetic_centers.keys()) > 1:
    #         arrays.append(self.interaction_matrix)
    #     return arrays
    
    # @property
    # def info(self):
    #     info_list = []
    #     for i in range(len(self._magnetic_centers.keys())):
    #         info_list.append(self._magnetic_centers[i][1])
    #     return (self._mode, info_list, self._local_states)

    def e(self): return self._slt_group["STATES_ENERGIES"]

    def s(self): return SltDatasetSLP(self._slt_group, "S")
    
    def sx(self): return SltDatasetSLP(self._slt_group, "S", 0)

    def sy(self): return SltDatasetSLP(self._slt_group, "S", 1)
    
    def sz(self): return SltDatasetSLP(self._slt_group, "S", 2)
    
    def l(self): return SltDatasetSLP(self._slt_group, "L")

    def lx(self): return SltDatasetSLP(self._slt_group, "L", 0)
    
    def ly(self): return SltDatasetSLP(self._slt_group, "L", 1)

    def lz(self): return SltDatasetSLP(self._slt_group, "L", 2)
    
    def p(self): return SltDatasetSLP(self._slt_group, "P")
    
    def px(self): return SltDatasetSLP(self._slt_group, "P", 0)

    def py(self): return SltDatasetSLP(self._slt_group, "P", 1)

    def pz(self): return SltDatasetSLP(self._slt_group, "P", 2)
    
    def j(self): return SltDatasetJM(self._slt_group, "J")
    
    def jx(self): return SltDatasetJM(self._slt_group, "J", 0)

    def jy(self): return SltDatasetJM(self._slt_group, "J", 1)

    def jz(self): return SltDatasetJM(self._slt_group, "J", 2)

    def m(self): return SltDatasetJM(self._slt_group, "M")

    def mx(self): return SltDatasetJM(self._slt_group, "M", 0)

    def my(self): return SltDatasetJM(self._slt_group, "M", 1)

    def mz(self): return SltDatasetJM(self._slt_group, "M", 2) ## from here args kwargs

    def states_energies_cm_1(self, start_state=0, stop_state=0, slt_save=None): return SltStatesEnergiesCm1(self._slt_group, start_state, stop_state, slt_save)
    
    def states_energies_au(self, start_state=0, stop_state=0, slt_save=None): return SltStatesEnergiesAu(self._slt_group, start_state, stop_state, slt_save)
    
    def spin_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltSpinMatrices(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)
    
    def states_spins(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltStatesSpins(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)
    
    def angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltAngularMomentumMatrices(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)
    
    def states_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltStatesAngularMomenta(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)

    def electric_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltElectricDipoleMomentumMatrices(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)
    
    def states_electric_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltStatesElectricDipoleMomenta(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)

    def total_angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltTotalAngularMomentumMatrices(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)

    def states_total_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltStatesTotalAngularMomenta(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)

    def magnetic_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltMagneticDipoleMomentumMatrices(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)

    def states_magnetic_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None): return SltStatesMagneticDipoleMomenta(self._slt_group, xyz, start_state, stop_state, rotation, slt_save)
    
    def zeeman_splitting(self, magnetic_fields: ndarray[Union[float32, float64]], orientations: ndarray[Union[float32, float64]], number_of_states: int = 0, states_cutoff: int = [0, "auto"], rotation: ndarray = None, electric_field_vector: ndarray = None, hyperfine: dict = None, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltZeemanSplitting:
        return SltZeemanSplitting(self._slt_group, magnetic_fields, orientations, number_of_states, states_cutoff, rotation, electric_field_vector, hyperfine, number_cpu, number_threads, autotune, slt_save)
    
    def magnetisation(self, magnetic_fields: ndarray[Union[float32, float64]], orientations: ndarray[Union[float32, float64]], temperatures: ndarray[Union[float32, float64]], states_cutoff: int = [0, "auto"], rotation: ndarray = None, electric_field_vector: ndarray = None, hyperfine: dict = None, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltMagnetisation:
        return SltMagnetisation(self._slt_group, magnetic_fields, orientations, temperatures, states_cutoff, rotation, electric_field_vector, hyperfine, number_cpu, number_threads, autotune, slt_save)


class SltExchangeHamiltonian(SltHamiltonian):
    _method_type = "EXCHANGE_HAMILTONIAN" ################################################ INPUT PARSER !!!!!!!!!!!!!! slt_group.type == "EXCHANGE_HAMILTONIAN" a nie "SLOTHPY" itd.

    __slots__ = ["_slt_group", "_hdf5", "_magnetic_centers", "_exchange_interactions", "_states", "_electric_dipole", "_magnetic_interactions", "_electric_interactions", "_mode", "_local_states"]

    def __init__(self, slt_group: SltGroup, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> None: ##### Here those other options are not needed probably
        self._slt_group = slt_group ### Only This has to stay
        self._hdf5: str = slt_group._hdf5
        self._magnetic_centers, self._exchange_interactions, self._states, self._electric_dipole, self._magnetic_interactions, self._electric_interactions, self._local_states = self._retrieve_hamiltonian_dict(states_cutoff, rotation, hyperfine, local_states)
        self._mode: str = None # "eslpjm"

        ############## Here it must  have self._slt_group
    def _retrieve_hamiltonian_dict(self, states_cutoff=[0,0], rotation=None, hyperfine=None, coordinates=None, local_states=True): ##################### here you must rotate also !!! every rotation in dict!!!!!!!!!!!!!!!!!!!
        ################################# Up rotation ################################################
        states = self._slt_group.attributes["States"]
        electric_dipole = False
        magnetic_interactions = False
        electric_interactions = False
        try:
            if self._slt_group.attributes["Additional"] == "ELECTRIC_DIPOLE_MOMENTA":
                electric_dipole = True
        except SltFileError:
            pass
        try:
            if "m" in self._slt_group.attributes["Interactions"]:
                magnetic_interactions = True
            if "p" in self._slt_group.attributes["Interactions"]:
                electric_interactions = True
        except SltFileError:
            pass
        if self._slt_group.attributes["Kind"] == "SLOTHPY": ######### This has to go away because we only load it as ExchangeHamiltonian now
            with File(self._slt_group._hdf5, 'r') as file:
                group = file[self._slt_group._group_name]
                
                def load_dict_from_group(group, subgroup_name):
                    data_dict = {}
                    subgroup = group[subgroup_name]
                    for key in subgroup.keys():
                        value = subgroup[key][()]
                        original_key = literal_eval(key.rsplit('_', 1)[0])
                        if original_key not in data_dict.keys():
                            data_dict[original_key] = []
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                            if value == 'None':
                                value = None
                        elif isinstance(value, ndarray) and value.shape == ():
                            value = value.item()
                        data_dict[original_key].append(value)
                    return data_dict
                
                magnetic_centers = load_dict_from_group(group, "MAGNETIC_CENTERS")
                if not local_states:
                    for center in magnetic_centers.values():
                        center[1][0] = center[1][1]
                exchange_interactions = load_dict_from_group(group, "EXCHANGE_INTERACTIONS")
        else:
            magnetic_centers = {0:(self._slt_group._group_name, (states_cutoff[0],0,states_cutoff[1]), rotation, coordinates, hyperfine)} ### no else 
            exchange_interactions = None
        
        return magnetic_centers, exchange_interactions, states, electric_dipole, magnetic_interactions, electric_interactions, local_states

    def _slt_hamiltonian_from_slt_group(self, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True):
            return SltExchangeHamiltonian(self._slt_group, states_cutoff, rotation, hyperfine, local_states)

    def _compute_data(self, matrix_class):
        data = []
        for center in self._magnetic_centers.values():
            arr = matrix_class(SltGroup(self._hdf5, center[0]), stop_state=center[1][0], rotation=center[2]).eval()
            conjugate(arr, out=arr)
            data.append(arr)
        return data

    @property
    def e(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltStatesEnergiesAu(SltGroup(self._hdf5, center[0]), stop_state=center[1][0]).eval())
        return data

    @property
    def s(self):
        return self._compute_data(SltSpinMatrices)

    @property
    def l(self):
        return self._compute_data(SltAngularMomentumMatrices)

    @property
    def p(self):
        return self._compute_data(SltElectricDipoleMomentumMatrices)

    @property
    def j(self):
        return self._compute_data(SltTotalAngularMomentumMatrices)

    @property
    def m(self):
        return self._compute_data(SltMagneticDipoleMomentumMatrices)

    @property
    def interaction_matrix(self):
        result = zeros((self._states, self._states), dtype=settings.complex)
        n = len(self._magnetic_centers.keys())
        if not any(value[3] is None for value in self._magnetic_centers.values()) and self._magnetic_interactions:
            dipole_magnetic_momenta_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).magnetic_dipole_momentum_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
            result = self._add_dipole_interaction(dipole_magnetic_momenta_dict, n, U_PI_A_AU, result)
            if self._electric_dipole and self._electric_interactions:
                dipole_electric_momenta_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).electric_dipole_momentum_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
                result = self._add_dipole_interaction(dipole_electric_momenta_dict, n, E_PI_A_AU, result)

        for (key1, key2), J in self._exchange_interactions.items():
            spin_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).spin_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
            for l in range(3):
                for m in range(3):
                    coeff = - J[0][l, m]
                    if abs(coeff) < 1e-13:
                        continue
                    op1 = coeff * spin_dict[key1][l]
                    op2 = spin_dict[key2][m]
                    ops = [op1 if k == key1 else op2 if k == key2 else spin_dict[k].shape[1] for k in range(n)]
                    result += _kron_mult(ops)
        
        result_tmp = result.copy()
        energy_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).states_energies_au(stop_state=self._magnetic_centers[key][1][1]).eval().astype(result.dtype) for key in self._magnetic_centers.keys()}
        for i in range(n):
            ops = [diag(energy_dict[k]) if k == i else energy_dict[k].shape[0] for k in range(n)]
            result_tmp += _kron_mult(ops)
        eigenvalues = eigvalsh(result_tmp, driver="evr", check_finite=False, overwrite_a=True, overwrite_b=True)
        _subtract_const_diagonal(result, eigenvalues[0])

        #TODO: hyperfine interactions and different types of interactions (J,L???)
        return result
    
    def _add_dipole_interaction(self, dipole_momenta_dict, n, coeff, result):
        for key1 in self._magnetic_centers.keys():
            for key2 in range(key1+1, n):
                r_vec = self._magnetic_centers[key1][3] - self._magnetic_centers[key2][3]
                r_norm = norm(r_vec)
                if r_norm <= 1e-2:
                    raise ValueError("Magnetic centers are closer than 0.01 Angstrom. Please double-check the SlothPy Hamiltonian dictionary. Quitting here.")
                coeff = coeff / r_norm ** 3
                r_vec = r_vec / r_norm
                op1 = tensordot(dipole_momenta_dict[key1], - 3. * coeff * r_vec ,axes=(0, 0))
                op2 = tensordot(dipole_momenta_dict[key2], r_vec, axes=(0, 0))
                ops = [op1 if k == key1 else op2 if k == key2 else dipole_momenta_dict[k].shape[1] for k in range(n)]
                result += _kron_mult(ops)
                for i in range(3):
                    ops[key1] = coeff * dipole_momenta_dict[key1][i]
                    ops[key2] = dipole_momenta_dict[key2][i]
                    result += _kron_mult(ops)
        
        return result

    @property
    def arrays_to_shared_memory(self):
        arrays = [item for property in self._mode for item in getattr(self, property)]
        if len(self._magnetic_centers.keys()) > 1:
            arrays.append(self.interaction_matrix)
        return arrays
    
    @property
    def info(self):
        info_list = []
        for i in range(len(self._magnetic_centers.keys())):
            info_list.append(self._magnetic_centers[i][1])
        return (self._mode, info_list, self._local_states)


#TODO From here consider using the input parser on top of the methods and move parts with validation to it


#################
# Topology Groups
#################


class SltXyz(metaclass=MethodTypeMeta):
    _method_type = "XYZ"

    __slots__ = ["_slt_group", "_atoms", "_charge", "_multiplicity"]

    def __init__(self, slt_group: SltGroup) -> None:
        self._slt_group = slt_group
        elements = slt_group["ELEMENTS"][:]
        self._atoms = Atoms(elements, slt_group["COORDINATES"][:])
        self._charge = None
        self._multiplicity = None
        if "Charge" in slt_group.attributes:
            self._charge = slt_group.attributes["Charge"]
        if "Multiplicity" in slt_group.attributes:
            self._multiplicity = slt_group.attributes["Multiplicity"]

    def atoms_object(self):
        return self._atoms
    
    def charge(self):
        return self._charge
    
    def multiplicity(self):
        return self._multiplicity

    def to_xyz(self, xyz_filepath: str):
        additional_info = ""
        if self._charge is not None:
            additional_info += f"Charge: {self._charge} "
        if self._multiplicity is not None:
            additional_info += f"Multiplicity: {self._multiplicity} "
        if self._method_type in ["UNIT_CELL", "SUPERCELL"]:
            additional_info += f"{"Cell" if self._method_type == "UNIT_CELL" else "Supercell"} parameters [a, b, c, alpha, beta, gamma]: {self._atoms.get_cell_lengths_and_angles().tolist()} "
        if self._method_type == "SUPERCELL":
            additional_info += f"Supercell_Repetitions [nx, ny, nz]: {self._nxnynz.tolist()} "
        
        write(xyz_filepath, self._atoms, comment=f"{additional_info}Created by SlothPy from File/Group '{self._slt_group._hdf5}/{self._slt_group._group_name}'")

    def replace_atoms(self, atom_indices: List[int], new_symbols: List[str]) -> None:
        current_symbols = self._atoms.get_chemical_symbols()
        for idx, new_sym in zip(atom_indices, new_symbols):
            print(f"Replacing atom at index {idx} ({current_symbols[idx]}) with '{new_sym}'.")
            current_symbols[idx] = new_sym
        self._atoms.set_chemical_symbols(current_symbols)
        
        try:
            elements_ds = self._slt_group["ELEMENTS"]
            elements = elements_ds[:]
            for idx, new_sym in zip(atom_indices, new_symbols):
                elements[idx] = new_sym
            elements_ds[:] = elements
            print(f"'ELEMENTS' dataset successfully updated in group '{self._slt_group._group_name}'.")
        except Exception as exc:
            raise SltFileError(self._slt_group._hdf5, exc, f"Failed to update 'ELEMENTS' dataset in the .slt group") from None
        
        return self._slt_group
        
    def generate_finite_stencil_displacements(self, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None, _supercell: bool = False, _nx: Optional[int] = None, _ny: Optional[int] = None, _nz: Optional[int] = None, _dof_dict: Optional[dict] = None) -> Optional[Iterator[Atoms]]:
        if output_option == 'xyz':
            makedirs(custom_directory, exist_ok=True)

        if self._method_type == "SUPERCELL":
            num_atoms = len(self._atoms) // self._nxnynz.prod()
        else:
            num_atoms = len(self._atoms)

        total_dofs = 3 * num_atoms
        n_checked = False
        if _supercell:
            n_checked = _check_n(_nx, _ny, _nz)
        if n_checked:
            atoms_tmp = self._atoms.repeat((_nx, _ny, _nz))
        else:
            atoms_tmp = self._atoms

        if self._method_type == "SUPERCELL":
            _nx, _ny, _nz = self._nxnynz
            _nx, _ny, _nz = int(_nx), int(_ny), int(_nz)
            n_checked = True

        def displacement_generator() -> Iterator[Atoms]:
            zero_geometry_flag = False
            if _dof_dict is not None:
                dof_range = _dof_dict.keys()
            else:
                dof_range = range(total_dofs)
            for dof in dof_range:
                axis = dof % 3
                atom_idx = dof // 3
                dof_mapped = dof if _dof_dict is None else _dof_dict[dof]
                for multiplier in range(-displacement_number, displacement_number + 1):
                    if multiplier == 0 and zero_geometry_flag:
                        continue
                    elif multiplier == 0:
                        zero_geometry_flag = True
                        yield (atoms_tmp, 0, multiplier, _nx, _ny, _nz) if n_checked else (atoms_tmp, 0, multiplier)
                        continue
                    displaced_atoms = atoms_tmp.copy()
                    displacement = multiplier * step
                    displaced_atoms.positions[atom_idx, axis] += displacement
                    yield (displaced_atoms, dof_mapped, multiplier, step, _nx, _ny, _nz) if n_checked else (displaced_atoms, dof_mapped, multiplier, step)

        if output_option == 'xyz':
            for displacement_info in displacement_generator():
                displaced_atoms, dof, multiplier = displacement_info[0], displacement_info[1], displacement_info[2]
                xyz_file_name = f"dof_{dof}_disp_{multiplier}.xyz"
                xyz_file_path = join(custom_directory, xyz_file_name)
                try:
                    additional_info = f"Step: {step} Displacement_Number: {displacement_number} "
                    if self._charge is not None:
                        additional_info += f"Charge: {self._charge} "
                    if self._multiplicity is not None:
                        additional_info += f"Multiplicity: {self._multiplicity} "
                    if n_checked:
                        additional_info += f"Supercell_Repetitions [nx, ny, nz]: {[_nx, _ny, _nz]} "
                    if self._method_type in ["UNIT_CELL", "SUPERCELL"]:
                        additional_info += f"{"Cell" if self._method_type == "UNIT_CELL" and not _supercell else "Supercell"} parameters [a, b, c, alpha, beta, gamma]: {atoms_tmp.get_cell_lengths_and_angles().tolist()} "
                    write(xyz_file_path, displaced_atoms, comment=f"{additional_info}Created by SlothPy from File/Group '{self._slt_group._hdf5}/{self._slt_group._group_name}")
                except Exception as exc:
                    raise IOError(f"Failed to write XYZ file '{xyz_file_path}': {exc}")
            return None

        elif output_option == 'iterator':
            return (displacement_info for displacement_info in displacement_generator())

        elif output_option == 'slt':
            try:
                with File(self._slt_group._hdf5, 'a') as slt:
                    displacement_group = slt.create_group(slt_group_name)
                    displacement_group.attrs["Type"] = "DISPLACEMENTS_XYZ" if not n_checked else "DISPLACEMENTs_SUPERCELL"
                    displacement_group.attrs["Description"] = "Group containing displaced XYZ coordinates groups."
                    displacement_group.attrs["Displacement_Number"] = displacement_number
                    displacement_group.attrs["Step"] = step
                    displacement_group.attrs["Original_Group"] = self._slt_group._group_name
                    if n_checked:
                        displacement_group.attrs["Supercell_Repetitions"] = [_nx, _ny, _nz]
                for displacement_info in displacement_generator():
                    displaced_atoms, dof, multiplier = displacement_info[0], displacement_info[1], displacement_info[2]
                    subgroup_name = f"{slt_group_name}/dof_{dof}_disp_{multiplier}"
                    _xyz_to_slt(self._slt_group._hdf5, subgroup_name, displaced_atoms.get_chemical_symbols(), displaced_atoms.get_positions(), self._charge, self._multiplicity)
            except Exception as exc:
                raise SltFileError(self._slt_group._hdf5, exc, f"Failed to write displacement Group '{slt_group_name}' to the .slt file") from None
            return SltGroup(self._slt_group._hdf5, slt_group_name)

    def generate_finite_stencil_displacements_reduced_to_unit_cell(self, unit_cell_group_name: str, central_atom: ndarray[Union[float32, float64]], displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        xyz_atoms_postions = self.atoms_object().get_positions()
        unit_cell = SltGroup(self._slt_group._hdf5, unit_cell_group_name).atoms_object
        unit_cell_atom_postions = unit_cell.get_positions()
        cell_vectors = unit_cell.get_cell()

        distances = norm(xyz_atoms_postions - central_atom, axis=1)
        difference = unit_cell_atom_postions[newaxis, :, :] - xyz_atoms_postions[:, newaxis, :]
        n_vectors = einsum('ijk,kl->ijl', difference, inv(cell_vectors))
        check_vectors = is_approximately_integer(n_vectors, 0.005)
        matching_indices = where(all(check_vectors, axis=2))
        match_candidates = list(zip(distances[matching_indices[0]], matching_indices[0], matching_indices[1]))
        match_candidates.sort()

        assigned_atoms_xyz = set()
        assigned_atoms_unit_cell = set()

        dof_dict = {}

        for (distance, atom_xyz, atom_unit_cell) in match_candidates:
            if atom_xyz not in assigned_atoms_xyz and atom_unit_cell not in assigned_atoms_unit_cell:
                dof_dict[3 * atom_xyz] = 3 * atom_unit_cell
                dof_dict[3 * atom_xyz + 1] = 3 * atom_unit_cell + 1
                dof_dict[3 * atom_xyz + 2] = 3 * atom_unit_cell + 2
                assigned_atoms_xyz.add(atom_xyz)
                assigned_atoms_unit_cell.add(atom_unit_cell)

        return self.generate_finite_stencil_displacements(displacement_number, step, output_option, custom_directory, slt_group_name, _dof_dict=dof_dict)


class SltUnitCell(SltXyz):
    _method_type = "UNIT_CELL"

    __slots__ = SltXyz.__slots__

    def __init__(self, slt_group: SltGroup) -> None:
        super().__init__(slt_group)
        self._atoms.set_cell(slt_group["CELL"][:])

    def cell_object(self):
        return self._atoms.get_cell()
    
    def supercell(self, nx: int, ny: int, nz: int, output_option: Literal["xyz", "slt"] = "slt", xyz_filepath: Optional[str] = None, slt_group_name: Optional[str] = None) -> Optional[SltGroup]:
        if self._method_type == "SUPERCELL":
            warnings.warn("You are trying to construct a supercell out of another supercell, creating a ... mega-cell with all parameters multiplied!")

        atoms: Atoms = self._atoms.repeat((nx, ny, nz))

        multiplicity = None
        if self._multiplicity:
            multiplicity = ((self._multiplicity - 1) * nx * ny * nz) + 1

        if output_option == "xyz":
            additional_info = f"Supercell parameters [a, b, c, alpha, beta, gamma]: {atoms.get_cell_lengths_and_angles().tolist()} "
            write(xyz_filepath, self._atoms, comment=f"{additional_info}Created by SlothPy from File/Group '{self._slt_group._hdf5}/{self._slt_group._group_name}'")
        else:
            _supercell_to_slt(self._slt_group._hdf5, slt_group_name, atoms.get_chemical_symbols(), atoms.get_positions(), atoms.get_cell().array, nx, ny, nz, multiplicity)
            return SltGroup(self._slt_group._hdf5, slt_group_name)

    def generate_supercell_finite_stencil_displacements(self, nx: int, ny: int, nz: int, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None, save_supercell_to_slt: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        if self._method_type == "SUPERCELL":
            warnings.warn("You are trying to construct a supercell for finite displacements out of another supercell, creating a ... mega-cell with all parameters multiplied! If you wish to make displacements within a given supercell, use generate_finite_stencil_displacements instead.")
        if save_supercell_to_slt:
            self.supercell(nx, ny, nz, 'slt', slt_group_name=save_supercell_to_slt)
        return self.generate_finite_stencil_displacements(displacement_number, step, output_option, custom_directory, slt_group_name, True, nx, ny, nz)
    
    def show_bandpath(self, brillouin_zone_path: str = None, npoints: int = None, density: float = None, special_points: Mapping[str, Sequence[float]] = None, symmetry_eps: float = 2e-4) -> None:
        self.atoms_object().cell.bandpath(path=brillouin_zone_path, npoints=npoints, special_points=special_points, density=density, eps=symmetry_eps).plot(show=True)


class SltSuperCell(SltUnitCell):
    _method_type = "SUPERCELL"

    __slots__ = SltUnitCell.__slots__ + ["_nxnynz"]

    def __init__(self, slt_group: SltGroup) -> None:
        super().__init__(slt_group)
        self._atoms.set_cell(slt_group["CELL"][:])
        self._nxnynz = slt_group.attributes["Supercell_Repetitions"]

    def hessian_from_finite_displacements(self, dirpath: str, format: Literal["CP2K"], slt_group_name: str, displacement_number: int, step: float, accoustic_sum_rule: Literal["symmetric", "self_term", "without"] = "symmetric", born_charges: bool = False, force_files_suffix: Optional[str] = None, dipole_momenta_files_suffix: Optional[str] = None):
        dof_number = 3 * len(self._atoms) // self._nxnynz.prod()
        hessian, born_charges = _read_hessian_born_charges_from_dir(dirpath, format, dof_number, self._nxnynz[0], self._nxnynz[1], self._nxnynz[2], displacement_number, step, accoustic_sum_rule, born_charges, force_files_suffix, dipole_momenta_files_suffix)
        _hessian_to_slt(self._slt_group._hdf5, slt_group_name, self._atoms.get_chemical_symbols(), self._atoms.get_positions(), self._atoms.get_cell().array, self._nxnynz[0], self._nxnynz[1], self._nxnynz[2], self._multiplicity, hessian, born_charges)
        
        return SltGroup(self._slt_group._hdf5, slt_group_name)


###############
# Forces Groups
###############


class SltHessian(SltSuperCell):
    _method_type = "HESSIAN"

    __slots__ = SltXyz.__slots__ + ["_bandpath"]

    def __init__(self, slt_group) -> None:
        super().__init__(slt_group)

    @property
    def masses(self):
        return self.atoms_object().get_masses()[:self.hessian().shape[3]//3].astype(settings.float)

    def hessian(self) -> ndarray:
        return self._slt_group["HESSIAN"]
    
    def born_charges(self) -> ndarray:
        if _dataset_exists(self._slt_group._hdf5, self._slt_group._group_name, "BORN_CHARGES"):
            return self._slt_group["BORN_CHARGES"]
        else:
            raise RuntimeError(f"Hessian from {BLUE}Group{RESET}: '{self._slt_group._group_name}' {GREEN}File{RESET}: '{self._slt_group._hdf5}' does not have born charges loaded.")
    
    def _masses_inv_sqrt(self):
        return 1.0 / sqrt(repeat(self.masses, 3))
    
    def phonon_frequencies(self, kpoint: ndarray[Union[float32, float64]] = [0, 0, 0], start_mode: Optional[int] = None, stop_mode: Optional[int] = None, slt_save: str = None) -> SltPhononFrequencies:
        return SltPhononFrequencies(self._slt_group, self.hessian()[:], self._masses_inv_sqrt(), kpoint, start_mode, stop_mode, slt_save)

    def phonon_dispersion(self, brillouin_zone_path: str = None, npoints: int = None, density: float = None, special_points: Mapping[str, Sequence[float]] = None, symmetry_eps: float = 2e-4, start_mode: int = 0, stop_mode: int = 0, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltPhononDispersion:
        self._bandpath = self.atoms_object().cell.bandpath(path=brillouin_zone_path, npoints=npoints, special_points=special_points, density=density, eps=symmetry_eps)
        return SltPhononDispersion(self._slt_group, self.hessian()[:], self._masses_inv_sqrt(), self._bandpath, start_mode, stop_mode, number_cpu, number_threads, autotune, slt_save)

    def phonon_density_of_states(self, kpoints_grid: Union[int, ndarray], start_wavenumber: float, stop_wavenumber: float, resolution: int, convolution: Optional[Literal["lorentzian", "gaussian"]] = None, fwhm: float = 3, number_cpu: int = None, number_threads: int = None, slt_save: str = None, autotune: bool = False) -> SltPhononDensityOfStates:
        return SltPhononDensityOfStates(self._slt_group, self.hessian()[:], self._masses_inv_sqrt(), kpoints_grid, start_wavenumber, stop_wavenumber, resolution, convolution, fwhm, number_cpu, number_threads, autotune, slt_save)

    def ir_spectrum(self, start_wavenumber: float, stop_wavenumber: float, convolution: Optional[Literal["lorentzian", "gaussian"]] = None, fwhm: float = 3, resolution: Optional[int] = None, slt_save: Optional[str] = None) -> SltIrSpectrum:
        return SltIrSpectrum(self._slt_group, self.hessian()[:], self._masses_inv_sqrt(), asfortranarray(self.born_charges()[:], dtype=settings.complex), start_wavenumber, stop_wavenumber, convolution, resolution, fwhm, slt_save)

    def animate_normal_modes(self, modes_list: list[int], output_directory: str, kpoint: ndarray[Union[float32, float64]] = [0, 0, 0], frames: int = 60, amplitude: float = 0.8, output_prefix: str = "", output_format: Literal["xyz", "pdb"] = "pdb") -> None:
        
        if not exists(output_directory):
            makedirs(output_directory)

        start_mode = min(modes_list)
        stop_mode = max(modes_list) + 1
        masses_sqrt_inv = self._masses_inv_sqrt()

        hessian_object = Hessian([self.hessian()[:], outer(masses_sqrt_inv, masses_sqrt_inv)], kpoint, start_mode=start_mode, stop_mode=stop_mode, eigen_range="I", single_process=True)
        _, eigenvectors = hessian_object.frequencies_eigenvectors

        atoms = self.atoms_object()
        positions = atoms.get_positions()
        nxnynz = self._nxnynz
        N_atoms_unit_cell = len(atoms) // (nxnynz[0] * nxnynz[1] * nxnynz[2])
        positions = positions.reshape((nxnynz[0], nxnynz[1], nxnynz[2], N_atoms_unit_cell, 3))

        R_vectors = stack(meshgrid(arange(nxnynz[0]), arange(nxnynz[1]), arange(nxnynz[2]), indexing='ij'), axis=-1)

        times = linspace(0, 1, frames, endpoint=False)
        phase = exp(2j * pi * tensordot(R_vectors, kpoint, axes=([3], [0])))

        topology_filename = join(output_directory, f"{output_prefix}topology.{output_format}")
        write(topology_filename, atoms)
        print(f"Initial topology saved to '{topology_filename}'")

        for mode_index in modes_list:
            mode_internal_index = mode_index - start_mode
            eigenvector = eigenvectors[:, mode_internal_index]
            eigenvector = (eigenvector * masses_sqrt_inv).reshape((N_atoms_unit_cell, 3))
            
            trajectory = Trajectory('.tmp_slt_animation.traj', 'w')

            for t in times:
                total_phase = phase * exp(-2j * pi * t)
                displacements = amplitude * real(eigenvector[None, None, None, :, :] * total_phase[:, :, :, None, None])
                new_positions = positions + displacements

                atoms_frame = atoms.copy()
                atoms_frame.set_positions(new_positions.reshape((-1, 3)))
                trajectory.write(atoms_frame)

            trajectory.close()
            trajectory_filename = join(output_directory, f"{output_prefix}mode_{mode_index}.{output_format}")
            trajectory = read('.tmp_slt_animation.traj@0:%d' % frames)
            write(trajectory_filename, trajectory)
            remove('.tmp_slt_animation.traj')
            print(f"Animation frames for mode {mode_index} saved to '{trajectory_filename}'")


####################
# Derivatives Groups
####################


class SltPropertyCoordinateDerivative(SltXyz):
    _method_type = "PROPERTY_DERIVATIVE"

    __slots__ = SltXyz.__slots__

    def __init__(self, slt_group) -> None:
        super().__init__(slt_group)


################
# Complex Groups
################


class SltCrystalLattice(SltHessian):
    _method_type = "CRYSTAL_LATTICE"

    __slots__ = ["_hdf5", "_magnetic_centers", "_exchange_interactions", "_states", "_electric_dipole", "_magnetic_interactions", "_electric_interactions", "_mode", "_local_states"]

    def __init__(self, slt_group: SltGroup, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> None:
        self._hdf5: str = slt_group._hdf5
        self._magnetic_centers, self._exchange_interactions, self._states, self._electric_dipole, self._magnetic_interactions, self._electric_interactions, self._local_states = slt_group._retrieve_hamiltonian_dict(states_cutoff, rotation, hyperfine, local_states)
        self._mode: str = None # "eslpjm"


