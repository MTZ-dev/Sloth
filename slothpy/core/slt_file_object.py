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

# TODO use annotations like Optional[Union[str, np.ndarray, Sequence]]

from os import path
from functools import partial
from typing import Tuple, Union, Literal
from h5py import File, Group, Dataset
from numpy import (
    ndarray,
    array,
    float32,
    float64,
    int64,
    complex128,
    linspace,
    arange,
    max,
    newaxis,
    allclose,
    identity,
    ones,
    bytes_
)
from scipy.spatial.transform import Rotation
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.animation import PillowWriter
from matplotlib.widgets import Slider
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import (
    plot,
    figure,
    subplots,
    rc,
    tight_layout,
    title,
    cla,
    close,
    Normalize,
)
from slothpy.core._slothpy_exceptions import (
    SltFileError,
    SltCompError,
    SltSaveError,
    SltReadError,
    SltInputError,
    SltPlotError,
    slothpy_exc,
    KeyError,
)
from slothpy._general_utilities._constants import (
    RED,
    GREEN,
    BLUE,
    PURPLE,
    YELLOW,
    RESET,
)
from slothpy._angular_momentum._rotation import SltRotation
from slothpy._magnetism._g_tensor import _g_tensor_and_axes_doublet

from slothpy._magnetism._magnetisation import _mag_3d
from slothpy._magnetism._susceptibility import (
    _chitht,
    _chitht_tensor,
    _chit_3d,
)
from slothpy._magnetism._zeeman import (
    _get_zeeman_matrix,
    _eth,
    _energy_3d,
)
from slothpy._general_utilities._io import _exchange_hamiltonian_to_slt

from slothpy._angular_momentum._pseudo_spin_ito import (
    _get_decomposition_in_z_pseudo_spin_basis,
    _ito_real_decomp_matrix,
    _ito_complex_decomp_matrix,
    _get_soc_matrix_in_z_pseudo_spin_basis,
    _get_zeeman_matrix_in_z_pseudo_spin_basis,
    _matrix_from_ito_complex,
    _matrix_from_ito_real,
)
from slothpy._general_utilities._math_expresions import (
    _normalize_grid_vectors,
    _normalize_orientations,
    _normalize_orientation,
)
from slothpy._general_utilities._auto_tune import _auto_tune
from slothpy._general_utilities._ploting_utilities import (
    color_map,
    _custom_color_cycler,
)
from slothpy._general_utilities._ploting_utilities import _display_plot
from slothpy._general_utilities._grids_over_sphere import (
    _meshgrid_over_sphere,
    _fibonacci_over_sphere,
)
from slothpy.core._input_parser import _parse_hamiltonian_dicts
from slothpy.core._slt_file import SltGroup, SltDataset
from slothpy.core._config import settings
from slothpy.core._delayed_methods import SltZeemanSplitting

class SltFile():
    """
    The core object constituting the API and access to all the methods.
    """

    __slots__ = ["_hdf5", "_groups", "_datasets", "_groups_and_datasets"]

    @classmethod
    def _new(cls, filepath: str):
        """
        This is a private method for initializing the SltFile object that
        should be only used by the creation_functions.

        Parameters
        ----------
        filepath : str
            A path of the file that will be associated with the created
            instance of the SltFile class.
        filename : str
            A name of the file that will be associated with the created
            instance of the SltFile class.

        Returns
        -------
        SltFile
            An instance of the SltFile class.
        """

        if not filepath.endswith(".slt"):
            filepath += ".slt"
        obj = super().__new__(cls)
        obj._hdf5 = filepath
        obj._get_hdf5_groups_datasets_and_attributes()

        return obj

    def __new__(cls, *args, **kwargs) -> None:
        """
        The definition of this method prevents direct instantialization of the
        SltFile class.

        Raises
        ------
        TypeError
            Prevents SltFile() from working.
        """

        raise TypeError(
            f"The {RED}SltFile{RESET} object should not be instantiated "
            "directly. Use a SltFile creation function instead."
        )

    @slothpy_exc("SltFileError")
    def __setitem__(self, key, value) -> None:
        """
        Performs the operation __setitem__.

        Provides a convenient method for setting groups and datasets in the
        .slt file associated with a SltFile instance in an array-like manner.
        """
        with File(self._hdf5, 'r+') as file:
            if isinstance(key, tuple) and len(key) == 2:
                group = file.require_group(key[0])
                if key[1] in group:
                    raise KeyError(f"{PURPLE}Dataset{RESET} '{key[1]}' already exists within the {BLUE}Group{RESET} '{key[0]}'. Delete it manually to ensure your data safety.")
                group.create_dataset(key[1], data=array(value))
            else:
                if key in file:
                    raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' already exists. Delete it manually to ensure data safety.")
                file.create_dataset(key, data=array(value))

    @slothpy_exc("SltFileError")
    def __getitem__(self, key) -> Union[SltGroup, SltDataset]:
        """
        Performs the operation __getitem__.

        Provides a convenient method for getting datasets from the .slt file
        associated with a SltFile instance in an array-like manner.
        """
        with File(self._hdf5, 'r+') as file:
            if isinstance(key, tuple) and len(key) == 2:
                dataset_path = f"{key[0]}/{key[1]}"
                if isinstance(file[dataset_path], Dataset):
                    return SltDataset(self._hdf5, f"{key[0]}/{key[1]}")
                else:
                    raise KeyError(f"{PURPLE}Dataset{RESET} '{dataset_path}' does not exist in the .slt file.")
            if key in file:
                item = file[key]
                if isinstance(item, Dataset):
                    return SltDataset(self._hdf5, key)
                elif isinstance(item, Group):
                    return SltGroup(self._hdf5, key)
            else:
                return SltGroup(self._hdf5, key)

    def __str__(self) -> str:
        """
        Performs the operation __str__.

        Creates a string representation of the SltFile object using names and
        attributes of the groups and datasets contained in the associated
        .slt file.

        Returns
        -------
        str
            A string representation in terms of the contents of the .slt file.
        """

        self._get_hdf5_groups_datasets_and_attributes()
        
        representation = f"{RED}SltFile{RESET} from File: {GREEN}{self._hdf5}{RESET} with the following {BLUE}Groups{RESET}/{PURPLE}Datasets{RESET}.\n"
        for group, attributes_group in self._groups.items():
            representation += f"Group: {BLUE}{group}{RESET}"
            for attribute_name, attribute_text in attributes_group.items():
                representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
            representation += "\n"

            if group in self._groups_and_datasets.keys():
                for dataset, attributes_dataset in self._groups_and_datasets[group].items():
                    representation += f"\t{PURPLE}{dataset}{RESET}"
                    for attribute_name, attribute_text in attributes_dataset.items():
                        representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
                    representation += "\n"
        
        for lone_dataset, attributes_lone_dataset in self._datasets.items():
            representation += f"Dataset: {PURPLE}{lone_dataset}{RESET}"
            for attribute_name, attribute_text in attributes_lone_dataset.items():
                representation += f"| {YELLOW}{attribute_name}{RESET}: {attribute_text}"
            representation += "\n"

        return representation.rstrip()

    def __repr__(self) -> str:
        return f"<{RED}SltFile{RESET} object for {GREEN}File{RESET} '{self._hdf5}'.>"

    def _get_hdf5_groups_datasets_and_attributes(self):
        self._groups = {}
        self._datasets = {}
        self._groups_and_datasets = {}

        def collect_groups(name, obj):
            if isinstance(obj, Group):
                if settings.print_level < 1 and "/" in name:
                    return
                else:
                    self._groups[name] = dict(obj.attrs)

        def collect_groups_datasets_atributes(name, obj):
            if isinstance(obj, Dataset):
                parts = name.split('/')
                if settings.print_level < 1:
                    if len(parts) > 2:
                        return
                if parts[0] in self._groups.keys():
                    if parts[0] not in self._groups_and_datasets.keys():
                        self._groups_and_datasets[parts[0]] = {parts[1]: dict(obj.attrs)}
                    else:
                        self._groups_and_datasets[parts[0]][parts[1]] = dict(obj.attrs)
                else:
                    self._datasets[parts[0]] = dict(obj.attrs)

        with File(self._hdf5, "r") as file:
            file.visititems(collect_groups)
            file.visititems(collect_groups_datasets_atributes)
    
    @slothpy_exc("SltFileError")
    def __delitem__(self, key):
        with File(self._hdf5, 'r+') as file:
            if isinstance(key, tuple) and len(key) == 2:
                key = f"{key[0]}/{key[1]}"
            if key not in file:
                raise KeyError(f"'{key}' does not exist in the .slt file.")
            del file[key]
    
    @slothpy_exc("SltFileError")
    def exchange_hamiltonian(self, magnetic_centers: dict, exchange_interactions: dict, slt_group_name: str, magnetic_dipole_interactions: bool = False, electric_dipole_interactions: bool = False):
        r"""
        Creates a custom, user-defined SlothPy exchange Hamiltonian and
        saves it as a group in the corresponding .slt file to be used for
        subsequent calculations.

        Parameters
        ----------
        magnetic_centers : dict
            A dictionary with integer keys (from 0) enumerating magnetic
            centers in the form: {center_number : [hamiltonian_group_name,
            cutoff_list, rotation, coordinates, hyperfine], ...} where:
            'hamiltonian_group_name' is the name of a Hamiltonian-type group
            in the .slt file representing the desired magnetic center; Note:
            here another SlothPy exchange Hamiltonians cannot be used!,
            'cutoff_list' is a list of three integers [local, mixing, exchange]
            where local represents the overall cutoff for matrices of operators
            for a given magnetic center to be considered in caluclations,
            mixing is the number of states that will be included in the
            exchange Hamiltonian, therefore, the dimension of exchange direct
            product basis for N centers is a product over their mixing
            parameters \prod_{i = 1}^{N} mixing_{i}, and exchange controls how
            many eigenvalues and eigenvectors for a particular center will be
            computed for subsequent calculations using the exchange basis (once
            again, the number of considered eigenvalues/vectors for N centers
            is given by \prod_{i = 1}^{N} exchange_{i}). All of the states that
            do not fit into the diemnsion of exchange space will be included as
            local (not participating in the exchange directly - only by mixing
            part), that is, for each center (local - exchange) number of states
            will be reated in this way. For more examples and the underlying
            theory of such a partitioning scheme, please see tutorials or
            theory section at slothpy.org.,
            'rotation' is a (3,3) orthogonal rotation matrix, an instance of
            SltRotation class or scipy.spatial.transform.Rotation (only single 
            rotation) used to rotate operator matrices for a given magnetic
            center. Note that the inverse matrix has to be given to rotate the
            reference frame instead, set it to None if rotation is not needed.
            'coordinates' is a list or 1d-array of coordinates in Angstrom
            [x,y,z] of magnetic centers to include long-range magnetic (and
            electric if available in the Hamiltonians) dipole interactions
            between all centers. For this to work, the user needs to provide
            coordinates for all the defined centers or set it to None if one
            do not need to include them.,
            hyperfine is scheduled to be released in the 0.4 major release. Now
            left it as None.
        exchange_interactions : dict
            A dictionary whose keys are tuples of integers (n,m) where n and m
            correspondst to magnetic centers numbers defined in the
            magnetic_centers dictionary and values are J anisotropic coupling
            tensor in the form {(n,m) : [[Jxx, Jxy, Jxz],[Jyx, Jyy, Jyz],
            [Jzx, Jzy, Jzz]], ...} with exchange Hamiltonian of the form
            /sum_nm -J_nm*S_n*S_m where S_n are spin vector operators of
            corresponding magnetic centers.
        slt_group_name : str
            Name of the group to which the Hamiltonian will be saved.
            The user can use this name for every available method, the same as 
            any other Hamiltonian groups.
        magnetic_dipole_interactions: bool, optional
            Turns on the inclusion of magnetic dipole interactions between
            centers if coordinates for all of them are given in the
            magnetic_centers dictionary., by default False
        electric_dipole_interactions: bool, optional
            Turns on the inclusion of electric dipole interactions between
            centers if coordinates for all of them are given in the
            magnetic_centers dictionary and all electric dipole momenta matrix
            elements are accessible., by default False
        
        Examples
        --------
        >>> import slothpy as slt
        >>> Dy = slt.hamiltonian_from_molcas('.', 'DyCo_bas3', '.', 'example', 'Dy')
        >>> Dy.hamiltonian({0:['Dy', [30,16,16], None, [0,0,0], None], 1:['Dy', [30,16,16], None, [2,2,2], None], 2:['Dy', [30,16,16], None, [4,4,4], None]},
          {(0,1):[[1,2,3],[1,2,3],[1,2,3]], (0,2):[[1,2,3],[1,2,3],[1,2,3]], (1,2):[[1,2,3],[1,2,3],[1,2,3]]}, 'Dy3_exchange', True)
        (3x Dy centers in a line with 16x16x16 exchange basis and higher term included as 'local' states)
        """

        states, contains_electric_dipole_momenta = _parse_hamiltonian_dicts(self, magnetic_centers, exchange_interactions)
        _exchange_hamiltonian_to_slt(self._hdf5, slt_group_name, states, magnetic_centers, exchange_interactions, contains_electric_dipole_momenta, electric_dipole_interactions, magnetic_dipole_interactions)
        
        return self[f"{slt_group_name}"]

    def zeeman_splitting(
        self,
        hamiltonian_group_name: str,
        magnetic_fields: ndarray[Union[float32, float64]],
        orientations: Union[ndarray[Union[float32, float64]], int, list],
        number_of_states: int = 0,
        states_cutoff: list = [0, 'auto'],
        rotation: Union[ndarray[Union[float32, float64]], SltRotation, Rotation] = None,
        electric_field_vector: ndarray[Union[float32, float64]] = None,
        hyperfine: dict = None,
        number_cpu: int = None,
        number_threads: int = None,
        slt_save: str = None,
        autotune: bool = False,
    ) -> SltZeemanSplitting:
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
            The resulting numpy array (after invoking eval or to_numpy_arrays methods)
            gives Zeeman splitting of number_of_states energy levels in cm⁻¹ for each
            orientation in the form [orientations, fields, energies] - the first
            dimension runs over different orientations, the second over field values,
            and the last gives energies of number_of_states states, unless the
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
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field and orientation values.
        """

        return self[hamiltonian_group_name].zeeman_splitting(magnetic_fields, orientations, number_of_states, states_cutoff, rotation, electric_field_vector, hyperfine, number_cpu, number_threads, slt_save, autotune)

    def calculate_g_tensor_and_axes_doublet(
        self, group: str, doublets: ndarray[int64], slt: str = None
    ) -> Tuple[ndarray[float64], ndarray[float64]]:
        """
        Calculates pseudo-g-tensor components (for S = 1/2) and
        main magnetic axes for a given list of doublet states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of g-tensors.
        doublets : ndarray[int64]
            ArrayLike structure (can be converted to numpy.NDArray) of integers
            corresponding to doublet labels (numbers).
        slt : str, optional
            If given, the results will be saved using this name to the .slt
            file with the suffix: _g_tensors_axes, by default None.

        Returns
        -------
        Tuple[ndarray[float64], ndarray[float64]]
            The first array (g_tensor_list) contains a list g-tensors in
            a format [doublet_number, gx, gy, gz], the second one
            (magnetic_axes_list) contains respective rotation matrices.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If doublets are not one-diemsional array.
        SltCompError
            If the calculation of g-tensors is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        Magnetic axes are returned in the form of rotation matrices that
        diagonalise the Abragam-Bleaney tensor (G = gg.T). Coordinates of the
        main axes XYZ in the initial xzy frame are columns of such matrices
        (0-X, 1-Y, 2-Z).

        See Also
        --------
        slothpy.exporting.table_energy_and_g,
        slothpy.exporting.axes_in_mol2, slothpy.exporting.axes_in_xyz
        """

        if slt is not None:
            slt_group_name = f"{slt}_g_tensors_axes"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually",
                ) from None
        try:
            doublets = array(doublets, dtype=int64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if doublets.ndim != 1:
            raise SltInputError(
                ValueError("The list of doublets has to be a 1D array.")
            ) from None

        try:
            (
                g_tensor_list,
                magnetic_axes_list,
            ) = _g_tensor_and_axes_doublet(self._hdf5, group, doublets)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute g-tensors and main magnetic axes from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_g_tensors",
                    "Dataset containing number of doublet and respective"
                    f" g-tensors from Group {group}.",
                    f"Group({slt}) containing g-tensors of doublets and"
                    f" their magnetic axes calculated from Group: {group}.",
                ] = g_tensor_list[:, :]

                self[
                    slt_group_name,
                    f"{slt}_axes",
                    "Dataset containing rotation matrices from the initial"
                    " coordinate system to the magnetic axes of respective"
                    f" g-tensors from Group: {group}.",
                ] = magnetic_axes_list[:, :, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save g-tensors and magnetic axes to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return g_tensor_list, magnetic_axes_list

    def magnetisation(
        self,
        hamiltonian_group_name: str,
        magnetic_fields: ndarray[float64],
        orientations: Union[int, ndarray[float64]],
        temperatures: ndarray[float64],
        states_cutoff: list = [0, "auto"],
        rotation: ndarray = None, ###################opisac z nowa klasa
        electric_field_vector: ndarray[Union[float32, float64]] = None,
        hyperfine: dict = None, #################opisac ze nie ma xd
        number_cpu: int = 0,
        number_threads: int = 1,
        slt_save: str = None,
        autotune: bool = False,
    ):
        """
        Calculates powder-averaged or directional molar magnetisation M(T,H)
        for a given list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetisation.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which magnetisation will be computed.
        grid : Union[int, ndarray[float64]]
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, user can provide
            an ArrayLike structure (can be converted to numpy.NDArray) with the
            convention: [[direction_x, direction_y, direction_z, weight],...]
            for powder-averaging. If one wants a calculation for a single,
            particular direction the list has to contain one entry like this:
            [[direction_x, direction_y, direction_z, 1.]]. Custom grids will be
            automatically normalized.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which magnetisation will be computed.
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        slt_save : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _magnetisation., by default None
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
        ndarray[float64]
            The resulting mth_array gives magnetisation in Bohr magnetons and
            is in the form [temperatures, fields] - the first dimension runs
            over temperature values, and the second over fields.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of magnetisation is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field values.

        See Also
        --------
        slothpy.SltFile.plot_magnetisation,
        slothpy.lebedev_laikov_grid : For the description of the prescribed
                                      Lebedev-Laikov grids.
        """
        return self[hamiltonian_group_name].magnetisation(magnetic_fields, orientations, temperatures, states_cutoff, rotation, electric_field_vector, hyperfine, number_cpu, number_threads, slt_save, autotune)

    def calculate_magnetisation_3d(
        self,
        group: str,
        fields: ndarray[float64],
        grid_type: Literal["mesh", "fibonacci"],
        grid_number: int,
        temperatures: ndarray[float64],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
    ) -> ndarray[float64]:
        """
        Calculates 3D magnetisation over a spherical grid for a given list of
        temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the 3D magnetisation.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which 3D magnetisation will be computed.
        grid_type: Literal["mesh", "fibonacci"]
            Determines the type of a spherical grid used for the 3D
            magnetisation simulation. Two grids can be used: a classical
            meshgrid and a Fibonacci sphere. The latter can only be plotted as
            a scatter but is uniformly distributed on the sphere, avoiding
            accumulation points near the poles - fewer points are needed.
        grid_number : int
            Controls the density (number of points) of the angular grid for the
            3D magnetisation calculation. A grid of dimension (spherical_grid*
            2*spherical_grid) for spherical angles, phi [0, pi] and theta
            [0, 2*pi] will be used for meshgrid or when Fibonacci sphere is
            chosen grid_number points will be distributed on the sphere.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temperature values (K) at which 3D magnetisation will be computed.
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _magnetisation_3d., by default None
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
        ndarray[float64]
            For the meshgrid the resulting mag_3d_array gives magnetisation in
            Bohr magnetons and is in the form [coordinates, fields,
            temperatures, mesh, mesh] - the first dimension runs over
            coordinates (0-x, 1-y, 2-z), the second over field values, and the
            third over temperatures. The last two dimensions are in the form of
            meshgrids over theta and phi, ready for 3D plots as xyz. For
            Fibonacci, the array has the form [fields, temperatures,
            points[x,y,z]] where points[x,y,z] are two-dimensional
            (grid_number, 3) arrays holding coordinates of grid_number points
            in the [x, y, z] convention.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If grid_type is not "mesh" or "fibonacci".
        SltInputError
            If grid_number is not a positive integer.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of 3D magnetisation is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the number of points on spherical grid.
        Be aware that the resulting arrays and computations can quickly consume
        much memory (e.g. for a calculation with 100 field values 1-10 T, 300
        temperatures 1-300 K, and mesh grid with grid_number = 60, the
        resulting array will take 3*100*300*2*60*60*8 bytes = 5.184 GB).

        See Also
        --------
        slothpy.SltFile.plot_3d, slothpy.SltFile.interactive_plot_3d,
        slothpy.SltFile.animate_3d
        """

        if slt is not None:
            slt_group_name = f"{slt}_magnetisation_3d"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            temperatures = array(temperatures, dtype=float64)
            fields = array(fields, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if grid_type != "mesh" and grid_type != "fibonacci":
            raise SltInputError(
                ValueError(
                    'The only allowed grid types are "mesh" or "fibonacci".'
                )
            ) from None

        if (not isinstance(grid_number, int)) or grid_number <= 0:
            raise SltInputError(
                ValueError("Grid number has to be a positive integer.")
            ) from None

        if autotune:
            try:
                if grid_type == "mesh":
                    grid_autotune = _meshgrid_over_sphere_flatten(grid_number)
                    num_to_parallelize = 2 * grid_number**2
                elif grid_type == "fibonacci":
                    grid_autotune = _fibonacci_over_sphere(grid_number)
                    num_to_parallelize = grid_number
                number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields,
                    grid_autotune,
                    temperatures,
                    states_cutoff,
                    number_cpu,
                    num_to_parallelize,
                    fields.shape[0],
                    "magnetisation_3d",
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        try:
            mag_3d_array = _mag_3d(
                self._hdf5,
                group,
                fields,
                grid_type,
                grid_number,
                temperatures,
                states_cutoff,
                number_cpu,
                number_threads,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute 3D magnetisation from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                if grid_type == "mesh":
                    self[
                        slt_group_name,
                        f"{slt}_mag_3d",
                        "Dataset containing 3D magnetisation as meshgird"
                        " (0-x,1-y,2-z) arrays over sphere (xyz, field,"
                        " temperature, meshgrid, meshgrid) calculated from"
                        f" group: {group}.",
                        f"Group({slt}) containing 3D magnetisation calculated"
                        f" from group: {group}.",
                    ] = mag_3d_array[:, :, :, :, :]
                else:
                    self[
                        slt_group_name,
                        f"{slt}_mag_3d",
                        "Dataset containing 3D magnetisation as xyz points"
                        " (field, temperature, points[x,y,z]) calculated from"
                        f" group: {group}.",
                        f"Group({slt}) containing 3D magnetisation calculated"
                        f" from group: {group}.",
                    ] = mag_3d_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of 3D magnetisation from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of 3D magnetisation from group: {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save 3D magnetisation to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return mag_3d_array

    def calculate_susceptibility(
        self,
        group: str,
        temperatures: ndarray[float64],
        fields: ndarray[float64],
        number_of_points: int = 1,
        delta_h: float = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        grid: Union[int, ndarray[float64]] = None,
        slt: str = None,
        autotune: bool = False,
    ) -> ndarray[float64]: ###################### Check for negative fields during the differentation!!!
        """
        Calculates powder-averaged or directional molar magnetic susceptibility
        chi(T)(H,T) for a given list of field and temperatures values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetisation.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which magnetic susceptibility will
            be computed.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which magnetic susceptibility will be computed.
        number_of_points : int, optional
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.,
            by default 1
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as a starting
            point., by default 0.0001
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        exp : bool, optional
            Turns on the experimentalist model for magnetic susceptibility.,
            by default False
        T : bool, optional
            Results are returned as a product with temperature chiT(H,T).,
            by default True
        grid : Union[int, ndarray[float64]], optional
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over the hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, the user can
            provide an ArrayLike structure (can be converted to numpy.NDArray)
            with the convention: [[direction_x, direction_y, direction_z,
            weight],...] for powder-averaging. If one wants a calculation for a
            single, particular direction the list has to contain one entry like
            this: [[direction_x, direction_y, direction_z, 1.]]. If not given
            the average is taken over xyz directions, which is sufficient for a
            second rank tensor. Custom grids will be automatically normalized.,
            by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _susceptibility., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 300 with
            a higher number of field values and number_of_points) where it
            becomes a necessity., by default False

        Returns
        -------
        ndarray[float64]
            The resulting chitht_array gives magnetic susceptibility (or
            product with temperature) in cm^3 (or * K) and is in the form
            [fields, temperatures] - the first dimension runs over field
            values, and the second over temperatures.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If the  number of points for finite difference method is not
            a possitive integer.
        SltInputError
            If the field step for the finite difference method is not
            a possitive real number.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of magnetic susceptibility is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over fields.size*(2*number_of_points+1) tasks.

        See Also
        --------
        slothpy.SltFile.plot_susceptibility
        """
        if slt is not None:
            slt_group_name = f"{slt}_susceptibility"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            fields = array(fields, dtype=float64)
            temperatures = array(temperatures, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if (not isinstance(number_of_points, int)) or number_of_points < 0:
            raise SltInputError(
                ValueError(
                    "The number of points for the finite difference method has"
                    " to be a possitive integer."
                )
            ) from None

        if (not isinstance(delta_h, float)) or delta_h <= 0:
            raise SltInputError(
                ValueError(
                    "The field step for finite difference method has to be a"
                    " possitive number."
                )
            ) from None

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
        elif grid is not None:
            grid = _normalize_grid_vectors(grid)

        if autotune:
            try:
                if exp or number_of_points == 0:
                    num_to_parallelize = fields.size
                else:
                    num_to_parallelize = (
                        2 * number_of_points + 1
                    ) * fields.size
                if grid is None:
                    grid_autotune = array(
                        [
                            [1.0, 0.0, 0.0, 0.3333333333333333],
                            [0.0, 1.0, 0.0, 0.3333333333333333],
                            [0.0, 0.0, 1.0, 0.3333333333333333],
                        ],
                        dtype=float64,
                    )
                    grid_shape = 3  # xyz grid in the inner loop
                else:
                    grid_shape = grid.shape[0]
                if not (exp or number_of_points == 0):
                    fields_autotune = (
                        arange(-number_of_points, number_of_points + 1).astype(
                            int64
                        )
                        * delta_h
                    )[:, newaxis] + fields
                    fields_autotune = fields_autotune.T.astype(float64)
                    fields_autotune = fields_autotune.flatten()
                number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields_autotune,
                    grid_autotune,
                    temperatures,
                    states_cutoff,
                    number_cpu,
                    num_to_parallelize,
                    grid_shape,
                    "magnetisation",
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        if T:
            chi_name = "chiT(H,T)"
            chi_file = "chit"
        else:
            chi_name = "chi(H,T)"
            chi_file = "chi"

        try:
            chitht_array = _chitht(
                self._hdf5,
                group,
                temperatures,
                fields,
                number_of_points,
                delta_h,
                states_cutoff,
                number_cpu,
                number_threads,
                exp,
                T,
                grid,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute {chi_name} from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_{chi_file}ht",
                    f"Dataset containing {chi_name} magnetic"
                    " susceptibility (H - rows, T - columns) calculated"
                    f" from group: {group}.",
                    f"Group({slt}) containing {chi_name} magnetic"
                    f" susceptibility calculated from group: {group}.",
                ] = chitht_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    " simulation of magnetic susceptibility from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of {chi_name} from group: {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {chi_name} to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return chitht_array

    def calculate_susceptibility_tensor(
        self,
        group: str,
        temperatures: ndarray[float64],
        fields: ndarray[float64],
        number_of_points: int = 1,
        delta_h: float = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
    ) -> ndarray[float64]:
        """
        Calculates magnetic susceptibility chi(H,T) (Van Vleck) tensor for
        a given list of field and temperature values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetisation.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which magnetic susceptibility tensor
            will be computed.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which magnetic susceptibility tensor will be
            computed.
        number_of_points : int, optional
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.,
            by default 1
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as a starting
            point., by default 0.0001,
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        exp : bool, optional
            Turns on the experimentalist model for magnetic susceptibility.,
            by default False
        T : bool, optional
            Results are returned as a product with temperature chiT(H,T).,
            by default True
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _susceptibility_tensor., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 500 with
            a higher number of field values and number_of_points) where it
            becomes a necessity., by default False

        Returns
        -------
        ndarray[float64]
            The resulting array gives magnetic susceptibility (Van Vleck)
            tensors (or products with temperature) in cm^3 (or * K) and is in
            the form [fields, temperatures, 3x3 tensor] - the first dimension
            runs over field values, the second over temperatures, and the last
            two accomodate 3x3 tensors.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If the  number of points for finite difference method is not
            a possitive integer
        SltInputError
            If the field step for the finite difference method is not
            a possitive real number.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of magnetic susceptibility tensor is
            unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over fields.size*(2*number_of_points+1) tasks.
        """
        if slt is not None:
            slt_group_name = f"{slt}_susceptibility_tensor"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            fields = array(fields, dtype=float64)
            temperatures = array(temperatures, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if (not isinstance(number_of_points, int)) or number_of_points < 0:
            raise SltInputError(
                ValueError(
                    "The number of points for the finite difference method has"
                    " to be a possitive integer."
                )
            ) from None

        if (not isinstance(delta_h, float)) or delta_h <= 0:
            raise SltInputError(
                ValueError(
                    "The field step for finite difference method has to be a"
                    " possitive number."
                )
            ) from None

        if autotune:
            try:
                if exp or number_of_points == 0:
                    num_to_parallelize = fields.size
                else:
                    num_to_parallelize = (
                        2 * number_of_points + 1
                    ) * fields.size
                grid_autotune = ones((9, 4))
                if not (exp or number_of_points == 0):
                    fields_autotune = (
                        arange(-number_of_points, number_of_points + 1).astype(
                            int64
                        )
                        * delta_h
                    )[:, newaxis] + fields
                    fields_autotune = fields_autotune.T.astype(float64)
                    fields_autotune = fields_autotune.flatten()
                number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields_autotune,
                    grid_autotune,
                    temperatures,
                    states_cutoff,
                    number_cpu,
                    num_to_parallelize,
                    9,
                    "magnetisation",
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        if T:
            chi_name = "chiT(H,T)"
            chi_file = "chit"
        else:
            chi_name = "chi(H,T)"
            chi_file = "chi"

        try:
            chitht_tensor_array = _chitht_tensor(
                self._hdf5,
                group,
                temperatures,
                fields,
                number_of_points,
                delta_h,
                states_cutoff,
                number_cpu,
                number_threads,
                exp,
                T,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute {chi_name} tensor from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_{chi_file}ht_tensor",
                    f"Dataset containing {chi_name}_tensor Van Vleck"
                    " susceptibility tensor (H, T, 3, 3) calculated from"
                    f" group: {group}.",
                    f"Group({slt}) containing {chi_name}_tensor Van Vleck"
                    " susceptibility tensor calculated from group:"
                    f" {group}.",
                ] = chitht_tensor_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of {chi_name}_tensor Van Vleck"
                    f" susceptibility tensor from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of {chi_name}_tensor Van Vleck"
                    f" susceptibility tensor from group: {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {chi_name} tensor to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return chitht_tensor_array

    def calculate_susceptibility_3d(
        self,
        group: str,
        temperatures: ndarray[float64],
        fields: ndarray[float64],
        grid_type: Literal["mesh", "fibonacci"],
        grid_number: int,
        number_of_points: int = 1,
        delta_h: float = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
    ) -> ndarray[float64]:
        """
        Calculates 3D magnetic susceptibility over a spherical grid for a given
        list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the 3D magnetic
            susceptibility.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temperature values (K) at which 3D magnetic susceptibility will be
            computed.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which 3D magnetic susceptibility will be computed.
        grid_type: Literal["mesh", "fibonacci"]
            Determines the type of a spherical grid used for the 3D
            susceptibility simulation. Two grids can be used: a classical
            meshgrid and a Fibonacci sphere. The latter can only be plotted as
            a scatter but is uniformly distributed on the sphere, avoiding
            accumulation points near the poles - fewer points are needed.
        grid_number : int
            Controls the density (number of points) of the angular grid for the
            3D susceptibility  calculation. A grid of dimension (spherical_grid*
            2*spherical_grid) for spherical angles, phi [0, pi] and theta
            [0, 2*pi] will be used for meshgrid or when Fibonacci sphere is
            chosen grid_number points will be distributed on the sphere.
        number_of_points : int, optional
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.,
            by default 1
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as a starting
            point., by default 0.0001
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        exp : bool, optional
            Turns on the experimentalist model for magnetic susceptibility.,
            by default False
        T : bool, optional
            Results are returned as a product with temperature chiT(H,T).,
            by default True
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _susceptibility_3d., by default None
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
        ndarray[float64]
            For the meshgrid the resulting chi(t)_3d_array gives susceptibility
            in cm^3 (or * K) and is in the form [coordinates, fields,
            temperatures, mesh, mesh] - the first dimension runs over
            coordinates (0-x, 1-y, 2-z), the second over field values, and the
            third over temperatures. The last two dimensions are in the form of
            meshgrids over theta and phi, ready for 3D plots as xyz. For
            Fibonacci, the array has the form [fields, temperatures,
            points[x,y,z]] where points[x,y,z] are two-dimensional
            (grid_number, 3) arrays holding coordinates of grid_number points
            in the [x, y, z] convention.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If grid_type is not "mesh" or "fibonacci".
        SltInputError
            If grid_number is not a positive integer
        SltInputError
            If the  number of points for finite difference method is not
            a possitive integer.
        SltInputError
            If the field step for the finite difference method is not
            a possitive real number.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of 3D magnetic susceptibility is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the number of points on spherical grid.
        Be aware that the resulting arrays and computations can quickly
        consume much memory (e.g. for calculation with 100 field values 1-10 T,
        300 temperatures 1-300 K, number_of_points=3, and spherical_grid = 60,
        the intermediate array (before numerical differentiation) will take
        7*100*300*2*60*60*8 bytes = 12.096 GB).

        See Also
        --------
        slothpy.SltFile.plot_3d, slothpy.SltFile.interactive_plot_3d,
        slothpy.SltFile.animate_3d
        """

        if slt is not None:
            slt_group_name = f"{slt}_susceptibility_3d"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            fields = array(fields, dtype=float64)
            temperatures = array(temperatures, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if grid_type != "mesh" and grid_type != "fibonacci":
            raise SltInputError(
                ValueError(
                    'The only allowed grid types are "mesh" or "fibonacci".'
                )
            ) from None

        if (not isinstance(grid_number, int)) or grid_number <= 0:
            raise SltInputError(
                ValueError("Grid number has to be a positive integer.")
            ) from None

        if (not isinstance(number_of_points, int)) or number_of_points < 0:
            raise SltInputError(
                ValueError(
                    "The number of points for the finite difference method has"
                    " to be a possitive integer."
                )
            ) from None

        if (not isinstance(delta_h, float)) or delta_h <= 0:
            raise SltInputError(
                ValueError(
                    "The field step for finite difference method has to be a"
                    " possitive number."
                )
            ) from None

        if autotune:
            try:
                if exp or number_of_points == 0:
                    inner_loop_size = fields.size
                else:
                    inner_loop_size = (2 * number_of_points + 1) * fields.size
                if grid_type == "mesh":
                    grid_autotune = _meshgrid_over_sphere_flatten(grid_number)
                    num_to_parallelize = 2 * grid_number**2
                elif grid_type == "fibonacci":
                    grid_autotune = _fibonacci_over_sphere(grid_number)
                    num_to_parallelize = grid_number
                if not (exp or number_of_points == 0):
                    fields_autotune = (
                        arange(-number_of_points, number_of_points + 1).astype(
                            int64
                        )
                        * delta_h
                    )[:, newaxis] + fields
                    fields_autotune = fields_autotune.T.astype(float64)
                    fields_autotune = fields_autotune.flatten()
                number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields_autotune,
                    grid_autotune,
                    temperatures,
                    states_cutoff,
                    number_cpu,
                    num_to_parallelize,
                    inner_loop_size,
                    "magnetisation_3d",
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        if T:
            chi_name = "chiT(H,T)"
            chi_file = "chit"
        else:
            chi_name = "chi(H,T)"
            chi_file = "chi"

        try:
            chit_3d_array = _chit_3d(
                self._hdf5,
                group,
                temperatures,
                fields,
                grid_type,
                grid_number,
                number_of_points,
                delta_h,
                states_cutoff,
                number_cpu,
                number_threads,
                exp,
                T,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute 3D magnetic susceptibility"
                f" {chi_name} from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                if grid_type == "mesh":
                    self[
                        slt_group_name,
                        f"{slt}_{chi_file}_3d",
                        "Dataset containing 3D magnetic susceptibility"
                        f" {chi_name} as meshgird (0-x,1-y,2-z) arrays over"
                        " sphere ((xyz, field, temperature, meshgrid,"
                        f" meshgrid) calculated from group: {group}.",
                        f"Group({slt}) containing 3D magnetic susceptibility"
                        f" {chi_name} calculated from group: {group}.",
                    ] = chit_3d_array[:, :, :, :, :]
                else:
                    self[
                        slt_group_name,
                        f"{slt}_{chi_file}_3d",
                        "Dataset containing 3D magnetic susceptibility as xyz"
                        " points (field, temperature, points[x,y,z])"
                        f" calculated from group: {group}.",
                        f"Group({slt}) containing 3D magnetic susceptibility"
                        f" calculated from group: {group}.",
                    ] = chit_3d_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    " simulation of 3D magnetic susceptibility"
                    f" {chi_name} from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    " simulation of 3D magnetic susceptibility"
                    f" {chi_name} from group: {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save 3D magnetic susceptibility {chi_name} to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return chit_3d_array

    def calculate_energy(
        self,
        group: str,
        fields: ndarray[float64],
        grid: Union[int, ndarray[float64]],
        temperatures: ndarray[float64],
        energy_type: Literal["helmholtz", "internal"],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        slt: str = None,
        autotune: bool = False,
    ) -> ndarray[float64]:
        """
        Calculates powder-averaged or directional Helmholtz or internal
        energy for a given list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the energy.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which energy will be computed.
        grid : ndarray[float64]
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, user can provide
            an ArrayLike structure (can be converted to numpy.NDArray) with the
            convention: [[direction_x, direction_y, direction_z, weight],...]
            for powder-averaging. If one wants a calculation for a single,
            particular direction the list has to contain one entry like this:
            [[direction_x, direction_y, direction_z, 1.]]. Custom grids will be
            automatically normalized.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which energy will be computed
        energy_type: Literal["helmholtz", "internal"]
            Determines which kind of energy, Helmholtz or internal, will be
            calculated.
        states_cutoff : int
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _helmholtz_energy or _internal_energy.,
            by default None
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
        ndarray[float64]
            The resulting eth_array gives energy in cm⁻¹ and is in the form
            [temperatures, fields] - the first dimension runs over temperature
            values, and the second over fields.

        Raises
        ------
        SltInputError
            if energy_type is not "helmholtz" or "internal".
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array
        SltInputError
            If temperatures are not a one-diemsional array
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of energy is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field values.

        See Also
        --------
        slothpy.SltFile.plot_energy
        slothpy.lebedev_laikov_grid : For the description of the prescribed
                                      Lebedev-Laikov grids.
        """
        if energy_type == "internal":
            group_suffix = "_internal_energy"
            name = "internal"
        elif energy_type == "helmholtz":
            group_suffix = "_helmholtz_energy"
            name = "Helmholtz"
        else:
            raise SltInputError(
                ValueError(
                    'Energy type must be set to "helmholtz" or "internal".'
                )
            ) from None

        if slt is not None:
            slt_group_name = f"{slt}{group_suffix}"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            fields = array(fields, dtype=float64)
            temperatures = array(temperatures, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
        else:
            grid = _normalize_grid_vectors(grid)

        if autotune:
            try:
                number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields,
                    grid,
                    temperatures,
                    states_cutoff,
                    number_cpu,
                    fields.shape[0],
                    grid.shape[0],
                    "energy",
                    energy_type=energy_type,
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        try:
            energyth_array = _eth(
                self._hdf5,
                group,
                fields,
                grid,
                temperatures,
                energy_type,
                states_cutoff,
                number_cpu,
                number_threads,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute {name} energy from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_eth",
                    f"Dataset containing E(T,H) {name} energy (T - rows,"
                    f" H - columns) calculated from group: {group}.",
                    f"Group({slt}) containing E(T,H) {name} energy"
                    f" calculated from group: {group}.",
                ] = energyth_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of E(T,H) {name} energy from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of E(T,H) {name} energy from group:"
                    f" {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {name} energy to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return energyth_array

    def calculate_energy_3d(
        self,
        group: str,
        fields: ndarray[float64],
        grid_type: Literal["mesh", "fibonacci"],
        grid_number: int,
        temperatures: ndarray[float64],
        energy_type: Literal["helmholtz", "internal"],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
    ) -> ndarray[float64]:
        """
        Calculates 3D Helmholtz or internal energy over a spherical grid for
        a given list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the 3D energy.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which 3D energy will be computed.
        grid_type: Literal["mesh", "fibonacci"]
            Determines the type of a spherical grid used for the 3D
            energy simulation. Two grids can be used: a classical
            meshgrid and a Fibonacci sphere. The latter can only be plotted as
            a scatter but is uniformly distributed on the sphere, avoiding
            accumulation points near the poles - fewer points are needed.
        grid_number : int
            Controls the density (number of points) of the angular grid for the
            3D magnetisation calculation. A grid of dimension (spherical_grid*
            2*spherical_grid) for spherical angles, phi [0, pi] and theta
            [0, 2*pi] will be used for meshgrid or when Fibonacci sphere is
            chosen grid_number points will be distributed on the sphere.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temperature values (K) at which 3D energy will be computed.
        energy_type: Literal["helmholtz", "internal"]
            Determines which kind of energy, Helmholtz or internal, will be
            calculated.
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0,
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        internal_energy : bool, optional
            Turns on the calculation of internal energy., by default False
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _helmholtz_energy_3d or _internal_energy_3d.,
            by default None
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
        ndarray[float64]
            For the meshgrid the resulting energy_3d_array gives energy in cm⁻¹
            and is in the form [coordinates, fields, temperatures, mesh, mesh]
            - the first dimension runs over coordinates (0-x, 1-y, 2-z), the
            second over field values, and the third over temperatures. The last
            two dimensions are in the form of meshgrids over theta and phi,
            ready for 3D plots as xyz. For Fibonacci, the array has the form
            [fields, temperatures, points[x,y,z]] where points[x,y,z] are
            two-dimensional (grid_number, 3) arrays holding coordinates of
            grid_number points in the [x, y, z] convention.

        Raises
        ------
        SltInputError
            if energy_type is not "helmholtz" or "internal".
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If grid_type is not "mesh" or "fibonacci".
        SltInputError
            If grid_number is not a positive integer.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of 3D energy is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the number of points on spherical grid.
        Be aware that the resulting arrays and computations can quickly consume
        much memory (e.g. for a calculation with 100 field values 1-10 T, 300
        temperatures 1-300 K, and mesh grid with grid_number = 60, the
        resulting array will take 3*100*300*2*60*60*8 bytes = 5.184 GB).

        See Also
        --------
        slothpy.SltFile.plot_3d, slothpy.SltFile.interactive_plot_3d,
        slothpy.SltFile.animate_3d
        """
        if energy_type == "internal":
            group_suffix = "_internal_energy_3d"
            name = "internal"
        elif energy_type == "helmholtz":
            group_suffix = "_helmholtz_energy_3d"
            name = "Helmholtz"
        else:
            raise SltInputError(
                ValueError(
                    'Energy type must be set to "helmholtz" or "internal".'
                )
            ) from None

        if slt is not None:
            slt_group_name = f"{slt}{group_suffix}"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            temperatures = array(temperatures, dtype=float64)
            fields = array(fields, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if grid_type != "mesh" and grid_type != "fibonacci":
            raise SltInputError(
                ValueError(
                    'The only allowed grid types are "mesh" or "fibonacci".'
                )
            ) from None

        if (not isinstance(grid_number, int)) or grid_number <= 0:
            raise SltInputError(
                ValueError("Grid number has to be a positive integer.")
            ) from None

        if autotune:
            try:
                if grid_type == "mesh":
                    grid_autotune = _meshgrid_over_sphere_flatten(grid_number)
                    num_to_parallelize = 2 * grid_number**2
                elif grid_type == "fibonacci":
                    grid_autotune = _fibonacci_over_sphere(grid_number)
                    num_to_parallelize = grid_number
                number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields,
                    grid_autotune,
                    temperatures,
                    states_cutoff,
                    number_cpu,
                    num_to_parallelize,
                    fields.shape[0],
                    "energy_3d",
                    energy_type=energy_type,
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        try:
            energy_3d_array = _energy_3d(
                self._hdf5,
                group,
                fields,
                grid_type,
                grid_number,
                temperatures,
                energy_type,
                states_cutoff,
                number_cpu,
                number_threads,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute 3D {name} energy from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                if grid_type == "mesh":
                    self[
                        slt_group_name,
                        f"{slt}_energy_3d",
                        f"Dataset containing 3D {name} energy as meshgird"
                        " (0-x,1-y,2-z) arrays over sphere (xyz, field,"
                        " temperature, meshgrid, meshgrid) calculated from"
                        f" group: {group}.",
                        f"Group({slt}) containing 3D {name}_energy"
                        f" calculated from group: {group}.",
                    ] = energy_3d_array[:, :, :, :, :]
                else:
                    self[
                        slt_group_name,
                        f"{slt}_energy_3d",
                        f"Dataset containing 3D {name} energy as xyz points"
                        " (field, temperature, points[x,y,z]) calculated from"
                        f" group: {group}.",
                        f"Group({slt}) containing 3D {name}_energy"
                        f" calculated from group: {group}.",
                    ] = energy_3d_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of 3D {name} energy from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of 3D {name} energy from group:"
                    f" {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save 3D {name} energy to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return energy_3d_array

    def zeeman_matrix(
        self,
        group: str,
        fields: ndarray[float64],
        orientations: ndarray[float64],
        states_cutoff: int = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates Zeeman matrices for a given list of magnetic fields and
        their orientations.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the Zeeman matrices.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) for which Zeeman matrices will be computed.
        orientations : ndarray[float64]
            List (ArrayLike structure) of particular magnetic field directions
            for which Zeeman matrices will be constructed. The list has to
            follow the format: [[direction_x, direction_y, direction_z],...].
            The vectors will be automatically normalized.
        states_cutoff : int
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _zeeman_matrix., by default None

        Returns
        -------
        ndarray[complex128]
            The resulting array gives Zeeman matrices for each field value and
            orientation in the form [fields, orientations, matrix, matrix] in
            atomic units a.u. (Hartree).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltCompError
            If the calculation of Zeeman matrices is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_zeeman_matrix"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            fields = array(fields, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        orientations = _normalize_orientations(orientations)

        try:
            zeeman_matrix_array = _get_zeeman_matrix(
                self._hdf5, group, states_cutoff, fields, orientations
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to calculate Zeeman matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_matrix",
                    "Dataset containing Zeeman matrices calculated from"
                    f" group: {group} in the form [fields, orientations,"
                    " matrix, matrix].",
                    f"Group({slt}) containing Zeeman matrices calculated"
                    f" from group: {group}.",
                ] = zeeman_matrix_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    " simulation of Zeeman matrices from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_orientations",
                    "Dataset containing magnetic field orientations"
                    " used in simulation of"
                    f" Zeeman matrices from group: {group}.",
                ] = orientations[:, :]

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save Zeeman matrix to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return zeeman_matrix_array

    def soc_energies_cm_1(
        self, group: str, number_of_states: int = 0, slt: str = None
    ) -> ndarray[float64]:
        """
        Returns energies for the given number of first spin-orbit
        states in cm⁻¹.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations.
        number_of_states : int, optional
            Number of states whose energy will be returned. If set to zero, all
            available states will be inculded., by default 0
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _soc_energies., by default None

        Returns
        -------
        ndarray[float64]
            The resulting array is one-dimensional and contains the energy of
            first number_of_states states in cm⁻¹.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltReadError
            If the program is unable to get SOC energies from the .slt file.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        See Also
        --------
        slothpy.exporting.table_energy_and_g
        """
        if slt is not None:
            slt_group_name = f"{slt}_soc_energies"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            soc_energies_array = _get_soc_energies_cm_1(
                self._hdf5, group, number_of_states
            )
        except Exception as exc:
            raise SltReadError(
                self._hdf5,
                exc,
                "Failed to read SOC energies from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_energies",
                    "Dataset containing SOC (Spin-Orbit Coupling) energies"
                    f" calculated from group: {group}.",
                    f"Group({slt}) containing SOC (Spin-Orbit Coupling)"
                    f" energies calculated from group: {group}.",
                ] = soc_energies_array
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save SOC energies to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return soc_energies_array

    def states_magnetic_momenta(
        self,
        group: str,
        states: Union[int, ndarray[int]] = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[float64]:
        """
        Calculates magnetic momenta of a given list (or number) of SOC states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetic momenta.
        states : Union[int, ndarray[int]], optional
            ArrayLike structure (can be converted to numpy.NDArray) of
            states indexes for which magnetic momenta will be calculated. If
            set to an integer it acts as a states cutoff (first n states will
            be given). For all available states set it to zero., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _states_magnetic_momenta., by default None

        Returns
        -------
        ndarray[float64]
            The resulting array is one-dimensional and contains the magnetic
            momenta corresponding to the given states indexes in atomic units.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltCompError
            If the calculation of magnetic momenta is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_states_magnetic_momenta"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        if not isinstance(states, int):
            try:
                states = array(states, dtype=int64)
            except Exception as exc:
                raise SltInputError(exc) from None

        try:
            magnetic_momenta_array = _get_states_magnetic_momenta(
                self._hdf5, group, states, rotation
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute states magnetic momenta from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_magnetic_momenta",
                    "Dataset containing states magnetic momenta"
                    f" (0-x,1-y,2-z) calculated from group: {group}.",
                    f"Group({slt}) containing states magnetic momenta"
                    f" calculated from group: {group}.",
                ] = magnetic_momenta_array
                self[
                    slt_group_name,
                    f"{slt}_states",
                    "Dataset containing indexes of states (or states"
                    " cutoff) used in simulation of magnetic momenta from"
                    f" group: {group}.",
                ] = array(states)

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states magnetic momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return magnetic_momenta_array

    def states_total_angular_momenta(
        self,
        group: str,
        states: Union[int, ndarray[int]] = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[float64]:
        """
        Calculates total angular momenta of a given list (or number) of SOC
        states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetic momenta.
        states : Union[int, ndarray[int]], optional
            ArrayLike structure (can be converted to numpy.NDArray) of
            states indexes for which total angular momenta will be calculated.
            If set to an integer it acts as a states cutoff (first n states
            will be given). For all available states set it to zero.
            , by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _states_total_angular_momenta., by default None

        Returns
        -------
        ndarray[float64]
            The resulting array is one-dimensional and contains the total
            angular momenta corresponding to the given states indexes in atomic
            units.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltCompError
            If the calculation of total angular momenta is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_states_total_angular_momenta"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        if not isinstance(states, int):
            try:
                states = array(states, dtype=int64)
            except Exception as exc:
                raise SltInputError(exc) from None

        try:
            total_angular_momenta_array = _get_states_total_angular_momenta(
                self._hdf5, group, states, rotation
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute states total angular momenta from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_total_angular_momenta",
                    "Dataset containing states total angular momenta"
                    f" (0-x,1-y,2-z) calculated from group: {group}.",
                    f"Group({slt}) containing states total angular momenta"
                    f" calculated from group: {group}.",
                ] = total_angular_momenta_array
                self[
                    slt_group_name,
                    f"{slt}_states",
                    "Dataset containing indexes of states (or states"
                    " cutoff) used in simulation of total angular momenta"
                    f" from group: {group}.",
                ] = array(states)

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states total angular momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return total_angular_momenta_array

    def magnetic_momenta_matrix(
        self,
        group: str,
        states_cutoff: ndarray = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates magnetic momenta matrix for a given number of SOC states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetic momenta
            matrix.
        states_cutoff : ndarray, optional
            Number of states that will be taken into account for construction
            of the magnetic momenta matrix. If set to zero, all available
            states from the file will be included., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _magnetic_momenta_matrix., by default None

        Returns
        -------
        ndarray[complex128]
            The resulting magnetic_momenta_matrix_array gives magnetic momenta
            in atomic units and is in the form [coordinates, matrix, matrix]
            - the first dimension runs over coordinates (0-x, 1-y, 2-z).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of magetic momenta matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_magnetic_momenta_matrix"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            magnetic_momenta_matrix_array = _get_magnetic_momenta_matrix(
                self._hdf5, group, states_cutoff, rotation
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute magnetic momenta matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_magnetic_momenta_matrix",
                    "Dataset containing magnetic momenta matrix"
                    f" (0-x, 1-y, 2-z) calculated from group: {group}.",
                    f"Group {group} containing magnetic momenta"
                    f" matrix calculated from group: {group}.",
                ] = magnetic_momenta_matrix_array[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states magnetic momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return magnetic_momenta_matrix_array

    def total_angular_momenta_matrix(
        self,
        group: str,
        states_cutoff: int = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates total angular momenta matrix for a given number of SOC
        states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the total angular momenta
            matrix.
        states_cutoff : ndarray, optional
            Number of states that will be taken into account for construction
            of the total angular momenta matrix. If set to zero, all available
            states from the file will be included., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _total angular_momenta_matrix., by default None

        Returns
        -------
        ndarray[complex128]
            The resulting total_angular_momenta_matrix_array gives total
            angular momenta in atomic units and is in the form [coordinates,
            matrix, matrix] - the first dimension runs over coordinates
            (0-x, 1-y, 2-z).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of total angular momenta matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_total_angular_momenta_matrix"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            total_angular_momenta_matrix_array = (
                _get_total_angular_momneta_matrix(
                    self._hdf5, group, states_cutoff, rotation
                )
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute total angular momenta matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_total_angular_momenta_matrix",
                    "Dataset containing total angular momenta matrix"
                    f" (0-x, 1-y, 2-z) calculated from group: {group}.",
                    f"Group {group} containing total angular momenta"
                    f" matrix calculated from group: {group}.",
                ] = total_angular_momenta_matrix_array[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states total angular momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return total_angular_momenta_matrix_array

    def matrix_decomposition_in_z_pseudo_spin_basis(
        self,
        group: str,
        matrix: Literal["soc", "zeeman"],
        pseudo_kind: Literal["magnetic", "total_angular"],
        start_state: int = 0,
        stop_state: int = 0,
        rotation: ndarray[float64] = None,
        field: float64 = None,
        orientation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[float64]:
        """
        Calculates decomposition of a given matrix in "z" pseudo-spin basis.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the construction of the matrix.
        matrix : Literal["soc", "zeeman"]
            Type of a matrix to be decomposed. Two options available: "soc" or
            "zeeman".
        pseudo_kind : Literal["magnetic", "total_angular"]
            Kind of a pseudo-spin basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis.
        start_state : int, optional
            Number of the first SOC state to be included., by default 0
        stop_state : int, optional
            Number of the last SOC state to be included. If both start and stop
            are set to zero all available states from the file will be used.
            , by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        field : float64, optional
            If matrix type = "zeeman" it controls a magnetic field value at
            which Zeeman matrix will be computed., by default None
        orientation : ndarray[float64], optional
            If matrix type = "zeeman" it controls the orientation of the
            magnetic field and has to be in the form [direction_x, direction_y,
            direction_z] and be an ArrayLike structure (can be converted to
            numpy.NDArray)., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _magnetic/total_angular_decomposition.
            , by default None

        Returns
        -------
        ndarray[float64]
            The resulting array gives decomposition in % where rows are
            SOC/Zeeman states and columns are associated with pseudo spin basis
            (from -Sz to Sz).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the decomposition of the matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_{pseudo_kind}_decomposition"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            if orientation is not None:
                orientation = _normalize_orientation(orientation)
            decomposition = _get_decomposition_in_z_pseudo_spin_basis(
                self._hdf5,
                group,
                matrix,
                pseudo_kind,
                start_state,
                stop_state,
                rotation,
                field,
                orientation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to decompose {matrix} matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        if slt is not None:
            dim = (decomposition.shape[1] - 1) / 2
            try:
                self[
                    slt_group_name,
                    f"{slt}_{pseudo_kind}_decomposition",
                    "Dataset containing decomposition (rows - SO-states,"
                    f' columns - basis) in "z" {pseudo_kind} momentum'
                    f" basis of {matrix} matrix from group: {group}.",
                    f'Group({slt}) containing decomposition in "z"'
                    f" {pseudo_kind} basis of {matrix} matrix calculated"
                    f" from group: {group}.",
                ] = decomposition[:, :]
                self[
                    slt_group_name,
                    f"{slt}_pseudo_spin_states",
                    "Dataset containing Sz pseudo-spin states"
                    " corresponding to the decomposition of"
                    f" {matrix} matrix from group: {group}.",
                ] = arange(-dim, dim + 1, step=1, dtype=float64)
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {pseudo_kind} decomposition of"
                    f" {matrix} matrix "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return decomposition

    def soc_crystal_field_parameters(
        self,
        group: str,
        start_state: int,
        stop_state: int,
        order: int,
        pseudo_kind: Literal["magnetic", "total_angular"],
        even_order: bool = True,
        complex: bool = False,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> list:
        """
        Calculates ITO decomposition (CFPs) of SOC matrix.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for obtaining the SOC matrix.
        start_state : int
            Number of the first SOC state to be included.
        stop_state : int
            Number of the last SOC state to be included. If both start and stop
            are set to zero all available states from the file will be used.
        order : int
            Order of the highest ITO (CFP) to be included in the decomposition.
        pseudo_kind : Literal["magnetic", "total_angular"]
            Kind of a pseudo-spin basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis.
        even_order : bool, optional
            If True, only even order ITOs (CFPs) will be included in the
            decomposition., by default True
        complex : bool, optional
            If True, instead of real ITOs (CFPs) complex ones will be given.,
            by default False
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _soc_ito_decomposition., by default None

        Returns
        -------
        list
            The resulting list gives CFP - B_k_q (ITO) in the form [k,q,B_k_q].

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltReadError
            If the program is unable to read SOC matrix from the file.
        SltInputError
            If the order exceeds 2S pseudo-spin value.
        SltCompError
            If the ITO decomposition of the matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        The decomposition is obtained using a projection method described in
        [1] (eq. 41) employing ITOs defined in [2] (eq. 29) with
        a normalization factor from eq. 17.

        References
        ----------
        .. [1] L. F. Chibotaru and L. Ungur
            "Ab initio calculation of anisotropic magnetic properties of
            complexes. I. Unique definition of pseudospin Hamiltonians and
            their derivation"
            J. Chem. Phys. 137, 064112 (2012).
        .. [2] I. D. Ryabov
            "On the Generation of Operator Equivalents and the Calculation
            of Their Matrix Elements"
            J. Magn. Reson. 140, 141–145 (1999).
        """
        if slt is not None:
            slt_group_name = f"{slt}_soc_ito_decomposition"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            soc_matrix = _get_soc_matrix_in_z_pseudo_spin_basis(
                self._hdf5,
                group,
                start_state,
                stop_state,
                pseudo_kind,
                rotation,
            )
        except Exception as exc:
            raise SltReadError(
                self._hdf5,
                exc,
                "Failed to read SOC matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        dim = (soc_matrix.shape[1] - 1) / 2

        if not isinstance(order, int) or order < 0 or order > 2 * dim:
            raise SltInputError(
                ValueError(
                    "Order of ITO parameters has to be a positive integer or"
                    " it exceeds 2S. Set it less or equal."
                )
            )

        try:
            if complex:
                cfp = _ito_complex_decomp_matrix(soc_matrix, order, even_order)
            else:
                cfp = _ito_real_decomp_matrix(soc_matrix, order, even_order)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to ITO decompose SOC matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        cfp_return = cfp

        if slt is not None:
            cfp = array(cfp)

            try:
                self[
                    slt_group_name,
                    f"{slt}_ito_parameters",
                    'Dataset containing ITO decomposition in "z"'
                    " pseudo-spin basis of SOC matrix from group:"
                    f" {group}.",
                    f'Group({slt}) containing ITO decomposition in "z"'
                    " pseudo-spin basis of SOC matrix calculated from"
                    f" group: {group}.",
                ] = cfp[:, :]
                self[
                    slt_group_name,
                    f"{slt}_pseudo_spin_states",
                    "Dataset containing S pseudo-spin number"
                    " corresponding to the decomposition of SOC matrix"
                    f" from group: {group}.",
                ] = array([dim])
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save ITO decomposition of SOC matrix to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return cfp_return

    def zeeman_matrix_ito_decpomosition(
        self,
        group: str,
        start_state: int,
        stop_state: int,
        field: float64,
        orientation: ndarray[float64],
        order: int,
        pseudo_kind: Literal["magnetic", "total_angular"],
        complex: bool = False,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> list:
        """
        Calculates ITO decomposition of Zeeman matrix.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for obtaining the Zeeman matrix.
        start_state : int
            Number of the first Zeeman state to be included.
        stop_state : int
            Number of the last Zeeman state to be included. If both start and
            stop are set to zero all available states from the file will be
            used.
        field : float64
            Magnetic field value at which Zeeman matrix will be computed.
        orientation : ndarray[float64]
            Orientation of the magnetic field in the form of an ArrayLike
            structure (can be converted to numpy.NDArray) [direction_x,
            direction_y, direction_z].
        order : int
            Order of the highest ITO (CFP) to be included in the decomposition.
        pseudo_kind : Literal["magnetic", "total_angular"]
            Kind of a pseudo-spin basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis.
        complex : bool, optional
            If True, instead of real ITOs (CFPs) complex ones will be given.,
            by default False
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _zeeman_ito_decomposition., by default None

        Returns
        -------
        list
            The resulting list gives ITOs - B_k_q in the form [k,q,B_k_q]

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the program is unable to calculate Zeeman matrix from the file.
        SltInputError
            If the order exceeds 2S pseudo-spin value
        SltCompError
            If the ITO decomposition of the matrix is unsuccessful
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        The decomposition is obtained using a projection method described in
        [1] (eq. 41) employing ITOs defined in [2] (eq. 29) with
        a normalization factor from eq. 17.

        References
        ----------
        .. [1] L. F. Chibotaru and L. Ungur
            "Ab initio calculation of anisotropic magnetic properties of
            complexes. I. Unique definition of pseudospin Hamiltonians and
            their derivation"
            J. Chem. Phys. 137, 064112 (2012).
        .. [2] I. D. Ryabov
            "On the Generation of Operator Equivalents and the Calculation
            of Their Matrix Elements"
            J. Magn. Reson. 140, 141–145 (1999).
        """
        if slt is not None:
            slt_group_name = f"{slt}_zeeman_ito_decomposition"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        orientation = _normalize_orientation(orientation)

        try:
            zeeman_matrix = _get_zeeman_matrix_in_z_pseudo_spin_basis(
                self._hdf5,
                group,
                field,
                orientation,
                start_state,
                stop_state,
                pseudo_kind,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to calculate Zeeman matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        dim = (zeeman_matrix.shape[1] - 1) / 2

        if not isinstance(order, int) or order < 0 or order > 2 * dim:
            raise SltInputError(
                ValueError(
                    "Order of ITO parameters has to be a positive integer or"
                    " it exceeds 2S. Set it less or equal."
                )
            )

        try:
            if complex:
                ito = _ito_complex_decomp_matrix(zeeman_matrix, order)
            else:
                ito = _ito_real_decomp_matrix(zeeman_matrix, order)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to ITO decompose Zeeman matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        ito_return = ito

        if slt is not None:
            ito = array(ito)

            try:
                self[
                    slt_group_name,
                    f"{slt}_ito_parameters",
                    'Dataset containing ITO decomposition in "z"'
                    " pseudo-spin basis of Zeeman matrix from group:"
                    f" {group}.",
                    f'Group({slt}) containing ITO decomposition in "z"'
                    " pseudo-spin basis of Zeeman matrix calculated from"
                    f" group: {group}.",
                ] = ito[:, :]
                self[
                    slt_group_name,
                    f"{slt}_pseudo_spin_states",
                    "Dataset containing S pseudo-spin number"
                    " corresponding to the decomposition of Zeeman matrix"
                    f" from group: {group}.",
                ] = array([dim])
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save ITO decomposition of Zeeman matrix to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return ito_return

    def matrix_from_ito(
        self,
        full_group_name: str,
        complex: bool,
        dataset_name: str = None,
        pseudo_spin: float64 = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates matrix from a given ITO decomposition.

        Parameters
        ----------
        full_group_name : str
            Full name of a group containing ITO decomposition.
        complex : bool
            Determines the type of ITOs in the dataset. If True, instead of
            real ITOs complex ones will be used., by default False
        dataset_name : str, optional
            A custom name for a user-created dataset within the group that
            contains list of B_k_q parameters in the form [k,q,B_k_q].,
            by default None
        pseudo_spin : float64, optional
            Pseudo spin S value for the user-defined dataset., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _matrix_from_ito., by default None

        Returns
        -------
        ndarray[complex128]
            Matrix reconstructed from a given ITO list.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of the matrix from ITOs is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        ITOs defined in [2] (eq. 29) with a normalization factor from eq. 17
        are used.

        References
        ----------
        .. [1] I. D. Ryabov
            "On the Generation of Operator Equivalents and the Calculation
            of Their Matrix Elements"
            J. Magn. Reson. 140, 141–145 (1999).
        """
        if slt is not None:
            slt_group_name = f"{slt}_matrix_from_ito"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            if (
                (dataset_name is not None)
                and (pseudo_spin is not None)
                and isinstance(pseudo_spin, int)
                and pseudo_spin > 0
            ):
                J = pseudo_spin
                coefficients = self[f"{full_group_name}", f"{dataset_name}"]
                if complex:
                    matrix = _matrix_from_ito_complex(J, coefficients)
                else:
                    matrix = _matrix_from_ito_real(J, coefficients)

            else:
                if full_group_name.endswith("_zeeman_ito_decomposition"):
                    dataset_name = full_group_name[
                        : -len("_zeeman_ito_decomposition")
                    ]
                elif full_group_name.endswith("_soc_ito_decomposition"):
                    dataset_name = full_group_name[
                        : -len("_soc_ito_decomposition")
                    ]
                else:
                    raise NameError(
                        f"Invalid group name: {full_group_name}. It must end"
                        " with _soc_ito_decomposition or"
                        " _zeeman_ito_decomposition."
                    )
                J = self[
                    f"{full_group_name}",
                    f"{dataset_name}_pseudo_spin_states",
                ]
                coefficients = self[
                    f"{full_group_name}",
                    f"{dataset_name}_ito_parameters",
                ]
                if complex:
                    matrix = _matrix_from_ito_complex(J[0], coefficients)
                else:
                    matrix = _matrix_from_ito_real(J[0], coefficients)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute matrix from ITOs from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{full_group_name}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_matrix",
                    "Dataset containing matrix from ITOs calculated from"
                    f" group: {full_group_name}.",
                    f"Group({slt}) containing matrix from ITO calculated"
                    f" from group: {full_group_name}.",
                ] = matrix[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save matrix from ITOs to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return matrix

    def soc_zeem_in_z_angular_magnetic_momentum_basis(
        self,
        group: str,
        start_state: int,
        stop_state: int,
        matrix_type: Literal["soc", "zeeman"],
        basis_kind: Literal["magnetic", "total_angular"],
        rotation: ndarray[float64] = None,
        field: float64 = None,
        orientation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates SOC or Zeeman matrix in "z" magnetic or total angular
        momentum basis.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for obtaining the SOC or Zeeman matrix.
        start_state : int
            Number of the first SOC state to be included.
        stop_state : int
            Number of the last SOC state to be included. If both start and stop
            are set to zero all available states from the file will be used
        matrix_type : Literal["soc", "zeeman"]
            Type of a matrix to be decomposed. Two options available: "soc" or
            "zeeman".
        basis_kind : Literal["magnetic", "total_angular"]
            Kind of a basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        field : float64, optional
            _description_, by default None
        orientation : ndarray[float64], optional
            Orientation of the magnetic field in the form of an ArrayLike
            structure (can be converted to numpy.NDArray) [direction_x,
            direction_y, direction_z]., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _{matrix_type}_matrix_in_{basis_kind}_basis.,
            by default None

        Returns
        -------
        ndarray[complex128]
            Matrix in a given kind of basis.

        Raises
        ------
        SltInputError
            If an unsuported type of matrix or basis is provided.
        SltInputError
            If there is no field value or orientation provided for Zeeman
            matrix.
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of a matrix in "z" basis is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if (matrix_type not in ["zeeman", "soc"]) or (
            basis_kind not in ["total_angular", "magnetic"]
        ):
            raise SltInputError(
                NotImplementedError(
                    "The only valid matrix types and pseudo spin kinds are"
                    ' "soc" or "zeeman" and "magnetic" or "total_angular"'
                    " respectively."
                )
            )

        if matrix_type == "zeeman" and (
            (field is None) or (orientation is None)
        ):
            raise SltInputError(
                ValueError(
                    "For Zeeman matrix provide field value and orientation."
                )
            )

        if slt is not None:
            slt_group_name = (
                f"{slt}_{matrix_type}_matrix_in_{basis_kind}_basis"
            )
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None

        try:
            if matrix_type == "zeeman":
                orientation = _normalize_orientation(orientation)
                matrix = _get_zeeman_matrix_in_z_pseudo_spin_basis(
                    self._hdf5,
                    group,
                    field,
                    orientation,
                    start_state,
                    stop_state,
                    basis_kind,
                    rotation,
                )
            elif matrix_type == "soc":
                matrix = _get_soc_matrix_in_z_pseudo_spin_basis(
                    self._hdf5,
                    group,
                    start_state,
                    stop_state,
                    basis_kind,
                    rotation,
                )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute matrix from ITOs from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_matrix",
                    f"Dataset containing {matrix_type} matrix in"
                    f' {basis_kind} momentum "z" basis calculated from'
                    f" group: {group}.",
                    f"Group({slt}) containing {matrix_type} matrix in"
                    f' {basis_kind} momentum "z" basis calculated from'
                    f" group: {group}.",
                ] = matrix[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f'Failed to save {matrix} matrix in "z"'
                    f" {basis_kind} basis to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return matrix

    def plot_magnetisation(
        self,
        group: str,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = None,
        color_map_name: str or list[str] = "rainbow",
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 1,
        yticks: int or float = 0,
        field: Literal["B", "H"] = "B",
    ):
        """
        Creates graphs of M(H,T) given a name of the group in .slt file, graphs
        can be optionally shown, saved, color palettes can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = None
            Determines name of the file that would be created if save = True,
            if left empty it will use the following format: "magnetisation_
            {group}.tiff".
        color_map_name: str or list[str] = "rainbow"
            Input of the color_map function.
        xlim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the x-axis if two floats
            are passed, or just the upper limit if one is passed.
        ylim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the y-axis if two floats
            are passed, or just the upper limit if one is passed.
        xticks: int or float = 1
            Determines the frequency of x major ticks.
        yticks: int or float = 0
            Determines the frequency of y major ticks.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.SltFile.calculate_magnetisation
        """
        try:
            # Getting data from .slt or sloth file
            mth = self[f"{group}_magnetisation", f"{group}_mth"]
            fields = self[f"{group}_magnetisation", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            temps = self[f"{group}_magnetisation", f"{group}_temperatures"]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load magnetisation data. "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        try:
            # Plotting in matplotlib
            fig, ax = subplots()
            # Defining color maps for graphs
            color = iter(color_map(color_map_name)(linspace(0, 1, len(temps))))
            # Creating a plot
            for i, mh in enumerate(mth):
                c = next(color)
                ax.plot(fields, mh, linewidth=2, c=c, label=f"{temps[i]} K")

            if yticks:
                ax.yaxis.set_major_locator(MultipleLocator(yticks))
            ax.xaxis.set_major_locator(MultipleLocator(xticks))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=7)
            ax.tick_params(which="minor", length=3.5)
            if field == "B":
                ax.set_xlabel(r"$B\ /\ \mathrm{T}$")
            elif field == "H":
                ax.set_xlabel(r"$H\ /\ \mathrm{kOe}$")
            ax.set_ylabel(r"$M\ /\ \mathrm{\mu_{B}}$")
            if xlim:
                if len(xlim) == 2:
                    ax.set_ylim(xlim[0], xlim[1])
                else:
                    ax.set_ylim(xlim[0])
            else:
                if len(temps) > 17:
                    ax.set_xlim(0)
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                else:
                    ax.set_xlim(0, fields[-1] + 0.3 * fields[-1])
                    ax.legend()
            if ylim:
                if len(ylim) == 2:
                    ax.set_ylim(ylim[0], ylim[1])
                else:
                    ax.set_ylim(ylim[0])
            else:
                ax.set_ylim(0)
            tight_layout()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot magnetisation data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                # Saving plot figure
                if save_name is None:
                    filename = path.join(
                        save_path, f"magnetisation_{group}.tiff"
                    )
                else:
                    filename = path.join(save_path, save_name)
                fig.savefig(filename, dpi=600)
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save magnetisation data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"{filename}",
                ) from None
        close("all")

    def plot_susceptibility(
        self,
        group: str,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = None,
        color_map_name: str or list[str] = "funmat",
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 100,
        yticks: int or float = 0,
        field: Literal["B", "H"] = "B",
    ):
        """
        Creates graphs of chiT(H,T) or chi(H,T) depending on the content of
        .slt file, given a name of the group in .slt file, graphs can be
        optionally saved, color palettes can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = None
            Determines name of the file that would be created if save = True,
            if left empty it will use the following format:
            "susceptibility_{group}.tiff".
        color_map_name: str or list[str] = 'funmat'
            Input of color_map function.
        xlim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the x-axis if two floats
            are passed, or just the upper limit if one is passed.
        ylim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the y-axis if two floats
            are passed, or just the upper limit if one is passed.
        xticks: int or float = 100
            Determines the frequency of x major ticks.
        yticks: int or float = 0
            Determines the frequency of y major ticks.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.SltFile.calculate_susceptibility
        """
        try:
            # Getting data from .slt or sloth file
            try:
                chi = self[f"{group}_susceptibility", f"{group}_chiht"]
                T = False
            except Exception as exc:
                chi = self[f"{group}_susceptibility", f"{group}_chitht"]
                T = True
            fields = self[f"{group}_susceptibility", f"{group}_fields"]
            if field == "H":
                fields *= 10
            temps = self[f"{group}_susceptibility", f"{group}_temperatures"]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load susceptibility data. "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        try:
            # Plotting in matplotlib
            fig, ax = subplots()
            # Defining color maps for graphs
            color = iter(
                color_map(color_map_name)(linspace(0, 1, len(fields)))
            )
            # Creating a plot
            for i, ch in enumerate(chi):
                c = next(color)
                ax.plot(
                    temps,
                    ch,
                    linewidth=2,
                    c=c,
                    label=(
                        f'{round(fields[i], 2)} {"kOe" if field == "H" else "T"}'
                    ),
                )
            ax.xaxis.set_major_locator(MultipleLocator(xticks))
            if yticks:
                ax.yaxis.set_major_locator(MultipleLocator(yticks))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=7)
            ax.tick_params(which="minor", length=3.5)
            ax.set_xlabel(r"$T\ /\ \mathrm{K}$")
            if T:
                ax.set_ylabel(
                    r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                )
            else:
                ax.set_ylabel(
                    r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$"
                )
            if xlim:
                if len(xlim) == 2:
                    ax.set_ylim(xlim[0], xlim[1])
                else:
                    ax.set_ylim(xlim[0])
            else:
                ax.set_xlim(0, temps[-1])
            if ylim:
                if len(ylim) == 2:
                    ax.set_ylim(ylim[0], ylim[1])
                else:
                    ax.set_ylim(ylim[0])
            else:
                ax.set_ylim(0)
            ax.legend()
            tight_layout()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot susceptibility data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                # Saving plot figure
                if save_name is None:
                    filename = path.join(
                        save_path, f"susceptibility_{group}.tiff"
                    )
                else:
                    filename = path.join(save_path, save_name)
                fig.savefig(filename, dpi=300)
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save susceptibility data plot from "
                    + BLUE
                    + "Group: "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", file: '
                    + PURPLE
                    + f"{filename}",
                ) from None
        close("all")

    def plot_energy(
        self,
        group: str,
        energy_type: Literal["helmholtz", "internal"],
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = None,
        color_map_name: str or list[str] = "PrOr",
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 1,
        yticks: int or float = 0,
        field: Literal["B", "H"] = "B",
    ):
        """
        Creates graphs of Helmholtz energy F(T,H) or internal energy U(T,H)
        given a name of the group in .slt file, graphs can be optionally saved,
        color palettes can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        energy_type: Literal["helmholtz", "internal"]
            Determines which kind of energy, Helmholtz or internal, will be
            calculated.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = None
            Determines name of the file that would be created if save = True,
            if left empty it will use following format: "{energy_type}_
            energy_{group}.tiff".
        color_map_name: str or list[str] = 'PrOr'
            Input of the color_map function.
        xlim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the x-axis if two floats
            are passed, or just the upper limit if one is passed.
        ylim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the y-axis if two floats
            are passed, or just the upper limit if one is passed.
        xticks: int or float = 100
            Determines the frequency of x major ticks.
        yticks: int = 0
            Determines the freqency of y major ticks.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create plot.
        SltSaveError
            If unable to save plot as image.

        See Also
        --------
        slothpy.SltFile.calculate_energy
        """
        if energy_type == "internal":
            name = "internal"
        elif energy_type == "helmholtz":
            name = "helmholtz"
        else:
            raise SltInputError(
                ValueError(
                    'Energy type must be set to "helmholtz" or "internal".'
                )
            ) from None
        try:
            # Getting data from .slt or sloth file
            eth = self[f"{group}_{name}_energy", f"{group}_eth"]
            fields = self[f"{group}_{name}_energy", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            temps = self[f"{group}_{name}_energy", f"{group}_temperatures"]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load energy data. "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + RED,
            ) from None

        try:
            # Plotting in matplotlib
            fig, ax = subplots()
            # Defining color maps for graphs
            color = iter(color_map(color_map_name)(linspace(0, 1, len(temps))))
            # Creating a plot
            for i, eh in enumerate(eth):
                c = next(color)
                ax.plot(fields, eh, linewidth=2, c=c, label=f"{temps[i]} K")

            if yticks:
                ax.yaxis.set_major_locator(MultipleLocator(yticks))
            ax.xaxis.set_major_locator(MultipleLocator(xticks))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="major", length=7)
            ax.tick_params(which="minor", length=3.5)
            if field == "B":
                ax.set_xlabel(r"$B\ /\ \mathrm{T}$")
            elif field == "H":
                ax.set_xlabel(r"$H\ /\ \mathrm{kOe}$")
            ax.set_ylabel(r"$E\ /\ \mathrm{cm^{-1}}$")
            if xlim:
                if len(xlim) == 2:
                    ax.set_ylim(xlim[0], xlim[1])
                else:
                    ax.set_ylim(xlim[0])
            else:
                if len(temps) > 17:
                    ax.set_xlim(0, fields[-1])
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                else:
                    ax.set_xlim(0, fields[-1] + 0.3 * fields[-1])
                    ax.legend()
            if ylim:
                if len(ylim) == 2:
                    ax.set_ylim(ylim[0], ylim[1])
                else:
                    ax.set_ylim(ylim[0])
            tight_layout()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot energy data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                # Saving plot figure
                if save_name is None:
                    filename = path.join(
                        save_path, f"{name}_energy_{group}.tiff"
                    )
                else:
                    filename = path.join(save_path, save_name)
                fig.savefig(filename, dpi=600)
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save energy data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"energyth_{group}.tiff",
                ) from None
        close("all")

    def plot_zeeman(
        self,
        group: str,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = None,
        color_map_name1: str or list[str] = "BuPi",
        color_map_name2: str or list[str] = "BuPi_r",
        single: bool = False,
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 1,
        yticks: int or float = 0,
        field: Literal["H", "B"] = "B",
    ):
        """
        Function that creates graphs of E(H,orientation) given a name of
        the group in .slt file, graphs can be optionally saved, color palettes
        can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which plot will be created.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = None
            Determines name of the file that would be created if save = True,
            if left empty it will use following format: f"zeeman_{group}.tiff"
            or f"zeeman_{group}_{orientation[i]}.tiff".
        color_map_name1: str or list[str] = 'BuPi'
            Input of the color_map function, determines a color of the lower
            set of split lines.
        color_map_name2: str or list[str] = 'BuPi_r'
            Input of the color_map function, determines a color of the higher
            set of split lines.
        single: bool = False
            Determines if all orientations are plotted together if plot is not
            a result of averaging.
        xlim: tuple of 1-2 floats = ()
            Determines the lower and upper limit of x-axis if two floats are
            passed, or just the upper limit if one is passed.
        ylim: tuple of 1-2 floats = ()
            Determines the lower and upper limit of y-axis if two floats are
            passed, or just the upper limit if one is passed.
        xticks: int or float = 1
            Determines the frequency of x major ticks.
        yticks: int or float = 0
            Determines the frequency of y major ticks.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.SltFile.calculate_zeeman_splitting
        """
        try:
            # Getting data from .slt
            zeeman = self[f"{group}", "ZEEMAN_SPLITTING"][:]
            fields = self[f"{group}", "MAGNETIC_FIELDS"][:]
            if field == "H":
                fields *= 10
                xticks *= 10
            orientations = self[
                f"{group}", f"ORIENTATIONS"
            ][:]

        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                f"Failed to load Zeeman splitting data. "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + RED,
            ) from None

        try:
            if orientations.shape[1] != 3:
                single = True
            # Plotting in matplotlib
            if not single:
                number_of_plots = orientations.shape[0]
                if number_of_plots % 5 == 0:
                    fig = figure(figsize=(16, 3.2 * (number_of_plots / 5)))
                    gs = GridSpec(int(number_of_plots / 5), 5)
                    divisor = 5
                elif number_of_plots % 3 == 0:
                    fig = figure(figsize=(9.6, 3.2 * (number_of_plots / 3)))
                    gs = GridSpec(int(number_of_plots / 3), 3)
                    divisor = 3
                elif number_of_plots % 2 == 0:
                    fig = figure(figsize=(6.4, 3.2 * (number_of_plots / 2)))
                    gs = GridSpec(int(number_of_plots / 2), 2)
                    divisor = 2
                else:
                    fig = figure(figsize=(6.4, 3.2 * number_of_plots))
                    gs = GridSpec(1, number_of_plots)
                    divisor = 1
                # Creating a plot
                for i, zee in enumerate(zeeman):
                    if i % divisor != 0:
                        rc(
                            "axes",
                            prop_cycle=_custom_color_cycler(
                                len(zeeman[0][0]),
                                color_map_name1,
                                color_map_name2,
                            ),
                        )
                        multiple_plots = fig.add_subplot(
                            gs[i // divisor, i % divisor]
                        )
                        plot(fields, zee, linewidth=0.75)
                        multiple_plots.xaxis.set_major_locator(
                            MultipleLocator(xticks * 2)
                        )
                        if yticks:
                            multiple_plots.yaxis.set_major_locator(
                                MultipleLocator(yticks)
                            )
                        multiple_plots.xaxis.set_minor_locator(
                            AutoMinorLocator(2)
                        )
                        multiple_plots.yaxis.set_minor_locator(
                            AutoMinorLocator(2)
                        )
                        multiple_plots.tick_params(
                            which="major",
                            left=False,
                            labelleft=False,
                            length=7,
                        )
                        multiple_plots.tick_params(
                            which="minor", left=False, length=3.5
                        )
                        if orientations.shape[1] != 3:
                            title("Averaged Splitting")
                        else:
                            title(
                                f"Orientation [{round(orientations[i][0], 3)} "
                                + f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                            )
                        if xlim:
                            if len(xlim) == 2:
                                multiple_plots.set_xlim(xlim[0], xlim[1])
                            else:
                                multiple_plots.set_xlim(xlim[0])
                        if ylim:
                            if len(ylim) == 2:
                                multiple_plots.set_ylim(ylim[0], ylim[1])
                            else:
                                multiple_plots.set_ylim(ylim[0])

                    else:
                        if (i // divisor) == 0:
                            rc(
                                "axes",
                                prop_cycle=_custom_color_cycler(
                                    len(zeeman[0][0]),
                                    color_map_name1,
                                    color_map_name2,
                                ),
                            )
                            multiple_plots = fig.add_subplot(
                                gs[i // divisor, i % divisor]
                            )
                            plot(fields, zee, linewidth=0.75)
                            multiple_plots.xaxis.set_major_locator(
                                MultipleLocator(xticks * 2)
                            )
                            if yticks:
                                multiple_plots.yaxis.set_major_locator(
                                    MultipleLocator(yticks)
                                )
                            multiple_plots.xaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            multiple_plots.tick_params(which="major", length=7)
                            multiple_plots.tick_params(
                                which="minor", length=3.5
                            )
                            multiple_plots.yaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            if orientations.shape[1] != 3:
                                title("Averaged Splitting")
                            else:
                                title(
                                    "Orientation"
                                    f" [{round(orientations[i][0], 3)} "
                                    f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                                )
                            if xlim:
                                if len(xlim) == 2:
                                    multiple_plots.set_xlim(xlim[0], xlim[1])
                                else:
                                    multiple_plots.set_xlim(xlim[0])
                            if ylim:
                                if len(ylim) == 2:
                                    multiple_plots.set_ylim(ylim[0], ylim[1])
                                else:
                                    multiple_plots.set_ylim(ylim[0])
                        else:
                            rc(
                                "axes",
                                prop_cycle=_custom_color_cycler(
                                    len(zeeman[0][0]),
                                    color_map_name1,
                                    color_map_name2,
                                ),
                            )
                            multiple_plots = fig.add_subplot(
                                gs[i // divisor, i % divisor]
                            )
                            plot(fields, zee, linewidth=0.75)
                            multiple_plots.xaxis.set_major_locator(
                                MultipleLocator(xticks * 2)
                            )
                            if yticks:
                                multiple_plots.yaxis.set_major_locator(
                                    MultipleLocator(yticks)
                                )
                            multiple_plots.xaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            multiple_plots.tick_params(which="major", length=7)
                            multiple_plots.tick_params(
                                which="minor", length=3.5
                            )
                            multiple_plots.yaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            if orientations.shape[1] != 3:
                                title("Averaged Splitting")
                            else:
                                title(
                                    "Orientation"
                                    f" [{round(orientations[i][0], 3)} "
                                    f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                                )
                            if xlim:
                                if len(xlim) == 2:
                                    multiple_plots.set_xlim(xlim[0], xlim[1])
                                else:
                                    multiple_plots.set_xlim(xlim[0])
                            if ylim:
                                if len(ylim) == 2:
                                    multiple_plots.set_ylim(ylim[0], ylim[1])
                                else:
                                    multiple_plots.set_ylim(ylim[0])
                if field == "B":
                    fig.supxlabel(r"$B\ /\ \mathrm{T}$")
                if field == "H":
                    fig.supxlabel(r"$H\ /\ \mathrm{kOe}$")
                fig.supylabel(r"$\mathrm{Energy\ /\ cm^{-1}}$")
                tight_layout()
                if show_fig:
                    _display_plot(fig, partial(close, "all"))
            elif single:
                for i, zee in enumerate(zeeman):
                    rc(
                        "axes",
                        prop_cycle=_custom_color_cycler(
                            len(zeeman[0][0]),
                            color_map_name1,
                            color_map_name2,
                        ),
                    )
                    fig, ax = subplots()
                    ax.plot(fields, zee, linewidth=0.75)
                    if orientations.shape[1] != 3:
                        title("Averaged Splitting")
                    else:
                        title(
                            "Orientation"
                            f" [{round(orientations[i][0], 3)} "
                            f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                        )
                    if field == "B":
                        ax.set_xlabel(r"$B\ /\ \mathrm{T}$")
                    elif field == "H":
                        ax.set_xlabel(r"$H\ /\ \mathrm{kOe}$")
                    ax.set_ylabel(r"$\mathrm{Energy\ /\ cm^{-1}}$")
                    ax.tick_params(which="major", length=7)
                    ax.tick_params(which="minor", length=3.5)
                    ax.xaxis.set_major_locator(MultipleLocator(xticks))
                    if yticks:
                        ax.yaxis.set_major_locator(MultipleLocator(yticks))
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    if xlim:
                        if len(xlim) == 2:
                            ax.set_xlim(xlim[0], xlim[1])
                        else:
                            ax.set_xlim(xlim[0])
                    if ylim:
                        if len(ylim) == 2:
                            ax.set_ylim(ylim[0], ylim[1])
                        else:
                            ax.set_ylim(ylim[0])
                    tight_layout()
                    if show_fig:
                        _display_plot(fig, partial(close, "all"))
                    if save:
                        try:
                            # Saving plot figure
                            if save_name is None:
                                filename = path.join(
                                    save_path,
                                    f"zeeman_{group}_Orientation"
                                    f" {orientations[i]}.tiff",
                                )
                            else:
                                filename = path.join(save_path, save_name)
                            fig.savefig(
                                filename,
                                dpi=600,
                            )
                        except Exception as exc:
                            close("all")
                            raise SltSaveError(
                                self._hdf5,
                                exc,
                                f"Failed to save zeeman splitting data plot"
                                + BLUE
                                + "Group "
                                + RESET
                                + '"'
                                + BLUE
                                + f"{group}"
                                + RESET
                                + '", filename: '
                                + PURPLE
                                + f"zeeman_{group}_Orientation"
                                + f" {filename}",
                            ) from None

        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                f"Failed to plot zeeman splitting data"
                + BLUE
                + "Group: "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save and not single:
            try:
                # Saving plot figure
                if save_name is None:
                    filename = path.join(save_path, f"zeeman_{group}.tiff")
                else:
                    filename = path.join(save_path, save_name)
                fig.savefig(filename, dpi=600)
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    f"Failed to save zeeman splitting data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"{filename}",
                ) from None
        close("all")

    def plot_3d(
        self,
        group: str,
        data_type: Literal[
            "chit",
            "chi",
            "helmholtz_energy",
            "internal_energy",
            "magnetisation",
        ],
        field_i: int,
        temp_i: int,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = None,
        color_map_name: str or list[str] = "dark_rainbow_r_l",
        round_temp: int = 3,
        round_field: int = 3,
        lim_scalar: float = 1.0,
        ticks: float = 1.0,
        plot_title: str = None,
        field: Literal["B", "H"] = "B",
        r_density: int = 0,
        c_density: int = 0,
        points_size: float = 0.2,
        solid_surface: bool = False,
        elev: int = 30,
        azim: int = -60,
        roll: int = 0,
        axis_off: bool = False,
        add_g_tensor_axes: bool = False,
        axes_group: str = "",
        axes_colors: list[str] = ["r", "g", "b"],
        doublet_number: int = None,
        axes_scale_factor: float64 = 1.0,
        rotation: ndarray[float64] = None,
    ):
        """
        Creates 3d plots of data dependent on field B[T] and temperature T[K].

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        data_type: Literal["chit", "chi", "helmholtz_energy", "internal_energy",
          "magnetisation"]
            Type of the data that will be used to create plot.
        field_i: int
            Index of the field from dataset that will be used for the plot.
        temp_i: int
            Index of the temperature from the dataset that will be used for the
            plot.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines path where file will be saved if save = True.
        save_name: str = None
            Determines name of a file that would be created if save = True,
            if left empty it will use following format:
            f"{group}_3d_{data_type}.tiff".
        color_map_name: str or list[str] = 'dark_rainbow_r_l'
            Input of the color_map function.
        round_temp: int = 3
            Determines how many digits will be rounded in the graph's title
            for temperature.
        round_field: int = 3
            Determines how many digits will be rounded in the graph's title
            for field.
        lim_scalar: float = 1.
            Scalar used to set limits of the axes, smaller values magnify the
            plotted figure.
        ticks: float = 1.
            Frequency of the ticks on all axes.
        plot_title: str = None
            Determines the title of the figure, if left blank automatic title is used.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].
        r_density: int = 0
            Determines the rcount of a 3D plot.
        c_density: int = 0
            Determines the ccount of a 3D plot.
        points_size: float = 0.2
            Determines points size for Fibonacci scatter plots.
        solid_surface: bool = False
            Makes surface plots using meshgrid appear as solid.
        elev: int = 30
            Determines an angle between a viewing position and the xy plane.
        azim: int = -60
            Determines a rotation of a viewing position in relation to z axis.
        roll: int = 0
            Determines a rotation of camera around the viewing (position) axis.
        axis_off: bool = False
            Determines if the axes are turned off.
        add_g_tensor_axes: bool = False
            Determines if add to the plot main magnetic axes scaled by the
            corresponding pseudo-g-tensor values.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        axes_colors: list[str] = ['r','g','b']
            Determines the colors of the magnetic axes in order of x, y, z.
            Accepts matplotlib colors inputs, for example HTML color codes.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining the length of the longest (main) magnetic
            axis concerning the maximal value of the loaded data and setting
            a maximal limit of the plot's xyz axes. It should be set > 1
            otherwise, some data will end up missing from the plot! The limit
            is max(loaded_data) * axes_scale_factor.
        rotation: ndarray[float64] = None
            Has to be given if 3d data was calculated with optional rotation of
            the coordinate frame and add_g_tensor_axes option is turned on.
            One must provide the same rotation as that used for the simulation
            to apply it to the magnetic axes.

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load data file. Most likely encountered if the group
            name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.SltFile.calculate_magnetisation_3d, slothpy.SltFile.calculate_susceptibility_3d,
        slothpy.SltFile.calculate_energy_3d,
        slothpy.SltFile.calculate_g_tensor_axes
        """
        if (not isinstance(axes_scale_factor, (float, int))) or (
            axes_scale_factor < 1
        ):
            raise SltInputError(
                ValueError(
                    "Axes scale factor has to be a float greater than 1."
                )
            )

        try:
            if (data_type == "chi") or (data_type == "chit"):
                group_name = f"{group}_susceptibility_3d"
                xyz = self[group_name, f"{group}_chit_3d"]
                if data_type == "chit":
                    label = (
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                    )
                else:
                    label = r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$"
                description1 = f"Susceptibility dependence on direction,"
            elif (data_type == "helmholtz_energy") or (
                data_type == "internal_energy"
            ):
                if data_type == "helmholtz_energy":
                    label = r"$F\ /\ \mathrm{cm^{-1}}$"
                    group_name = f"{group}_helmholtz_energy_3d"
                    energy_type = "Helmholtz"
                else:
                    group_name = f"{group}_internal_energy_3d"
                    label = r"$U\ /\ \mathrm{cm^{-1}}$"
                    energy_type = "Internal"
                xyz = self[group_name, f"{group}_energy_3d"]
                description1 = f"{energy_type} energy dependence on direction,"
            elif data_type == "magnetisation":
                group_name = f"{group}_magnetisation_3d"
                label = r"$M\ /\ \mathrm{\mu_{B}}$"
                xyz = self[group_name, f"{group}_mag_3d"]
                description1 = "Magnetisation dependence on direction,"
            else:
                raise ValueError
            field_scalar = 10 if field == "H" else 1
            description2 = (
                f" {field}={round(self[group_name, f'{group}_fields'][field_i]*field_scalar, round_field)} {'T' if field == 'B' else 'kOe'}, "
                f"T={round(self[group_name, f'{group}_temperatures'][temp_i], round_temp)} K"
            )
            description = description1 + description2
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load the 3D data file "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '". ',
            ) from None
        try:
            fig = figure()
            ax = fig.add_subplot(projection="3d")
            if xyz.ndim == 5:
                x = xyz[0, field_i, temp_i, :, :]
                y = xyz[1, field_i, temp_i, :, :]
                z = xyz[2, field_i, temp_i, :, :]
            elif xyz.ndim == 4:
                x = xyz[field_i, temp_i, :, 0]
                y = xyz[field_i, temp_i, :, 1]
                z = xyz[field_i, temp_i, :, 2]
            max_array = array([max(x), max(y), max(z)])
            lim = max(max_array) * axes_scale_factor
            norm = Normalize(z.min(), z.max())
            colors = color_map(color_map_name)(norm(z))
            if z.ndim == 1:
                rcount, ccount = colors.shape
            else:
                rcount, ccount, _ = colors.shape
            if not r_density:
                r_density = rcount
            if not c_density:
                c_density = ccount
            if xyz.ndim == 5:
                surface = ax.plot_surface(
                    x,
                    y,
                    z,
                    rcount=r_density,
                    ccount=c_density,
                    facecolors=colors,
                    shade=False,
                )
                if not solid_surface:
                    surface.set_facecolor((0, 0, 0, 0))
            if xyz.ndim == 4:
                ax.scatter(x, y, z, s=points_size, facecolors=colors)
            ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
            # Important order of operations!
            labelpad = 20 * len(str(ticks)) / 4
            labelpad_scalar = 1
            if data_type == "magnetisation":
                labelpad_scalar = labelpad_scalar * 0.5

            ax.set_xlabel(label, labelpad=labelpad * labelpad_scalar)
            ax.set_ylabel(label, labelpad=labelpad * labelpad_scalar)
            ax.set_zlabel(label, labelpad=labelpad * labelpad_scalar)

            if ticks == 0:
                for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis_i.set_ticklabels([])
                    axis_i._axinfo["axisline"]["linewidth"] = 1
                    axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                    axis_i._axinfo["grid"]["linewidth"] = 0.5
                    axis_i._axinfo["grid"]["linestyle"] = "-"
                    axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                    axis_i._axinfo["tick"]["inward_factor"] = 0.0
                    axis_i._axinfo["tick"]["outward_factor"] = 0.0
                    axis_i.set_pane_color((0.95, 0.95, 0.95))
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                if ticks != 1:
                    ax.xaxis.set_major_locator(MultipleLocator(ticks))
                    ax.yaxis.set_major_locator(MultipleLocator(ticks))
                    ax.zaxis.set_major_locator(MultipleLocator(ticks))
            ax.grid(False)

            ax.set_box_aspect([1, 1, 1])
            ax.azim = azim
            ax.elev = elev
            ax.roll = roll
            if plot_title is None:
                title(description)
            else:
                title(plot_title)
            if add_g_tensor_axes:
                doublets = axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ].shape[0]
                if (
                    (not isinstance(doublet_number, int))
                    or (doublet_number < 0)
                    or (doublet_number > doublets - 1)
                ):
                    raise SltInputError(
                        ValueError(
                            "Doublet number must be a not negative integer"
                            " less or equal to the number of doublets in the"
                            " axes group."
                        )
                    ) from None
                axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ][doublet_number]
                g_tensor = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_g_tensors"
                ][doublet_number]

                vec = axes_matrix * g_tensor[newaxis, 1:]
                if rotation is not None:
                    if rotation.shape != (3, 3):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be a 3x3 matrix."
                            )
                        ) from None
                    product = rotation.T @ rotation
                    if not allclose(product, identity(3), atol=1e-2, rtol=0):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be orthogonal."
                            )
                        ) from None
                    vec = rotation @ vec
                max_vec = max(vec)
                vec = vec * lim / max_vec
                for i in range(3):
                    ax.plot(
                        [vec[0, i], -vec[0, i]],
                        [vec[1, i], -vec[1, i]],
                        [vec[2, i], -vec[2, i]],
                        axes_colors[i],
                        linewidth=3,
                    )
            if axis_off:
                ax.set_axis_off()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot 3D data "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                if axis_off:
                    transp = True
                else:
                    transp = False
                if save_name == "":
                    save_name = f"{group}_3d_{data_type}.tiff"
                fig.savefig(
                    path.join(save_path, save_name),
                    dpi=600,
                    transparent=transp,
                )
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save the 3D data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"{group}_3d_{data_type}.tiff",
                ) from None
        close("all")

    def animate_3d(
        self,
        group: str,
        data_type: Literal[
            "chit",
            "chi",
            "helmholtz_energy",
            "internal_energy",
            "magnetisation",
        ],
        animation_variable: Literal["temperature", "field"],
        filename: str,
        i_start: int = 0,
        i_end: int = -1,
        i_constant: int = 0,
        color_map_name: str or list[str] = "dark_rainbow_r_l",
        lim_scalar: float = 1.0,
        ticks: int or float = 1,
        plot_title: str = None,
        field: Literal["B", "H"] = "B",
        r_density: int = 0,
        c_density: int = 0,
        points_size: float = 0.2,
        solid_surface: bool = False,
        axis_off: bool = False,
        fps: int = 15,
        dpi: int = 300,
        bar: bool = True,
        bar_scale: bool = False,
        bar_color_map_name: str or list[str] = "dark_rainbow_r",
        temp_rounding: int = 0,
        field_rounding: int = 0,
        elev: int = 30,
        azim: int = -60,
        roll: int = 0,
        add_g_tensor_axes: bool = False,
        axes_group: str = "",
        axes_colors: list[str] = ["r", "g", "b"],
        doublet_number: int = None,
        axes_scale_factor: float64 = 1.0,
        rotation: ndarray[float64] = None,
    ):
        """
        Creates animations of 3d plots dependent on field B[T]
        and temperature T[K].

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which plot will be created.
        data_type: Literal["chit", "chi", "helmholtz_energy", "internal_energy",
          "magnetisation"]
            Type of data that will be used to create plot.
        animation_variable: Literal["temperature", "field"]
            Variable changing during animation, can take one of two values:
            temperature or field.
        filename: str
            Name of the output .gif file.
        i_start: int = 0
            Index of first frame's field/temperature.
        i_end: int = -1
            Index of last frame's field/temperature.
        i_constant: int
            Index of constant temperature/field.
        color_map_name: str or list = 'dark_rainbow_r_l'
            Input of color_map function, determines color of main figure.
        lim_scalar: float = 1.
            Scalar used to set limits of axes, smaller values magnify plotted
            figure.
        ticks: float = 1
            Determines the ticks spacing.
        plot_title: str = None
            Determines the title of the figure, if left blank automatic title is used.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].
        r_density: int = 0
            Determines rcount of 3D plot.
        c_density: int = 0
            Determines ccount of 3D plot.
        points_size: float = 0.2
            Determines points size for Fibonacci scatter plots.
        solid_surface: bool = False
            Makes surface plots using meshgrid appear as solid.
        axis_off: bool = False
            Determines if axes are turned off.
        fps: int
            Number of frames per second in animation.
        dpi: int
            Dots per inch resolution of frames.
        bar: bool = True
            Determines if bar representing animation variable is shown.
        bar_scale: bool = False
            Determines if a scale should be shown for bar.
        bar_color_map_name: str or list = 'dark_rainbow_r_l'
            Input of the color_map function, determines the color of the bar.
        temp_rounding: int = 0
            Determines how many decimal places are shown in bar/plot labels
            for temperatures.
        field_rounding: int = 0
            Determines how many decimal places are shown in bar/plot labels
            for fields.
        elev: int = 30
            Determines an angle between a viewing position and the xy plane.
        azim: int = -60
            Determines a rotation of a viewing position in ralation to z axis.
        roll: int = 0
            Determines a rotation of camera around the viewing (position) axis.
        add_g_tensor_axes: bool = False
            Determines if add to the plot main magnetic axes scaled by the
            corresponding pseudo-g-tensor values.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        axes_colors: list[str] = ['r','g','b']
            Determines the colors of the magnetic axes in order of x, y, z.
            Accepts matplotlib colors inputs, for example HTML color codes.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining the length of the longest (main) magnetic
            axis concerning the maximal value of the loaded data and setting
            a maximal limit of the plot's xyz axes. It should be set > 1
            otherwise, some data will end up missing from the plot! The limit
            is max(loaded_data) * axes_scale_factor.
        rotation: ndarray[float64] = None
            Has to be given if 3d data was calculated with optional rotation of
            the coordinate frame and add_g_tensor_axes option is turned on.
            One must provide the same rotation as that used for the simulation
            to apply it to the magnetic axes.

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.

        See Also
        --------
        slothpy.SltFile.calculate_magnetisation_3d, slothpy.SltFile.calculate_susceptibility_3d,
        slothpy.SltFile.calculate_energy_3d,
        slothpy.SltFile.calculate_g_tensor_axes
        """
        if (not isinstance(axes_scale_factor, (float, int))) or (
            axes_scale_factor < 1
        ):
            raise SltInputError(
                ValueError(
                    "Axes scale factor has to be a float greater than 1."
                )
            )

        try:
            if (data_type == "chi") or (data_type == "chit"):
                group_name = f"{group}_susceptibility_3d"
                x0y0z0 = self[group_name, f"{group}_chit_3d"]
                if data_type == "chit":
                    label = (
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                    )
                else:
                    label = r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$"
                data_description = "Susceptibility dependence on direction, "
            elif (data_type == "helmholtz_energy") or (
                data_type == "internal_energy"
            ):
                if data_type == "helmholtz_energy":
                    group_name = f"{group}_helmholtz_energy_3d"
                    label = r"$F\ /\ \mathrm{cm^{-1}}$"
                    data_description = (
                        "Helmholtz energy dependence on direction, "
                    )
                else:
                    group_name = f"{group}_internal_energy_3d"
                    label = r"$U\ /\ \mathrm{cm^{-1}}$"
                    data_description = (
                        "Internal energy dependence on direction, "
                    )
                x0y0z0 = self[group_name, f"{group}_energy_3d"]
            elif data_type == "magnetisation":
                group_name = f"{group}_magnetisation_3d"
                label = r"$M\ /\ \mathrm{\mu_{B}}$"
                data_description = (
                    "Magnetisation energy dependence on direction, "
                )
                x0y0z0 = self[group_name, f"{group}_mag_3d"]
            else:
                raise ValueError(
                    "Acceptable data types: chit, chi, helmholtz_energy and"
                    " magnetisation"
                )
            fields = self[group_name, f"{group}_fields"]
            field_unit = "T"
            if field == "H":
                fields = fields * 10
                field_unit = "kOe"
            temps = self[group_name, f"{group}_temperatures"]
            if x0y0z0.ndim == 5:
                plot_style = "globe"
            elif x0y0z0.ndim == 4:
                plot_style = "scatter"
            if add_g_tensor_axes:
                doublets = axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ].shape[0]
                if (
                    (not isinstance(doublet_number, int))
                    or (doublet_number < 0)
                    or (doublet_number > doublets - 1)
                ):
                    raise SltInputError(
                        ValueError(
                            "Doublet number must be a not negative integer"
                            " less or equal to the number of doublets in the"
                            " axes group."
                        )
                    ) from None
                axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ][doublet_number]
                g_tensor = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_g_tensors"
                ][doublet_number]

        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load the 3D data file "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '". ',
            ) from None

        if animation_variable == "temperature":
            description = f"{field}={fields[i_constant]:.4f} {field_unit}"
        elif animation_variable == "field":
            description = f"T={temps[i_constant]:.4f} K"
        else:
            raise ValueError(
                "There exist only two animation variables: field and"
                " temperature"
            )
        if plot_title is None:
            plot_title = data_description + description

        fig = figure()
        ax = fig.add_subplot(projection="3d")
        if i_end == -1:
            i_end = (
                len(fields) - 1
                if animation_variable == "field"
                else len(temps) - 1
            )
        if bar:
            color = iter(
                color_map(bar_color_map_name)(linspace(0, 1, i_end - i_start))
            )
            indicator = linspace(0, 1, i_end - i_start)
            if bar_scale:

                def my_ticks(x, pos):
                    if animation_variable == "temperature":
                        return f"{round(x * temps[-1], temp_rounding)} K"
                    else:
                        return (
                            f"{round(x * fields[-1], field_rounding)} {field_unit}"
                        )

        labelpad = 20 * len(str(ticks)) / 4
        labelpad_scalar = 1
        if data_type == "magnetisation":
            labelpad_scalar = labelpad_scalar * 0.5

        try:
            writer = PillowWriter(fps=fps)
            with writer.saving(fig, f"{filename}.gif", dpi):
                surface = None
                if animation_variable == "temperature":
                    for i_temp in range(i_start, i_end):
                        if plot_style == "globe":
                            x = x0y0z0[0, i_constant, i_temp, :, :]
                            y = x0y0z0[1, i_constant, i_temp, :, :]
                            z = x0y0z0[2, i_constant, i_temp, :, :]
                            max_array = array([max(x), max(y), max(z)])
                            lim = max(max_array) * axes_scale_factor
                            norm = Normalize(z.min(), z.max())
                            colors = color_map(color_map_name)(norm(z))
                            rcount, ccount, _ = colors.shape
                            if not r_density:
                                r_density = rcount
                            if not c_density:
                                c_density = ccount
                            surface = ax.plot_surface(
                                x,
                                y,
                                z,
                                rcount=r_density,
                                ccount=c_density,
                                facecolors=colors,
                                shade=False,
                            )
                        elif plot_style == "scatter":
                            x = x0y0z0[i_constant, i_temp, :, 0]
                            y = x0y0z0[i_constant, i_temp, :, 1]
                            z = x0y0z0[i_constant, i_temp, :, 2]
                            max_array = array([max(x), max(y), max(z)])
                            lim = max(max_array) * axes_scale_factor
                            norm = Normalize(z.min(), z.max())
                            colors = color_map(color_map_name)(norm(z))
                            surface = ax.scatter(
                                x, y, z, s=points_size, facecolors=colors
                            )

                        ax.set_xlabel(
                            label, labelpad=labelpad * labelpad_scalar
                        )
                        ax.set_ylabel(
                            label, labelpad=labelpad * labelpad_scalar
                        )
                        ax.set_zlabel(
                            label, labelpad=labelpad * labelpad_scalar
                        )

                        if add_g_tensor_axes:
                            vec = axes_matrix * g_tensor[newaxis, 1:]
                            if rotation is not None:
                                if rotation.shape != (3, 3):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be a"
                                            " 3x3 matrix."
                                        )
                                    ) from None
                                product = rotation.T @ rotation
                                if not allclose(
                                    product, identity(3), atol=1e-2, rtol=0
                                ):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be"
                                            " orthogonal."
                                        )
                                    ) from None
                                vec = rotation @ vec
                            max_vec = max(vec)
                            vec = vec * lim / max_vec
                            for i in range(3):
                                ax.plot(
                                    [vec[0, i], -vec[0, i]],
                                    [vec[1, i], -vec[1, i]],
                                    [vec[2, i], -vec[2, i]],
                                    axes_colors[i],
                                )
                        if plot_style == "globe" and not solid_surface:
                            surface.set_facecolor((0, 0, 0, 0))
                        ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                        ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                        ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                        # Important order of operations!

                        if ticks == 0:
                            for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                                axis_i.set_ticklabels([])
                                axis_i._axinfo["axisline"]["linewidth"] = 1
                                axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                                axis_i._axinfo["grid"]["linewidth"] = 0.5
                                axis_i._axinfo["grid"]["linestyle"] = "-"
                                axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                                axis_i._axinfo["tick"]["inward_factor"] = 0.0
                                axis_i._axinfo["tick"]["outward_factor"] = 0.0
                                axis_i.set_pane_color((0.95, 0.95, 0.95))
                        else:
                            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                            if ticks != 1:
                                ax.xaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.yaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.zaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                        ax.grid(False)

                        ax.set_box_aspect([1, 1, 1])
                        ax.azim = azim
                        ax.elev = elev
                        ax.roll = roll
                        ax.set_title(plot_title)
                        if axis_off:
                            ax.set_axis_off()

                        if bar:
                            c = next(color)
                            axins = ax.inset_axes([0, 0.6, 0.098, 0.2])
                            axins.bar(
                                1,
                                indicator[i_temp - i_start],
                                width=0.2,
                                color=c,
                            )
                            axins.set_ylim(0, 1)
                            if not bar_scale:
                                axins.text(
                                    1,
                                    1,
                                    s=(
                                        f"{round(temps[i_end], temp_rounding)} K"
                                    ),
                                    verticalalignment="bottom",
                                    horizontalalignment="center",
                                )
                                axins.text(
                                    1,
                                    -0.03,
                                    s=(
                                        f"{round(temps[i_start], temp_rounding)} K"
                                    ),
                                    verticalalignment="top",
                                    horizontalalignment="center",
                                )
                                axins.axison = False
                            if bar_scale:
                                axins.get_xaxis().set_visible(False)
                                axins.xaxis.set_tick_params(labelbottom=False)
                                axins.yaxis.set_major_formatter(
                                    FuncFormatter(my_ticks)
                                )
                                axins.yaxis.set_minor_locator(
                                    AutoMinorLocator(2)
                                )

                        writer.grab_frame()
                        cla()

                elif animation_variable == "field":
                    for i_field in range(i_start, i_end):
                        if plot_style == "globe":
                            x = x0y0z0[0, i_field, i_constant, :, :]
                            y = x0y0z0[1, i_field, i_constant, :, :]
                            z = x0y0z0[2, i_field, i_constant, :, :]
                            max_array = array([max(x), max(y), max(z)])
                            lim = max(max_array) * axes_scale_factor
                            norm = Normalize(z.min(), z.max())
                            colors = color_map(color_map_name)(norm(z))
                            rcount, ccount, _ = colors.shape
                            if not r_density:
                                r_density = rcount
                            if not c_density:
                                c_density = ccount
                            surface = ax.plot_surface(
                                x,
                                y,
                                z,
                                rcount=r_density,
                                ccount=c_density,
                                facecolors=colors,
                                shade=False,
                            )
                        elif plot_style == "scatter":
                            x = x0y0z0[i_field, i_constant, :, 0]
                            y = x0y0z0[i_field, i_constant, :, 1]
                            z = x0y0z0[i_field, i_constant, :, 2]
                            max_array = array([max(x), max(y), max(z)])
                            lim = max(max_array) * axes_scale_factor
                            norm = Normalize(z.min(), z.max())
                            colors = color_map(color_map_name)(norm(z))
                            ax.scatter(
                                x, y, z, s=points_size, facecolors=colors
                            )

                        ax.set_xlabel(
                            label, labelpad=labelpad * labelpad_scalar
                        )
                        ax.set_ylabel(
                            label, labelpad=labelpad * labelpad_scalar
                        )
                        ax.set_zlabel(
                            label, labelpad=labelpad * labelpad_scalar
                        )

                        if add_g_tensor_axes:
                            vec = axes_matrix * g_tensor[newaxis, 1:]
                            if rotation is not None:
                                if rotation.shape != (3, 3):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be a"
                                            " 3x3 matrix."
                                        )
                                    ) from None
                                product = rotation.T @ rotation
                                if not allclose(
                                    product, identity(3), atol=1e-2, rtol=0
                                ):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be"
                                            " orthogonal."
                                        )
                                    ) from None
                                vec = rotation @ vec
                            max_vec = max(vec)
                            vec = vec * lim / max_vec
                            for i in range(3):
                                ax.plot(
                                    [vec[0, i], -vec[0, i]],
                                    [vec[1, i], -vec[1, i]],
                                    [vec[2, i], -vec[2, i]],
                                    axes_colors[i],
                                )
                        if plot_style == "globe" and not solid_surface:
                            surface.set_facecolor((0, 0, 0, 0))
                        ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                        ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                        ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                        # Important order of operations!

                        if ticks == 0:
                            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                                axis.set_ticklabels([])
                                axis._axinfo["axisline"]["linewidth"] = 1
                                axis._axinfo["axisline"]["color"] = (0, 0, 0)
                                axis._axinfo["grid"]["linewidth"] = 0.5
                                axis._axinfo["grid"]["linestyle"] = "-"
                                axis._axinfo["grid"]["color"] = (0, 0, 0)
                                axis._axinfo["tick"]["inward_factor"] = 0.0
                                axis._axinfo["tick"]["outward_factor"] = 0.0
                                axis.set_pane_color((0.95, 0.95, 0.95))
                        else:
                            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                            if ticks != 1:
                                ax.xaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.yaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.zaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                        ax.grid(False)

                        ax.set_box_aspect([1, 1, 1])
                        ax.azim = azim
                        ax.elev = elev
                        ax.roll = roll
                        ax.set_title(plot_title)

                        if axis_off:
                            axis("off")

                        if bar:
                            c = next(color)
                            axins = ax.inset_axes([0, 0.6, 0.098, 0.2])
                            axins.bar(
                                1,
                                indicator[i_field - i_start],
                                width=0.2,
                                color=c,
                            )
                            axins.set_ylim(0, 1)

                            if not bar_scale:
                                axins.text(
                                    1,
                                    1,
                                    s=(
                                        f"{round(fields[i_end], field_rounding)} {field_unit}"
                                    ),
                                    verticalalignment="bottom",
                                    horizontalalignment="center",
                                )
                                axins.text(
                                    1,
                                    -0.03,
                                    s=(
                                        f"{round(fields[i_start], field_rounding)} {field_unit}"
                                    ),
                                    verticalalignment="top",
                                    horizontalalignment="center",
                                )
                                axins.axison = False
                            if bar_scale:
                                axins.get_xaxis().set_visible(False)
                                axins.xaxis.set_tick_params(labelbottom=False)
                                axins.yaxis.set_major_formatter(
                                    FuncFormatter(my_ticks)
                                )
                        writer.grab_frame()
                        cla()
                else:
                    raise ValueError(
                        "There exist only two animation variables: field and"
                        " temperature"
                    )
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                f"Failed to plot and save 3d data"
                + BLUE
                + " Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        close("all")

    def interactive_plot_3d(
        self,
        group: str,
        data_type: Literal[
            "chit",
            "chi",
            "helmholtz_energy",
            "internal_energy",
            "magnetisation",
        ],
        color_map_name: str or list[str] = "dark_rainbow_r",
        T_slider_color: str = "#77f285",
        B_slider_color: str = "#794285",
        temp_bar_color_map_name: str or list[str] = "BuRd",
        field_bar_color_map_name: str or list[str] = "BuPi",
        lim_scalar: float = 1.0,
        ticks: int or float = 1,
        field: Literal["B", "H"] = "B",
        points_size: float = 0.2,
        solid_surface: bool = False,
        bar: bool = True,
        bar_scale: bool = False,
        temp_rounding: int = 2,
        field_rounding: int = 2,
        axis_off: int = False,
        add_g_tensor_axes: bool = False,
        axes_group: str = "",
        axes_colors: list[str] = ["r", "g", "b"],
        doublet_number: int = None,
        axes_scale_factor: float64 = 1.0,
        rotation: ndarray[float64] = None,
    ):
        """
        Creates interactive widget plot dependent on field and temperature
        values.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which plot will be created.
        data_type: Literal["chit", "chi", "helmholtz_energy", "internal_energy",
          "magnetisation"]
            Type of the data that will be used to create the plot.
        color_map_name: str or list = 'dark_rainbow_r_l'
            Input of the color_map function, determines a color of the main
            figure.
        T_slider_color: str
            Determines a color of the temperature slider.
        B_slider_color: str
            Determines a color of the field slider.
        temp_bar_color_map_name: str or list[str] = 'BuRd'
            Input of the color_map function, determines a color map of the
            temperature bar.
        field_bar_color_map_name: str or list[str] = 'BuPi'
            Input of the color_map function, determines a color map of the
            field bar.
        lim_scalar: float = 1.
            Scalar used to set limits of the axes, smaller values magnify the
            plotted figure.
        ticks: float = 1
            Determines the ticks spacing.
        field: Literal['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].
        points_size: float = 0.2
            Determines points size for Fibonacci scatter plots.
        solid_surface: bool = False
            Makes surface plots using meshgrid appear as solid.
        bar: bool = True
            Determines if the bar is shown.
        bar_scale: bool = False
            Determines if the bar scale is shown.
        temp_rounding: int = 2
            Determines how many significant digits are shown relative to the
            int(value) for temperature.
        temp_rounding: int = 2
            Determines how many significant digits are shown relative to the
            int(value) for temperature.
        field_rounding: int = 2
            Determines how many significant digits are shown relative to the
            int(value) for field.
        axis_off: bool = False
            Determines if the axes are turned off.
        add_g_tensor_axes: bool = False
            Determines if add to the plot main magnetic axes scaled by the
            corresponding pseudo-g-tensor values.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        axes_colors: list[str] = ['r','g','b']
            Determines the colors of the magnetic axes in order of x, y, z.
            Accepts matplotlib colors inputs, for example HTML color codes.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining the length of the longest (main) magnetic
            axis concerning the maximal value of the loaded data and setting
            a maximal limit of the plot's xyz axes. It should be set > 1
            otherwise, some data will end up missing from the plot! The limit is
            max(loaded_data) * axes_scale_factor.
        rotation: ndarray[float64] = None
            Has to be given if 3d data was calculated with optional rotation of
            the coordinate frame and add_g_tensor_axes option is turned on.
            One must provide the same rotation as that used for the simulation
            to apply it to the magnetic axes.

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.

        See Also
        --------
        slothpy.SltFile.calculate_magnetisation_3d, slothpy.SltFile.calculate_susceptibility_3d,
        slothpy.SltFile.calculate_energy_3d,
        slothpy.SltFile.calculate_g_tensor_axes
        """
        if (not isinstance(axes_scale_factor, (float, int))) or (
            axes_scale_factor < 1
        ):
            raise SltInputError(
                ValueError(
                    "Axes scale factor has to be a float greater than 1."
                )
            )

        field_i, temp_i = 0, 0
        try:
            global label
            if (data_type == "chi") or (data_type == "chit"):
                group_name = f"{group}_susceptibility_3d"
                x0y0z0 = self[group_name, f"{group}_chit_3d"]
                if data_type == "chit":
                    label = (
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                    )
                else:
                    label = r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$"
            elif (data_type == "helmholtz_energy") or (
                data_type == "internal_energy"
            ):
                if data_type == "helmholtz_energy":
                    group_name = f"{group}_helmholtz_energy_3d"
                    label = r"$F\ /\ \mathrm{cm^{-1}}$"
                else:
                    group_name = f"{group}_internal_energy_3d"
                    label = r"$U\ /\ \mathrm{cm^{-1}}$"
                x0y0z0 = self[group_name, f"{group}_energy_3d"]
            elif data_type == "magnetisation":
                group_name = f"{group}_magnetisation_3d"
                label = r"$M\ /\ \mathrm{\mu_{B}}$"
                x0y0z0 = self[group_name, f"{group}_mag_3d"]
            else:
                raise ValueError(
                    "Acceptable data types: chit, chi, helmholtz_energy and"
                    " magnetisation"
                )
            fields = self[group_name, f"{group}_fields"]
            field_unit = "T"
            if field == "H":
                fields = fields * 10
                field_unit = "kOe"
            temps = self[group_name, f"{group}_temperatures"]
            if x0y0z0.ndim == 5:
                plot_style = "globe"
            elif x0y0z0.ndim == 4:
                plot_style = "scatter"
            if add_g_tensor_axes:
                doublets = axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ].shape[0]
                if (
                    (not isinstance(doublet_number, int))
                    or (doublet_number < 0)
                    or (doublet_number > doublets - 1)
                ):
                    raise SltInputError(
                        ValueError(
                            "Doublet number must be a nonnegative integer less"
                            " or equal to the number of doublets in the axes"
                            " group."
                        )
                    ) from None
                axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ][doublet_number]
                g_tensor = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_g_tensors"
                ][doublet_number]

        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load the 3D data file "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        try:
            fig = figure()
            global ax
            ax = fig.add_subplot(projection="3d")
            if bar:
                color1 = color_map(temp_bar_color_map_name)(
                    linspace(0, 1, len(temps))
                )
                color2 = color_map(field_bar_color_map_name)(
                    linspace(0, 1, len(fields))
                )

                indicator1 = linspace(0, 1, len(temps))
                indicator2 = linspace(0, 1, len(fields))
                if bar_scale:

                    def my_ticks(x, pos):
                        return f"{round(x * temps[-1], temp_rounding)} K"

                    def my_ticks2(x, pos):
                        return (
                            f"{round(x * fields[-1], field_rounding)} {field_unit}"
                        )

            if plot_style == "globe":
                x = x0y0z0[0, field_i, temp_i, :, :]
                y = x0y0z0[1, field_i, temp_i, :, :]
                z = x0y0z0[2, field_i, temp_i, :, :]
                max_array = array([max(x), max(y), max(z)])
                lim = max(max_array) * axes_scale_factor
                norm = Normalize(z.min(), z.max())
                colors = color_map(color_map_name)(norm(z))
                rcount, ccount, _ = colors.shape
                surface = ax.plot_surface(
                    x,
                    y,
                    z,
                    facecolors=colors,
                    shade=False,
                )
            elif plot_style == "scatter":
                x = x0y0z0[field_i, temp_i, :, 0]
                y = x0y0z0[field_i, temp_i, :, 1]
                z = x0y0z0[field_i, temp_i, :, 2]
                max_array = array([max(x), max(y), max(z)])
                lim = max(max_array) * axes_scale_factor
                norm = Normalize(z.min(), z.max())
                colors = color_map(color_map_name)(norm(z))
                surface = ax.scatter(x, y, z, s=points_size, facecolors=colors)
            if add_g_tensor_axes:
                vec = axes_matrix * g_tensor[newaxis, 1:]
                if rotation is not None:
                    if rotation.shape != (3, 3):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be a 3x3 matrix."
                            )
                        ) from None
                    product = rotation.T @ rotation
                    if not allclose(product, identity(3), atol=1e-2, rtol=0):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be orthogonal."
                            )
                        ) from None
                    vec = rotation @ vec
                max_vec = max(vec)
                vec = vec * lim / max_vec
                for i in range(3):
                    ax.plot(
                        [vec[0, i], -vec[0, i]],
                        [vec[1, i], -vec[1, i]],
                        [vec[2, i], -vec[2, i]],
                        axes_colors[i],
                    )
            if plot_style == "globe" and not solid_surface:
                surface.set_facecolor((0, 0, 0, 0))
            ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
            # Important order of operations!
            global labelpad
            labelpad = 20 * len(str(ticks)) / 4
            global labelpad_scalar
            labelpad_scalar = 1
            if data_type == "magnetisation":
                labelpad_scalar = labelpad_scalar * 0.5

            ax.set_xlabel(label, labelpad=labelpad * labelpad_scalar)
            ax.set_ylabel(label, labelpad=labelpad * labelpad_scalar)
            ax.set_zlabel(label, labelpad=labelpad * labelpad_scalar)

            if ticks == 0:
                for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis_i.set_ticklabels([])
                    axis_i._axinfo["axisline"]["linewidth"] = 1
                    axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                    axis_i._axinfo["grid"]["linewidth"] = 0.5
                    axis_i._axinfo["grid"]["linestyle"] = "-"
                    axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                    axis_i._axinfo["tick"]["inward_factor"] = 0.0
                    axis_i._axinfo["tick"]["outward_factor"] = 0.0
                    axis_i.set_pane_color((0.95, 0.95, 0.95))
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                if ticks != 1:
                    ax.xaxis.set_major_locator(MultipleLocator(ticks))
                    ax.yaxis.set_major_locator(MultipleLocator(ticks))
                    ax.zaxis.set_major_locator(MultipleLocator(ticks))
            ax.grid(False)

            ax.set_box_aspect([1, 1, 1])
            fig.subplots_adjust(left=0.1)
            if bar:
                c = color1[temp_i]
                axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
                axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                axins.set_ylim(0, 1)
                c = color2[field_i]
                axins2 = ax.inset_axes([-0.05, 0.2, 0.098, 0.2])
                axins2.bar(1, indicator2[field_i], width=0.2, color=c)
                axins2.set_ylim(0, 1)

                if not bar_scale:
                    axins.text(
                        1,
                        1,
                        s=f"{round(temps[-1], 1)} K",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                    )
                    axins.text(
                        1,
                        -0.03,
                        s=f"{round(temps[0], 1)} K",
                        verticalalignment="top",
                        horizontalalignment="center",
                    )
                    axins.axison = False
                    axins2.text(
                        1,
                        1,
                        s=f"{round(fields[-1], 1)} {field_unit}",
                        verticalalignment="bottom",
                        horizontalalignment="center",
                    )
                    axins2.text(
                        1,
                        -0.03,
                        s=f"{round(fields[0], 1)} {field_unit}",
                        verticalalignment="top",
                        horizontalalignment="center",
                    )
                    axins2.axison = False
                if bar_scale:
                    axins.get_xaxis().set_visible(False)
                    axins.xaxis.set_tick_params(labelbottom=False)
                    axins.yaxis.set_major_formatter(FuncFormatter(my_ticks))
                    axins.yaxis.set_minor_locator(AutoMinorLocator(2))
                    axins2.get_xaxis().set_visible(False)
                    axins2.xaxis.set_tick_params(labelbottom=False)
                    axins2.yaxis.set_major_formatter(FuncFormatter(my_ticks2))
                    axins2.yaxis.set_minor_locator(AutoMinorLocator(2))

            if bar:
                c = color1[temp_i]
                axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
                axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                axins.set_ylim(0, 1)
                axins.text(
                    1,
                    1,
                    s=f"{round(temps[-1], 1)} K",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                )
                axins.text(
                    1,
                    -0.03,
                    s=f"{round(temps[0], 1)} K",
                    verticalalignment="top",
                    horizontalalignment="center",
                )

            ax_temp = fig.add_axes([0.05, 0.6, 0.1, 0.2])
            ax_temp.axison = False

            ax_field = fig.add_axes([0.05, 0.3, 0.1, 0.2])
            ax_field.axison = False

            slider_temp = Slider(
                ax_temp,
                "T [index]",
                valmin=0,
                valmax=temps.size - 1,
                orientation="vertical",
                valstep=1,
                initcolor=None,
                color=T_slider_color,
            )
            slider_field = Slider(
                ax_field,
                f"{field} [index]",
                valmin=0,
                valmax=fields.size - 1,
                orientation="vertical",
                initcolor=None,
                valstep=1,
                color=B_slider_color,
            )

            def slider_update_globe(val):
                temp_i = slider_temp.val
                field_i = slider_field.val

                ax.cla()

                x = x0y0z0[0, field_i, temp_i, :, :]
                y = x0y0z0[1, field_i, temp_i, :, :]
                z = x0y0z0[2, field_i, temp_i, :, :]
                max_array = array([max(x), max(y), max(z)])
                lim = max(max_array) * axes_scale_factor
                norm = Normalize(z.min(), z.max())
                colors = color_map(color_map_name)(norm(z))
                rcount, ccount, _ = colors.shape
                surface = ax.plot_surface(
                    x,
                    y,
                    z,
                    facecolors=colors,
                    shade=False,
                )
                if add_g_tensor_axes:
                    vec = axes_matrix * g_tensor[newaxis, 1:]
                    if rotation is not None:
                        if rotation.shape != (3, 3):
                            raise SltInputError(
                                ValueError(
                                    "Input rotation matrix must be a 3x3"
                                    " matrix."
                                )
                            ) from None
                        product = rotation.T @ rotation
                        if not allclose(
                            product, identity(3), atol=1e-2, rtol=0
                        ):
                            raise SltInputError(
                                ValueError(
                                    "Input rotation matrix must be orthogonal."
                                )
                            ) from None
                        vec = rotation @ vec
                    max_vec = max(vec)
                    vec = vec * lim / max_vec
                    for i in range(3):
                        ax.plot(
                            [vec[0, i], -vec[0, i]],
                            [vec[1, i], -vec[1, i]],
                            [vec[2, i], -vec[2, i]],
                            axes_colors[i],
                        )
                if not solid_surface:
                    surface.set_facecolor((0, 0, 0, 0))
                ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_title(
                    f"{field}={fields[field_i]:.4f} {field_unit},"
                    f" T={temps[temp_i]:.4f} K"
                )
                ax.set_xlabel(label, labelpad=labelpad * labelpad_scalar)
                ax.set_ylabel(label, labelpad=labelpad * labelpad_scalar)
                ax.set_zlabel(label, labelpad=labelpad * labelpad_scalar)

                if ticks == 0:
                    for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                        axis_i.set_ticklabels([])
                        axis_i._axinfo["axisline"]["linewidth"] = 1
                        axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                        axis_i._axinfo["grid"]["linewidth"] = 0.5
                        axis_i._axinfo["grid"]["linestyle"] = "-"
                        axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                        axis_i._axinfo["tick"]["inward_factor"] = 0.0
                        axis_i._axinfo["tick"]["outward_factor"] = 0.0
                        axis_i.set_pane_color((0.95, 0.95, 0.95))
                else:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                    if ticks != 1:
                        ax.xaxis.set_major_locator(MultipleLocator(ticks))
                        ax.yaxis.set_major_locator(MultipleLocator(ticks))
                        ax.zaxis.set_major_locator(MultipleLocator(ticks))
                ax.grid(False)
                fig.subplots_adjust(left=0.1)
                if bar:
                    c = color1[temp_i]
                    axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
                    axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                    axins.set_ylim(0, 1)
                    c = color2[field_i]
                    axins2 = ax.inset_axes([-0.05, 0.2, 0.098, 0.2])
                    axins2.bar(1, indicator2[field_i], width=0.2, color=c)
                    axins2.set_ylim(0, 1)

                    if not bar_scale:
                        axins.text(
                            1,
                            1,
                            s=f"{round(temps[-1], 1)} K",
                            verticalalignment="bottom",
                            horizontalalignment="center",
                        )
                        axins.text(
                            1,
                            -0.03,
                            s=f"{round(temps[0], 1)} K",
                            verticalalignment="top",
                            horizontalalignment="center",
                        )
                        axins.axison = False
                        axins2.text(
                            1,
                            1,
                            s=f"{round(fields[-1], 1)} {field_unit}",
                            verticalalignment="bottom",
                            horizontalalignment="center",
                        )
                        axins2.text(
                            1,
                            -0.03,
                            s=f"{round(fields[0], 1)} {field_unit}",
                            verticalalignment="top",
                            horizontalalignment="center",
                        )
                        axins2.axison = False
                    if bar_scale:
                        axins.get_xaxis().set_visible(False)
                        axins.xaxis.set_tick_params(labelbottom=False)
                        axins.yaxis.set_major_formatter(
                            FuncFormatter(my_ticks)
                        )
                        axins.yaxis.set_minor_locator(AutoMinorLocator(2))
                        axins2.get_xaxis().set_visible(False)
                        axins2.xaxis.set_tick_params(labelbottom=False)
                        axins2.yaxis.set_major_formatter(
                            FuncFormatter(my_ticks2)
                        )
                        axins2.yaxis.set_minor_locator(AutoMinorLocator(2))

                fig.canvas.draw()

            def slider_update_scatter(val):
                temp_i = slider_temp.val
                field_i = slider_field.val

                ax.cla()

                x = x0y0z0[field_i, temp_i, :, 0]
                y = x0y0z0[field_i, temp_i, :, 1]
                z = x0y0z0[field_i, temp_i, :, 2]
                max_array = array([max(x), max(y), max(z)])
                lim = max(max_array) * axes_scale_factor
                norm = Normalize(z.min(), z.max())
                colors = color_map(color_map_name)(norm(z))
                surface = ax.scatter(x, y, z, s=points_size, facecolors=colors)
                if add_g_tensor_axes:
                    vec = axes_matrix * g_tensor[newaxis, 1:]
                    if rotation is not None:
                        if rotation.shape != (3, 3):
                            raise SltInputError(
                                ValueError(
                                    "Input rotation matrix must be a 3x3"
                                    " matrix."
                                )
                            ) from None
                        product = rotation.T @ rotation
                        if not allclose(
                            product, identity(3), atol=1e-2, rtol=0
                        ):
                            raise SltInputError(
                                ValueError(
                                    "Input rotation matrix must be orthogonal."
                                )
                            ) from None
                        vec = rotation @ vec
                    max_vec = max(vec)
                    vec = vec * lim / max_vec
                    for i in range(3):
                        ax.plot(
                            [vec[0, i], -vec[0, i]],
                            [vec[1, i], -vec[1, i]],
                            [vec[2, i], -vec[2, i]],
                            axes_colors[i],
                        )
                ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_title(
                    f"{field}={fields[field_i]:.4f} {field_unit},"
                    f" T={temps[temp_i]:.4f} K"
                )
                ax.set_xlabel(label, labelpad=labelpad * labelpad_scalar)
                ax.set_ylabel(label, labelpad=labelpad * labelpad_scalar)
                ax.set_zlabel(label, labelpad=labelpad * labelpad_scalar)

                if ticks == 0:
                    for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                        axis_i.set_ticklabels([])
                        axis_i._axinfo["axisline"]["linewidth"] = 1
                        axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                        axis_i._axinfo["grid"]["linewidth"] = 0.5
                        axis_i._axinfo["grid"]["linestyle"] = "-"
                        axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                        axis_i._axinfo["tick"]["inward_factor"] = 0.0
                        axis_i._axinfo["tick"]["outward_factor"] = 0.0
                        axis_i.set_pane_color((0.95, 0.95, 0.95))
                else:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                    if ticks != 1:
                        ax.xaxis.set_major_locator(MultipleLocator(ticks))
                        ax.yaxis.set_major_locator(MultipleLocator(ticks))
                        ax.zaxis.set_major_locator(MultipleLocator(ticks))
                ax.grid(False)
                fig.subplots_adjust(left=0.1)
                if bar:
                    c = color1[temp_i]
                    axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
                    axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                    axins.set_ylim(0, 1)
                    c = color2[field_i]
                    axins2 = ax.inset_axes([-0.05, 0.2, 0.098, 0.2])
                    axins2.bar(1, indicator2[field_i], width=0.2, color=c)
                    axins2.set_ylim(0, 1)

                    if not bar_scale:
                        axins.text(
                            1,
                            1,
                            s=f"{round(temps[-1], 1)} K",
                            verticalalignment="bottom",
                            horizontalalignment="center",
                        )
                        axins.text(
                            1,
                            -0.03,
                            s=f"{round(temps[0], 1)} K",
                            verticalalignment="top",
                            horizontalalignment="center",
                        )
                        axins.axison = False
                        axins2.text(
                            1,
                            1,
                            s=f"{round(fields[-1], 1)} {field_unit}",
                            verticalalignment="bottom",
                            horizontalalignment="center",
                        )
                        axins2.text(
                            1,
                            -0.03,
                            s=f"{round(fields[0], 1)} {field_unit}",
                            verticalalignment="top",
                            horizontalalignment="center",
                        )
                        axins2.axison = False
                    if bar_scale:
                        axins.get_xaxis().set_visible(False)
                        axins.xaxis.set_tick_params(labelbottom=False)
                        axins.yaxis.set_major_formatter(
                            FuncFormatter(my_ticks)
                        )
                        axins.yaxis.set_minor_locator(AutoMinorLocator(2))
                        axins2.get_xaxis().set_visible(False)
                        axins2.xaxis.set_tick_params(labelbottom=False)
                        axins2.yaxis.set_major_formatter(
                            FuncFormatter(my_ticks2)
                        )
                        axins2.yaxis.set_minor_locator(AutoMinorLocator(2))

                fig.canvas.draw()

            if plot_style == "globe":
                slider_temp.on_changed(slider_update_globe)
                slider_field.on_changed(slider_update_globe)
            else:
                slider_temp.on_changed(slider_update_scatter)
                slider_field.on_changed(slider_update_scatter)
            if axis_off:
                ax.set_axis_off()
            _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot 3D data"
                + BLUE
                + " Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        close("all")
