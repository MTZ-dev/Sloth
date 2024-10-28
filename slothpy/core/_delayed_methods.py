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
from typing import Literal, Union, Optional
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.synchronize import Event

from numpy import ndarray, outer, zeros, empty, repeat, sqrt
from pandas import DataFrame
import matplotlib.pyplot as plt
from ase.dft.kpoints import BandPath

from slothpy.core._config import settings
from slothpy.core._slothpy_exceptions import SltCompError, SltFileError
from slothpy.core._drivers import _SingleProcessed, _MultiProcessed, ensure_ready
from slothpy._general_utilities._constants import RED, BLUE, GREEN, RESET
from slothpy._general_utilities._constants import H_CM_1
from slothpy._general_utilities._utils import slpjm_components_driver
from slothpy._magnetism._zeeman import _zeeman_splitting_proxy
from slothpy._magnetism._magnetisation import _magnetisation_proxy
from slothpy._lattice_dynamics._phonon_dispersion import _phonon_dispersion_proxy

#################
# SingleProcessed
#################


class SltStatesEnergiesCm1(_SingleProcessed):
    _method_name = "States' Energies in cm-1"
    _method_type = "STATES_ENERGIES_CM1"

    __slots__ = _SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._start_state = start_state
        self._stop_state = stop_state

    def _executor(self):
        return self._slt_group.e[self._start_state:self._stop_state] * H_CM_1
    
    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "CM_1",
            "States": self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' energies in cm-1 from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ENERGIES_CM_1": (self._result,  "States' energies in cm-1.")}

    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_ENERGIES_CM_1"][:]

    def _plot(self, show=True, **kwargs):
        from slothpy._general_utilities._plot import _plot_energy_levels
        fig, ax = _plot_energy_levels(self._result, **kwargs)
        from slothpy._gui._plot_gui import _display_plot
        if show:
            _display_plot(self._result, 'states_energy_cm_1')
        else:
            return fig, ax
    
    def _to_data_frame(self):
        self._df = DataFrame({'Energy (cm^-1)': self._result})
        self._df.index.name = 'State Number'
        return self._df


class SltStatesEnergiesAu(_SingleProcessed):
    _method_name = "States' Energies in a.u."
    _method_type = "STATES_ENERGIES_AU"

    __slots__ = _SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._start_state = start_state
        self._stop_state = stop_state

    def _executor(self):
        return self._slt_group.e[self._start_state:self._stop_state]

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AU",
            "States": self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' energies in a.u. from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ENERGIES_AU": (self._result, "States' energies in cm-1.")}

    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_ENERGIES_AU"][:]

    #TODO: plot
    def _plot(self):
        fig, ax = plt.subplots()
        x_min = 0
        x_max = 1
        for energy in self._result:
            ax.hlines(y=energy, xmin=x_min, xmax=x_max, colors='skyblue', linestyles='solid', linewidth=2)
            ax.text(x_max + 0.1, energy, f'{energy:.1f}', va='center', ha='left')
        ax.set_ylabel('Energy (cm$^{-1}$)')
        ax.set_title('Energy Levels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticks([])
        plt.tight_layout()
        plt.show()

    def _to_data_frame(self):
        self._df = DataFrame({'Energy (a.u.)': self._result})
        self._df.index.name = 'State Number'
        return self._df


class SltSpinMatrices(_SingleProcessed):
    _method_name = "Spin matrices"
    _method_type = "SPINS"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "s", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Spin matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"SPIN_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the spin.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the spin components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["SPIN_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesSpins(_SingleProcessed):
    _method_name = "States' spins"
    _method_type = "STATES_SPINS"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "s", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the spin from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_SPINS": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's spins.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the spin components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_SPINS"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltAngularMomentumMatrices(_SingleProcessed):
    _method_name = "Angular momentum matrices"
    _method_type = "ANGULAR_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "l", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Angular momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"ANGULAR_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the angular momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the angular momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["ANGULAR_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesAngularMomenta(_SingleProcessed):
    _method_name = "States' angular momenta"
    _method_type = "STATES_ANGULAR_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "l", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the angular momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ANGULAR_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's angular momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the angular momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_ANGULAR_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltElectricDipoleMomentumMatrices(_SingleProcessed):
    _method_name = "Electric dipole momentum matrices"
    _method_type = "ELECTRIC_DIPOLE_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "p", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Electric dipole momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"ELECTRIC_DIPOLE_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the electric dipole momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the electric dipole momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["ELECTRIC_DIPOLE_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesElectricDipoleMomenta(_SingleProcessed):
    _method_name = "States' electric dipole momenta"
    _method_type = "STATES_ELECTRIC_DIPOLE_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "p", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the electric dipole momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ELECTRIC_DIPOLE_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's electric dipole momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the electric dipole momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_ELECTRIC_DIPOLE_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltTotalAngularMomentumMatrices(_SingleProcessed):
    _method_name = "Total angular momentum matrices"
    _method_type = "TOTAL_ANGULAR_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "j", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Total angular momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"TOTAL_ANGULAR_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the total angular momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the total angular momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["TOTAL_ANGULAR_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesTotalAngularMomenta(_SingleProcessed):
    _method_name = "States' total angular momenta"
    _method_type = "STATES_TOTAL_ANGULAR_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "j", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the total angular momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_TOTAL_ANGULAR_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's total angular momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the total angular momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_TOTAL_ANGULAR_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltMagneticDipoleMomentumMatrices(_SingleProcessed):
    _method_name = "Magnetic dipole momentum matrices"
    _method_type = "MAGNETIC_DIPOLE_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "m", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Magnetic dipole momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"MAGNETIC_DIPOLE_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the magnetic dipole momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the magnetic dipole momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["MAGNETIC_DIPOLE_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesMagneticDipoleMomenta(_SingleProcessed):
    _method_name = "States' magnetic dipole momenta"
    _method_type = "STATES_MAGNETIC_DIPOLE_MOMENTA"

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "m", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the magnetic dipole momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_MAGNETIC_DIPOLE_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's magnetic dipole momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the magnetic dipole momentum components.")
    
    def _load_from_slt_file(self):
        self._result = self._slt_group["STATES_MAGNETIC_DIPOLE_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltIrSpectrum(_SingleProcessed):
    _method_name = "Infrared Spectrum"
    _method_type = "IR_SPECTRUM"

    __slots__ = _SingleProcessed.__slots__ + ["_hessian", "_masses", "_start_wavenumber", "_stop_wavenumber", "_resolution", "_convolution", "_fwhm"]
     
    def __init__(self, slt_group, hessian: ndarray, masses: ndarray, start_wavenumber: float, stop_wavenumber: float, resolution: int, convolution: Optional[Literal["lorentzian", "gaussian"]] = "lorentizan", fwhm: float = None, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._hessian = hessian
        self._masses = masses
        self._start_wavenumber = start_wavenumber
        self._stop_wavenumber = stop_wavenumber
        self._resolution = resolution
        self._convolution = convolution
        self._fwhm = fwhm

    def _executor(self): #return tuple wave spectrum etc
        return slpjm_components_driver(self._slt_group, "diagonal", "m", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the magnetic dipole momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_MAGNETIC_DIPOLE_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's magnetic dipole momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the magnetic dipole momentum components.")
    
    def _load_from_slt_file(self): #load tuple
        self._result = self._slt_group["STATES_MAGNETIC_DIPOLE_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


################
# MultiProcessed
################


class SltPropertyUnderMagneticField(_MultiProcessed):
    #_method_name = ...................................... (here I assume it depends)
    _method_type = "PROPERTY_UNDER_MAGNETIC_FIELD"

    __slots__ = _MultiProcessed.__slots__ + ["_mode", "_matrix", "_return_energies", "_energies", "_direction", "_magnetic_fields", "_orientations", "_number_of_states", "_states_cutoff", "_rotation", "_electric_field_vector", "_hyperfine", "_dims"]

    def __init__(self, slt_group,
        mode: Union[Literal["s", "l", "p", "j", "m"], str],
        full_matrix: bool,
        return_energies: bool,
        direction: Union[ndarray, Literal["xyz"]],
        magnetic_fields: ndarray,
        orientations: ndarray,
        number_of_states: int,
        states_cutoff: list = [0,0],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = 1,
        number_threads: int = 1,
        autotune: bool = False,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0], number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._mode = mode
        self._method_name = f"{self._mode.upper()} Under Magnetic Field"
        self._full_matrix = full_matrix
        self._return_energies = return_energies
        self._direction = direction
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        if self._orientations.shape[1] == 4:
            self._returns = True
        self._number_of_states = number_of_states
        self._states_cutoff = states_cutoff
        self._rotation = rotation
        self._electric_field_vector = electric_field_vector
        self._hyperfine = hyperfine
        self._args = [self._mode, self._full_matrix, self._return_energies, self._direction, self._number_of_states, self._electric_field_vector]
        self._executor_proxy = _property_under_magnetic_field_proxy
        self._slt_hamiltonian = self._slt_group._slt_hamiltonian_from_slt_group(self._states_cutoff, self._rotation, self._hyperfine, False) ### moze SltHamiltonian(slt_group)!!!!!!!!!!!!
        self._slt_hamiltonian._mode = "em" if electric_field_vector is None else "emp"
        for mod in self._mode:
            if mod not in self._slt_hamiltonian._mode:
                self._slt_hamiltonian._mode += mod
        self._dims = [] if self._direction != "xyz" else [3]
        self._energies = empty((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._number_of_states), dtype=settings.float, order="C") if self._return_energies and not self._returns else empty((1))
        self._additional_result = True
        self._additional_result_shape = (self._magnetic_fields.shape[0], self._orientations.shape[0], self._number_of_states)
        if full_matrix:
            self._dims += [self._number_of_states] * 2
        else:
            self._dims.append(self._number_of_states)
    
    def __repr__(self):
        return f"<{RED}Slt{self._mode.upper()}UnderMagneticField{RESET} object from {BLUE}Group{RESET} '{self._group_name}' {GREEN}File{RESET} '{self._hdf5}'.>"
    
    def _load_args_arrays(self):
        self._args_arrays = [*self._slt_hamiltonian.arrays_to_shared_memory, self._magnetic_fields, self._orientations, self._energies] # additional result must be the last
        if not self._returns:
            self._result = empty((len(self._mode), self._magnetic_fields.shape[0] * self._orientations.shape[0], *self._dims), dtype=settings.complex if self._full_matrix else settings.float, order="C")
            self._result_shape = (len(self._mode), self._orientations.shape[0], self._magnetic_fields.shape[0], *self._dims)
    
    def _gather_results(self, result_queue):
        property_array = zeros((len(self._mode), self._magnetic_fields.shape[0], *self._dims), dtype=settings.complex if self._full_matrix else settings.float, order="C")
        self._energies = zeros((self._magnetic_fields.shape[0], self._number_of_states), dtype=settings.float, order="C")
        while not result_queue.empty():
            start_field_index, end_field_index, property_array_result, energies_array_result = result_queue.get()
            for i, j in enumerate(range(start_field_index, end_field_index)):
                property_array[:, j] += property_array_result[:, i]
                self._energies[j] += energies_array_result[i]
        return property_array

    def _save(self):
        mode_dict = {"s": "SPIN", "l": "ANGULAR_MOMENTA", "p": "ELECTRIC_DIPOLE_MOMENTA", "j": "TOTAL_ANGULAR_MOMENTA", "m": "MAGNETIC_DIPOLE_MOMENTA"}
        xyz_string = 'xyz, ' if self._direction == 'xyz' else ''
        field_orientations_format = f'[fields, {xyz_string}:]' if self._orientations.shape[1] == 4 else f'[orientations, fields, {xyz_string}:]'
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AVERAGE" if self._orientations.shape[1] == 4 else "DIRECTIONAL",
            "Precision": settings.precision.upper(),
            "Mode": self._mode.upper(),
            "Description": f"Group containing {self._mode.upper()} {'matrices' if self._full_matrix else ''} under magnetic field calculated from Group '{self._group_name}'."
        }
        self._data_dict = {
            "MAGNETIC_FIELDS": (self._magnetic_fields, "Dataset containing magnetic field (T) values used in the simulation."),
            "ORIENTATIONS": (self._orientations, "Dataset containing magnetic fields' orientation grid used in the simulation."),
            "DIRECTION": (self._direction, "Dataset containing information about the direction of the calculated properties."),
        }
        if self._return_energies:
            self._data_dict["ENERGIES"] = (self._energies, f"Dataset containing energies (a.u.) of states under magnetic fields in the form {field_orientations_format[-2]} energies].")
        for index, mode in enumerate(self._mode):
            self._data_dict[mode_dict[mode]] = (self._result[index], f"Dataset containing {mode.upper()} {'matrices' if self._full_matrix else 'expectation values'} under magnetic fields in the form {field_orientations_format}.")
    
    def _load_from_slt_file(self):
        mode_dict = {"s": "SPIN", "l": "ANGULAR_MOMENTA", "p": "ELECTRIC_DIPOLE_MOMENTA", "j": "TOTAL_ANGULAR_MOMENTA", "m": "MAGNETIC_DIPOLE_MOMENTA"}
        self._mode = self._slt_group.attributes["Mode"]
        dims = self._slt_group[self._mode[0]].shape
        self._result = empty((len(self._mode), *dims), dtype=settings.complex if len(dims) == 3 else settings.float, order="C")
        for index, mode in enumerate(self._mode):
            self._result[index] = self._slt_group[mode_dict[mode]][:]
        self._magnetic_fields = self._slt_group["MAGNETIC_FIELDS"][:]
        self._orientations = self._slt_group["ORIENTATIONS"][:]
        self._direction = self._slt_group["DIRECTION"][:]
        try:
            self._energies = self._slt_group["ENERGIES"][:]
        except SltFileError:
            self._return_energies = False
    
    @property
    @ensure_ready
    def energies(self):
        if self._return_energies:
            return self._energies
        else:
            raise SltCompError(self._hdf5, RuntimeError("Computation of energies was not requested. To obtain them run calculations again with return_energies = True."))

    def _plot(self, **kwargs):
        pass
 
    def _to_data_frame(self):
        pass

    # also in input parser add to error with direct acces to properties to use property_under_magnetic_field instead!! for slothpy hamiltonians with field [0,0,0]


class SltZeemanSplitting(_MultiProcessed):
    _method_name = "Zeeman Splitting"
    _method_type = "ZEEMAN_SPLITTING"

    __slots__ = _MultiProcessed.__slots__ + ["_magnetic_fields", "_orientations", "_number_of_states", "_states_cutoff", "_rotation", "_electric_field_vector", "_hyperfine"]
     
    def __init__(self, slt_hamiltonian, ####### Teraz slt_group i argumenty boezposrednio przekazywane (zmień całość tutaj i w magnetyzacji)
        magnetic_fields: ndarray, ########################## Also slt.hamiltonian.info is removed from create jobs from first entry so probably move it into args!!!!!!!!!!!!!!
        orientations: ndarray,
        number_of_states: int,
        states_cutoff: list = [0,0],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = 1,
        number_threads: int = 1,
        autotune: bool = False,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_hamiltonian, magnetic_fields.shape[0] * orientations.shape[0], number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        self._number_of_states = number_of_states
        if self._orientations.shape[1] == 4:
            self._returns = True
        self._states_cutoff = states_cutoff
        self._rotation = rotation
        self._electric_field_vector = electric_field_vector
        self._hyperfine = hyperfine
        self._args = [self._number_of_states, self._electric_field_vector]
        self._executor_proxy = _zeeman_splitting_proxy
        self._slt_hamiltonian = slt_hamiltonian._slt_hamiltonian_from_slt_group(self._states_cutoff, self._rotation, self._hyperfine, False) ######## This should be at least renamed to set rotation or something like this
        self._slt_hamiltonian._mode = "em" if electric_field_vector is None else "emp"
    
    def _load_args_arrays(self):
        self._args_arrays = [*self._slt_hamiltonian.arrays_to_shared_memory, self._magnetic_fields, self._orientations]
        if not self._returns:
            self._result = empty((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._number_of_states), dtype=self._magnetic_fields.dtype, order="C")
            self._result_shape = (self._orientations.shape[0], self._magnetic_fields.shape[0], self._number_of_states)
    
    def _gather_results(self, result_queue):
        zeeman_splitting_array = zeros((self._magnetic_fields.shape[0], self._args[0]), dtype=self._magnetic_fields.dtype)
        while not result_queue.empty():
            start_field_index, end_field_index, zeeman_array = result_queue.get()
            for i, j in enumerate(range(start_field_index, end_field_index)):
                zeeman_splitting_array[j, :] += zeeman_array[i, :]
        return zeeman_splitting_array

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AVERAGE" if self._orientations.shape[1] == 4 else "DIRECTIONAL",
            "Precision": settings.precision.upper(),
            "Description": f"Group containing Zeeman splitting calculated from Group '{self._group_name}'."
        }
        self._data_dict = { # tutaj jednak wybrałem {slt._method_type itd. patrz na phonons}
            "ZEEMAN_SPLITTING": (self._result, "Dataset containing Zeeman splitting in the form {}".format("[fields, energies]" if self._orientations.shape[1] == 4 else "[orientations, fields, energies]")),
            "MAGNETIC_FIELDS": (self._magnetic_fields, "Dataset containing magnetic field (T) values used in the simulation."),
            "ORIENTATIONS": (self._orientations, "Dataset containing magnetic fields' orientation grid used in the simulation."),
        }

    def _load_from_slt_file(self):
        self._result = self._slt_group["ZEEMAN_SPLITTING"][:]
        self._magnetic_fields = self._slt_group["MAGNETIC_FIELDS"][:]
        self._orientations = self._slt_group["ORIENTATIONS"][:]

    def _plot(self, **kwargs):
        from slothpy._general_utilities._ploting_utilities import _plot_zeeman_splitting
        _plot_zeeman_splitting(self._, self._result, self._magnetic_fields, **kwargs)
 
    def _to_data_frame(self):
        pass


class SltMagnetisation(_MultiProcessed):
    _method_name = "Magnetisation"
    _method_type = "MAGNETISATION"

    __slots__ = _MultiProcessed.__slots__ + ["_magnetic_fields", "_orientations", "_temperatures", "_states_cutoff", "_rotation", "_electric_field_vector", "_hyperfine"]
     
    def __init__(self, slt_group,
        magnetic_fields: ndarray,
        orientations: ndarray,
        temperatures: ndarray,
        states_cutoff: list = [0,0],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = 1,
        number_threads: int = 1,
        autotune: bool = False,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0] , number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        if self._orientations.shape[1] == 4:
            self._returns = True
        self._temperatures = temperatures
        self._states_cutoff = states_cutoff
        self._rotation = rotation
        self._electric_field_vector = electric_field_vector
        self._hyperfine = hyperfine
        self._args = [self._electric_field_vector]
        self._executor_proxy = _magnetisation_proxy
        self._slt_hamiltonian = self._slt_group._slt_hamiltonian_from_slt_group(self._states_cutoff, self._rotation, self._hyperfine, True)
        self._slt_hamiltonian._mode = "em" if electric_field_vector is None else "emp"
    
    def _load_args_arrays(self):
        self._args_arrays = [*self._slt_hamiltonian.arrays_to_shared_memory, self._magnetic_fields, self._orientations, self._temperatures]
        if not self._returns:
            self._result = empty((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._temperatures.shape[0]), dtype=self._magnetic_fields.dtype, order="C")
            self._result_shape = (self._orientations.shape[0], self._magnetic_fields.shape[0], self._temperatures.shape[0])
            self._transpose_result = (0, 2, 1)
    
    def _gather_results(self, result_queue):
        result_magnetisation_array = zeros((self._magnetic_fields.shape[0], self._temperatures.shape[0]), dtype=self._temperatures.dtype)
        while not result_queue.empty():
            start_field_index, end_field_index, magnetisation_array = result_queue.get()
            for i, j in enumerate(range(start_field_index, end_field_index)):
                result_magnetisation_array[j, :] += magnetisation_array[i, :]
        return result_magnetisation_array.T

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AVERAGE" if self._orientations.shape[1] == 4 else "DIRECTIONAL",
            "Precision": settings.precision.upper(),
            "Description": f"Group containing magnetisation calculated from Group '{self._group_name}'."
        }
        self._data_dict = {
            "MAGNETISATION": (self._result, "Dataset containing magnetisation in the form {}".format("[temperatures, fields]" if self._orientations.shape[1] == 4 else "[orientations, temperatures, fields]")),
            "MAGNETIC_FIELDS": (self._magnetic_fields, "Dataset containing magnetic field (T) values used in the simulation."),
            "ORIENTATIONS": (self._orientations, "Dataset containing magnetic fields' orientation grid used in the simulation."),
            "TEMPERATURES": (self._temperatures, "Dataset containing temperature (K) values used in the simulation.")
        }

    def _load_from_slt_file(self):
        self._result = self._slt_group["MAGNETISATION"][:]
        self._magnetic_fields = self._slt_group["MAGNETIC_FIELDS"][:]
        self._orientations = self._slt_group["ORIENTATIONS"][:]

    def _plot(self):
        pass
 
    def _to_data_frame(self):
        pass


class SltPhononDispersion(_MultiProcessed):
    _method_name = "Phonon Dispersion"
    _method_type = "PHONON_DISPERSION"

    __slots__ = _MultiProcessed.__slots__ + ["_hessian", "_kpts", "_masses", "_bandpath", "_modes_cutoff", "_x", "_x_coords", "_x_labels"]
     
    def __init__(self, slt_group, hessian: ndarray, masses: ndarray, bandpath: BandPath, modes_cutoff: int = 0, number_cpu: int = 1, number_threads: int = 1, autotune: bool = False, slt_save: str = None, smm: SharedMemoryManager = None, terminate_event: Event = None) -> None:
        super().__init__(slt_group, len(bandpath.kpts), number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._hessian = hessian
        self._masses = masses
        self._bandpath = bandpath
        self._kpts = bandpath.kpts.astype(settings.float)
        self._x, self._x_coords, self._x_labels = bandpath.get_linear_kpoint_axis()
        self._modes_cutoff = modes_cutoff
        self._args = [self._modes_cutoff]
        self._executor_proxy = _phonon_dispersion_proxy
    
    def _load_args_arrays(self):
        masses = repeat(self._masses, 3)
        masses_inv_sqrt = 1.0 / sqrt(masses)
        masses_inv_sqrt = outer(masses_inv_sqrt, masses_inv_sqrt)
        self._args_arrays = [self._hessian[:], masses_inv_sqrt, self._kpts]
        self._result = empty((len(self._x), self._modes_cutoff), dtype=settings.float, order="C")

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Precision": settings.precision.upper(),
            "Description": f"Group containing {self._method_name} calculated from Group '{self._group_name}'."
        }
        self._data_dict = {
            f"{self._method_type}": (self._result, f"Dataset containing {self._method_name} in the form [kpts, frequencies] in cm-1."),
            "X": (self._x, "Dataset containing X coordinates for the dispersion plotting."),
            "X_COORDS": (self._x_coords, "Dataset containing X coordiantes of the special point labels."),
            "X_LABELS": (self._x_labels, "Dataset containing the special point labels."),
        }

    def _load_from_slt_file(self):
        self._result = self._slt_group[f"{self._method_type}"][:]
        self._x = self._slt_group["X"][:]
        self._x_coords = self._slt_group["X_COORDS"][:]
        self._x_labels = self._slt_group["X_LABELS"][:]

    def _plot(self, **kwargs):
        plt.figure(figsize=(8, 6))
        for mode in range(self._result.shape[1]):
            plt.plot(self._x, self._result[:, mode], color='b')

        plt.xticks(self._x_coords, self._x_labels)
        plt.xlabel('Wave Vector Fraction along Path')
        plt.ylabel('Frequency (cm$^{-1}$)')  # Adjust units as needed
        plt.title('Phonon Dispersion')
        plt.grid(True)
        plt.show()
 
    def _to_data_frame(self):
        pass