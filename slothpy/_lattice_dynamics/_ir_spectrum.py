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

from typing import Optional, Literal

from numpy import ndarray, asarray, zeros, outer, sum, hstack, linspace, sign, max, abs, sqrt, complex64, log

from slothpy.core._hessian_object import Hessian
from slothpy._general_utilities._constants import AU_BOHR_CM_1
from slothpy._general_utilities._lapack import _cgemmt, _zgemmt
from slothpy._general_utilities._math_expresions import _convolve_gaussian_xyz, _convolve_lorentzian_xyz

def _ir_spectrum(hessian: ndarray, masses_inv_sqrt: ndarray, born_charges: ndarray, start_wavenumber: float, stop_wavenumber: float, convolution: Optional[Literal["lorentzian", "gaussian"]] = "lorentizan", resolution: int = None, fwhm: float = None):
    au_bohr_cm_1 = asarray(AU_BOHR_CM_1, dtype=hessian.dtype)

    hessian_object = Hessian([hessian, outer(masses_inv_sqrt, masses_inv_sqrt)], kpoint=asarray([0., 0., 0.], dtype=hessian.dtype), start_frequency=start_wavenumber, stop_frequency=stop_wavenumber, single_process=True, eigen_range="V")

    frequencies, eigenvectors = hessian_object.frequencies_eigenvectors
    frequencies = frequencies * au_bohr_cm_1

    born_charges_weighted = born_charges * masses_inv_sqrt[:, None]
    Z_modes = _cgemmt(eigenvectors, born_charges_weighted) if eigenvectors.dtype == complex64 else _zgemmt(eigenvectors, born_charges_weighted)
    IR_intensities_xyz = (abs(Z_modes)**2)
    IR_intensities_av = sum(IR_intensities_xyz, axis=1)

    frequencies_intensities = hstack((frequencies.reshape(-1, 1), IR_intensities_xyz, IR_intensities_av.reshape(-1, 1))).T
    frequencies_intensities[1:4,:] = frequencies_intensities[1:4,:] / max(frequencies_intensities[1:4,:])
    frequencies_intensities[4,:] = frequencies_intensities[4,:] / max(frequencies_intensities[4,:])

    if convolution is not None:
        frequency_range_convolution = zeros((5, resolution), dtype=hessian.dtype)
        start_wavenumber = sign(start_wavenumber) * sqrt(abs(start_wavenumber)) * au_bohr_cm_1
        stop_wavenumber = sign(stop_wavenumber) * sqrt(abs(stop_wavenumber)) * au_bohr_cm_1
        frequency_range_convolution[0, :] = linspace(start_wavenumber, stop_wavenumber, resolution, dtype=hessian.dtype)

        if convolution == "lorentzian":
            gamma = fwhm / 2
            _convolve_lorentzian_xyz(frequencies_intensities, frequency_range_convolution, gamma)
        elif convolution == "gaussian":
            sigma = fwhm / (2 * sqrt(2 * log(2)))
            _convolve_gaussian_xyz(frequencies_intensities, frequency_range_convolution, sigma)

        frequency_range_convolution[1:4,:] = frequency_range_convolution[1:4,:] / max(frequency_range_convolution[1:4,:])
        frequency_range_convolution[4,:] = frequency_range_convolution[4,:] / max(frequency_range_convolution[4,:])

        return frequencies_intensities, frequency_range_convolution
    
    return (frequencies_intensities,)