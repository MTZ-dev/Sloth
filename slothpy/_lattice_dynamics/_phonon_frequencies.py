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

from numpy import ndarray, asarray, arange, where, outer, abs, sqrt, int64

from slothpy.core._hessian_object import Hessian
from slothpy._general_utilities._constants import AU_BOHR_CM_1

def _phonon_frequencies(hessian: ndarray, masses_inv_sqrt: ndarray, kpoint: ndarray, start_mode: int, stop_mode: int):
    au_bohr_cm_1 = asarray(AU_BOHR_CM_1, dtype=hessian.dtype)
    hessian_object = Hessian(hessian, outer(masses_inv_sqrt, masses_inv_sqrt), kpoint=kpoint, start_mode=start_mode, stop_mode=stop_mode, eigen_range="I")

    frequencies = hessian_object.frequencies * au_bohr_cm_1

    mode_numbers = arange(start_mode, stop_mode, 1, dtype=int64)

    return mode_numbers, frequencies