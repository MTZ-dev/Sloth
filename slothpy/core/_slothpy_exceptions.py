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

"""Module for storing custom exception classes with error reporting modes."""
from functools import wraps
from typing import Literal
from slothpy._general_utilities._constants import RED, GREEN, YELLOW, RESET


class SltInputError(Exception):
    """
    A custom exception class for errors in input data.

    Parameters
    ----------
    None
    """

    def __init__(self, exception: Exception, message: str = ""):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """
        self.error_type = type(exception).__name__ if exception is not None else ""
        self.error_message = str(exception) if exception is not None else ""
        self.slt_message = f"{RED}SlothInputError{RESET},{'' if self.error_type == '' else ' '}{YELLOW}{self.error_type}{RESET}{': ' if self.error_type != '' else ''}{self.error_message}"
        self.final_message = f"{self.slt_message}{'' if self.slt_message.endswith(' ') else ' '}{message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """
        return self.final_message


class SltFileError(Exception):
    """
    A custom exception class for errors connected to operations on .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = "", error_type_override=None):
        """
        Initialize for the custom message printing.

        Parameters
        ----------
        file : str
            A file to which the error corresponds.
        exception : Exception
            An exception that initially caused the error.
        message : str, optional
            A message to be printed., by default ""
        """

        self._slt_error_type = error_type_override if error_type_override is not None else "SlothFileError"
        self.error_type = type(exception).__name__ if exception is not None else ""
        self.error_message = str(exception) if exception is not None else ""
        self.slt_message = f"{RED}{self._slt_error_type}{RESET}, {GREEN}File{RESET} '{file}',{'' if self.error_type == '' else ' '}{YELLOW}{self.error_type}{RESET}{': ' if self.error_type != '' else ''}{self.error_message}"
        self.final_message = f"{self.slt_message}{'' if self.slt_message.endswith(' ') else ' '}{message}"
        super().__init__(self.final_message)

    def __str__(self) -> str:
        """
        Perform the operation __str__.

        Overwrites the default Exception __str__ method to provide a custom
        message for printing.

        Returns
        -------
        str
            Custom error message.
        """

        return self.final_message


class SltCompError(SltFileError):
    """
    A custom exception class for runtime errors during computations.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        super().__init__(file, exception, message, "SlothComputationError")


class SltSaveError(SltFileError):
    """
    A custom exception class for errors in saving data to .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        super().__init__(file, exception, message, "SlothSaveError")


class SltReadError(SltFileError):
    """
    A custom exception class for errors in reading data from .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        super().__init__(file, exception, message, "SlothReadError")
    

class SltPlotError(SltFileError):
    """
    A custom exception class for errors in data plotting from .slt files.

    Parameters
    ----------
    None
    """

    def __init__(self, file: str, exception: Exception, message: str = ""):
        super().__init__(file, exception, message, "SlothPlotError")
    

def slothpy_exc(slt_exception: Literal["SltFileError", "SltCompError", "SltSaveError", "SltReadError", "SltInputError", "SltPlotError"], slt_message: str = "") -> callable:
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                exception_mapping = {
                    "SltInputError": SltInputError,
                    "SltFileError": SltFileError,
                    "SltCompError": SltCompError,
                    "SltSaveError": SltSaveError,
                    "SltReadError": SltReadError,
                    "SltPlotError": SltPlotError,
                }
                if slt_exception in exception_mapping:
                    if slt_exception == "SltInputError":
                        raise exception_mapping[slt_exception](exc, slt_message) from None
                    else:
                        raise exception_mapping[slt_exception](args[0]._hdf5, exc, slt_message) from None
                else:
                    raise ValueError(f"Unsupported {RED}SltException{RESET} provided.") from None
        return wrapper
    return decorator


class KeyError(Exception):
    """
    A custom KeyError to overwrite the standard library to get rid of its weird str processing and quoting.

    Parameters
    ----------
    None
    """
    def __init__(self, message):
        super().__init__(message)

