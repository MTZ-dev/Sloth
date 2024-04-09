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
from typing import List, Tuple
from abc import ABC, abstractmethod
from contextlib import contextmanager, ExitStack
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Process
from multiprocessing import Event as terminate_event
from os.path import join
from time import perf_counter_ns, sleep
from datetime import datetime

from numpy import array, zeros, any, all, median, int64
from threadpoolctl import threadpool_limits
from numba import set_num_threads

from slothpy.core._slothpy_exceptions import slothpy_exc
from slothpy.core._slt_file import SltGroup
from slothpy._general_utilities._system import SltProcessPool, SharedMemoryArrayInfo, ChunkInfo, _get_number_of_processes_threads, _to_shared_memory, _from_shared_memory, _distribute_chunks, _from_shared_memory_to_array
from slothpy._general_utilities._constants import RED, GREEN, BLUE, YELLOW, PURPLE, RESET

def ensure_ready(func):
    def wrapper(self, *args, **kwargs):
        if not self._ready:
            self.run()
        return func(self, *args, **kwargs)
    
    return wrapper

class SingleProcessed(ABC):

    __slots__ = ["_slt_group", "_hdf5", "_group_name", "_driver", "_result", "_ready", "_slt_save", "_df"]

    @abstractmethod
    def __init__(self, slt_group: SltGroup, slt_save: str = None) -> None:
        super().__init__()
        self._slt_group = slt_group
        self._hdf5 = slt_group._hdf5
        self._group_name = slt_group._group_name
        self._driver = "single"
        self._result = None
        self._ready = False
        self._slt_save = slt_save
        self._df = None

    @abstractmethod
    def __repr__(self) -> str:
        pass
    
    @classmethod
    def _from_file(cls, slt_group: SltGroup) -> SingleProcessed:
        instance = cls.__new__(cls)
        instance._slt_group = slt_group
        instance._hdf5 = slt_group._hdf5
        instance._group_name = slt_group._group_name
        instance._driver = None
        instance._result = None
        instance._slt_save = None
        instance._df = None
        instance._load()
        instance._ready = True

        return instance
    
    @abstractmethod
    def _executor():
        pass
    
    @slothpy_exc("SltCompError")
    def run(self):
        if not self._ready:
            self._result = self._executor()
            self._ready = True
        if self._slt_save is not None:
            self.save()
        return self._result
    
    @ensure_ready
    def eval(self):
        return self._result

    @abstractmethod
    def save(self, slt_save = None):
        pass

    @abstractmethod
    def _load(self):
        pass
    
    @abstractmethod
    def plot(self):
        pass
    
    @ensure_ready
    def to_numpy_array(self):
        return self._result
    
    @abstractmethod
    def to_data_frame(self):
        pass
    
    def to_csv(self, file_path=".", file_name="states_energies_cm_1.csv", separator=","):
        if self._df is None:
            self.to_data_frame()
        self._df.to_csv(join(file_path, file_name), sep=separator)
    
    def dataframe_to_slt_file(self, group_name):
        if self._df is None:
            self.to_data_frame()
        self._df.to_hdf(self._hdf5)
       

class MulitProcessed(SingleProcessed):

    __slots__ = SingleProcessed.__slots__ + ["_number_to_parallelize", "_number_cpu", "_number_processes", "_number_threads", "_autotune", "_smm", "_sm_info", "_sm", "_terminate_event", "_returns", "_args_arrays", "_args", "_sm_result_info", "_result_shape", "_sm_progress_array_info", "_method_no_return", "_method_return"]

    @abstractmethod
    def __init__(self, slt_group: SltGroup, number_to_parallelize: int, number_cpu: int, number_threads: int, autotune: bool, smm: SharedMemoryManager = None, terminate_event: Event = None, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._driver = "multi"
        self._number_to_parallelize = number_to_parallelize
        self._number_cpu = number_cpu
        self._number_processes, self._number_threads = _get_number_of_processes_threads(number_cpu, number_threads, number_to_parallelize)
        self._autotune = autotune
        self._smm = smm
        self._sm_info = []
        self._sm = []
        self._terminate_event = terminate_event
        self._returns = False
        self._args_arrays = ()
        self._args = ()
        self._sm_result_info = None
        self._result_shape = ()
        self._sm_progress_array_info = None
        self._method_no_return = None
        self._method_return = None

    @contextmanager
    def _ensure_shared_memory_manager(self):
        if self._smm is None:
            with SharedMemoryManager() as smm:
                self._smm = smm
                yield
                self._smm = None
        else:
            yield

    def _create_shared_memory(self):
        for array in self._args_arrays:
            self._sm_info.append(_to_shared_memory(self._smm, array))
        del self._args_arrays
        self._sm_progress_array_info = _to_shared_memory(self._smm, zeros((self._number_processes,), dtype=int64, order="C"))
        if not self._returns:
            self._sm_result_info = _to_shared_memory(self._smm, self._result)
        del self._result

    def _retrieve_shared_memory_arrays(self):
        arrays = []
        for sm_array_info in self._sm_info:
            self._sm.append(SharedMemory(sm_array_info.name))
            arrays.append(_from_shared_memory(self._sm[-1], sm_array_info))
        self._sm.append(SharedMemory(self._sm_progress_array_info.name))
        arrays.append(_from_shared_memory(self._sm[-1], self._sm_progress_array_info))
        if not self._returns:
            self._sm.append(SharedMemory(self._sm_result_info.name))
            arrays.append(_from_shared_memory(self._sm[-1], self._sm_result_info))
        return arrays
    
    def _retrieve_args_arrays_and_results_from_shared_memory(self):
        self._args_arrays = []
        for sm_array_info in self._sm_info:
            self._args_arrays.append(_from_shared_memory_to_array(sm_array_info))
        self._sm_info = []
        self._result = _from_shared_memory_to_array(self._sm_result_info)
        self._sm_result_info = None

    def _create_jobs(self):
        return [(process_index, chunk.start, chunk.end) for process_index, chunk in enumerate(_distribute_chunks(self._number_to_parallelize, self._number_processes))]
    
    @abstractmethod
    def _gather_results(self, results):
        pass

    def _executor(self, process_index, chunk_start, chunk_end):
        sm_arrays = self._retrieve_shared_memory_arrays()
        with threadpool_limits(limits=self._number_threads):
            set_num_threads(self._number_threads)
            if self._returns:
                return self._method_return(*sm_arrays, *self._args, process_index, chunk_start, chunk_end)
            else:
                self._method_no_return(*sm_arrays, *self._args, process_index, chunk_start, chunk_end)

    def _parallel_executor(self):
        return SltProcessPool(self._executor, self._create_jobs(), self._returns, self._terminate_event).start_and_collect()

    @slothpy_exc("SltCompError")
    def autotune(self, _from_run: bool = False):
        final_number_of_processes = self._number_cpu
        final_number_of_threads = 1
        best_time = float("inf")
        old_processes = 0
        worse_counter = 0
        with ExitStack() as stack:
            stack.enter_context(self._ensure_shared_memory_manager())
            self._load_args_arrays
            self._create_shared_memory()
            for number_threads in range(min(64, self._number_cpu), 0, -1):
                number_processes = self._number_cpu // number_threads
                if number_processes >= self._number_to_parallelize:
                    number_processes = self._number_to_parallelize
                    number_threads = self._number_cpu // number_processes
                if number_processes != old_processes:
                    old_processes = number_processes
                    chunk_size = self._number_to_parallelize // number_processes
                    remainder = self._number_to_parallelize % number_processes
                    max_tasks_per_process = array([(chunk_size + (1 if i < remainder else 0)) for i in range(number_processes)])
                    if any(max_tasks_per_process < 5):
                        print(f"The job for {number_processes} {BLUE}Processes{RESET} and {number_threads} {PURPLE}Threads{RESET} is already too small to be autotuned! Quitting here.")
                        break
                    self._number_processes = number_processes
                    self._number_threads = number_threads
                    progress_array = zeros((number_processes,), dtype=int64, order="C")
                    self._sm_progress_array_info = _to_shared_memory(self._smm, progress_array)
                    self._terminate_event = terminate_event()
                    benchmark_process = Process(target=self._parallel_executor)
                    sm_progress = SharedMemory(self._sm_progress_array_info.name)
                    progress_array = _from_shared_memory(sm_progress, self._sm_progress_array_info)
                    benchmark_process.start()
                    while any(progress_array <= 1):
                        sleep(0.001)
                    start_time = perf_counter_ns()
                    start_progress = progress_array.copy()
                    final_progress = start_progress
                    stop_time = start_time
                    while any(progress_array - start_progress <= 4) and all(progress_array < max_tasks_per_process):
                        stop_time = perf_counter_ns()
                        final_progress = progress_array.copy()
                        sleep(0.01)
                    self._terminate_event.set()
                    overall_time = stop_time - start_time
                    progress = final_progress - start_progress
                    if any(progress <= 1) or overall_time == 0:
                        print(f"Jobs iterations for {number_processes} {BLUE}Processes{RESET} and {number_threads} {PURPLE}Threads{RESET} are too fast to be reliably autotuned! Quitting here.")
                        break
                    current_estimated_time = overall_time * (max_tasks_per_process/(progress))
                    current_estimated_time = median(current_estimated_time[:remainder] if remainder != 0 else current_estimated_time)
                    benchmark_process.join()
                    benchmark_process.close()
                    info = f"{BLUE}Processes{RESET}: {number_processes}, {PURPLE}Threads{RESET}: {number_threads}. Estimated execution time of the main loop: "
                    if current_estimated_time < best_time:
                        best_time = current_estimated_time
                        final_number_of_processes = number_processes
                        final_number_of_threads = number_threads
                        info += f"{GREEN}{current_estimated_time/1e9:.2f}{RESET} s."
                        worse_counter = 0
                    else:
                        info += f"{RED}{current_estimated_time/1e9:.2f}{RESET} s."
                        worse_counter += 1
                    if worse_counter > 3:
                        break
                    info += f" The best time: {GREEN}{best_time/1e9:.2f}{RESET} s."
                    print(info)
            time_info = f" (starting from now - [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}])" if _from_run else ''
            print(f"Job will run using{YELLOW} {final_number_of_processes * final_number_of_threads}{RESET} logical{YELLOW} CPU(s){RESET} with{BLUE} {final_number_of_processes}{RESET} parallel{BLUE} Processe(s){RESET} each utilizing{PURPLE} {final_number_of_threads} Thread(s){RESET}.\nThe calculation time{time_info} is estimated to be at least: {GREEN}{best_time/1e9} s{RESET}.")
            self._retrieve_args_arrays_and_results_from_shared_memory()
            self._number_processes, self._number_threads = final_number_of_processes, final_number_of_threads
            self._sm_progress_array_info = None
            self._autotune = False
            #### jak from run to nie czyścić argumentów tylko retrieve jak nie from run to
            
    
    @slothpy_exc("SltCompError")
    def run(self):
        if self._autotune:
            self.autotune(True)
        else:
            _load_args_arrays
        if not self._ready:
            with ExitStack() as stack:
                stack.enter_context(self._ensure_shared_memory_manager())
                self._create_shared_memory()
                results = self._parallel_executor()
                if self._returns:
                    self._result = self._gather_results(results)
                else:
                    self._result = _from_shared_memory_to_array(self._sm_result_info, reshape=(self._result_shape))
                self._ready = True
        if self._slt_save is not None:
            self.save()
        return self._result

    


    