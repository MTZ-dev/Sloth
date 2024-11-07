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

import sys
import signal
from time import sleep
from typing import Iterable
from multiprocessing import Process, Queue
from multiprocessing.synchronize import Event

from threadpoolctl import threadpool_limits
from numba import set_num_threads

from slothpy.core._system import SltTemporarySignalHandler


class SltProcessPool:
    def __init__(self, worker: callable, jobs: Iterable, number_threads: int, returns: bool, gather_results: callable, terminate_event: Event = None):
        self._worker = worker
        self._jobs = jobs
        self._number_threads = number_threads
        self._terminate_event = terminate_event
        self._processes = []
        self._returns = returns
        if returns:
            self._result_queue = Queue()
            self._gather_results = gather_results
        self._result = None

    def termiante_and_close_pool(self, signum = None, frame = None, silent = False):
        if not silent:
            print("\nSltProcessPool interrupted. Clearing and terminating...")
        for process in self._processes:
            if process is not None:
                process.terminate()
        for process in self._processes:
            if process is not None:
                process.join()
                process.close()
        sys.exit(1)

    def start_and_collect(self):
        with SltTemporarySignalHandler([signal.SIGTERM, signal.SIGINT], self.termiante_and_close_pool):
            for job_args in self._jobs:
                if self._returns:
                    process = Process(target=_worker_wrapper, args=(self._worker, job_args, self._number_threads, self._result_queue))
                else:
                    process = Process(target=_worker_wrapper, args=(self._worker, job_args, self._number_threads))
                process.start()
                self._processes.append(process)

            if self._terminate_event:
                while True:
                    if self._terminate_event.is_set():
                        self.termiante_and_close_pool(silent=True)
                        break
                    if all(process is None or not process.is_alive() for process in self._processes):
                        break
                sleep(0.2)
            else:
                if self._returns:
                    self._result = self._gather_results(self._result_queue, len(self._processes))
                for process in self._processes:
                    process.join()

        return self._result
    

def _worker_wrapper(worker, args, number_threads, result_queue=None):
    with threadpool_limits(limits=number_threads):
        set_num_threads(number_threads)
        result = worker(*args)
    if result_queue is not None:
        result_queue.put(result)