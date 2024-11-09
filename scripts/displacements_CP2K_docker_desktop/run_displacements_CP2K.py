#!/usr/bin/env python3

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

import os
import sys
import signal
import argparse
import glob
import re
import docker
import datetime
import uuid
from multiprocessing import Pool

import docker.errors
from tqdm import tqdm

def generate_input_file(dof_number, disp_number, wfn_start=None):
    project_name = f'dof_{dof_number}_disp_{disp_number}'
    input_filename = f'{project_name}.inp'
    xyz_filename = f'dof_{dof_number}_disp_{disp_number}.xyz'

    if dof_number == 0 and disp_number == 0:
        restart_wfn_line = '' if wfn_start is None else f"    WFN_RESTART_FILE_NAME {wfn_start}"
    else:
        restart_wfn_line = '    WFN_RESTART_FILE_NAME dof_0_disp_0-RESTART.wfn'

    input_template = """&GLOBAL
    PROJECT {project_name}
    RUN_TYPE ENERGY_FORCE
    PRINT_LEVEL MEDIUM
    WALLTIME 255000
    # PREFERRED_DIAG_LIBRARY ScaLAPACK
&END GLOBAL

&FORCE_EVAL
    METHOD QS
    STRESS_TENSOR  ANALYTICAL

    &DFT
    BASIS_SET_FILE_NAME   BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME   POTENTIAL_UZH
{restart_wfn_line}
    CHARGE  0
    MULTIPLICITY 1
    UKS  F

    &MGRID
        CUTOFF 1800
        NGRIDS 5
        REL_CUTOFF 90
    &END MGRID

    &POISSON
        PERIODIC XYZ
    &END POISSON

    &QS
        EPS_DEFAULT 1.0E-10
        METHOD GAPW
        EXTRAPOLATION_ORDER  4
        # MIN_PAIR_LIST_RADIUS -1 # Set this for hybrids: see https://www.cp2k.org/faq:hfx_eps_warning
        # EPS_PGF_ORB 1.0E-14
    &END QS

    &SCF
        @INCLUDE scf.inc
    &END SCF

    &XC
        @INCLUDE XC.inc
    &END XC

    &PRINT
        &MOMENTS
        FILENAME
        &END
        &END

    &END DFT

    &SUBSYS
    @INCLUDE subsys.inc

    &TOPOLOGY
        COORD_FILE_NAME {xyz_filename}
        COORD_FILE_FORMAT  XYZ
        NUMBER_OF_ATOMS  -1
        MULTIPLE_UNIT_CELL  1 1 1
        &END TOPOLOGY

    &END SUBSYS

    &PRINT
    &FORCES
        FILENAME
        NDIGITS 12
    &END
    &END

&END FORCE_EVAL
    """

    with open(input_filename, 'w') as f:
        f.write(input_template.format(project_name=project_name, xyz_filename=xyz_filename, restart_wfn_line=restart_wfn_line))

    return input_filename

def run_cp2k(input_file, output_file, mpi_processes, threads, cp2k_version, dof_number, disp_number, main_process=False):

    image = f'cp2k/cp2k:{cp2k_version}'  ###!!!### Replace that with your format if it's different than cp2k/cp2k:{cp2k_version} ###!!!###

    client = docker.from_env()
    container = None

    # Generate a unique container name
    container_name = f'cp2k_{input_file}_{uuid.uuid4().hex}'

    volumes = {
        os.getcwd(): {
            'bind': '/mnt',
            'mode': 'rw'
        }
    }

    user = f"{os.getuid()}:{os.getgid()}"

    # Prepare the command to run inside the container
    command = [
        'mpirun',
        '-bind-to', 'none',
        '-np', str(mpi_processes),
        '-x', f'OMP_NUM_THREADS={threads}',
        'cp2k',
        '-i', input_file
    ]

    try:
        existing_container = client.containers.get(container_name)
        print(f"Removing existing container with name {container_name}")
        existing_container.remove(force=True)
    except docker.errors.NotFound:
        pass  # No existing container, proceed

    print(f"Starting container for {input_file}...")

    container = client.containers.create(
        image=image,
        command=command,
        volumes=volumes,
        user=user,
        working_dir='/mnt',
        detach=True,
        shm_size='4g',
        name=container_name,
    )

    def handle_sigterm(signum, frame):
        print(f"KeyboardInterrupt or termination signal caught in run_cp2k for {input_file} closing docker container and client...")
        try:
            container.remove(force=True)
            client.close()
        except Exception as e:
            print(f"Error stopping container or client: {e}. Stop it and all the others manually e.g. using task manager.")
        print(f"Terminating process and container for dof {dof_number} disp {disp_number}...")
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    container.start()
    # Stream logs to the output file
    with open(output_file, 'wb') as outfile:
        for log in container.logs(stream=True):
            outfile.write(log)
            outfile.flush()

    exit_code = container.wait()['StatusCode']

    if exit_code != 0:
        print(f"Container exited with code {exit_code} for {input_file}")
        raise Exception(f"Container error with exit code {exit_code}")

def process_dof_disp(dof_disp):
    dof_number, disp_number = dof_disp
    project_name = f'dof_{dof_number}_disp_{disp_number}'

    # Safety check: if calculation is already done, skip
    moments_file = f'{project_name}-moments-1_0.dat'
    xyz_file_out = f'{project_name}-1_0.xyz'
    if os.path.exists(moments_file) and os.path.exists(xyz_file_out):
        print(f"Skipping {project_name}, calculation already completed.")
        return

    input_file = generate_input_file(dof_number, disp_number)
    output_file = f'{project_name}.out'
    threads_per_process = args.threads
    mpi_processes = (args.cpus // args.processes) // threads_per_process
    cp2k_version = args.version
    run_cp2k(input_file, output_file, mpi_processes, threads_per_process, cp2k_version, dof_number, disp_number)

    # Remove temporary files except for dof_0_disp_0
    if not (dof_number == 0 and disp_number == 0):
        files_to_remove = glob.glob(f'{project_name}-RESTART.wfn*')
        for f in files_to_remove:
            if os.path.isfile(f):
                os.remove(f)
    print(f'Completed calculation for {project_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CP2K calculations in Docker containers.')
    parser.add_argument('--cpus', type=int, default=16, help='Total number of CPUs to use.')
    parser.add_argument('--processes', type=int, default=1, help='Number of concurrent processes (containers) where (cpus//processes)//threads mpi processes will be used per container.')
    parser.add_argument('--threads', type=int, default=2, help='Number of OMP threads per each mpi process within container.')
    parser.add_argument('--version', type=str, required=True, help='CP2K Docker image version to use.')
    parser.add_argument('--wfn_start', type=str, default=None, help='Optional CP2K -RESTART.wfn file with starting guess for the relaxed geometry.')

    args = parser.parse_args()

    # Get list of xyz files
    xyz_files = glob.glob('dof_*_disp_*.xyz')
    pattern = re.compile(r'dof_(-?\d+)_disp_(-?\d+)\.xyz')

    dof_disp_list = []

    for filename in xyz_files:
        match = pattern.match(filename)
        if match:
            dof_number = int(match.group(1))
            disp_number = int(match.group(2))
            if (dof_number == 0 and disp_number == 0):
                continue  # We handle dof_0_disp_0 separately
            else:
                # Check if calculation is already done
                moments_file = f'dof_{dof_number}_disp_{disp_number}-moments-1_0.dat'
                xyz_file_out = f'dof_{dof_number}_disp_{disp_number}-1_0.xyz'
                if os.path.exists(moments_file) or os.path.exists(xyz_file_out):
                    print(f"Skipping dof_{dof_number}_disp_{disp_number}, calculation already completed.")
                    continue
                else:
                    dof_disp_list.append((dof_number, disp_number))

# Check if dof_0_disp_0.xyz exists
if not os.path.exists('dof_0_disp_0.xyz'):
    raise ValueError('dof_0_disp_0.xyz file is missing. It is required to proceed.')

# Check if dof_0_disp_0 calculation is already completed
restart_file = 'dof_0_disp_0-RESTART.wfn'
if os.path.exists(restart_file):
    print('Calculation for dof_0_disp_0 is already completed, skipping.')
else:
    print(f"Job started {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('Processing dof_0_disp_0...')
    input_file = generate_input_file(0, 0, args.wfn_start)
    output_file = 'dof_0_disp_0.out'
    threads = args.threads
    mpi_processes = args.cpus // threads  # Use allocated CPUs
    cp2k_version = args.version
    run_cp2k(input_file, output_file, mpi_processes, threads, cp2k_version, 0, 0, True)
    print('Completed calculation for dof 0 disp 0')
    # After completion, check if the restart file was generated
    if not os.path.exists(restart_file):
        raise ValueError('Calculation for dof_0_disp_0 failed to generate the restart file. Cannot proceed.')

# Process remaining dof and disp in parallel
if dof_disp_list:
    print('Processing remaining calculations in parallel...')
    try:
        pool_args = [dof_disp_list[i] for i in range(len(dof_disp_list))]

        with Pool(processes=args.processes) as pool:
            for _ in tqdm(pool.imap_unordered(process_dof_disp, pool_args), total=len(pool_args)):
                pass
    except KeyboardInterrupt:
        print("\nTerminating pool...")
        pool.terminate()
        pool.join()
        sys.exit(1)
else:
    print('No other dof_disp combinations to process.')
