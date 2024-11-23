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
import time
import signal
import argparse
import glob
import re
import subprocess
import shutil
from multiprocessing import Pool

from tqdm import tqdm
from scipy.special import binom

element_to_atomic_number = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92
}

lanthanide_data = {
    'Ce': {'nel':1, 'S_list':[0.5], 'NDoubGtensor':3},
    'Pr': {'nel':2, 'S_list':[1, 0], 'NDoubGtensor':5},
    'Nd': {'nel':3, 'S_list':[1.5, 0.5], 'NDoubGtensor':5},
    'Pm': {'nel':4, 'S_list':[2, 1, 0], 'NDoubGtensor':5},
    'Sm': {'nel':5, 'S_list':[2.5, 1.5, 0.5], 'NDoubGtensor':3},
    'Eu': {'nel':6, 'S_list':[3, 2, 1, 0], 'NDoubGtensor':1},
    'Gd': {'nel':7, 'S_list':[3.5, 2.5, 1.5, 0.5], 'NDoubGtensor':4},
    'Tb': {'nel':8, 'S_list':[3, 2, 1, 0], 'NDoubGtensor':7},
    'Dy': {'nel':9, 'S_list':[2.5, 1.5, 0.5], 'NDoubGtensor':8},
    'Ho': {'nel':10, 'S_list':[2, 1, 0], 'NDoubGtensor':9},
    'Er': {'nel':11, 'S_list':[1.5, 0.5], 'NDoubGtensor':8},
    'Tm': {'nel':12, 'S_list':[1, 0], 'NDoubGtensor':7},
    'Yb': {'nel':13, 'S_list':[0.5], 'NDoubGtensor':4}
}

def weyl_formula(n, S, N):
    return int((2*S+1)/(n+1) * binom(n+1, N/2 - S) * binom(n+1, N/2 + S + 1))

def parse_xyz_file(xyz_filename):
    # Function to extract charge, multiplicity, atoms, and lanthanide information from the XYZ file
    with open(xyz_filename, 'r') as xyz_file:
        lines = xyz_file.readlines()

    if len(lines) < 2:
        raise ValueError(f"XYZ file {xyz_filename} is incomplete.")

    # Extract charge and multiplicity from the second line
    comment_line = lines[1]
    charge_match = re.search(r'Charge:\s*(-?\d+)', comment_line)
    multiplicity_match = re.search(r'Multiplicity:\s*(\d+)', comment_line)

    if charge_match:
        charge = int(charge_match.group(1))
    else:
        raise ValueError(f"Charge not found in XYZ comment line: {comment_line}")

    # Parse atom information
    atom_lines = lines[2:]  # Skip the first two lines (atom count and comment)
    atoms = []
    lanthanide_indices = []

    for index, line in enumerate(atom_lines):
        parts = line.strip().split()
        if len(parts) != 4:
            continue  # Skip invalid lines
        element = parts[0]
        atoms.append({'element': element, 'index': index})

        if element in lanthanide_data.keys():
            lanthanide_indices.append(index)

    if len(lanthanide_indices) == 0:
        raise ValueError("No open-shell lanthanide found in the XYZ file.")

    if len(lanthanide_indices) > 1:
        raise ValueError("More than one open-shell lanthanide found. Only one is allowed.")

    # Get the lanthanide element and its index
    lanthanide_index = lanthanide_indices[0]
    lanthanide_element = atoms[lanthanide_index]['element']

    if multiplicity_match:
        multiplicity = int(multiplicity_match.group(1))
    else:
        multiplicity = int(2*lanthanide_data[lanthanide_element]['S_list'][0] + 1)

    return charge, multiplicity, atoms, lanthanide_element, lanthanide_index

def build_basis_section(atoms, lanthanide_element, expbas=False):
    # Build the BASIS section
    basis_section = "%basis\n"
    if expbas:
        basis_section += f"  newGTO {lanthanide_element} \"SARC2-DKH-QZVP\" end\n"
    else:
        basis_section += f"  newGTO {lanthanide_element} \"SARC2-DKH-QZV\" end\n"

    # Track elements already added to the basis section
    basis_elements = []

    for atom in atoms:
        element = atom['element']
        atomic_number = element_to_atomic_number.get(element)
        if atomic_number is None:
            raise ValueError(f"Unknown element symbol: {element}")

        # Include basis for elements with atomic number > 36 (beyond Kr), excluding the lanthanide
        if atomic_number > 36 and element not in basis_elements and element != lanthanide_element:
            basis_section += f"  newGTO {element} \"SARC-DKH-TZVP\" end\n"
            basis_elements.append(element)

    basis_section += "  DecontractAuxJ true\n"
    basis_section += "  DecontractAuxC true\n"
    basis_section += "end\n"

    return basis_section

def generate_input_file_pbe_guess(cpus, max_memory, tmp_dir):
    # Function to generate the initial PBE input file for dof_0_disp_0_guess
    project_name = 'dof_0_disp_0_guess'
    input_filename = f'{project_name}.inp' 
    input_file = os.path.join(tmp_dir, input_filename)
    xyz_filename = os.path.join('dof_0_disp_0.xyz')

    charge, multiplicity, atoms, lanthanide_element, lanthanide_index = parse_xyz_file(xyz_filename)

    nel = lanthanide_data[lanthanide_element]['nel']
    norb = 7  # Always 7 f orbitals

    # Build the AVAS section
    shell_list = ', '.join(['4'] * norb)
    l_list = ', '.join(['3'] * norb)
    m_l_list = '0,1,-1,2,-2,3,-3'
    center_list = ', '.join([str(lanthanide_index)] * norb)

    # Build the BASIS section
    basis_section = build_basis_section(atoms, lanthanide_element)

    # Build the ORCA input content for the initial PBE calculation
    input_content = f"""! PBE DKH2 DKH-def2-SVP AutoAux RIJCOSX NormalSCF NoFrozenCore SlowConv UNO

{basis_section}

%maxcore {int(max_memory // cpus)}

%pal
 nprocs {cpus}
end

%scf
 guess HCore
 AutoTrahIter 400
 maxiter 2000
 AVAS
   system
    nel {nel}
    norb {norb}
    shell {shell_list}
    l {l_list}
    m_l {m_l_list}
    center {center_list}
    end
  end
end

* xyzfile {charge} {multiplicity} ../{xyz_filename}
"""

    with open(input_file, 'w') as f:
        f.write(input_content)

    return input_filename

def generate_input_file_casscf(project_name, cpus, max_memory, use_nevpt2, moinp_file, tmp_dir, expbas, expbas_guess=False, run_expbas=False):
    # Function to generate the CASSCF input file using the specified %moinp file
    input_filename = f'{project_name}.inp' 
    input_file = os.path.join(tmp_dir, input_filename)
    xyz_filename = os.path.join(f'{project_name}.xyz')

    charge, multiplicity, atoms, lanthanide_element, lanthanide_index = parse_xyz_file(xyz_filename)

    nel = lanthanide_data[lanthanide_element]['nel']
    S_list = lanthanide_data[lanthanide_element]['S_list']
    NDoubGtensor = lanthanide_data[lanthanide_element]['NDoubGtensor']

    norb = 7  # Always 7 f orbitals

    # Calculate multiplicities and nroots
    mult_list = []
    nroots_list = []
    for S in S_list:
        multiplicity_casscf = int(2*S + 1)
        nroots = weyl_formula(norb, S, nel)
        mult_list.append(str(multiplicity_casscf))
        nroots_list.append(str(nroots))

    mult_str = ','.join(mult_list)
    nroots_str = ','.join(nroots_list)

    # Build the AVAS section
    shell_list = ', '.join(['4'] * norb)
    l_list = ', '.join(['3'] * norb)
    m_l_list = '0,1,-1,2,-2,3,-3'
    center_list = ', '.join([str(lanthanide_index)] * norb)

    # Build the BASIS section
    basis_section = build_basis_section(atoms, lanthanide_element, expbas)

    # Handle NEVPT2 option
    if use_nevpt2:
        ptmethod_line = "\n  ptmethod SC_NEVPT2"
    else:
        ptmethod_line = ""
    
    rel = "" if expbas_guess else f"""
 rel
  printlevel 3
  dosoc true
  gtensor true
  NDoubGtensor {NDoubGtensor}
 end"""
    
    avas = "" if moinp_file.endswith(".gbw") and not run_expbas else f"""AVAS
   system
    nel {nel}
    norb {norb}
    shell {shell_list}
    l {l_list}
    m_l {m_l_list}
    center {center_list}
    end
  end"""

    # Build the ORCA input content for the CASSCF calculation
    input_content = f"""! CASSCF DKH2 {"ma-" if expbas else ""}DKH-def2-SVP AutoAux RIJCOSX NoFrozenCore VerySlowConv {"StrongSCF" if expbas_guess else "VeryTightSCF"}

%maxcore {int(max_memory // cpus)}

%pal
  nprocs {cpus}
end

%scf
  AutoTrahIter 150
  maxiter 3000
  {avas}
end

%rel
  method dkh
  picturechange 2
end

%casscf
 nel  {nel}
 norb {norb}
 mult {mult_str}
 nroots {nroots_str}
 maxiter 3000
{rel}
 
 DoCD false
 DoDipoleLength false
 DoDipoleVelocity false
 DoHigherMoments false
 DecomposeFoscLength false
 DecomposeFoscVelocity false
 DoFullSemiclassical false
{ptmethod_line}

end

{basis_section}

! MOREAD
%moinp "{moinp_file}"

* xyzfile {charge} {multiplicity} ../{xyz_filename}
"""

    with open(input_file, 'w') as f:
        f.write(input_content)

    return input_filename

def run_orca(input_file, output_file, orca_path, tmp_dir):
    command = [orca_path, input_file]

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'  # ORCA uses MPI parallelization

    try:
        with open(output_file, 'w') as outfile:
            process = subprocess.Popen(command, stdout=outfile, stderr=subprocess.STDOUT, env=env, cwd=tmp_dir)
            process.communicate()
            exit_code = process.wait()
            time.sleep(1)
            if exit_code != 0:
                print(f"ORCA exited with code {exit_code} for {input_file}")
                raise KeyboardInterrupt(f"ORCA error with exit code {exit_code}")
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in run_orca for {input_file}. Terminating process.")
        process.terminate()
        process.communicate()
        process.wait()
        time.sleep(1)
        raise

def cleanup_files(tmp_dir, project_name):
    # Remove the temporary directory and all its contents
    if os.path.exists(tmp_dir):
        try:
            output_file = os.path.join(tmp_dir, f'{project_name}.out')
            if os.path.exists(output_file):
                # print(f"Moving {project_name}.out file back to the main directory for inspection. If it is not complete remove or rename it before restarting the calculation!")
                shutil.move(output_file, f'{project_name}.out')
            # print(f"Clearing the temporary directory {tmp_dir}")
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Error moving .out file or deleting temporary directory {tmp_dir}: {e}")
            raise

def process_initial_pbe_guess(cpus, max_memory, orca_path):
    project_name = 'dof_0_disp_0_guess'
    qro_file = f'{project_name}.qro'
    gbw_file = f'dof_0_disp_0.gbw'
    if os.path.exists(qro_file) or os.path.exists(gbw_file):
        print(f"Skipping initial PBE calculation for {project_name}, {".gbw" if os.path.exists(gbw_file) else ".qro"} file already exists.")
        return
    
    tmp_dir = f'tmp_{project_name}'
    os.makedirs(tmp_dir, exist_ok=True)

    raise_on_exit = False

    try:
        input_file = generate_input_file_pbe_guess(cpus, max_memory, tmp_dir)
        output_file = os.path.join(tmp_dir, f'{project_name}.out')

        run_orca(input_file, output_file, orca_path, tmp_dir)
    except Exception:
        raise
    finally:
        # Move the .qro file back to the main directory
        qro_file = os.path.join(tmp_dir, f'{project_name}.qro')
        if os.path.exists(qro_file):
            print("Moving .qro file back to the main directory...")
            shutil.move(qro_file, f'{project_name}.qro')
        else:
            raise_on_exit = True
        # Clean up the temporary directory and move the .out file back to the main directory
        cleanup_files(tmp_dir, project_name)
        if raise_on_exit:
            raise ValueError(f"Calculation for {project_name} failed to generate the .qro file. Cannot proceed.")

def process_initial_casscf(cpus, max_memory, orca_path, use_nevpt2, start_from_different_lanthanide, expbas):
    project_name = 'dof_0_disp_0'
    gbw_file = f'{project_name}.gbw'
    if os.path.exists(gbw_file) and not start_from_different_lanthanide:
        print(f"Skipping initial CASSCF calculation for {project_name}, .gbw file already exists.")
        return

    tmp_dir = f'tmp_{project_name}'
    os.makedirs(tmp_dir, exist_ok=True)

    raise_on_exit = False

    try:
        start_extension = ".gbw" if start_from_different_lanthanide else "_guess.qro"
        start_file = f'dof_0_disp_0{start_extension}'
        if not os.path.exists(start_file):
            raise ValueError(f"Required {start_extension} file {start_file} not found. Ensure the initial {".gbw file (from different lanthanide calculation) is in the directory" if start_from_different_lanthanide else "PBE calculation has completed"}.")
        # Copy the .qro file into the temporary directory
        shutil.copy(start_file, tmp_dir)

        input_file = generate_input_file_casscf(project_name, cpus, max_memory, use_nevpt2, start_file, tmp_dir, False, expbas)
        output_file = os.path.join(tmp_dir, f'{project_name}.out')

        run_orca(input_file, output_file, orca_path, tmp_dir)
    except Exception:
        raise
    finally:
        # Move .gbw file back to the main directory
        gbw_file = os.path.join(tmp_dir, f'{project_name}.gbw')
        if os.path.exists(gbw_file):
            print("Moving .gbw file back to the main directory...")
            shutil.move(gbw_file, f'{project_name}.gbw')
        else:
            raise_on_exit = True
        # Clean up the temporary directory and move the .out file back to the main directory
        cleanup_files(tmp_dir, project_name)
        if raise_on_exit:
            raise ValueError(f"Calculation for {project_name} failed to generate the .gbw file. Cannot proceed.")

def process_expbas_casscf(cpus, max_memory, orca_path, use_nevpt2):
    project_name = 'dof_0_disp_0'
    gbw_file = f'{project_name}.gbw'
    if not os.path.exists(gbw_file):
        raise ValueError(f"The .gbw file {gbw_file} required for the basis expansion not found. Ensure the CASSCF calculation for dof_0_disp_0 has completed or you correctly provided your own .gbw file.")

    tmp_dir = f'tmp_{project_name}'
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        gbw_file_tmp = os.path.join(tmp_dir, 'dof_0_disp_0_guess.gbw')
        shutil.copy(gbw_file, gbw_file_tmp)

        input_file = generate_input_file_casscf(project_name, cpus, max_memory, use_nevpt2, 'dof_0_disp_0_guess.gbw', tmp_dir, True, run_expbas=True)
        output_file = os.path.join(tmp_dir, f'{project_name}.out')

        run_orca(input_file, output_file, orca_path, tmp_dir)

        # Move .gbw file back to the main directory
        gbw_file = os.path.join(tmp_dir, f'{project_name}.gbw')
        if os.path.exists(gbw_file):
            shutil.move(gbw_file, f'{project_name}.gbw')
        else:
            raise ValueError(f"Expand basis calculation for {project_name} failed to generate the .gbw file. Cannot proceed.")
    except Exception:
        raise
    finally:
        # Clean up the temporary directory and move the .out file back to the main directory
        cleanup_files(tmp_dir, project_name)

def process_dof_disp(args_tuple):

    dof_disp, cpus, max_memory, orca_path, use_nevpt2, expbas = args_tuple
    dof_number, disp_number = dof_disp
    project_name = f'dof_{dof_number}_disp_{disp_number}'
    tmp_dir = f'tmp_{project_name}'
    os.makedirs(tmp_dir, exist_ok=True)

    def handle_sigterm(signum, frame):
        print(f"KeyboardInterrupt or termination signal caught in {project_name}...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        gbw_file = f'dof_0_disp_0.gbw'
        if not os.path.exists(gbw_file):
            raise ValueError(f"Required .gbw file {gbw_file} not found. Ensure the CASSCF calculation for dof_0_disp_0 has completed or you correctly provided your own .gbw file.")
        # Copy the .gbw file into the temporary directory
        shutil.copy(gbw_file, tmp_dir)

        input_file = generate_input_file_casscf(project_name, cpus, max_memory, use_nevpt2, gbw_file, tmp_dir, expbas)
        output_file = os.path.join(tmp_dir, f'{project_name}.out')

        run_orca(input_file, output_file, orca_path, tmp_dir)
    except Exception:
        raise
    finally:
        # Clean up the temporary directory and move the .out file back to the main directory
        cleanup_files(tmp_dir, project_name)

def main():
    parser = argparse.ArgumentParser(description='Run ORCA calculations.')
    parser.add_argument('--cpus', type=int, default=1, help='Total number of CPUs to use.')
    parser.add_argument('--processes', type=int, default=1, help='Number of concurrent processes.')
    parser.add_argument('--orca_path', type=str, required=True, help='Path to the ORCA executable.')
    parser.add_argument('--max_memory', type=float, required=True, help='Total maximum memory in MB for the calculation.')
    parser.add_argument('--use_nevpt2', action='store_true', help='Use NEVPT2 with ptmethod SC_NEVPT2 in the input file.')
    parser.add_argument('--expbas', action='store_true', help='Expand basis to SARC2-DKH-QZVP for the lanthanide ion and ma-DKH-def2-SVP for others up to Kr.')
    parser.add_argument('--start_from_different_lanthanide', action='store_true', help='If a .gbw file is present in the directory the script recalculates the guess assuming that it corresponds to a different lanthanide ion from a previous calculation. The initial .gbw file is being overwritten! (This can also be simply used to recalculate the guess)')

    args = parser.parse_args()

    total_cpus = args.cpus
    processes = args.processes

    if (args.start_from_different_lanthanide):
        print("Skipping the initial PBE calculation for dof_0_disp_0_guess as restart from different lanthanide's .gbw was requested.")
    else:
        # First, process the initial PBE calculation for dof_0_disp_0_guess using all CPUs
        print("Starting initial PBE calculation for dof_0_disp_0_guess...")
        process_initial_pbe_guess(total_cpus, args.max_memory, args.orca_path)

    # Then, process the CASSCF calculation for dof_0_disp_0 using all CPUs
    print("Starting CASSCF calculation for dof_0_disp_0...")
    process_initial_casscf(total_cpus, args.max_memory, args.orca_path, args.use_nevpt2, args.start_from_different_lanthanide, args.expbas)

    if args.expbas:
        print("Starting expand basis CASSCF calculation for dof_0_disp_0...")
        process_expbas_casscf(total_cpus, args.max_memory, args.orca_path, args.use_nevpt2)

    # Prepare the list of dof_disp combinations to process in parallel (excluding dof_0_disp_0)
    xyz_files = glob.glob('dof_*_disp_*.xyz')
    pattern = re.compile(r'dof_(-?\d+)_disp_(-?\d+)\.xyz')

    dof_disp_list = []

    for filename in xyz_files:
        match = pattern.match(filename)
        if match:
            dof_number = int(match.group(1))
            disp_number = int(match.group(2))
            if dof_number == 0 and disp_number == 0:
                continue  # Already processed
            project_name = f'dof_{dof_number}_disp_{disp_number}'
            output_file = f'{project_name}.out'
            if os.path.exists(output_file):
                print(f"Skipping {project_name}, calculation already completed.")
                continue
            else:
                dof_disp_list.append((dof_number, disp_number))

    # Now process the rest in parallel
    if dof_disp_list:
        print('Processing calculations in parallel...')
        try:
            cpus_per_process = total_cpus // processes
            pool_args = [((dof_disp_list[i]), cpus_per_process, args.max_memory, args.orca_path, args.use_nevpt2, args.expbas) for i in range(len(dof_disp_list))]

            with Pool(processes=processes) as pool:
                for _ in tqdm(pool.imap_unordered(process_dof_disp, pool_args), total=len(pool_args)):
                    pass
        except KeyboardInterrupt:
            print("\nTerminating pool...")
            pool.terminate()
            pool.join()
            raise
    else:
        print('No dof_disp combinations to process.')

if __name__ == '__main__':
    main()
