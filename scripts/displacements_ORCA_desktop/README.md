# README for ORCA Displacements Automation Script

This script automates ORCA calculations of displacement directories created by SlothPy. It reads `.xyz` files in the `dof_*_disp_*` format, generates ORCA input files, and runs calculations in parallel to obtain properties such as SOC Hamiltonian matrix, angular, spin, and dipole moments for open-shell lanthanide complexes.

**Features:**

- Automates initial PBE and CASSCF calculations required for open-shell lanthanide systems.
- Manages computational resources by adjusting the number of CPUs and processes.
- Cleans up unnecessary files after each calculation to save disk space.
- Provides an option to include NEVPT2 calculations.
- Supports parallel processing to expedite calculations.
- **Allows users to provide their own starting `.qro` or `.gbw` files to skip initial calculations.**

---

## Prerequisites

- **ORCA Quantum Chemistry Program:**
  - Ensure ORCA is installed on your system.
  - The path to the ORCA executable is required when running the script.

- **Python 3 Environment:**
  - The script is written in Python 3.
  - Install the following Python modules:
    ```bash
    pip install tqdm scipy
    ```
  - **Modules Used:**
    - `os`, `sys`, `argparse`, `glob`, `re`, `subprocess`, `multiprocessing`, `shutil`, `tqdm`, `scipy.special`

- **SlothPy Displacement Directories:**
  - The script operates on displacement directories with (`dof_*_disp_*`) files created by SlothPy.
  - Ensure all your `dof_*_disp_*.xyz` files are present in the working directory.
  - The file `dof_0_disp_0.xyz` must be present, as it is required for the initial calculations.

---

## Important Instructions

### Customize Input Parameters

- **Adjust Computational Parameters:**
  - Modify parameters such as `BASIS_SET` and `METHOD`, and  directly in the script if necessary.
  - The script uses predefined settings suitable for open-shell lanthanide systems, but you can adjust them according to your needs.

- **Lanthanide Data:**
  - The script includes a `lanthanide_data` dictionary containing necessary data for each open-shell lanthanide element.
  - Ensure that your system's lanthanide is included and that the data (number of electrons `nel`, spin states `S_list`, etc.) are correct.

### Prepare Your Working Directory

- **Include Required Files:**
  - Place all `dof_*_disp_*.xyz` files in the working directory where the script is present.
  - Ensure that `dof_0_disp_0.xyz` is present.
  - **Provide Your Own Starting Files (Optional):**
  - You can provide your own `dof_0_disp_0_guess.qro` or `dof_0_disp_0.gbw` files if you wish to skip the initial PBE or CASSCF calculations.
  - If these files are present, the script will detect them and skip the corresponding calculations.
  - If you choose the `--start_from_different_lanthanide` option your .gbw file can correspond to the different lanthanide (presumably easier to converge such as Ce(III) or Yb(III)) and the script will recalculate the guess overwriting your initial .gbw file!
  - You can set the `--expbas` option to further expand the initial basis to SARC2-DKH-QZVP for the lanthanide ion and ma-DKH-def2-SVP for others up to Kr.

- **File Structure:**
  - The script expects files to be named in the format `dof_X_disp_Y.xyz`, where `X` and `Y` are integers.
  - Do not rename the generated files unless you also adjust the script accordingly.

### Resource Management

- **Specify Computational Resources:**
  - When running the script, specify the total number of CPUs, number of processes, and maximum memory using command-line arguments.
  - **Total CPUs (`--cpus`):** Total number of CPU cores available for calculations.
  - **Processes (`--processes`):** Number of parallel processes to run (for the displacement calculations).
  - **Maximum Memory (`--max_memory`):** Total memory available for calculations (in MB).
  - **NEVPT2 Option (`--use_nevpt2`):** Include this flag if you want to perform NEVPT2 calculations.

- **Resource Allocation:**
  - The initial calculations (`dof_0_disp_0_guess` and `dof_0_disp_0`) use all available CPUs.
  - Subsequent calculations distribute the CPUs evenly across the specified number of processes.
    - Each process uses `cpus_per_process = total_cpus // processes`.
    - Ensure that `total_cpus` is divisible by `processes` for optimal resource utilization.

### Cleanup and Disk Space Management

- **Automatic File Cleanup:**
  - After each displacement calculation (except for `dof_0_disp_0`), the script deletes unnecessary files to save disk space by removing the temporary directory.
  - Only the `.out` files are kept for analysis.

- **Preservation of Essential Files:**
  - The file `dof_0_disp_0.gbw` is preserved, as it is required for the initial calculations and as a starting point for subsequent calculations.

### Adjusting the Script

- **Custom Modifications:**
  - If you need to modify the computational methods, basis sets, or any other settings, you can edit the functions `generate_input_file_pbe_guess` and `generate_input_file_casscf` within the script.
  - Ensure that any changes are consistent throughout the script to avoid errors.

---

## Example Usage

Run the script from the command line, specifying the required arguments:

```bash
python run_displacements_ORCA.py --cpus 64 --processes 4 --orca_path /path/to/orca --max_memory 8000 --expbas
```

or when you provide .gbw starting file from the calculation corresponding to the different lanthanide and want to skip the basis expansion:

```bash
python run_displacements_ORCA.py --cpus 64 --processes 4 --orca_path /path/to/orca --max_memory 8000 --start_from_different_lanthanide
```

- **Arguments:**
  - `--cpus`: Total number of CPUs to use (e.g., `64`).
  - `--processes`: Number of concurrent processes for parallel calculations (e.g., `4`).
  - `--orca_path`: Full path to the ORCA executable (e.g., `/usr/local/orca/orca`).
  - `--max_memory`: Total maximum memory in MB (e.g., `8000` for 8 GB).
  - `--use_nevpt2`: Include this flag to perform NEVPT2 calculations.

**Example with NEVPT2 Enabled:**

```bash
python run_displacements_ORCA.py --cpus 64 --processes 4 --orca_path /usr/local/orca/orca --max_memory 8000 --use_nevpt2
```

---

## Script Overview

- **Initial Calculations:**
  1. **PBE Calculation (`dof_0_disp_0_guess`):**
     - Generates the `.qro` file required for subsequent CASSCF calculations.
     - Uses all available CPUs.
  2. **CASSCF Calculation (`dof_0_disp_0`):**
     - Starts from the `.qro` file generated in the previous step.
     - Produces the `.gbw` file used as the starting point for displacement calculations.
     - Uses all available CPUs.

- **Displacement Calculations:**
  - **Parallel CASSCF Calculations (`dof_*_disp_*`):**
    - Start from `dof_0_disp_0.gbw`.
    - Distributed across the specified number of processes.
    - Each process uses `cpus_per_process = total_cpus // processes`.

- **File Cleanup:**
  - After each displacement calculation, unnecessary files are deleted to conserve disk space.
  - Only the `.out` files and `dof_0_disp_0.gbw` are retained.

---

## Additional Notes

- **Error Handling:**
  - The script includes checks to ensure that required files are present before proceeding.
  - Informative error messages are provided if a calculation fails or if a required file is missing.

- **Dependencies:**
  - Ensure all dependencies are installed:
    ```bash
    pip install tqdm scipy
    ```
  - The script uses the `tqdm` module to display progress bars during parallel processing.

- **Adjusting for Non-Divisible CPU Counts:**
  - If `total_cpus` is not evenly divisible by `processes`, the script uses integer division for `cpus_per_process`.
  - You may need to adjust `total_cpus` or `processes` to ensure even distribution or handle the remaining CPUs as desired.

- **Atom Indexing:**
  - The script accounts for ORCA's 0-based atom indexing when specifying atoms in the input files.
  - Ensure that any modifications maintain this indexing to avoid errors.

- **XYZ File Format:**
  - The `.xyz` files must include charge and multiplicity in the comment line (second line) in the format:
    ```
    Charge: <charge> Multiplicity: <multiplicity>
    ```

- **License Compliance:**
  - Ensure you comply with ORCA's licensing terms when using the software.

---

## Customization and Support

- **Custom Systems:**
  - For systems not included in the `lanthanide_data` dictionary, you can add entries with the necessary data.
  - Ensure that all required parameters (`nel`, `S_list`, `NDoubGtensor`) are provided.

- **Support:**
  - For assistance with the script, you may need to consult with me mikolaj,zychowicz@uj.edu.pl.

---

## Example Workflow

1. **Prepare the Working Directory:**
   - Place all `dof_*_disp_*.xyz` files in the directory.
   - Ensure `dof_0_disp_0.xyz` is present.

2. **Adjust Script Settings (if necessary):**
   - Modify computational parameters in the script functions if needed.

3. **Run the Script:**
   - Execute the script with the desired command-line arguments.

4. **Monitor Progress:**
   - The script will display progress bars for the parallel calculations.
   - Check the output files (`.out`) for results and any error messages.

5. **Analyze Results:**
   - After the calculations are complete, analyze the `.out` files to extract the required data.

6. **Cleanup (Optional):**
   - If additional disk space management is needed, verify that unnecessary files have been deleted.
   - Manually remove any other files that are not needed.

---

## Disclaimer

- **Usage Responsibility:**
  - The user is responsible for ensuring that the script and calculations comply with all relevant licenses and regulations.
  - Always verify the results and consult the ORCA documentation for any advanced configurations.

---

**Note:** This script automates complex quantum chemistry calculations and should be used by individuals familiar with computational chemistry and the ORCA software.