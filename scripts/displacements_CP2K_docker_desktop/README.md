# README for CP2K Displacements Automation Script

This script automates CP2K calculations using Docker containers of displacement directories created by SlothPy. It reads `.xyz` files in the `dof_*_disp_*` format, generates CP2K input files, and runs calculations in parallel to obtain forces and dipole moments.

**Prerequisites:**

- **Docker:** Ensure Docker is installed and properly configured on your system.

- **CP2K Docker Image:**

  - **Build from Source:**
    - You need to have the CP2K Docker image available. You can build the image yourself using the Dockerfiles provided at [https://github.com/cp2k/cp2k-containers](https://github.com/cp2k/cp2k-containers).
    - **Steps to Build:**
      ```bash
      # Clone the repository
      git clone https://github.com/cp2k/cp2k-containers.git

      # Navigate to the desired version directory
      cd cp2k-containers/docker

      # Build the Docker image
      docker build -f ./2024.2_openmpi_native_psmp.Dockerfile -t cp2k/cp2k:2024.2_openmpi_native_psmp .
      ```
    - Replace `2024.2_openmpi_native_psmp` with the version you wish to build either from the prepared files (commands to install is in the header Usage:) or create your own with the python script generate_docker_files.py.
  - **Use Precompiled Image:**
    - Alternatively, if a precompiled CP2K Docker image is available, you can use that.
    - Pull the image from Docker Hub (if available):
      ```bash
      docker pull cp2k/cp2k:2024.3_openmpi_generic_psmp
      ```
    - Replace `2024.3_openmpi_generic_psmp` with the tag of the image you wish to use.
  - **Docker Image Path in Script:**
    - In the script, the Docker image is specified via the `--version` argument.
    - By default, it is assumed that the image name starts with `cp2k/cp2k:`.
    - If your Docker image has a different name or tag, make sure to modify image=f'cp2k/cp2k:{cp2k_version}' line in the script.
    - **Example:**
      ```bash
      # If your image is named differently, e.g., my_cp2k_image:latest
      python run_displacements_CP2K.py --cpus 64 --processes 2 --threads 2 --version my_cp2k_image:latest
      ```

- **Docker SDK for Python:** Install the Docker SDK by running:

  ```bash
  pip install docker
  ```

- **Python Modules:** The script uses Python modules such as `tqdm` and `docker`. Ensure these modules are available in your Python environment.

**Important Instructions:**

- **Customize Input Parameters:**
  - Modify the input parameters in the `generate_input_file` function within the script to match your computational needs.
  - Adjust parameters such as:
    - `BASIS_SET_FILE_NAME`: e.g., `BASIS_MOLOPT_UZH`
    - `POTENTIAL_FILE_NAME`: e.g., `POTENTIAL_UZH`
    - `CHARGE`, `MULTIPLICITY`, `UKS`
    - `&MGRID` settings: `CUTOFF`, `NGRIDS`, `REL_CUTOFF`
    - `&QS` settings: `METHOD`, `EXTRAPOLATION_ORDER`
    - `PREFERRED_DIAG_LIBRARY`, etc.
  - You can replace or add parameters specific to your system.

- **Provide Necessary `.inc` Files:**
  - Ensure all required `.inc` files (e.g., `scf.inc`, `XC.inc`, `subsys.inc`) are present and contain settings tuned for your system.
  - Place these files in the working directory or adjust the script to point to their locations.
  - You can optionally include the CP2K -RESTART.wfn file as the initial guess for the relaxed geometry and pass its name using --wfn_start argument.

- **Prepare Your Working Directory:**
  - Include all your `dof_*_disp_*.xyz` files in the directory.
  - Ensure `dof_0_disp_0.xyz` is present, as it is required for the initial wavefunction.

- **Adjust Resource Settings:**
  - When running the script, specify the appropriate number of CPUs, processes, threads, and the CP2K Docker image version using command-line arguments.
  - **Docker Image Path:**
    - If your CP2K Docker image has a different name or is located in a different repository, adjust the `--version` argument to match the full image name.
    - **Example:**
      ```bash
      python run_displacements_CP2K.py --cpus 64 --processes 2 --threads 2 --version my_cp2k_image:latest
      ```

**Example Usage:**

```bash
python run_displacements_CP2K.py --cpus 64 --processes 2 --threads 2 --version 2024.3_openmpi_native_psmp --wfn_start YCo_relaxed_supercell-RESTART.wfn
```

**Note:** Customize the script and input files according to your computational requirements to ensure accurate and efficient simulations.

**Additionally:** Directory contains an example of an input file for unit cell/supercell optimization example_opt.inp (with XYZ file YCo.xyz) that can be used to obtain relaxed geometries for finite displacement force calculations - remember to adjust cell parameters in subsys.inc.