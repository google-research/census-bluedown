# Census BlueDown: Differentially Private Block Hierarchical Aggregation

This repository contains the reference Python implementation for the _BlueDown_
algorithm applied for post-processing the
[Noisy Measurement File](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/00-2020-Redistricting-Noisy-Measurement-File/Noisy-Measurement-File-2020-Census-Redistricting-Data.pdf)
for the U.S. Census P.L. 94-171 Redistricting Data.

For technical details on the algorithm, privacy guarantees, and performance,
please refer to our [paper](https://arxiv.org/abs/2603.10099).

**Note:** This is not an officially supported Google product. This project is
not eligible for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

## Installation

Clone the repository and install the dependencies listed in `requirements.txt`:

```bash
git clone https://github.com/google-research/census-bluedown.git
cd census-bluedown
pip install -r requirements.txt
```

## Obtaining Data

Recreating the experiments done in the aforementioned paper requires downloading
two files:

* [Noisy Measurement File (NMF)](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/00-2020-Redistricting-Noisy-Measurement-File/Noisy-Measurement-File-2020-Census-Redistricting-Data.pdf).
For example, these could be downloaded from
[Harvard Dataverse](https://doi.org/10.7910/DVN/5LAVKV). For using this
codebase, we require that `2020-pl94-nmf-parquets.zip` is downloaded
and extracted at the location `data/2020-pl94-nmf-parquets`.
* [Privacy-Protected Microdata File (PPMF)](https://www.census.gov/data/tables/2024/dec/2020-census-ppmf.html).
Specifically, for using this codebase, we require that `2020_ppmf_bystate.zip`
available at
[this link](https://www2.census.gov/programs-surveys/decennial/2020/data/privacy-protected-microdata-file),
is saved at the location `data/2020_ppmf_bystate.zip`. **Note:** This
should remain as a ZIP file and should not be extracted.

## Gurobi Solver License Setup

The non-linear optimization steps in `nonlinear_solver.py` uses the
[Gurobi](https://www.gurobi.com/) solver engine via Google OR-Tools by default
(`constants.SOLVER = SolverType.GUROBI`).

### Providing a Gurobi License
To run Gurobi locally, you must provide a valid Gurobi license file
(`gurobi.lic`):

1. **Obtain a License:**
   - Academic users can request a free academic license at
     [gurobi.com/academics](https://www.gurobi.com/academics).
   - Commercial users require a valid commercial or evaluation license key from
     [Gurobi](https://www.gurobi.com).
2. **Install the License:**
   - Run the license retriever tool provided by Gurobi:

     ```bash
     grbgetkey <YOUR-LICENSE-KEY>
     ```
   - By default, Gurobi saves `gurobi.lic` to your home directory
     (`~/gurobi.lic`).
3. **Set the License Environment Variable (Optional):**
   - If your license file is located in a custom directory, set
   `GRB_LICENSE_FILE`:

     ```bash
     export GRB_LICENSE_FILE=/path/to/gurobi.lic
     ```

**Note:** The authors have not tested the Gurobi license setup workflow
thoroughly, since they used Gurobi using a different mechanism. Please contact
the authors by commenting on this repository in case there are any issues in
this described workflow.

### Switching to Open-Source Solver (SCIP)
If you do not have a Gurobi license, you can switch to the open-source **SCIP**
solver by modifying `census_bluedown/constants.py`:

```python
SOLVER = SolverType.SCIP
```

This is intended for testing purposes only. Our experiments only use Gurobi,
and we have not evaluated the performance and error metrics when using SCIP.

## Usage

Set `PYTHONPATH=.` and define path constants and state lists in your bash
environment before running scripts:

```bash
export PYTHONPATH=.

PPMF_STATES=(
    '01' '02' '04' '05' '06' '08' '09' '10' '11' '12' '13' '15' '16' '17' '18'
    '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33'
    '34' '35' '36' '37' '38' '39' '40' '41' '42' '44' '45' '46' '47' '48' '49'
    '50' '51' '53' '54' '55' '56'
)

FOLDER_IDS_2020=(
    '001' '002' '004' '005' '006' '008' '009' '010' '011' '012' '013' '015'
    '016' '017' '018' '019' '020' '021' '022' '023' '024' '025' '026' '027'
    '028' '029' '030' '031' '032' '033' '034' '035' '036' '037' '038' '039'
    '040' '041' '042' '044' '045' '046' '047' '048' '049' '050' '051' '053'
    '054' '055' '056' '101' '102' '104' '106' '108' '109' '112' '113' '115'
    '116' '118' '119' '120' '122' '123' '125' '126' '127' '128' '130' '131'
    '132' '135' '136' '137' '138' '140' '141' '145' '146' '147' '148' '149'
    '151' '153' '155' '156'
)

PPMF_ZIP_PATH="data/2020_ppmf_bystate.zip"
NMF_DIR="data/2020-pl94-nmf-parquets/US_Person_PL_PROD/"
SUBTREE_TOTALS_DIR="data/state-totals/"
SYNTHETIC_GROUND_TRUTH="output/synthetic-ground-truth/"
SYNTHETIC_NMF="output/synthetic-nmf/"
SYNTHETIC_PROCESSED="output/synthetic-processed/"
```

### 1. Synthetic Data Generation (`generate_synthetic_data.py`)

Our experiments use the Privacy-Protected Microdata File (PPMF) provided by the
Census Bureau as the "ground truth", and generates synthetic versions of the
Noisy Measurement File (NMF) by adding discrete Gaussian noise of the
appropriate scale for each query.

The following commands prepare the ground truth dataset, and synthetic datasets.

**Note:** While we provide the commands to run the entire algorithms on a single
machine, in practice, this will take a prohibitively long time. We parallelized
this by running the command presented inside `for` loops (each loop step for a
separate state) on independent machines. We do not provide the code for this
parallelization infrastructure, since we used a custom parallelization
infrastructure that cannot be open-sourced.

* **Pass 1: Ground Truth Generation (`--pass=ground_truth`)**

    * **Ground Truth for all States**

      The command below reads microdata Census ZIP archives (`--zip_path`) and
      NMF structure (`--nmf_dir`) to generate ground truth tables for each
      state:

      ```bash
      for state in "${PPMF_STATES[@]}"; do
        python3 census_bluedown/generate_synthetic_data.py \
          --pass=ground_truth \
          --state="${state}" \
          --nmf_dir="${NMF_DIR}" \
          --out_dir="${SYNTHETIC_GROUND_TRUTH}" \
          --zip_path="${PPMF_ZIP_PATH}"
      done
      ```

    * **Ground Truth for US (Root)**

      Generates ground truth tables for the US national level (root):

      ```bash
      python3 census_bluedown/generate_synthetic_data.py \
        --pass=ground_truth \
        --nmf_dir="${NMF_DIR}" \
        --out_dir="${SYNTHETIC_GROUND_TRUTH}" \
        --zip_path="${PPMF_ZIP_PATH}"
      ```

* **Pass 2: Synthetic NMF Generation (`--pass=nmf`)**

    * **Synthetic NMF for all States**

      Generates synthetic noisy measurement files (NMF) from ground-truth
      tables (`--ground_truth_dir`) for each state subtree:

      ```bash
      for state in "${FOLDER_IDS_2020[@]}"; do
        python3 census_bluedown/generate_synthetic_data.py \
          --pass=nmf \
          --state="${state}" \
          --ground_truth_dir="${SYNTHETIC_GROUND_TRUTH}" \
          --out_dir="${SYNTHETIC_NMF}" \
          --seed=1234
      done
      ```

    * **Synthetic NMF for US (Root)**

      Generates synthetic NMF for the US national level (root):

      ```bash
      python3 census_bluedown/generate_synthetic_data.py \
        --pass=nmf \
        --ground_truth_dir="${SYNTHETIC_GROUND_TRUTH}" \
        --out_dir="${SYNTHETIC_NMF}" \
        --seed=1234
      ```

### 2. Postprocessing Pipelines (BlueDown and TopDown)

#### 2.1. BlueDown Algorithm

Postprocessing runs in **2 stages** (each with 2 sub-steps).

* **Pass 1: Bottom-Up Pass (`--pass=bottom_up`)**

    * **Bottom-Up Pass for all States**

      Initializes input estimates and aggregates variances bottom-up for
      individual state subtrees:

      ```bash
      for state in "${FOLDER_IDS_2020[@]}"; do
        python3 census_bluedown/run_pass.py \
          --pass=bottom_up \
          --state="${state}" \
          --nmf_dir="${SYNTHETIC_NMF}" \
          --out_dir="${SYNTHETIC_PROCESSED}"
      done
      ```

    * **Bottom-Up Pass for US (Root)**

      Combines state-level totals to construct national (US root) bottom-up
      totals:

      ```bash
      python3 census_bluedown/run_pass.py \
        --pass=bottom_up \
        --nmf_dir="${SYNTHETIC_NMF}" \
        --out_dir="${SYNTHETIC_PROCESSED}" \
        --subtree_totals_dir="${SUBTREE_TOTALS_DIR}"
      ```

* **Pass 2: Top-Down Pass (`--pass=top_down`)**

    * **Top-Down Pass for US (Root)**

      Runs non-linear optimization at the root level to establish consistent
      national upper bounds:

      ```bash
      python3 census_bluedown/run_pass.py \
        --pass=top_down \
        --in_dir="${SYNTHETIC_PROCESSED}" \
        --out_dir="${SYNTHETIC_PROCESSED}" \
        --subtree_totals_dir="${SUBTREE_TOTALS_DIR}"
      ```

    * **Top-Down Pass for all States**

      Runs top-down constrained optimization down to individual census blocks
      using root bounds:

      ```bash
      for state in "${FOLDER_IDS_2020[@]}"; do
        python3 census_bluedown/run_pass.py \
          --pass=top_down \
          --state="${state}" \
          --in_dir="${SYNTHETIC_PROCESSED}" \
          --out_dir="${SYNTHETIC_PROCESSED}"
      done
      ```

#### 2.2. Baseline Census Top-Down Algorithm (`--pass=baseline_census_topdown`)

The commands below run our implementation of the Census Top-Down algorithm of
[Abowd et al. (2022)](https://www2.census.gov/adrm/CED/Papers/CY22/2022-002-AbowdAshmeadCumingMenonGarfinkelEtal.pdf).

* **Top-Down Pass for US (Root)**

  ```bash
  python3 census_bluedown/run_pass.py \
    --pass=baseline_census_topdown \
    --nmf_dir="${SYNTHETIC_NMF}" \
    --out_dir="${SYNTHETIC_PROCESSED}" \
    --subtree_totals_dir="${SUBTREE_TOTALS_DIR}"
  ```

* **Top-Down Pass for all States**

  ```bash
  for state in "${FOLDER_IDS_2020[@]}"; do
    python3 census_bluedown/run_pass.py \
      --pass=baseline_census_topdown \
      --state="${state}" \
      --nmf_dir="${SYNTHETIC_NMF}" \
      --out_dir="${SYNTHETIC_PROCESSED}"
  done
  ```

### 3. Error Evaluation

Computes error metrics comparing postprocessed outputs with ground-truth
tables:

* **Compute Errors for each State**

  ```bash
  for pass in compute_errors compute_alternate_errors; do
    for state in "${FOLDER_IDS_2020[@]}"; do
      python3 census_bluedown/run_pass.py \
        --pass="${pass}" \
        --state="${state}" \
        --in_dir="${SYNTHETIC_GROUND_TRUTH}" \
        --out_dir="${SYNTHETIC_PROCESSED}"
    done
  done
  ```

* **Aggregating Errors across States**

  Aggregates state-level error metrics into national summary reports:

  ```bash
  for pass in compute_errors compute_alternate_errors; do
    python3 census_bluedown/run_pass.py \
      --pass="${pass}" \
      --out_dir="${SYNTHETIC_PROCESSED}"
  done
  ```

## License

Licensed under the [Apache 2.0 License](LICENSE).

## Paper & Citation

If you use this code or algorithm in your research, or just want to refer to it,
please cite our paper:

<!-- mdlint off(SNIPPET_INVALID_LANGUAGE) -->
```bibtex
@misc{ghazi2026bluedown,
  title={Denoising the US Census: Succinct Block Hierarchical Regression},
  author={Badih Ghazi and Pritish Kamath and Ravi Kumar and
          Pasin Manurangsi and Adam Sealfon},
  year={2026},
  eprint={2603.10099},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.10099},
}
```