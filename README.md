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

*   **Statewide Population Totals:** This file is already included in the
    repository under `data/state-totals/US/state_totals.parquet` and contains
    the **2020 U.S. Census** statewide resident population totals from the U.S.
    Census Bureau's
    [2020 Census Apportionment Data](https://www.census.gov/data/tables/2020/dec/2020-apportionment-data.html)
    ([Table 2](https://www2.census.gov/programs-surveys/decennial/2020/data/apportionment/apportionment-2020-table02.pdf))
    (for a different year or dataset, one would have to create a file in the
    same format with the appropriate totals). The included file has 51 rows
    (the 50 U.S. states and the District of Columbia) with three columns:

    *   `geocode` (`str`): 2-digit state FIPS code (`'01'` through `'56'`).
    *   `name` (`str`): State or territory name.
    *   `value` (`int64`): Invariant statewide total resident population.

    This file is required by the root-level (US national) passes
    (`--subtree_totals_dir="data/state-totals"`). In the NMF hierarchy, states
    with American Indian & Alaska Native (AIAN) areas are split into separate
    AIAN (`1XX`) and non-AIAN (`0XX`) subtrees; while each state's combined
    population is an exact invariant, individual subtree populations are not.
    The BlueDown passes and baseline Census TopDown passes read
    `state_totals.parquet` to enforce that the sum of the AIAN and non-AIAN
    subtree estimates for each state equals its exact statewide population
    `value` (and that the US root total equals the sum across all states).

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
NMF_DIR="data/2020-pl94-nmf-parquets/US_Person_PL_PROD"
SUBTREE_TOTALS_DIR="data/state-totals"
SYNTHETIC_GROUND_TRUTH="output/synthetic-ground-truth"
SYNTHETIC_NMF="output/synthetic-nmf-replicates"
SYNTHETIC_PROCESSED="output/synthetic-processed-replicates"
SEED="1234"
```

### 1. Synthetic Data Generation (`generate_synthetic_data.py`)

Our experiments use the Privacy-Protected Microdata File (PPMF) provided by the
Census Bureau as the "ground truth", and generates 10 independent synthetic
replicates (`0` through `9`) of the Noisy Measurement File (NMF) by adding
discrete Gaussian noise of the appropriate scale for each query.

The following commands prepare the ground truth dataset, and synthetic datasets.

**Note:** While we provide the commands to run the entire algorithms on a single
machine, in practice, this will take a prohibitively long time. We parallelized
this by running the command presented inside `for` loops (each loop step for a
separate state and replicate) on independent machines. We do not provide the
code for this parallelization infrastructure.

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

    For each replicate `i` in `{0..9}`, synthetic NMFs are generated using seed
    `${i}${SEED}` (which is concatenated with the 3-digit `state` subtree ID for
    state-level jobs) and saved under `output/synthetic-nmf-replicates/${i}`.

    * **Synthetic NMF for all States**

        Generates synthetic noisy measurement files (NMF) from ground-truth
        tables (`--ground_truth_dir`) for each state subtree and replicate:

        ```bash
        for replicate in {0..9}; do
          for state in "${FOLDER_IDS_2020[@]}"; do
            python3 census_bluedown/generate_synthetic_data.py \
              --pass=nmf \
              --state="${state}" \
              --ground_truth_dir="${SYNTHETIC_GROUND_TRUTH}" \
              --out_dir="${SYNTHETIC_NMF}/${replicate}" \
              --seed="${replicate}${SEED}"
          done
        done
        ```

    * **Synthetic NMF for US (Root)**

        Generates synthetic NMF for the US national level (root) for each
        replicate:

        ```bash
        for replicate in {0..9}; do
          python3 census_bluedown/generate_synthetic_data.py \
            --pass=nmf \
            --ground_truth_dir="${SYNTHETIC_GROUND_TRUTH}" \
            --out_dir="${SYNTHETIC_NMF}/${replicate}" \
            --seed="${replicate}${SEED}"
        done
        ```

### 2. Postprocessing Pipelines (BlueDown and TopDown)

For each replicate `i` in `{0..9}`, postprocessed outputs are saved under
`output/synthetic-processed-replicates/${i}`.

#### 2.1. BlueDown Algorithm

Postprocessing runs in **2 stages** (each with 2 sub-steps).

* **Pass 1: Bottom-Up Pass (`--pass=bottom_up`)**

    * **Bottom-Up Pass for all States**

        Initializes input estimates and aggregates variances bottom-up for
        individual state subtrees:

        ```bash
        for replicate in {0..9}; do
          for state in "${FOLDER_IDS_2020[@]}"; do
            python3 census_bluedown/run_pass.py \
              --pass=bottom_up \
              --state="${state}" \
              --nmf_dir="${SYNTHETIC_NMF}/${replicate}" \
              --out_dir="${SYNTHETIC_PROCESSED}/${replicate}"
          done
        done
        ```

    * **Bottom-Up Pass for US (Root)**

        Combines state-level totals to construct national (US root) bottom-up
        totals:

        ```bash
        for replicate in {0..9}; do
          python3 census_bluedown/run_pass.py \
            --pass=bottom_up \
            --nmf_dir="${SYNTHETIC_NMF}/${replicate}" \
            --out_dir="${SYNTHETIC_PROCESSED}/${replicate}" \
            --subtree_totals_dir="${SUBTREE_TOTALS_DIR}"
        done
        ```

* **Pass 2: Top-Down Pass (`--pass=top_down`)**

    * **Top-Down Pass for US (Root)**

        Runs non-linear optimization at the root level to establish consistent
        national upper bounds:

        ```bash
        for replicate in {0..9}; do
          python3 census_bluedown/run_pass.py \
            --pass=top_down \
            --in_dir="${SYNTHETIC_PROCESSED}/${replicate}" \
            --out_dir="${SYNTHETIC_PROCESSED}/${replicate}" \
            --subtree_totals_dir="${SUBTREE_TOTALS_DIR}"
        done
        ```

    * **Top-Down Pass for all States**

        Runs top-down constrained optimization down to individual census blocks
        using root bounds:

        ```bash
        for replicate in {0..9}; do
          for state in "${FOLDER_IDS_2020[@]}"; do
            python3 census_bluedown/run_pass.py \
              --pass=top_down \
              --state="${state}" \
              --in_dir="${SYNTHETIC_PROCESSED}/${replicate}" \
              --out_dir="${SYNTHETIC_PROCESSED}/${replicate}"
          done
        done
        ```

#### 2.2. Baseline Census Top-Down Algorithm (`--pass=baseline_census_topdown`)

The commands below run our implementation of the Census Top-Down algorithm of
[Abowd et al. (2022)](https://www2.census.gov/adrm/CED/Papers/CY22/2022-002-AbowdAshmeadCumingMenonGarfinkelEtal.pdf).

* **Top-Down Pass for US (Root)**

    ```bash
    for replicate in {0..9}; do
      python3 census_bluedown/run_pass.py \
        --pass=baseline_census_topdown \
        --nmf_dir="${SYNTHETIC_NMF}/${replicate}" \
        --out_dir="${SYNTHETIC_PROCESSED}/${replicate}" \
        --subtree_totals_dir="${SUBTREE_TOTALS_DIR}"
    done
    ```

* **Top-Down Pass for all States**

    ```bash
    for replicate in {0..9}; do
      for state in "${FOLDER_IDS_2020[@]}"; do
        python3 census_bluedown/run_pass.py \
          --pass=baseline_census_topdown \
          --state="${state}" \
          --nmf_dir="${SYNTHETIC_NMF}/${replicate}" \
          --out_dir="${SYNTHETIC_PROCESSED}/${replicate}"
      done
    done
    ```

### 3. Error Evaluation

Computes error metrics comparing postprocessed outputs with ground-truth tables
across geographic levels (`Block`, `Block_Group`, `Tract`, `County`, and
`State`):

<!-- mdformat off(prevent table wrapping) -->

| Pass | Description |
| :--- | :--- |
| `compute_errors` | Mean L1 error across geographic units for standard marginal and joint workload queries (`TOTAL`, `HOUSING_TYPE`, `VOTINGAGE`, `HISPANIC`, `CENRACE`, `HISPANICxCENRACE`, `AGExHISPxCENRACE` and `DETAILED`). |
| `compute_alternate_errors` | Elementwise mean absolute error across geographic units for a subset of the [2020 Census Production Disclosure Avoidance System Detailed Summary Metrics](https://www2.census.gov/programs-surveys/decennial/2020/data/demographic-and-housing-characteristics-file/2020-Census-Disclosure-Avoidance-System-Detailed-Summary-Metrics.xlsx) ([2020 Census technical documentation](https://www.census.gov/programs-surveys/decennial-census/technical-documentation/complete-technical-documents.2020.html)), covering race alone, race in combination, number of races, cross-tabulations with Hispanic origin and voting age, and grouped group-quarters counts. |
| `compute_errors_list` | Per-region L1 errors, signed population biases, and ground-truth totals for each geographic unit (used for distributional and population-binned analyses; per-state only). |

<!-- mdformat on -->

* **Compute Errors for each State**

    ```bash
    for replicate in {0..9}; do
      for pass in compute_errors compute_alternate_errors compute_errors_list; do
        for state in "${FOLDER_IDS_2020[@]}"; do
          python3 census_bluedown/run_pass.py \
            --pass="${pass}" \
            --state="${state}" \
            --in_dir="${SYNTHETIC_GROUND_TRUTH}" \
            --out_dir="${SYNTHETIC_PROCESSED}/${replicate}"
        done
      done
    done
    ```

* **Aggregating Errors across States**

    Aggregates state-level error metrics into national summary reports:

    ```bash
    for replicate in {0..9}; do
      for pass in compute_errors compute_alternate_errors; do
        python3 census_bluedown/run_pass.py \
          --pass="${pass}" \
          --out_dir="${SYNTHETIC_PROCESSED}/${replicate}"
      done
    done
    ```

### 4. Analysis Plots and Tables (`analysis/`)

The figures and LaTeX tables in the paper summarize error metrics over several
independent replicates of the synthetic NMF generation (with different `--seed`
values) and postprocessing pipeline. The analysis scripts expect replicate `i`
to be written under `<input_dir>/<i>/` with the layout produced by the commands
above (i.e. run sections 1–3 once per replicate with
`--out_dir="${SYNTHETIC_PROCESSED}/${i}"`).

*   **Generate Plots and Tables from Replicate Outputs**

    Reads the per-replicate error parquets and writes figures (`--formats=pdf`
    and/or `png`), LaTeX tables, and optionally tidy CSV exports of every
    plotted series (`--formats=csv`):

    ```bash
    PLOTS_DIR="output/plots"

    python3 census_bluedown/analysis/generate_plots.py \
      --input_dir="${SYNTHETIC_PROCESSED}" \
      --gt_parquet_path="${SYNTHETIC_GROUND_TRUTH}/US.parquet/DPQuery/ground_truth.parquet" \
      --output_dir="${PLOTS_DIR}" \
      --num_replicates=10 \
      --plots=all \
      --formats=pdf,png,csv
    ```

    `--plots` accepts any comma-separated subset of `boxplots`,
    `detailed_queries`, `bias`, `binned_error`, `ridgeline`, `tables`, and
    `--levels` any subset of `US`, `State`, `County`, `Tract`, `Block_Group`,
    `Block`.

*   **Regenerate Plots and Tables from Exported CSVs**

    Re-creates the same figures and tables purely from the CSV files exported
    above, without access to the replicate parquets:

    ```bash
    python3 census_bluedown/analysis/generate_plots_from_csv.py \
      --input_dir="${PLOTS_DIR}" \
      --output_dir="output/plots_from_csv" \
      --plots=all \
      --formats=pdf,png
    ```

## Repository Structure

All Python modules reside under `census_bluedown/`.

<!-- mdformat off(prevent table wrapping) -->

### Core Data Structures and Succinct Representations

| Module | Description | Unit Tests |
| :--- | :--- | :--- |
| `estimate.py` | Defines `Estimate`, `Covariance`, `DenseCovariance`, `Constraint`, and `DenseConstraint` for representing estimates, covariance matrices, and linear equality constraints. | `estimate_test.py` |
| `succinct.py` | Defines `SuccinctCovariance`, `SuccinctInputCovariance`, `SuccinctWorkload`, and `SuccinctConstraint` for representing covariance matrices, query workloads, and linear equality constraints succinctly in Kronecker product form. | `succinct_test.py` |
| `query_encoding.py` | Defines the encoding of Census query shapes (including the grouped `HHINSTLEVELS` query), projections of detailed histograms and optimization variables to a query shape, and utilities for compressing query shapes, variances, and values into succinct workloads and covariance structures. | `query_encoding_test.py` |
| `constants.py` | Defines constants and configuration settings used in the algorithm, including the solver type, column names for Pandas DataFrames, query shapes, and parameters for the tree hierarchy. | — |
| `io.py` | Defines `InputFormatIO` and `ProcessingFormatIO` for reading and writing input Noisy Measurement Files (NMF), constraints, and processed estimates in Parquet and pickle formats. | `io_test.py` |

### Main Algorithm

| Module | Description | Unit Tests |
| :--- | :--- | :--- |
| `regression.py` | Implements generalized least squares (GLS) query processing and minimum-variance linear combination of independent estimates using succinct covariance matrices. | `regression_test.py` |
| `constrain.py` | Applies linear equality constraints (sum constraints, first-feature zero constraints, and multifeature zero constraints) to block estimates via GLS projection. | `constrain_test.py` |
| `linear_postprocessing.py` | Implements the tree-structured Best Linear Unbiased Estimator (BLUE) passes: block-level GLS processing, bottom-up aggregation, and top-down exclusive/combined estimate propagation. | `linear_postprocessing_test.py` |
| `nonlinear_solver.py` | Formulates and solves the non-negative constrained L2 and rounder optimization problems at each node using MathOpt (Gurobi or SCIP) with succinct precision matrices. | `nonlinear_solver_test.py` |
| `nonlinear_postprocessing.py` | Orchestrates non-linear postprocessing across parent-child levels of a geographic subtree. | — |
| `root_optimization.py` | Formulates and solves the root-level (US national total) non-linear optimization and rounder passes. | — |
| `root.py` | Coordinates root-level bottom-up, top-down, and baseline Census TopDown passes across state subtrees. | — |
| `main_passes.py` | Coordinates the end-to-end input, bottom-up, optimization top-down, and baseline Census TopDown passes for individual state subtrees. | — |

### Synthetic Data Generation, Evaluation, and CLI Entry Points

| Module | Description | Unit Tests |
| :--- | :--- | :--- |
| `ground_truth.py` | Generates ground-truth histograms from Census Privacy-Protected Microdata Files (PPMF). | `ground_truth_test.py` |
| `nmf.py` | Generates synthetic Noisy Measurement Files (NMF) by adding discrete Gaussian noise to ground-truth histograms. | — |
| `generate_synthetic_data.py` | CLI entry point for generating ground-truth and synthetic NMF datasets. | — |
| `compute_errors.py` | Computes and aggregates L1 error metrics across geographic levels and queries comparing postprocessed estimates against ground truth. | `compute_errors_test.py` |
| `run_pass.py` | Main CLI entry point for running bottom-up, top-down, baseline Census TopDown, and error computation passes. | — |

### Analysis Plots and Tables (`analysis/`)

| Module | Description | Unit Tests |
| :--- | :--- | :--- |
| `common.py` | Geographic level and query constants, output layout, and shared helpers for extracting and normalizing replicate errors. | `common_test.py` |
| `style.py` | Color palettes and blending helpers. | `style_test.py` |
| `data_loader.py` | Loads per-replicate error parquets into tidy DataFrames (summary, detailed, population-binned, and per-unit error histograms). | `data_loader_test.py` |
| `csv_utils.py` | Exports those DataFrames to tidy CSVs and loads them back. | `csv_utils_test.py` |
| `boxplots.py` | Generates baseline-median-normalized error boxplots across geographic levels and detailed summary queries. | — |
| `population_bin_plots.py` | Generates population-binned signed bias and absolute L1 error plots. | `population_bin_plots_test.py` |
| `ridgeline_plots.py` | Generates stacked dual-series KDE ridgeline error distribution plots. | `ridgeline_plots_test.py` |
| `tables.py` | Generates LaTeX summary tables with paired t-test statistics. | `tables_test.py` |
| `plot_pipeline.py` | Defines the `PlotSource` protocol, `CsvSource`, and shared `generate()` pipeline used by both CLI entry points. | `plot_pipeline_test.py` |
| `generate_plots.py` | CLI entry point for generating plots and tables from per-replicate error parquets. | — |
| `generate_plots_from_csv.py` | CLI entry point for regenerating plots and tables from exported CSVs. | — |

<!-- mdformat on -->

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