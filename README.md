# Census BlueDown: Differentially Private Block Hierarchical Aggregation

This repository contains the reference Python implementation for the _BlueDown_
algorithm applied for post-processing the
[Noisy Measurement File](https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/00-2020-Redistricting-Noisy-Measurement-File/Noisy-Measurement-File-2020-Census-Redistricting-Data.pdf)
for the U.S. Census P.L. 94-171 Redistricting Data.

**Note:** This is not an officially supported Google product. This project is
not eligible for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

## Paper & Citation

For technical details on the algorithm, privacy guarantees, and performance,
please refer to our [paper](https://arxiv.org/abs/2603.10099).

If you use this code or algorithm in your research, please cite our paper:

<!-- mdlint off(SNIPPET_INVALID_LANGUAGE) -->
```bibtex
@misc{ghazi2026bluedown,
  title={Denoising the US Census: Succinct Block Hierarchical Regression},
  author={Badih Ghazi and Pritish Kamath and Ravi Kumar and Pasin Manurangsi and Adam Sealfon},
  year={2026},
  eprint={2603.10099},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.10099},
}
```

## Installation

Clone the repository and install the dependencies listed in `requirements.txt`:

```bash
git clone https://github.com/google-research/census-bluedown.git
cd census-bluedown
pip install -r requirements.txt
```

## Usage

Set `PYTHONPATH=.` before running scripts from the repository root:

```bash
export PYTHONPATH=.
```

---

### 1. Synthetic Data Generation (`generate_synthetic_data.py`)

Generate synthetic datasets or ground-truth evaluation tables:

* **Pass 1: Ground Truth Generation (`--pass=ground_truth`)**
  Reads microdata Census ZIP archives (`--zip_path`) and NMF structure
  (`--nmf_dir`) to generate ground truth tables for a specified state
  (`--state`):

  ```bash
  python3 census_bluedown/generate_synthetic_data.py \
    --pass=ground_truth \
    --state=02 \
    --nmf_dir=/path/to/raw_nmf/ \
    --out_dir=/path/to/synthetic-ground-truth/ \
    --zip_path=/path/to/2020_ppmf_bystate.zip
  ```

* **Pass 2: Synthetic NMF Generation (`--pass=nmf`)**
  Generates synthetic noisy measurement files (NMF) from ground-truth
  tables (`--ground_truth_dir`):

  ```bash
  python3 census_bluedown/generate_synthetic_data.py \
    --pass=nmf \
    --state=002 \
    --ground_truth_dir=/path/to/synthetic-ground-truth/ \
    --out_dir=/path/to/synthetic-nmf/
  ```

---

### 2. Block Hierarchical Postprocessing Pipeline (`run_pass.py`)

Postprocessing runs in **4 sequential steps**, followed by optional error
evaluation:

#### **Step 1: Bottom-Up Pass (State Subtrees)**
Initializes input estimates and aggregates variances bottom-up for
individual state subtrees:

```bash
python3 census_bluedown/run_pass.py \
  --pass=bottom_up \
  --state=125 \
  --nmf_dir=/path/to/synthetic-nmf/ \
  --out_dir=/path/to/synthetic-processed/
```

#### **Step 2: Bottom-Up Pass (Root Node)**
Combines state-level totals to construct national (US root) bottom-up totals:

```bash
python3 census_bluedown/run_pass.py \
  --pass=bottom_up \
  --out_dir=/path/to/synthetic-processed/ \
  --subtree_totals_dir=/path/to/synthetic-processed/
```

#### **Step 3: Top-Down Pass (Root Node)**
Runs non-linear optimization at the root level to establish consistent
national upper bounds:

```bash
python3 census_bluedown/run_pass.py \
  --pass=top_down \
  --in_dir=/path/to/synthetic-processed/ \
  --out_dir=/path/to/synthetic-processed/ \
  --subtree_totals_dir=/path/to/synthetic-processed/
```

#### **Step 4: Top-Down Pass (State Subtrees)**
Runs top-down constrained optimization down to individual census blocks
using root bounds:

```bash
python3 census_bluedown/run_pass.py \
  --pass=top_down \
  --state=125 \
  --in_dir=/path/to/synthetic-processed/ \
  --out_dir=/path/to/synthetic-processed/
```

#### **Alternative Baseline & Error Evaluation Passes**

* **Baseline Census Top-Down Pass**:

  ```bash
  python3 census_bluedown/run_pass.py \
    --pass=baseline_census_topdown \
    --state=125 \
    --nmf_dir=/path/to/synthetic-nmf/ \
    --out_dir=/path/to/synthetic-processed/
  ```
* **Error Evaluation**:

  ```bash
  python3 census_bluedown/run_pass.py \
    --pass=compute_errors \
    --state=125 \
    --in_dir=/path/to/synthetic-ground-truth/ \
    --out_dir=/path/to/synthetic-processed/
  ```

### Running Tests

To run the unit test suite:

```bash
PYTHONPATH=. python3 census_bluedown/estimate_test.py
PYTHONPATH=. python3 census_bluedown/nonlinear_solver_test.py
```

## License

Licensed under the [Apache 2.0 License](LICENSE).
