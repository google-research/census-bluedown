# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loading utilities for block hierarchical analysis and plotting."""

import ast
import bisect
from collections.abc import Iterable, Sequence
import itertools
import os
from typing import Any

import numpy as np
import pandas as pd

from census_bluedown.analysis import common

_NON_AIAN_STATE_IDS = (
    '001',
    '002',
    '004',
    '005',
    '006',
    '008',
    '009',
    '010',
    '011',
    '012',
    '013',
    '015',
    '016',
    '017',
    '018',
    '019',
    '020',
    '021',
    '022',
    '023',
    '024',
    '025',
    '026',
    '027',
    '028',
    '029',
    '030',
    '031',
    '032',
    '033',
    '034',
    '035',
    '036',
    '037',
    '038',
    '039',
    '040',
    '041',
    '042',
    '044',
    '045',
    '046',
    '047',
    '048',
    '049',
    '050',
    '051',
    '053',
    '054',
    '055',
    '056',
)


def _safe_parse_list(val: Any) -> list[Any]:
  """Safely parses string representations of lists or returns list directly."""
  if isinstance(val, str):
    return ast.literal_eval(val)
  elif isinstance(val, (list, np.ndarray)):
    return list(val)
  return [val]


def _compute_us_level_replicate_errors(
    base_dir: str,
    gt_parquet_path: str,
    num_replicates: int = 10,
) -> pd.DataFrame:
  """Computes error metrics for the US level across replicates.

  Reads the US ground truth array and per-replicate estimate parquets,
  aggregates them across the dimensions specified by US_QUERIES_AND_SHAPES,
  and computes absolute L1 errors.

  Args:
    base_dir: Root directory containing replicate subdirectories (e.g. '0',
      '1'). Each replicate subdirectory base_dir/<rep>/US/ must contain: *
      'optimized_estimate.parquet' with column 'value' reshapeable to (8, 2, 2,
      63). * 'baseline_estimate.parquet' with column 'value' reshapeable to (8,
      2, 2, 63).
    gt_parquet_path: Path to the US ground truth parquet file. Expected columns:
      * 'query_name' (str): Query identifier (must include 'detailed_dpq'). *
      'value' (ndarray): Flattened array reshapeable to (8, 2, 2, 63).
    num_replicates: Number of replicates to process.

  Returns:
    DataFrame with columns:
    * 'level' (str): Geographic level ('US').
    * 'query' (str): Query name from US_QUERIES_AND_SHAPES.
    * 'baseline' (list[float]): Raw absolute errors for baseline across
      replicates.
    * 'optimized' (list[float]): Raw absolute errors for optimized across
      replicates.
  """
  with open(gt_parquet_path, 'rb') as in_f:
    us_gt_df = pd.read_parquet(in_f)
    is_detailed = us_gt_df[common.GT_QUERY_NAME_COL] == common.GT_DETAILED_QUERY
    us_gt = (
        us_gt_df[is_detailed][common.ESTIMATE_VALUE_COL]
        .item()
        .reshape(common.US_DETAILED_SHAPE)
    )

  # Read each replicate's estimates once; shape (num_replicates, 8, 2, 2, 63).
  opt_estimates = []
  base_estimates = []
  for i in range(num_replicates):
    rep_us_dir = os.path.join(base_dir, str(i), common.REPLICATE_US_DIR)
    opt_path = os.path.join(rep_us_dir, common.OPTIMIZED_ESTIMATE_FILE)
    base_path = os.path.join(rep_us_dir, common.BASELINE_ESTIMATE_FILE)
    opt_estimates.append(_read_us_tensor(opt_path))
    base_estimates.append(_read_us_tensor(base_path))
  opt_stack = np.stack(opt_estimates)
  base_stack = np.stack(base_estimates)

  us_baseline_data = []
  us_optimized_data = []
  for shape in common.US_QUERIES_AND_SHAPES.values():
    # Marginalise the tensor dimensions this query does not distinguish. Axis
    # 0 of the stacks is the replicate axis, so tensor axes shift by one.
    tensor_axes = tuple(i for i, dim in enumerate(shape) if dim == 1)
    agg_gt = np.sum(us_gt, axis=tensor_axes, keepdims=True)
    stack_axes = tuple(ax + 1 for ax in tensor_axes)
    agg_opt = np.sum(opt_stack, axis=stack_axes, keepdims=True)
    agg_base = np.sum(base_stack, axis=stack_axes, keepdims=True)
    # L1 error per replicate: sum over all tensor axes.
    tensor_dims = tuple(range(1, agg_opt.ndim))
    opt_err = np.sum(np.abs(agg_opt - agg_gt), axis=tensor_dims)
    base_err = np.sum(np.abs(agg_base - agg_gt), axis=tensor_dims)
    us_optimized_data.append([float(x) for x in opt_err])
    us_baseline_data.append([float(x) for x in base_err])

  return pd.DataFrame({
      common.COL_LEVEL: [common.LEVEL_US] * len(common.US_QUERIES_AND_SHAPES),
      common.COL_QUERY: list(common.US_QUERIES_AND_SHAPES.keys()),
      common.COL_BASELINE: us_baseline_data,
      common.COL_OPTIMIZED: us_optimized_data,
  })


def _read_us_tensor(path: str) -> np.ndarray:
  """Reads a single-row parquet holding a flattened US detailed tensor.

  Args:
    path: Path to a parquet with one row whose ESTIMATE_VALUE_COL cell is the
      flattened tensor.

  Returns:
    The tensor reshaped to common.US_DETAILED_SHAPE.
  """
  with open(path, 'rb') as in_f:
    flat = pd.read_parquet(in_f)[common.ESTIMATE_VALUE_COL].item()
  return np.asarray(flat).reshape(common.US_DETAILED_SHAPE)


def load_summary_replicate_errors(
    base_dir: str,
    num_replicates: int = 10,
    gt_parquet_path: str | None = None,
) -> pd.DataFrame:
  """Loads and aggregates standard query errors across replicates.

  Reads 'query_errors.parquet' files for each replicate under
  base_dir/<rep>/US/, aggregates them into list-valued columns per
  (level, query) pair, and optionally appends US-level errors.

  Args:
    base_dir: Root directory containing replicate output folders ('0', '1',
      ...). Each folder is expected to contain
      base_dir/<rep>/US/query_errors.parquet with columns: * 'level' (str):
      Geographic level name (e.g. 'State', 'County', 'Tract'). * 'query' (str):
      Query name. * 'baseline' (float): Error value for baseline method. *
      'optimized' (float): Error value for optimized method.
    num_replicates: Number of replicate runs to load.
    gt_parquet_path: Optional path to US ground truth parquet file. If provided
      and exists, US-level errors will be computed and appended.

  Returns:
    DataFrame grouped by ['level', 'query'] with list-valued columns:
      * 'level' (str): Geographic level name.
      * 'query' (str): Query name.
      * 'baseline' (list[float]): Baseline error values across replicates.
      * 'optimized' (list[float]): Optimized error values across replicates.
  """
  replicate_errs = []
  for i in range(num_replicates):
    err_file = os.path.join(
        base_dir, str(i), common.REPLICATE_US_DIR, common.QUERY_ERRORS_FILE
    )
    with open(err_file, 'rb') as in_f:
      next_errs = pd.read_parquet(in_f)
    replicate_errs.append(next_errs)

  summary_df = (
      pd.concat(replicate_errs)
      .groupby([common.COL_LEVEL, common.COL_QUERY])
      .agg(
          baseline=(common.COL_BASELINE, list),
          optimized=(common.COL_OPTIMIZED, list),
      )
      .reset_index()
  )

  if gt_parquet_path and os.path.exists(gt_parquet_path):
    us_df = _compute_us_level_replicate_errors(
        base_dir=base_dir,
        gt_parquet_path=gt_parquet_path,
        num_replicates=num_replicates,
    )
    summary_df = pd.concat([summary_df, us_df], ignore_index=True)

  return summary_df


def load_detailed_query_errors(
    base_dir: str, num_replicates: int = 10
) -> pd.DataFrame:
  """Loads and aggregates detailed summary query errors across replicates.

  Args:
    base_dir: Root directory path containing replicate subdirectories.
    num_replicates: Number of replicate subdirectories to load.

  Returns:
    DataFrame grouped by ['level', 'query'] with list-valued columns:
    * 'level' (str): Geographic level name.
    * 'query' (str): Query name.
    * 'baseline' (list[float]): Baseline error values across replicates.
    * 'optimized' (list[float]): Optimized error values across replicates.
  """
  detailed_replicate_errs = []
  for i in range(num_replicates):
    err_file = os.path.join(
        base_dir,
        str(i),
        common.REPLICATE_US_DIR,
        common.ALTERNATE_QUERY_ERRORS_FILE,
    )
    with open(err_file, 'rb') as in_f:
      overall_alt = pd.read_parquet(in_f)
    detailed_replicate_errs.append(overall_alt)

  return (
      pd.concat(detailed_replicate_errs)
      .groupby([common.COL_LEVEL, common.COL_QUERY])
      .agg(
          baseline=(common.COL_BASELINE, list),
          optimized=(common.COL_OPTIMIZED, list),
      )
      .reset_index()
  )


def _partition_by_thresholds(
    values: Sequence[float],
    condition_values: Sequence[float],
    thresholds: Sequence[float],
) -> list[list[float]]:
  """Partitions values based on condition_values relative to thresholds.

  Args:
    values: Sequence of numerical values to partition.
    condition_values: Sequence of numerical conditioning values (e.g. ground
      truth population sizes) of the same length as values.
    thresholds: Ascending sequence of numerical threshold boundaries.

  Returns:
    List of partitioned float lists of length len(thresholds) + 1.
  """
  if len(values) != len(condition_values):
    raise ValueError('values and condition_values must be of the same length.')

  sorted_thresholds = sorted(thresholds)
  num_partitions = len(sorted_thresholds) + 1
  partitions: list[list[float]] = [[] for _ in range(num_partitions)]

  for val, cond in zip(values, condition_values):
    idx = bisect.bisect_right(sorted_thresholds, cond)
    partitions[idx].append(val)

  return partitions


def _mean_or_zero(values: Sequence[float], absolute: bool = False) -> float:
  """Returns the mean of values (or of |values|), or 0.0 if values is empty.

  Args:
    values: Sequence of numbers.
    absolute: If True, average the absolute values instead.

  Returns:
    Mean as a float, or 0.0 for an empty sequence.
  """
  if not values:
    return 0.0
  arr = np.asarray(values, dtype=float)
  return float(np.mean(np.abs(arr) if absolute else arr))


def load_population_binned_data(
    base_dir: str,
    levels: Sequence[str] = common.POPULATION_BINNED_LEVELS,
    state_ids: Sequence[str] = _NON_AIAN_STATE_IDS,
    num_replicates: int = 10,
) -> pd.DataFrame:
  """Loads signed bias and L1 error of TOTAL, binned by population size.

  For every replicate and level, concatenates per-unit TOTAL biases across the
  requested states, partitions units into population bins using
  common.BIAS_THRESHOLDS on the ground-truth population, and records the mean
  signed bias and mean absolute bias (L1 error) of each bin.

  Args:
    base_dir: Root directory containing replicate output folders. Each
      base_dir/<rep>/State=<sid>/<level>/query_errors_list.parquet is expected
      to be indexed by query name with list-valued (or stringified list) columns
      'base_bias', 'opt_bias', and 'ground_truth'.
    levels: Sequence of geographic levels to process.
    state_ids: List of state subtree IDs to include.
    num_replicates: Number of replicate runs.

  Returns:
    Tidy DataFrame with one row per (level, metric, replicate, population_bin)
    and columns:
      * 'level' (str): Geographic level name.
      * 'metric' (str): 'bias' (mean signed bias) or 'binned_error' (mean
        absolute bias).
      * 'replicate' (int): Replicate index.
      * 'population_bin' (str): Bin label from common.BIAS_LABELS.
      * 'bluedown_value' (float): Mean metric value for BlueDown.
      * 'topdown_value' (float): Mean metric value for TopDown.
    Bins with no units in a replicate are recorded as 0.0.
  """
  rows = []
  for level in levels:
    thresholds = common.BIAS_THRESHOLDS[level]
    labels = common.BIAS_LABELS[level]

    for replicate in range(num_replicates):
      base_bias: list[float] = []
      opt_bias: list[float] = []
      gt: list[float] = []

      for sid in state_ids:
        err_list_path = os.path.join(
            base_dir,
            str(replicate),
            common.STATE_DIR_TEMPLATE.format(sid=sid),
            level,
            common.QUERY_ERRORS_LIST_FILE,
        )
        if not os.path.exists(err_list_path):
          continue
        with open(err_list_path, 'rb') as in_f:
          df = pd.read_parquet(in_f)
        if common.QUERY_TOTAL not in df.index:
          continue
        total_row = df.loc[common.QUERY_TOTAL]
        base_bias.extend(_safe_parse_list(total_row[common.UNIT_BASE_BIAS_COL]))
        opt_bias.extend(_safe_parse_list(total_row[common.UNIT_OPT_BIAS_COL]))
        gt.extend(_safe_parse_list(total_row[common.UNIT_GROUND_TRUTH_COL]))

      base_bins = _partition_by_thresholds(base_bias, gt, thresholds)
      opt_bins = _partition_by_thresholds(opt_bias, gt, thresholds)

      for label, base_vals, opt_vals in zip(labels, base_bins, opt_bins):
        rows.append({
            common.COL_LEVEL: level,
            common.COL_METRIC: common.METRIC_BIAS,
            common.COL_REPLICATE: replicate,
            common.COL_POPULATION_BIN: label,
            common.COL_BLUEDOWN_VALUE: _mean_or_zero(opt_vals),
            common.COL_TOPDOWN_VALUE: _mean_or_zero(base_vals),
        })
        rows.append({
            common.COL_LEVEL: level,
            common.COL_METRIC: common.METRIC_BINNED_ERROR,
            common.COL_REPLICATE: replicate,
            common.COL_POPULATION_BIN: label,
            common.COL_BLUEDOWN_VALUE: _mean_or_zero(opt_vals, absolute=True),
            common.COL_TOPDOWN_VALUE: _mean_or_zero(base_vals, absolute=True),
        })

  return pd.DataFrame(rows)


def load_sliced_query_errors(
    base_dir: str,
    levels: Sequence[str],
    state_ids: Sequence[str] = _NON_AIAN_STATE_IDS,
    num_replicates: int = 10,
) -> dict[str, pd.DataFrame]:
  """Loads per-geographic-unit error distributions across states and replicates.

  Args:
    base_dir: Root directory containing replicate output folders.
    levels: Sequence of geographic levels to load (e.g., County, Tract).
    state_ids: List of state subtree IDs to include.
    num_replicates: Number of replicate runs.

  Returns:
    Dict mapping geographic level to aggregated DataFrame indexed by query name
    containing columns:
    * 'baseline' (list[float]): Concatenated baseline error list across units.
    * 'optimized' (list[float]): Concatenated optimized error list across units.
    * 'n' (int): Total number of geographic units evaluated.
  """
  ridgeline_errs: dict[str, pd.DataFrame] = {}

  for level in levels:
    frames = []
    for i in range(num_replicates):
      for sid in state_ids:
        err_list_path = os.path.join(
            base_dir,
            str(i),
            common.STATE_DIR_TEMPLATE.format(sid=sid),
            level,
            common.QUERY_ERRORS_LIST_FILE,
        )
        if not os.path.exists(err_list_path):
          continue
        with open(err_list_path, 'rb') as in_f:
          next_errs = pd.read_parquet(in_f)[
              [common.COL_BASELINE, common.COL_OPTIMIZED, common.UNIT_COUNT_COL]
          ].copy()
        next_errs[common.COL_BASELINE] = next_errs[common.COL_BASELINE].apply(
            _safe_parse_list
        )
        next_errs[common.COL_OPTIMIZED] = next_errs[common.COL_OPTIMIZED].apply(
            _safe_parse_list
        )
        frames.append(next_errs)

    if frames:
      # Concatenate all (replicate, state) frames once and merge per query.
      # chain.from_iterable is linear in the total number of units, whereas
      # sum(lists, []) is quadratic.
      combined = pd.concat(frames)
      ridgeline_errs[level] = combined.groupby(combined.index).agg({
          common.COL_BASELINE: _concat_lists,
          common.COL_OPTIMIZED: _concat_lists,
          common.UNIT_COUNT_COL: 'sum',
      })

  return ridgeline_errs


def _concat_lists(lists: Iterable[list[Any]]) -> list[Any]:
  """Concatenates an iterable of lists in linear time."""
  return list(itertools.chain.from_iterable(lists))


def histogram_query_errors(level: str, level_df: pd.DataFrame) -> pd.DataFrame:
  """Bins per-unit errors of one level into integer-valued histograms.

  This is the single source of the (value, count) pairs that ridgeline plots
  are drawn from, whether plotting directly or via an exported CSV.

  Args:
    level: Geographic level name.
    level_df: DataFrame indexed by query name, as returned per level by
      load_sliced_query_errors, with list-valued columns 'optimized' and
      'baseline' holding one error per geographic unit.

  Returns:
    Tidy DataFrame with one row per (query, error_value) and columns:
      * 'level' (str): Geographic level name.
      * 'query' (str): Query name, in common.RIDGELINE_QUERY_ORDER.
      * 'error_value' (int): Error rounded to the nearest integer.
      * 'bluedown_count' (int): Number of units with that error (BlueDown).
      * 'topdown_count' (int): Number of units with that error (TopDown).
    The TOTAL query is omitted for levels in common.LEVELS_WITH_TRIVIAL_TOTAL.
  """
  queries = [q for q in common.RIDGELINE_QUERY_ORDER if q in level_df.index]
  if level in common.LEVELS_WITH_TRIVIAL_TOTAL:
    queries = [q for q in queries if q != common.QUERY_TOTAL]

  frames = []
  for q in queries:
    opt = pd.Series(level_df.loc[q, common.COL_OPTIMIZED]).round().astype(int)
    base = pd.Series(level_df.loc[q, common.COL_BASELINE]).round().astype(int)
    counts = (
        pd.DataFrame({
            common.COL_BLUEDOWN_COUNT: opt.value_counts(),
            common.COL_TOPDOWN_COUNT: base.value_counts(),
        })
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    counts.index.name = common.COL_ERROR_VALUE
    counts = counts.reset_index()
    counts.insert(0, common.COL_QUERY, q)
    counts.insert(0, common.COL_LEVEL, level)
    frames.append(counts)

  columns = [
      common.COL_LEVEL,
      common.COL_QUERY,
      common.COL_ERROR_VALUE,
      common.COL_BLUEDOWN_COUNT,
      common.COL_TOPDOWN_COUNT,
  ]
  if not frames:
    return pd.DataFrame(columns=columns)
  return pd.concat(frames, ignore_index=True)[columns]
