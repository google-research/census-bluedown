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

"""LaTeX summary tables of replicate errors with paired t-tests."""

from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy import stats
from census_bluedown.analysis import common

# p-value rendering: below this threshold the table shows '<0.0001'.
_P_VALUE_FLOOR = 1e-4
_P_VALUE_BELOW_FLOOR = '<0.0001'
_P_VALUE_NOT_AVAILABLE = 'N/A'

_STATS_COLUMNS = (
    common.COL_GEOGRAPHIC_LEVEL,
    common.COL_QUERY,
    common.COL_BLUEDOWN_MEAN,
    common.COL_BLUEDOWN_STD,
    common.COL_TOPDOWN_MEAN,
    common.COL_TOPDOWN_STD,
    common.COL_P_VALUE,
)


def _format_p_value(
    baseline: Sequence[float], optimized: Sequence[float]
) -> str:
  """Formats the paired t-test p-value of two replicate error sequences."""
  if len(baseline) < 2 or len(baseline) != len(optimized):
    return _P_VALUE_NOT_AVAILABLE
  _, p_value = stats.ttest_rel(baseline, optimized)
  if np.isnan(p_value):
    return _P_VALUE_NOT_AVAILABLE
  if p_value < _P_VALUE_FLOOR:
    return _P_VALUE_BELOW_FLOOR
  return f'{p_value:.4f}'


def compute_summary_stats(
    summary_df: pd.DataFrame,
    queries: Sequence[str],
    levels: Sequence[str],
) -> pd.DataFrame:
  """Computes per-(query, level) error means, stds, and paired t-test p-values.

  Args:
    summary_df: Replicate summary DataFrame containing columns: * 'level' (str):
      Geographic level name. * 'query' (str): Query name. * 'optimized'
      (list[float]): Error values across replicates. * 'baseline' (list[float]):
      Error values across replicates.
    queries: Query names to include, in row order (outer loop).
    levels: Geographic levels to include, in row order (inner loop).

  Returns:
    DataFrame with one row per (query, level) present in summary_df and
    columns:
      * 'geographic_level' (str): Geographic level name.
      * 'query' (str): Query name.
      * 'bluedown_mean', 'bluedown_std' (float): Error statistics for BlueDown.
      * 'topdown_mean', 'topdown_std' (float): Error statistics for TopDown.
      * 'p_value' (str): Paired t-test p-value formatted to 4 decimals,
        '<0.0001' below that, 'N/A' if fewer than two paired replicates, or ''
        for the TOTAL query at levels in common.LEVELS_WITH_TRIVIAL_TOTAL,
        where both methods reproduce the invariant exactly and all statistics
        are reported as 0.
  """
  present_levels = set(summary_df[common.COL_LEVEL])
  rows = []
  for query in queries:
    for level in levels:
      if (
          query == common.QUERY_TOTAL
          and level in common.LEVELS_WITH_TRIVIAL_TOTAL
          and level in present_levels
      ):
        rows.append({
            common.COL_GEOGRAPHIC_LEVEL: level,
            common.COL_QUERY: query,
            common.COL_BLUEDOWN_MEAN: 0.0,
            common.COL_BLUEDOWN_STD: 0.0,
            common.COL_TOPDOWN_MEAN: 0.0,
            common.COL_TOPDOWN_STD: 0.0,
            common.COL_P_VALUE: '',
        })
        continue

      match = summary_df[
          (summary_df[common.COL_LEVEL] == level)
          & (summary_df[common.COL_QUERY] == query)
      ]
      if match.empty:
        continue
      opt = list(match[common.COL_OPTIMIZED].iloc[0])
      base = list(match[common.COL_BASELINE].iloc[0])

      rows.append({
          common.COL_GEOGRAPHIC_LEVEL: level,
          common.COL_QUERY: query,
          common.COL_BLUEDOWN_MEAN: float(np.mean(opt)),
          common.COL_BLUEDOWN_STD: float(np.std(opt)),
          common.COL_TOPDOWN_MEAN: float(np.mean(base)),
          common.COL_TOPDOWN_STD: float(np.std(base)),
          common.COL_P_VALUE: _format_p_value(base, opt),
      })
  return pd.DataFrame(rows, columns=list(_STATS_COLUMNS))


def _format_stat(value: float) -> str:
  """Formats a mean/std for the table; exact zeros print as a bare '0'."""
  return '0' if value == 0 else f'{value:.2f}'


def render_latex_table(
    stats_df: pd.DataFrame, include_pvalues: bool = True
) -> str:
  """Renders a summary-statistics DataFrame as a LaTeX tabular.

  Args:
    stats_df: DataFrame as returned by compute_summary_stats.
    include_pvalues: Whether to render the p-value column.

  Returns:
    LaTeX tabular source. Means and stds are shown as 'mean ± std'.
  """
  col_spec = r'r@{\hspace{1em}$\pm$\,}r'
  header = (
      r'Geographic level & Query & \multicolumn{2}{c|}{BlueDown error} &'
      r' \multicolumn{2}{c|}{TopDown error}'
  )
  if include_pvalues:
    tabular = rf'\begin{{tabular}}{{|ll|{col_spec}|{col_spec}|r|}}'
    header += r' & p-value \\'
  else:
    tabular = rf'\begin{{tabular}}{{|ll|{col_spec}|{col_spec}|}}'
    header += r' \\'
  lines = [tabular, r'\hline', header, r'\hline']

  for _, row in stats_df.iterrows():
    level = str(row[common.COL_GEOGRAPHIC_LEVEL])
    query = str(row[common.COL_QUERY])
    cells = [
        common.TABLE_NAMES.get(level, level),
        common.TABLE_NAMES.get(query, query),
        _format_stat(float(row[common.COL_BLUEDOWN_MEAN])),
        _format_stat(float(row[common.COL_BLUEDOWN_STD])),
        _format_stat(float(row[common.COL_TOPDOWN_MEAN])),
        _format_stat(float(row[common.COL_TOPDOWN_STD])),
    ]
    if include_pvalues:
      p_value = row[common.COL_P_VALUE]
      p_value = '' if pd.isna(p_value) else str(p_value)
      if p_value.startswith('<'):
        p_value = f'${p_value}$'
      cells.append(p_value)
    lines.append(' & '.join(cells) + r' \\')

  lines += [r'\hline', r'\end{tabular}']
  return '\n'.join(lines)
