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

"""Common constants, query specifications, and CLI helpers for analysis."""

from collections.abc import Sequence
import dataclasses
import os

from absl import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Type aliases
StringTuple = tuple[str, ...]

# DataFrame and CSV column names
COL_LEVEL = 'level'
COL_QUERY = 'query'
COL_OPTIMIZED = 'optimized'  # Column name for bluedown errors in saved parquet.
COL_BASELINE = 'baseline'  # Column name for topdown errors in saved parquet.
COL_REPLICATE = 'replicate'
COL_BLUEDOWN_ERROR = 'bluedown_error'
COL_TOPDOWN_ERROR = 'topdown_error'
COL_BENCHMARK = 'benchmark'
COL_MEASUREMENT = 'measurement'
COL_QUERY_KEY = 'query_key'
COL_ERROR_VALUE = 'error_value'
COL_BLUEDOWN_COUNT = 'bluedown_count'
COL_TOPDOWN_COUNT = 'topdown_count'
COL_METRIC = 'metric'
COL_POPULATION_BIN = 'population_bin'
COL_BLUEDOWN_VALUE = 'bluedown_value'
COL_TOPDOWN_VALUE = 'topdown_value'
COL_GEOGRAPHIC_LEVEL = 'geographic_level'
COL_BLUEDOWN_MEAN = 'bluedown_mean'
COL_BLUEDOWN_STD = 'bluedown_std'
COL_TOPDOWN_MEAN = 'topdown_mean'
COL_TOPDOWN_STD = 'topdown_std'
COL_P_VALUE = 'p_value'

# Layout of replicate input directories: base_dir/<replicate>/...
REPLICATE_US_DIR = 'US'
QUERY_ERRORS_FILE = 'query_errors.parquet'
ALTERNATE_QUERY_ERRORS_FILE = 'alternate_query_errors.parquet'
# Per-state outputs: base_dir/<replicate>/State=<sid>/<level>/...
STATE_DIR_TEMPLATE = 'State={sid}'
QUERY_ERRORS_LIST_FILE = 'query_errors_list.parquet'
# Columns of query_errors_list.parquet (list-valued, one entry per unit).
UNIT_BASE_BIAS_COL = 'base_bias'
UNIT_OPT_BIAS_COL = 'opt_bias'
UNIT_GROUND_TRUTH_COL = 'ground_truth'
UNIT_COUNT_COL = 'n'
OPTIMIZED_ESTIMATE_FILE = 'optimized_estimate.parquet'
BASELINE_ESTIMATE_FILE = 'baseline_estimate.parquet'
# Columns of the estimate / ground-truth parquets and the US detailed query.
ESTIMATE_VALUE_COL = 'value'
GT_QUERY_NAME_COL = 'query_name'
GT_DETAILED_QUERY = 'detailed_dpq'
# Tensor shape of the US detailed histogram: housing x voting age x hispanic
# x race.
US_DETAILED_SHAPE = (8, 2, 2, 63)

# CLI tokens shared by the plot-generation binaries.
FLAG_ALL = 'all'
PLOT_BOXPLOTS = 'boxplots'
PLOT_DETAILED_QUERIES = 'detailed_queries'
PLOT_BIAS = 'bias'
PLOT_BINNED_ERROR = 'binned_error'
PLOT_RIDGELINE = 'ridgeline'
PLOT_TABLES = 'tables'
FORMAT_PDF = 'pdf'
FORMAT_CSV = 'csv'

# Layout of generated outputs: <output_dir>/<subdir>/<stem>.<ext>. Stems are
# format strings taking the keyword arguments named in their placeholders.
BOXPLOTS_DIR = 'box-plots'
LEVEL_QUERIES_STEM = '{level}_queries'
DETAILED_QUERIES_DIR = 'detailed_queries'
DETAILED_QUERIES_STEM = 'detailed_queries_{name}'
BIAS_DIR = 'bias'
BIAS_STEM = 'bias_{level}'
ERR_POPULATION_BINS_DIR = 'err_population_bins'
ERR_POPULATION_BINS_STEM = 'err_population_bins_{level}'
RIDGELINE_DIR = 'ridgeline'
RIDGELINE_STEM = '{level}_ridgeline'
TABLES_DIR = 'tables'
SUMMARY_TABLE_STEM = 'summary_table_{index}'

# Geographic level names, in hierarchical order.
LEVEL_US = 'US'
LEVEL_STATE = 'State'
LEVEL_COUNTY = 'County'
LEVEL_TRACT = 'Tract'
LEVEL_BLOCK_GROUP = 'Block_Group'
LEVEL_BLOCK = 'Block'

ALL_LEVELS: StringTuple = (
    LEVEL_US,
    LEVEL_STATE,
    LEVEL_COUNTY,
    LEVEL_TRACT,
    LEVEL_BLOCK_GROUP,
    LEVEL_BLOCK,
)

# Levels at which the TOTAL query is an exact invariant for both methods, so
# its error is identically zero and it is excluded from level-query plots.
LEVELS_WITH_TRIVIAL_TOTAL: StringTuple = (LEVEL_US, LEVEL_STATE)

# Human-readable level names used in axis labels.
LEVEL_DISPLAY_NAMES: dict[str, str] = {
    LEVEL_US: 'US',
    LEVEL_STATE: 'state',
    LEVEL_COUNTY: 'county',
    LEVEL_TRACT: 'tract',
    LEVEL_BLOCK_GROUP: 'block group',
    LEVEL_BLOCK: 'block',
}

# Population-binned metrics (signed bias, L1 error) are computed for the
# sub-state levels only.
METRIC_BIAS = 'bias'
METRIC_BINNED_ERROR = 'binned_error'
POPULATION_BINNED_LEVELS: StringTuple = (
    LEVEL_COUNTY,
    LEVEL_TRACT,
    LEVEL_BLOCK_GROUP,
    LEVEL_BLOCK,
)

# Population size thresholds that split units into bins, and the bin labels.
BIAS_THRESHOLDS: dict[str, list[int]] = {
    LEVEL_COUNTY: [5000, 10000, 50000, 100000],
    LEVEL_TRACT: [500, 1000, 5000, 10000],
    LEVEL_BLOCK_GROUP: [50, 100, 500, 1000, 5000],
    LEVEL_BLOCK: [1, 10, 50, 100, 500],
}

BIAS_LABELS: dict[str, list[str]] = {
    LEVEL_COUNTY: ['<5000', '<10000', '<50000', '<100000', '≥100000'],
    LEVEL_TRACT: ['<500', '<1000', '<5000', '<10000', '≥10000'],
    LEVEL_BLOCK_GROUP: ['<50', '<100', '<500', '<1000', '<5000', '≥5000'],
    LEVEL_BLOCK: ['0', '<10', '<50', '<100', '<500', '≥500'],
}

# Query names
QUERY_TOTAL = 'TOTAL'
QUERY_HOUSING_TYPE = 'HOUSING_TYPE'
QUERY_VOTINGAGE = 'VOTINGAGE'
QUERY_HISPANIC = 'HISPANIC'
QUERY_CENRACE = 'CENRACE'
QUERY_HISPANICxCENRACE = 'HISPANICxCENRACE'  # pylint: disable=invalid-name
QUERY_DETAILED = 'DETAILED'
QUERY_AGExHISPxCENRACE = 'AGExHISPxCENRACE'  # pylint: disable=invalid-name

# Query labels
LABEL_TOTAL = 'total'
LABEL_HOUSING_TYPE = 'housing type'
LABEL_VOTINGAGE = 'voting age'
LABEL_HISPANIC = 'hispanic'
LABEL_CENRACE = 'race'
LABEL_HISPANICxCENRACE = 'hispanic x race'  # pylint: disable=invalid-name
LABEL_DETAILED = 'detailed'

# Standard level query definitions
STANDARD_LEVEL_QUERIES: StringTuple = (
    QUERY_TOTAL,
    QUERY_HOUSING_TYPE,
    QUERY_VOTINGAGE,
    QUERY_HISPANIC,
    QUERY_CENRACE,
    QUERY_HISPANICxCENRACE,
    QUERY_DETAILED,
)

STANDARD_LEVEL_QUERY_LABELS: StringTuple = (
    LABEL_TOTAL,
    LABEL_HOUSING_TYPE,
    LABEL_VOTINGAGE,
    LABEL_HISPANIC,
    LABEL_CENRACE,
    LABEL_HISPANICxCENRACE,
    LABEL_DETAILED,
)

# Target query names and multidimensional tensor shapes for US-level queries.
# Each shape entry is either 1 (axis marginalised out) or the corresponding
# entry of US_DETAILED_SHAPE (axis kept).
US_QUERIES_AND_SHAPES: dict[str, tuple[int, int, int, int]] = {
    QUERY_HOUSING_TYPE: (8, 1, 1, 1),
    QUERY_VOTINGAGE: (1, 2, 1, 1),
    QUERY_HISPANIC: (1, 1, 2, 1),
    QUERY_CENRACE: (1, 1, 1, 63),
    QUERY_HISPANICxCENRACE: (1, 1, 2, 63),
    QUERY_DETAILED: US_DETAILED_SHAPE,
    QUERY_AGExHISPxCENRACE: (1, 2, 2, 63),
    QUERY_TOTAL: (1, 1, 1, 1),
}

# Ridgeline query order and display relabeling
RIDGELINE_QUERY_ORDER: StringTuple = (
    QUERY_TOTAL,
    QUERY_HOUSING_TYPE,
    QUERY_VOTINGAGE,
    QUERY_HISPANIC,
    QUERY_CENRACE,
    QUERY_HISPANICxCENRACE,
    QUERY_AGExHISPxCENRACE,
    QUERY_DETAILED,
)

RIDGELINE_RELABEL: dict[str, str] = {
    QUERY_AGExHISPxCENRACE: 'v. age x hisp. x race',
    QUERY_CENRACE: LABEL_CENRACE,
    QUERY_DETAILED: LABEL_DETAILED,
    QUERY_HISPANIC: LABEL_HISPANIC,
    QUERY_HISPANICxCENRACE: LABEL_HISPANICxCENRACE,
    QUERY_HOUSING_TYPE: LABEL_HOUSING_TYPE,
    QUERY_TOTAL: LABEL_TOTAL,
    QUERY_VOTINGAGE: LABEL_VOTINGAGE,
}

# Ridgelines are drawn for every level below US; x-axis bounds per level.
RIDGELINE_LEVELS: StringTuple = tuple(l for l in ALL_LEVELS if l != LEVEL_US)
RIDGELINE_X_BOUNDS: dict[str, tuple[float, float]] = {
    LEVEL_STATE: (-40.0, 2500.0),
    LEVEL_COUNTY: (-20.0, 500.0),
    LEVEL_TRACT: (-5.0, 200.0),
    LEVEL_BLOCK_GROUP: (-2.0, 175.0),
    LEVEL_BLOCK: (-2.0, 40.0),
}

# Summary tables: which queries go in which table, and display names for
# levels and queries in the LaTeX output.
SUMMARY_TABLE_QUERIES: tuple[StringTuple, ...] = (
    (QUERY_TOTAL, QUERY_HOUSING_TYPE, QUERY_VOTINGAGE, QUERY_HISPANIC),
    (
        QUERY_CENRACE,
        QUERY_HISPANICxCENRACE,
        QUERY_AGExHISPxCENRACE,
        QUERY_DETAILED,
    ),
)
TABLE_NAMES: dict[str, str] = {
    LEVEL_US: 'US',
    LEVEL_STATE: 'State',
    LEVEL_COUNTY: 'County',
    LEVEL_TRACT: 'Tract',
    LEVEL_BLOCK_GROUP: 'Block group',
    LEVEL_BLOCK: 'Block',
    QUERY_TOTAL: 'Total',
    QUERY_VOTINGAGE: 'Voting age',
    QUERY_HISPANIC: 'Hispanic',
    QUERY_CENRACE: 'Race',
    QUERY_HISPANICxCENRACE: 'Hispanic x race',
    QUERY_DETAILED: 'Detailed',
    QUERY_HOUSING_TYPE: 'Housing type',
    QUERY_AGExHISPxCENRACE: 'V x H x R',
}

# Image formats accepted by save_figure. Non-image entries in --formats (such
# as 'csv') are ignored here and handled by the caller.
IMAGE_FORMATS = frozenset(('pdf', 'png', 'svg', 'eps', 'jpg', 'jpeg'))


def save_figure(
    fig: matplotlib.figure.Figure,
    output_dir: str,
    filename_stem: str,
    formats: Sequence[str],
) -> None:
  """Saves a figure to disk in all requested image formats and closes it.

  Args:
    fig: Matplotlib Figure object to save.
    output_dir: Directory path where files will be created.
    filename_stem: Base filename without extension (e.g. 'County_queries').
    formats: Sequence of file format extensions (e.g. ['pdf', 'png']). Entries
      not in IMAGE_FORMATS are skipped.
  """
  for fmt in formats:
    if fmt in IMAGE_FORMATS:
      out_path = os.path.join(output_dir, f'{filename_stem}.{fmt}')
      with open(out_path, 'wb') as out_f:
        fig.savefig(out_f, format=fmt, bbox_inches='tight')
      logging.info('Saved figure to %s', out_path)
  plt.close(fig)


def resolve_levels(levels_flag: Sequence[str]) -> list[str]:
  """Resolves the list of geographic levels to process from flag values.

  Args:
    levels_flag: Sequence of level names, or ['all'] for every level.

  Returns:
    List of matched geographic level strings in hierarchical order (the order
    of ALL_LEVELS).
  """
  if FLAG_ALL in levels_flag:
    return list(ALL_LEVELS)
  return [lvl for lvl in ALL_LEVELS if lvl in levels_flag]


def level_queries(level: str) -> tuple[list[str], list[str]]:
  """Returns the standard queries and their labels to plot for a level.

  Args:
    level: Geographic level name.

  Returns:
    A tuple of (queries, labels) from STANDARD_LEVEL_QUERIES /
    STANDARD_LEVEL_QUERY_LABELS, omitting TOTAL for levels in
    LEVELS_WITH_TRIVIAL_TOTAL.
  """
  pairs = zip(STANDARD_LEVEL_QUERIES, STANDARD_LEVEL_QUERY_LABELS)
  if level in LEVELS_WITH_TRIVIAL_TOTAL:
    pairs = (p for p in pairs if p[0] != QUERY_TOTAL)
  queries, labels = zip(*pairs)
  return list(queries), list(labels)


def detailed_queries_stem(spec_name: str) -> str:
  """Returns the output filename stem for a detailed query spec.

  Args:
    spec_name: Human-readable DetailedQuerySpec.name (e.g. 'Single race,
      County').

  Returns:
    DETAILED_QUERIES_STEM filled with a filesystem-safe, lower-cased version
    of the name (spaces and '/' become '_', commas are dropped).
  """
  safe = spec_name.replace(' ', '_').replace(',', '').replace('/', '_').lower()
  return DETAILED_QUERIES_STEM.format(name=safe)


@dataclasses.dataclass(frozen=True)
class DetailedQuerySpec:
  """Specification for a detailed summary query comparison.

  Attributes:
    name: Identifier name for the detailed query group.
    level: Geographic level name.
    measurement_names: Display names for the measurements.
    query_keys: Keys identifying each query.
    x_label: Optional x-axis label for the generated figure; defaults to
      name.lower().
    show_legend: Whether to render the legend on the generated figure.
  """

  name: str
  level: str
  measurement_names: StringTuple
  query_keys: StringTuple
  x_label: str | None = None
  show_legend: bool = True


def extract_query_errors(
    df: pd.DataFrame,
    level: str,
    queries: Sequence[str],
) -> tuple[list[list[float]], list[list[float]]]:
  """Extracts per-query replicate errors for one geographic level.

  Args:
    df: Replicate summary DataFrame containing columns: * 'level' (str):
      Geographic level name. * 'query' (str): Query name or key. * 'optimized'
      (list[float]): Error values across replicates. * 'baseline' (list[float]):
      Error values across replicates.
    level: Geographic level to select.
    queries: Query names to extract, in output order.

  Returns:
    A tuple of (optimized_errors, baseline_errors), each a list with one entry
    per query holding that query's replicate errors as a list of floats.

  Raises:
    ValueError: If any query is not present for the given level.
  """
  level_df = df[df[COL_LEVEL] == level]
  optimized = []
  baseline = []
  for query in queries:
    row = level_df[level_df[COL_QUERY] == query]
    if row.empty:
      raise ValueError(
          f'Query {query!r} not found for level {level!r} in DataFrame.'
      )
    optimized.append(list(row[COL_OPTIMIZED].iloc[0]))
    baseline.append(list(row[COL_BASELINE].iloc[0]))
  return optimized, baseline


def normalize_by_baseline_median(
    optimized: Sequence[Sequence[float]],
    baseline: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]]]:
  """Normalizes paired error lists by the median baseline error per query.

  Args:
    optimized: Replicate errors for the optimized method, one list per query.
    baseline: Replicate errors for the baseline method, one list per query.

  Returns:
    A tuple of (optimized_normalized, baseline_normalized) where each value is
    divided by the median baseline error of its query.

  Raises:
    ValueError: If any baseline error list is empty or has a non-positive
      median.
  """
  medians = []
  for idx, b in enumerate(baseline):
    if not b:
      raise ValueError(f'Baseline errors for query index {idx} are empty.')
    m = float(np.median(b))
    if m <= 0.0:
      raise ValueError(
          f'Cannot normalize by non-positive baseline median ({m}) for query'
          f' index {idx}.'
      )
    medians.append(m)

  opt_norm = [[x / m for x in errs] for errs, m in zip(optimized, medians)]
  base_norm = [[x / m for x in errs] for errs, m in zip(baseline, medians)]
  return opt_norm, base_norm


# Levels at which the detailed (race / group-quarters) queries are compared.
_DETAILED_LEVELS_RACE: StringTuple = (LEVEL_STATE, LEVEL_COUNTY, LEVEL_TRACT)
_DETAILED_LEVELS_GQ: StringTuple = (
    LEVEL_STATE,
    LEVEL_COUNTY,
    LEVEL_TRACT,
    LEVEL_BLOCK_GROUP,
    LEVEL_BLOCK,
)
_RACE_ALONE_NAMES: StringTuple = (
    'White alone',
    'Black alone',
    'AIAN alone',
    'Asian alone',
    'NHPI alone',
    'SOR alone',
    'Two or more races',
)
_RACE_COMBINED_NAMES: StringTuple = (
    'White',
    'Black',
    'AIAN',
    'Asian',
    'NHPI',
    'SOR',
)
_GQ_TYPE_NAMES: StringTuple = ('HH', 'CF', 'JF', 'NF', 'OI', 'SH', 'MQ', 'ON')
_GQ_GROUPED_NAMES: StringTuple = ('Institutionalized', 'Noninstitutionalized')


def standard_detailed_query_specs() -> list[DetailedQuerySpec]:
  """Returns the canonical detailed summary query specifications.

  Returns:
    DetailedQuerySpec objects for single race, race in combination, group
    quarters type, and grouped GQ population, each at the levels where that
    comparison is made. Spec names are unique across levels and are the basis
    of output filenames (see detailed_queries_stem).
  """
  specs = []
  for level in _DETAILED_LEVELS_RACE:
    display = LEVEL_DISPLAY_NAMES[level]
    specs.append(
        DetailedQuerySpec(
            name=f'Single race, {level}',
            level=level,
            measurement_names=_RACE_ALONE_NAMES,
            query_keys=tuple(
                f'Race_alone_{i}' for i in range(len(_RACE_ALONE_NAMES))
            ),
            x_label=f'Race queries, {display}',
            show_legend=level != LEVEL_COUNTY,
        )
    )
    specs.append(
        DetailedQuerySpec(
            name=(
                'Race alone or in combination with one or more other races,'
                f' {level}'
            ),
            level=level,
            measurement_names=_RACE_COMBINED_NAMES,
            query_keys=tuple(
                f'Race_combined_{i}' for i in range(len(_RACE_COMBINED_NAMES))
            ),
            x_label=f'Race in combination queries, {display}',
        )
    )
    specs.append(
        DetailedQuerySpec(
            name=(
                'Institutionalized and non-institutionalized GQ population,'
                f' {level}'
            ),
            level=level,
            measurement_names=_GQ_GROUPED_NAMES,
            query_keys=tuple(
                f'GQTYPE_GROUPED_{i}' for i in range(len(_GQ_GROUPED_NAMES))
            ),
            x_label=f'Grouped GQ population, {display}',
        )
    )
  for level in _DETAILED_LEVELS_GQ:
    specs.append(
        DetailedQuerySpec(
            name=f'Group quarters type, {level}',
            level=level,
            measurement_names=_GQ_TYPE_NAMES,
            query_keys=tuple(
                f'HOUSING_TYPE_{i}' for i in range(len(_GQ_TYPE_NAMES))
            ),
            x_label=f'housing type queries, {LEVEL_DISPLAY_NAMES[level]}',
            show_legend=level == LEVEL_STATE,
        )
    )
  return specs
