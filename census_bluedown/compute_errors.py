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

"""Error computation for block hierarchical postprocessing algorithm.
"""

from collections.abc import Iterator, Sequence
import itertools
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from census_bluedown import constants
from census_bluedown import io
from census_bluedown import query_encoding


VALUE = constants.VALUE
ID = constants.ID
ID_PREFIX = constants.ID_PREFIX
REGIONS_PER_LEVEL = constants.REGIONS_PER_LEVEL

GEO16_PREFIX_LENGTH_BY_LEVEL = {
    'Block': 16,
    'Block_Group': 12,
    'Tract': 11,
    'County': 5,
    'State': 2,
}

GEO_LENGTH_BY_LEVEL = {
    'Block': 31,
    'Block_Group': 15,
    'Tract': 13,
    'County': 9,
    'State': 3,
}

QUERY_DICT = {
    'TOTAL': (1, 1, 1, 1),
    'VOTINGAGE': (1, 2, 1, 1),
    'HISPANIC': (1, 1, 2, 1),
    'CENRACE': (1, 1, 1, 63),
    'HISPANICxCENRACE': (1, 1, 2, 63),
    'DETAILED': (8, 2, 2, 63),
    'HOUSING_TYPE': (8, 1, 1, 1),
    'AGExHISPxCENRACE': (1, 2, 2, 63),
}


# Names of a subset of detailed summary queries released by the Census Bureau.
# https://www.census.gov/programs-surveys/decennial-census/technical-documentation/complete-technical-documents.2020.html
# Detailed Summary Metrics available in the following spreadsheet:
# "2020 Census Production Disclosure Avoidance System Detailed Summary Metrics"
# https://www2.census.gov/programs-surveys/decennial/2020/data/demographic-and-housing-characteristics-file/2020-Census-Disclosure-Avoidance-System-Detailed-Summary-Metrics.xlsx
VOTINGAGE_QUERY = 'VOTINGAGE_vector'
HISPANIC_QUERY = 'HISP_vector'
CENRACE_ALONE_QUERY = 'Race_alone'  # Each option alone, or >= 2 races
CENRACE_COMBINED_QUERY = 'Race_combined'  # Each option, alone or in combination
CENRACE_NUMBER_QUERY = 'Race_number'  # Number of races selected
HISP_CENRACE_ALONE_QUERY = 'HISP_Race_alone'
HISP_CENRACE_COMBINED_QUERY = 'HISP_Race_combined'
HISP_CENRACE_NUMBER_QUERY = 'HISP_Race_number'
VOTINGAGE_HISP_CENRACE_ALONE_QUERY = 'VOTINGAGE_HISP_Race_alone'
VOTINGAGE_HISP_CENRACE_COMBINED_QUERY = 'VOTINGAGE_HISP_Race_combined'
VOTINGAGE_HISP_CENRACE_NUMBER_QUERY = 'VOTINGAGE_HISP_Race_number'
HOUSING_TYPE_QUERY = 'HOUSING_TYPE'
GQ_GROUPED_QUERY = 'GQTYPE_GROUPED'

ALTERNATE_QUERIES = (
    [f'{VOTINGAGE_QUERY}_{i}' for i in range(2)] +
    [f'{HISPANIC_QUERY}_{i}' for i in range(2)] +
    [f'{HOUSING_TYPE_QUERY}_{i}' for i in range(8)] +
    [f'{CENRACE_ALONE_QUERY}_{i}' for i in range(7)] +
    [f'{CENRACE_COMBINED_QUERY}_{i}' for i in range(6)] +
    [f'{CENRACE_NUMBER_QUERY}_{i}' for i in range(6)] +
    [f'{HISP_CENRACE_ALONE_QUERY}_{i}' for i in range(14)] +
    [f'{HISP_CENRACE_COMBINED_QUERY}_{i}' for i in range(12)] +
    [f'{HISP_CENRACE_NUMBER_QUERY}_{i}' for i in range(12)] +
    [f'{VOTINGAGE_HISP_CENRACE_ALONE_QUERY}_{i}' for i in range(28)] +
    [f'{VOTINGAGE_HISP_CENRACE_COMBINED_QUERY}_{i}' for i in range(24)] +
    [f'{VOTINGAGE_HISP_CENRACE_NUMBER_QUERY}_{i}' for i in range(24)] +
    [f'{GQ_GROUPED_QUERY}_{i}' for i in range(2)]
)


def get_combined_df(
    input_io: io.AbstractBlockHierarchicalIO,
    filename: str,
    state_id: str,
    level: str,
    value_column: str = VALUE,
) -> pd.DataFrame | None:
  """Returns a DataFrame for the specified state and level.

  This DataFrame combines the AIAN and non-AIAN regions of the state, if both
  exist. If only one region exists, the DataFrame for that region is returned.
  If neither region exists, returns None.

  Args:
    input_io: The input IO object.
    filename: The name of the file to read.
    state_id: The state ID.
    level: The level of the data to read.
    value_column: The name of the column containing the values.

  Returns:
    A DataFrame containing the combined data for the specified state and level.
  """
  dfs = []
  for pattern in constants.SUBTREE_GEOCODE_PATTERNS:
    folder_id = pattern % state_id
    input_io.set_subtree_folder(constants.SUBTREE_FOLDER_PATTERN % folder_id)
    if input_io.exists(level, filename):
      next_df = input_io.read(level, filename=filename)
      if filename == constants.GROUND_TRUTH_FNAME:
        next_df = next_df[next_df['query_name'] == 'detailed_dpq'].copy()
      dfs.append(next_df[[ID, value_column]])

  if len(dfs) == 1:
    return dfs[0]
  elif len(dfs) == 2:
    fill_value = 0 if value_column == VALUE else ''
    out_df = pd.merge(left=dfs[0], right=dfs[1], on=ID, how='outer',
                      suffixes=('_a', '_b'))
    out_df[value_column] = (
        out_df[value_column + '_a'].fillna(fill_value) +
        out_df[value_column + '_b'].fillna(fill_value))
    return out_df.drop(columns=[value_column + '_a', value_column + '_b'])
  else:
    return None


def get_complete_block_df(
    folder_id: str,
    input_io: io.AbstractBlockHierarchicalIO,
    filename: str,
    out_df: pd.DataFrame,
) -> pd.DataFrame:
  """Generate DataFrame for all Block-level geocodes in the specified subtree.
  """
  out_df['value'] = None

  for level in reversed(constants.LEVELS):

    next_df = get_combined_df(
        input_io=input_io,
        filename=filename,
        state_id=folder_id[1:],
        level=level)
    if next_df is None:
      continue

    prefix_length = GEO_LENGTH_BY_LEVEL[level]

    out_df[ID_PREFIX] = out_df[ID].str[:prefix_length]

    na_keys = out_df[out_df[VALUE].isna()][ID_PREFIX]  # pyrefly: ignore[missing-attribute]
    next_df = next_df[next_df[ID].isin(na_keys)]

    if not next_df[ID].is_unique:
      raise ValueError('Expected unique geocodes in after filtering; '
                       f'level {level}, state {folder_id}')

    out_df = out_df.set_index(ID_PREFIX)
    next_df = next_df.set_index(ID)

    out_df.update(next_df[VALUE], overwrite=False)
    out_df = out_df[[ID, VALUE]].reset_index()

  if out_df[VALUE].isna().any():
    raise ValueError('Expected all geocodes to have values after filtering; '
                     f'state {folder_id}')
  return out_df.drop(columns=[ID_PREFIX])


def get_complete_block_dfs(
    folder_id: str,
    ground_truth_io: io.AbstractBlockHierarchicalIO,
    processed_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Generate DataFrames for all Block-level geocodes in the specified subtree.

  Includes blocks that are omitted in the Block-level data files because there
  is only a single block in the corresponding Block_Group, Tract, or County.

  Args:
    folder_id: The state ID.
    ground_truth_io: The input IO object for the ground truth data.
    processed_io: The input IO object for the processed data.
    constraint_io: The input IO object for the constraint data, used to
      determine the set of all block geocodes for the subtree.

  Returns:
    A tuple of DataFrames containing the ground truth, baseline, and optimized
    estimates for all Block-level geocodes in the specified subtree.
  """
  df = get_combined_df(
      input_io=constraint_io,
      filename=constants.CONSTRAINT_FNAME,
      state_id=folder_id[1:],
      level=constants.LEAF_LEVEL,
      value_column='query_name')
  if df is None:
    raise ValueError(f'No constraint data found for state {folder_id}')

  df = df[df['query_name'] == 'hhgq_total_lb_con'][[ID]]
  ground_truth_df = get_complete_block_df(
      folder_id=folder_id,
      input_io=ground_truth_io,
      filename=constants.GROUND_TRUTH_FNAME,
      out_df=df.copy())
  base_df = get_complete_block_df(
      folder_id=folder_id,
      input_io=processed_io,
      filename=constants.BASELINE_EST_FNAME,
      out_df=df.copy())
  opt_df = get_complete_block_df(
      folder_id=folder_id,
      input_io=processed_io,
      filename=constants.OPTIMIZED_EST_FNAME,
      out_df=df.copy())
  return ground_truth_df, base_df, opt_df


def _histograms_by_geocode16(df: pd.DataFrame) -> pd.DataFrame:
  """Stacks the VALUE histograms of df into a numeric DataFrame.

  Args:
    df: A DataFrame with columns ID and VALUE, where each VALUE is a histogram
      and each ID is a block geocode.

  Returns:
    A DataFrame with one row per row of df and one numeric column per histogram
    entry, indexed by the geocode16 (last 16 characters of the ID).

  Raises:
    ValueError: If an ID does not have the block geocode length, or if two IDs
      share the same geocode16.
  """
  # The last 16 characters of the block ID are its Census GEOID or geocode_16
  # (state 2 + county 3 + tract 6 + block group 1 + block 4); the preceding 15
  # characters identify the node used specifically in the 2020 TopDown
  # encoding, including the AIAN/non-AIAN subtree and custom block groups.
  # Keying by the GEOID lets _aggregate_to_prefix combine the AIAN and non-AIAN
  # parts of the same geography.
  if (df[ID].str.len() != GEO_LENGTH_BY_LEVEL['Block']).any():
    raise ValueError(
        f'Expected block geocodes of length {GEO_LENGTH_BY_LEVEL["Block"]}.'
    )
  geocode16 = df[ID].str[-16:]
  if not geocode16.is_unique:
    raise ValueError('Expected unique geocode16 values.')
  return pd.DataFrame(
      np.stack(df[VALUE].to_numpy()), index=geocode16.to_numpy()
  )


def _aggregate_to_prefix(
    histograms: pd.DataFrame, prefix_length: int
) -> pd.DataFrame:
  """Sums the rows of histograms whose index labels share a prefix."""
  return histograms.groupby(histograms.index.str[:prefix_length]).sum()


def _diffs_from_aggregates(
    ground_truth: pd.DataFrame, base: pd.DataFrame, opt: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
  """Returns base and optimized minus ground truth.

  Args:
    ground_truth: Aggregated ground truth histograms, indexed by region.
    base: Aggregated baseline histograms, indexed by region.
    opt: Aggregated optimized histograms, indexed by region.

  Returns:
    A tuple of matrices with one row per region, containing the baseline and
    optimized diffs respectively.

  Raises:
    ValueError: If the three inputs are not indexed by the same regions.
  """
  if not (
      ground_truth.index.equals(base.index)
      and ground_truth.index.equals(opt.index)
  ):
    raise ValueError(
        'Expected ground truth, baseline, and optimized estimates for the same'
        ' regions.'
    )
  ground_truth_values = ground_truth.to_numpy()
  return (
      base.to_numpy() - ground_truth_values,
      opt.to_numpy() - ground_truth_values,
  )


class LevelDiffs(NamedTuple):
  """Ground truth and estimate errors for the regions at one level.

  Attributes:
    level: The name of the geographic level.
    ground_truth: Matrix with one row per region at the level, sorted by
      geocode, containing the ground truth detailed histogram.
    base_diffs: Matrix of the same shape containing the baseline estimates minus
      ground truth.
    opt_diffs: Matrix of the same shape containing the optimized estimates minus
      ground truth.
  """

  level: str
  ground_truth: np.ndarray
  base_diffs: np.ndarray
  opt_diffs: np.ndarray


def diffs_by_level(
    ground_truth_df: pd.DataFrame,
    base_df: pd.DataFrame,
    opt_df: pd.DataFrame,
) -> Iterator[LevelDiffs]:
  """Yields ground truth and diffs for each level, from leaf to root.

  Since the geocode16 prefixes for successive levels are nested, each level is
  aggregated from the previous (finer) level rather than from the blocks.

  Args:
    ground_truth_df: The DataFrame containing the ground truth estimates.
    base_df: The DataFrame containing the baseline estimates.
    opt_df: The DataFrame containing the optimized estimates. All three
      DataFrames must contain the same set of geocodes.

  Yields:
    A LevelDiffs for each level.
  """
  ground_truth, base, opt = (
      _histograms_by_geocode16(df) for df in (ground_truth_df, base_df, opt_df)
  )
  for level in reversed(constants.LEVELS):
    prefix = GEO16_PREFIX_LENGTH_BY_LEVEL[level]
    ground_truth, base, opt = (
        _aggregate_to_prefix(df, prefix) for df in (ground_truth, base, opt)
    )
    base_diffs, opt_diffs = _diffs_from_aggregates(
        ground_truth=ground_truth, base=base, opt=opt
    )
    yield LevelDiffs(level, ground_truth.to_numpy(), base_diffs, opt_diffs)


def query_errors(diffs: np.ndarray) -> dict[str, float]:
  """Computes the mean total absolute error of each query in QUERY_DICT.

  Args:
    diffs: Matrix with one row per region, containing the estimate minus the
      ground truth for each detailed histogram entry.

  Returns:
    A dictionary mapping each query name to the total absolute error of its
    projected entries, averaged across regions.
  """
  return {
      query: float(
          np.abs(query_encoding.project_to_query_shape(diffs, query_shape))
          .sum(axis=1)
          .mean()
      )
      for query, query_shape in QUERY_DICT.items()
  }


def compute_errors(
    folder_id: str,
    ground_truth_io: io.AbstractBlockHierarchicalIO,
    processed_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
):
  """Error computation for the block hierarchical postprocessing algorithm.

  Postprocesses each input block individually in the specified subtree.

  Args:
    folder_id: The state ID.
    ground_truth_io: The input IO object for the ground truth data.
    processed_io: The input IO object for the processed data.
    constraint_io: The input IO object for the constraint data, used to
      determine the set of all block geocodes for the subtree.
  """
  if folder_id[0] == '1':
    # AIAN region errors will be computed along with the corresponding non-AIAN
    # region.
    return

  ground_truth_df, base_df, opt_df = get_complete_block_dfs(
      folder_id=folder_id,
      ground_truth_io=ground_truth_io,
      processed_io=processed_io,
      constraint_io=constraint_io)

  for diffs in diffs_by_level(ground_truth_df, base_df, opt_df):
    df_combined = _errors_dataframe(
        query_errors(diffs.base_diffs),
        query_errors(diffs.opt_diffs),
        len(diffs.base_diffs),
    )
    processed_io.set_subtree_folder(
        constants.SUBTREE_FOLDER_PATTERN % folder_id)
    processed_io.write(diffs.level, constants.ERRORS_FNAME, df_combined)


def _int_list_str(arr: np.ndarray) -> str:
  return str(arr.astype(int).tolist())


def errors_list_dataframe(
    ground_truth: np.ndarray,
    base_diffs: np.ndarray,
    opt_diffs: np.ndarray,
) -> pd.DataFrame:
  """Computes per-region error, bias, and ground truth lists for each query."""
  query_err_lists = lambda diffs: {
      query: _int_list_str(
          np.abs(query_encoding.project_to_query_shape(diffs, query_shape)).sum(
              axis=1
          )
      )
      for query, query_shape in QUERY_DICT.items()
  }
  return _errors_dataframe(
      query_err_lists(base_diffs),
      query_err_lists(opt_diffs),
      len(base_diffs),
  ).assign(
      base_bias=_int_list_str(base_diffs.sum(axis=1)),
      opt_bias=_int_list_str(opt_diffs.sum(axis=1)),
      ground_truth=_int_list_str(ground_truth.sum(axis=1)),
  )


def compute_errors_list(
    folder_id: str,
    ground_truth_io: io.AbstractBlockHierarchicalIO,
    processed_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
):
  """Computes per-region error lists across geographic units at each level."""
  if folder_id[0] == '1':
    # AIAN region errors will be computed along with the corresponding non-AIAN
    # region.
    return

  ground_truth_df, base_df, opt_df = get_complete_block_dfs(
      folder_id=folder_id,
      ground_truth_io=ground_truth_io,
      processed_io=processed_io,
      constraint_io=constraint_io,
  )

  for diffs in diffs_by_level(ground_truth_df, base_df, opt_df):
    df_combined = errors_list_dataframe(
        diffs.ground_truth, diffs.base_diffs, diffs.opt_diffs
    )
    processed_io.set_subtree_folder(
        constants.SUBTREE_FOLDER_PATTERN % folder_id
    )
    processed_io.write(diffs.level, constants.ERRORS_LIST_FNAME, df_combined)


def race_counts_combined_transformation_matrix(
    num_races: int = 6,
) -> np.ndarray:
  """Produces a transformation matrix for aggregating per-race counts.

  Returns a 63 x 6 matrix applying a transformation to count how many
  individuals belong to each possible primary race category (White, Black, AIAN,
  Asian, NHPI, SOR), either alone or in combination with the other races.

  Args:
    num_races: The number of primary race categories.

  Returns:
    A 63 x 6 numpy array, or more generally a (2^num_races - 1) x num_races
    numpy array.
  """
  combos = list(
      itertools.chain.from_iterable(
          itertools.combinations(range(num_races), r)
          for r in range(1, num_races + 1)
      )
  )

  transformation_matrix = np.zeros((len(combos), num_races), dtype=int)
  for row, combo in enumerate(combos):
    transformation_matrix[row, list(combo)] = 1

  return transformation_matrix


def alternate_query_errors(diffs: np.ndarray) -> dict[str, np.ndarray]:
  """Computes the elementwise mean absolute error of each detailed summary metric.

  Args:
    diffs: Matrix with one row per region, containing the estimate minus the
      ground truth for each detailed histogram entry.

  Returns:
    A dictionary mapping each metric name (VOTINGAGE_QUERY, HISPANIC_QUERY,
    HOUSING_TYPE_QUERY, the race queries, and GQ_GROUPED_QUERY) to the array of
    mean absolute errors of its entries, averaged across regions.
  """
  transformation_matrix = race_counts_combined_transformation_matrix()

  def mean_abs(x: np.ndarray) -> np.ndarray:
    # Elementwise mean absolute error across regions (rows).
    return np.abs(x).mean(axis=0)

  num_regions = len(diffs)
  errs = {}
  # VOTINGAGE query: separately compute error for age < 18 and age >= 18
  # HISPANIC query: separately compute error for Hispanic and not Hispanic
  # HOUSING_TYPE query: separately compute error for each housing type
  for query, query_name in zip(
      [VOTINGAGE_QUERY, HISPANIC_QUERY, HOUSING_TYPE_QUERY],
      ['VOTINGAGE', 'HISPANIC', 'HOUSING_TYPE'],
  ):
    errs[query] = mean_abs(
        query_encoding.project_to_query_shape(diffs, QUERY_DICT[query_name])
    )

  # Race queries, alone and crossed with HISP and with VOTINGAGE x HISP.
  # Each projection is reshaped to (regions, groups, 63), and the race
  # transformations are applied along the last axis.
  for alone_query, combined_query, number_query, query_name, groups in [
      (
          CENRACE_ALONE_QUERY,
          CENRACE_COMBINED_QUERY,
          CENRACE_NUMBER_QUERY,
          'CENRACE',
          1,
      ),
      (
          HISP_CENRACE_ALONE_QUERY,
          HISP_CENRACE_COMBINED_QUERY,
          HISP_CENRACE_NUMBER_QUERY,
          'HISPANICxCENRACE',
          2,
      ),
      (
          VOTINGAGE_HISP_CENRACE_ALONE_QUERY,
          VOTINGAGE_HISP_CENRACE_COMBINED_QUERY,
          VOTINGAGE_HISP_CENRACE_NUMBER_QUERY,
          'AGExHISPxCENRACE',
          4,
      ),
  ]:
    races = query_encoding.project_to_query_shape(
        diffs, QUERY_DICT[query_name]
    ).reshape(num_regions, groups, 63)

    # The 63 CENRACE codes enumerate the nonempty subsets of the 6 OMB race
    # categories (White, Black, AIAN, Asian, NHPI, SOR), ordered by subset
    # size and then lexicographically; see the encoding scheme (for codes
    # 1--63, before converting to indices 0--62) in
    # https://www2.census.gov/census_1940/2018-End-to-End-Test-Disclosure-Avoidance-System-Design-Specification.pdf
    # Indices 0--5 are therefore the six single-race codes and indices 6--62
    # are all multi-race combinations.

    # Race alone query: compute error for each single race alone and for
    # "two or more races", i.e. the sum over indices 6--62.
    alone = np.concatenate(
        [races[..., :6], races[..., 6:].sum(axis=-1, keepdims=True)],
        axis=-1,
    )
    errs[alone_query] = mean_abs(alone.reshape(num_regions, -1))

    # Race combination query: compute error for each race, either alone or
    # in combination with the other races.
    combined = races @ transformation_matrix
    errs[combined_query] = mean_abs(combined.reshape(num_regions, -1))

    # Race number query: compute error for number of individuals with each
    # possible number of races selected (1 through 6). Since codes are
    # ordered by subset size, the codes with k races form a contiguous
    # range of length C(6, k): 6, 15, 20, 15, 6, 1, so splitting at the
    # cumulative boundaries [6, 21, 41, 56, 62] groups them.
    number = np.stack(
        [
            arr.sum(axis=-1)
            for arr in np.split(races, [6, 21, 41, 56, 62], axis=-1)
        ],
        axis=-1,
    )
    errs[number_query] = mean_abs(number.reshape(num_regions, -1))

  # GQ grouped query: compute errors for number of invidiuals in
  # institutionalized and non-institutionalized group quarters facilities.
  # Coding scheme is described in
  # https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/complete-tech-docs/privacy-protected-microdata-file/2020census-privacy-protected-microdata-file.pdf
  # These are the last two bins of the HHINSTLEVELS query, whose first bin
  # is households.
  hhinstlevels = query_encoding.project_to_query_shape(
      diffs, constants.CENSUS_ASYMMETRIC_QUERY_SHAPE
  )
  errs[GQ_GROUPED_QUERY] = mean_abs(hhinstlevels[:, 1:])
  return errs


def _errors_dataframe(
    base_errs: dict[str, Any], opt_errs: dict[str, Any], num_regions: int
) -> pd.DataFrame:
  """Combines baseline and optimized errors into one DataFrame.

  Args:
    base_errs: Baseline errors keyed by query name.
    opt_errs: Optimized errors keyed by the same query names.
    num_regions: The number of regions the errors were averaged over.

  Returns:
    A DataFrame indexed by query name with columns 'baseline', 'optimized',
    and 'n' (the number of regions).
  """
  base_errs_df = pd.DataFrame(
      {'baseline': list(base_errs.values())}, index=list(base_errs.keys())
  )
  opt_errs_df = pd.DataFrame(
      {'optimized': list(opt_errs.values())}, index=list(opt_errs.keys())
  )
  df_combined = pd.concat([base_errs_df, opt_errs_df], axis=1)
  return df_combined.assign(n=num_regions)


def compute_alternate_errors(
    folder_id: str,
    ground_truth_io: io.AbstractBlockHierarchicalIO,
    processed_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
):
  """Compute detailed summary metrics released by the Census Bureau.

  Postprocesses each input block individually in the specified subtree.

  Args:
    folder_id: The state ID.
    ground_truth_io: The input IO object for the ground truth data.
    processed_io: The input IO object for the processed data.
    constraint_io: The input IO object for the constraint data, used to
      determine the set of all block geocodes for the subtree.
  """
  if folder_id[0] == '1':
    # AIAN region errors will be computed along with the corresponding non-AIAN
    # region.
    return

  ground_truth_df, base_df, opt_df = get_complete_block_dfs(
      folder_id=folder_id,
      ground_truth_io=ground_truth_io,
      processed_io=processed_io,
      constraint_io=constraint_io)

  for level, _, base_diffs, opt_diffs in diffs_by_level(
      ground_truth_df, base_df, opt_df
  ):
    df_combined = _errors_dataframe(
        alternate_query_errors(base_diffs),
        alternate_query_errors(opt_diffs),
        len(base_diffs),
    )

    # Unpack columns for writing
    df_combined.index.name = 'index'
    df_combined = df_combined.explode(column=['baseline', 'optimized'])
    group_counter = df_combined.groupby('index').cumcount()
    new_index = df_combined.index + '_' + group_counter.astype(str)
    df_combined.index = new_index

    processed_io.set_subtree_folder(
        constants.SUBTREE_FOLDER_PATTERN % folder_id)
    processed_io.write(level, constants.ALTERNATE_ERRORS_FNAME, df_combined)


def aggregate_errors(
    processed_io: io.AbstractBlockHierarchicalIO,
    queries: Sequence[str] | None = None,
    file_name: str | None = None,
):
  """Aggregate errors across all states.

  Compute the mean error for each query across all states, and write the
  results to a single DataFrame at the root level.

  Args:
    processed_io: The input IO object for the processed data.
    queries: The names of the queries to aggregate, or None for default.
    file_name: The name of the file to read and write the aggregated errors, or
      None for default.

  Raises:
    FileNotFoundError: If no per-state error files are found at some level.
  """
  if queries is None:
    queries = QUERY_DICT.keys()  # pyrefly: ignore[bad-assignment]
  if file_name is None:
    file_name = constants.ERRORS_FNAME
  combined_dfs = {}
  for level in constants.LEVELS:
    error_dfs = []
    for folder_id in constants.FOLDER_IDS:
      if folder_id[0] == '1':
        continue
      processed_io.set_subtree_folder(
          constants.SUBTREE_FOLDER_PATTERN % folder_id)
      if processed_io.exists(level, file_name):
        next_df = processed_io.read(level, file_name)
        error_dfs.append(next_df)
    if not error_dfs:
      raise FileNotFoundError(
          f'No {file_name} files found at level {level} in'
          f' {processed_io.path_prefix}.'
      )

    regions_at_level = REGIONS_PER_LEVEL[level]
    base_errs = {}
    opt_errs = {}
    for query in queries:  # pyrefly: ignore[not-iterable]
      base_errs[query] = (
          sum(df.loc[query]['baseline'] *
              df.loc[query]['n'] for df in error_dfs) / regions_at_level)
      opt_errs[query] = (
          sum(df.loc[query]['optimized'] *
              df.loc[query]['n'] for df in error_dfs) / regions_at_level)
    base_errs_df = pd.DataFrame.from_dict(base_errs, orient='index',
                                          columns=['baseline'])
    opt_errs_df = pd.DataFrame.from_dict(opt_errs, orient='index',
                                         columns=['optimized'])
    combined_dfs[level] = pd.concat([base_errs_df, opt_errs_df], axis=1)
  combined_df = pd.concat(combined_dfs, ignore_index=False)
  combined_df = combined_df.reset_index()
  combined_df = combined_df.rename(columns={'level_0': 'level',
                                            'level_1': 'query'})
  processed_io.write(constants.ROOT_LEVEL, file_name, combined_df)
