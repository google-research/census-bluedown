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

from collections.abc import Sequence
import itertools

import numpy as np
import pandas as pd

from census_bluedown import constants
from census_bluedown import io


VALUE = constants.VALUE
ID = constants.ID
ID_PREFIX = constants.ID_PREFIX
DIFFS = 'diffs'
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

QUERY_LAMBDAS = {
    query: (
        lambda x, query_name=query: np.reshape(x, (8, 2, 2, 63))
        .sum(axis=tuple(np.flatnonzero(np.array(QUERY_DICT[query_name]) == 1)))
        .reshape(-1)
    )
    for query in QUERY_DICT
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


def get_diffs(
    ground_truth_df: pd.DataFrame,
    base_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    level: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compute diffs between estimates at the specified level.

  Compute diffs between ground truth, baseline, and optimized estimates at the
  specified level.

  Args:
    ground_truth_df: The DataFrame containing the ground truth estimates.
    base_df: The DataFrame containing the baseline estimates.
    opt_df: The DataFrame containing the optimized estimates.
    level: The level of the data to compute diffs for.

  Returns:
    A tuple of DataFrames containing the diffs between baseline and ground truth
    and between optimized and ground truth, at the specified level.
  """
  # Generate geocode16
  ground_truth_df[ID_PREFIX] = ground_truth_df[ID].str[-16:]
  base_df[ID_PREFIX] = base_df[ID].str[-16:]
  opt_df[ID_PREFIX] = opt_df[ID].str[-16:]

  ground_truth_df[VALUE] = ground_truth_df[VALUE].apply(np.asarray)
  base_df[VALUE] = base_df[VALUE].apply(np.asarray)
  opt_df[VALUE] = opt_df[VALUE].apply(np.asarray)

  prefix = GEO16_PREFIX_LENGTH_BY_LEVEL[level]
  ground_truth_df[ID_PREFIX] = ground_truth_df[ID_PREFIX].str[:prefix]
  base_df[ID_PREFIX] = base_df[ID_PREFIX].str[:prefix]
  opt_df[ID_PREFIX] = opt_df[ID_PREFIX].str[:prefix]

  ground_truth_df = (
      ground_truth_df[[ID_PREFIX, VALUE]].groupby(ID_PREFIX).sum().reset_index()
  )
  base_df = base_df[[ID_PREFIX, VALUE]].groupby(ID_PREFIX).sum().reset_index()
  opt_df = opt_df[[ID_PREFIX, VALUE]].groupby(ID_PREFIX).sum().reset_index()

  combined_base_df = pd.merge(ground_truth_df, base_df, on=ID_PREFIX,
                              how='left', suffixes=('_g', '_b'))
  combined_opt_df = pd.merge(ground_truth_df, opt_df, on=ID_PREFIX,
                             how='left', suffixes=('_g', '_o'))

  combined_base_df[DIFFS] = (combined_base_df[VALUE + '_b'] -
                             combined_base_df[VALUE + '_g'])
  combined_opt_df[DIFFS] = (combined_opt_df[VALUE + '_o'] -
                            combined_opt_df[VALUE + '_g'])

  return combined_base_df[[DIFFS]], combined_opt_df[[DIFFS]]


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

  for level in reversed(constants.LEVELS):
    base_diffs, opt_diffs = get_diffs(ground_truth_df, base_df, opt_df, level)

    base_errs, opt_errs = {}, {}
    for df, errs in zip([base_diffs, opt_diffs], [base_errs, opt_errs]):
      for query in QUERY_DICT:
        df[query] = df[DIFFS].apply(QUERY_LAMBDAS[query])
        # Compute the total error for each row, and average across rows.
        errs[query] = df[query].abs().apply(lambda x: sum(x)).mean()  # pylint: disable=unnecessary-lambda

    base_errs_df = pd.DataFrame({'baseline': list(base_errs.values())},
                                index=list(base_errs.keys()))
    opt_errs_df = pd.DataFrame({'optimized': list(opt_errs.values())},
                               index=list(opt_errs.keys()))
    df_combined = pd.concat([base_errs_df, opt_errs_df], axis=1)
    df_combined = df_combined.assign(n=len(base_diffs[DIFFS]))
    processed_io.set_subtree_folder(
        constants.SUBTREE_FOLDER_PATTERN % folder_id)
    processed_io.write(level, constants.ERRORS_FNAME, df_combined)


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

  transformation_matrix = race_counts_combined_transformation_matrix()
  for level in reversed(constants.LEVELS):
    base_diffs, opt_diffs = get_diffs(ground_truth_df, base_df, opt_df, level)

    base_errs, opt_errs = {}, {}
    for df, errs in zip([base_diffs, opt_diffs], [base_errs, opt_errs]):
      # VOTINGAGE query: separately compute error for age < 18 and age >= 18
      # HISPANIC query: separately compute error for Hispanic and not Hispanic
      # HOUSING_TYPE query: separately compute error for each housing type
      for query, query_name in zip(
          [VOTINGAGE_QUERY, HISPANIC_QUERY, HOUSING_TYPE_QUERY],
          ['VOTINGAGE', 'HISPANIC', 'HOUSING_TYPE']):
        df[query] = df[DIFFS].apply(QUERY_LAMBDAS[query_name])
        errs[query] = df[query].abs().mean()

      # Race queries: compute error for each single-race option and for two or
      # more races. For encoding scheme (before converting 1--63 to 0--62), see
      # https://www2.census.gov/census_1940/2018-End-to-End-Test-Disclosure-Avoidance-System-Design-Specification.pdf
      query = CENRACE_ALONE_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['CENRACE']).apply(
          lambda x: np.append(x[:6], np.sum(x[6:]))
      )
      errs[query] = df[query].abs().mean()

      # Race combination query: compute error for each race, either alone or in
      # combination with the other races.
      query = CENRACE_COMBINED_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['CENRACE']).apply(
          lambda x: x @ transformation_matrix
      )
      errs[query] = df[query].abs().mean()

      # Race number query: compute error for number of individuals with each
      # possible number of races selected.
      query = CENRACE_NUMBER_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['CENRACE']).apply(
          lambda x: np.array([np.sum(arr)
                              for arr in np.split(x, [6, 21, 41, 56, 62])])
      )
      errs[query] = df[query].abs().mean()

      # HISP and race combination queries:
      query = HISP_CENRACE_ALONE_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['HISPANICxCENRACE']).apply(
          lambda x: x.reshape(2, 63)
      ).apply(
          lambda x: np.concatenate(
              [x[:, :6], np.sum(x[:, 6:], axis=1, keepdims=True)],
              axis=1
          ).flatten()
      )
      errs[query] = df[query].abs().mean()

      query = HISP_CENRACE_COMBINED_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['HISPANICxCENRACE']).apply(
          lambda x: x.reshape(2, 63)
      ).apply(
          lambda x: (x @ transformation_matrix).flatten()
      )
      errs[query] = df[query].abs().mean()

      query = HISP_CENRACE_NUMBER_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['HISPANICxCENRACE']).apply(
          lambda x: x.reshape(2, 63)
      ).apply(
          lambda x: np.column_stack(
              [np.sum(arr, axis=1)
               for arr in np.split(x, [6, 21, 41, 56, 62], axis=1)]).flatten()
      )
      errs[query] = df[query].abs().mean()

      # VOTINGAGE, HISP and race combination queries:
      query = VOTINGAGE_HISP_CENRACE_ALONE_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['AGExHISPxCENRACE']).apply(
          lambda x: x.reshape(4, 63)
      ).apply(
          lambda x: np.concatenate(
              [x[:, :6], np.sum(x[:, 6:], axis=1, keepdims=True)],
              axis=1
          ).flatten()
      )
      errs[query] = df[query].abs().mean()

      query = VOTINGAGE_HISP_CENRACE_COMBINED_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['AGExHISPxCENRACE']).apply(
          lambda x: x.reshape(4, 63)
      ).apply(
          lambda x: (x @ transformation_matrix).flatten()
      )
      errs[query] = df[query].abs().mean()

      query = VOTINGAGE_HISP_CENRACE_NUMBER_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['AGExHISPxCENRACE']).apply(
          lambda x: x.reshape(4, 63)
      ).apply(
          lambda x: np.column_stack(
              [np.sum(arr, axis=1)
               for arr in np.split(x, [6, 21, 41, 56, 62], axis=1)]).flatten()
      )
      errs[query] = df[query].abs().mean()

      # GQ grouped query: compute errors for number of invidiuals in
      # institutionalized and non-institutionalized group quarters facilities.
      # Coding scheme is described in
      # https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/complete-tech-docs/privacy-protected-microdata-file/2020census-privacy-protected-microdata-file.pdf
      query = GQ_GROUPED_QUERY
      df[query] = df[DIFFS].apply(QUERY_LAMBDAS['HOUSING_TYPE']).apply(
          lambda x: np.array([np.sum(x[1:5]), np.sum(x[5:])])
      )
      errs[query] = df[query].abs().mean()

    base_errs_df = pd.DataFrame({'baseline': list(base_errs.values())},
                                index=list(base_errs.keys()))
    opt_errs_df = pd.DataFrame({'optimized': list(opt_errs.values())},
                               index=list(opt_errs.keys()))
    df_combined = pd.concat([base_errs_df, opt_errs_df], axis=1)
    df_combined = df_combined.assign(n=len(base_diffs[DIFFS]))

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
