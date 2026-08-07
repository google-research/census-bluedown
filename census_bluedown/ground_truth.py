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

"""Generate aggregated ground truth from census microdata format."""

from collections.abc import Sequence
import os
import zipfile

import numpy as np
import pandas as pd

from census_bluedown import constants
from census_bluedown import io


GEOCODE16 = 'geocode16'
GEOCODE_PREFIX = 'geocode_prefix'
REDISTRICTING_FEATURE = 'redistricting_feature'
VALUE = constants.VALUE
ID = constants.ID


def onehot_encode(x: int) -> np.ndarray:
  """Encode an integer in [0, 2015] as one-hot vector of length 2016."""
  encoding = np.zeros(2016, dtype=int)
  encoding[x] = 1
  return encoding


def handle_water_block_patch(series: pd.Series) -> pd.Series:
  """Fix geocodes that are inconsistent due to the water block patch.

  Due to the water block patch described in paragraph 27 of the README file
  https://www2.census.gov/programs-surveys/decennial/2020/data/demographic-and-housing-characteristics-file/00-2020-DHC-Noisy-Measurement-File/2020_DHC_NMF_README.html.
  a few of the geocodes in the PPMF microdata are not found in the NMF. This
  function replaces those geocodes with the corresponding geocodes in the NMF.

  Args:
    series: A pandas Series containing the geocode16 values.

  Returns:
    A pandas Series containing the geocode16 values corrected for the water
    block patch.
  """
  return series.replace({
      '4502997080033004': '4502999010000001',
      '4505395030244009': '4505399010000002',
      '5306104010044007': '5306199000200017',
      '4816772390033135': '4816799010000039',
      '0607501790311031': '0607599020000001'})


def aggregate_microdata_format(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Aggregate the microdata format census data.

  Arguments:
    df: A pandas dataframe containing the census microdata formatted as strings.
      The dataframe should contain the following columns:
      - TABBLKST: The state code.
      - TABBLKCOU: The county code.
      - TABTRACTCE: The tract code.
      - TABBLKGRPCE: The group code.
      - TABBLK: The block code.
      - GQTYPE_PL: The household or group quarters type.
      - VOTING_AGE: Voting age feature.
      - CENHISP: Hispanic origin feature.
      - CENRACE: Race feature.

  Returns:
    A pandas dataframe containing the aggregated block-level census data. The
    dataframe will contain the following columns:
    - geocode16: The length-16 geocode, as type str. This is the index.
    - value: The redistricting feature value, as a length-2016 numpy array.
  """
  geo16_cols = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLKGRPCE', 'TABBLK']
  df[GEOCODE16] = handle_water_block_patch(df[geo16_cols].agg(''.join, axis=1))

  # Combine the four feature values into a single field.
  # The encoding from most-significant-bit to least-significant-bit is:
  #   hhgq: 0--7
  #   voting_age: 1--2
  #   hispanic: 1--2
  #   cenrace: 01--63
  # The range of the combined feature is 0--2015.
  hhgq = 'GQTYPE_PL'
  voting_age = 'VOTING_AGE'
  hispanic = 'CENHISP'
  cenrace = 'CENRACE'
  df[REDISTRICTING_FEATURE] = df.apply(
      lambda x: (
          (
              (
                  int(x[hhgq]) * 2
                  + int(x[voting_age]) - 1
              ) * 2
              + int(x[hispanic]) - 1
          ) * 63
          + int(x[cenrace]) - 1
      ), axis=1
  )

  # Encode the redistricting feature as a histogram so we can aggregate it.
  df[VALUE] = df[REDISTRICTING_FEATURE].apply(onehot_encode)

  # Return data aggregated over geocode.
  return df[[GEOCODE16, VALUE]].groupby(GEOCODE16).sum()


def read_state_microdata_format(
    state: str,
    zip_path: str,
    filename_suffix_pattern: str = '2020_ppmf_bystate/2020_ppmf_person_%s.csv'
) -> pd.DataFrame:
  """Load the state-level census data in microdata format.

  Args:
    state: The state code, as a length-2 string between '01' and '56'.
    zip_path: The path to the zip file containing the microdata format census
      data.
    filename_suffix_pattern: The pattern for the filename of the state-level
      census data in the zip file. The pattern should contain a single
      substitution slot for the state code.

  Returns:
    A pandas dataframe containing the state-level census data in microdata
    format.
  """
  ppmf_file = open(zip_path, mode='rb')
  with zipfile.ZipFile(ppmf_file, 'r') as z:
    filename = filename_suffix_pattern % state
    return pd.read_csv(z.open(filename), dtype=str)


def split_df_by_constraint_dfs(
    df: pd.DataFrame,
    constraint_dfs: Sequence[pd.DataFrame],
    suffix_length: int = 16
) -> list[pd.DataFrame]:
  """Split the aggregated microdata format data by subtree.

  Each geocode16 in aggregated_microdata_df should match a geocode in exactly
  one of the constraint_dfs. The records in aggregated_microdata_df are to be
  partitioned according to this split.

  In the census application, the constraint_dfs are the dataframes for the AIAN
  and non-AIAN subtrees for a single state.

  Args:
    df: The aggregated microdata format data, with index corresponding to the
      geocode16.
    constraint_dfs: The constraint dataframes, one for each subtree.
    suffix_length: The length of the suffix of the geocode to use for
      partitioning, i.e. 16 for geocode16.

  Returns:
    A list of dataframes, one for each subtree. Each dataframe will contain the
    aggregated microdata format data for that subtree and contains the full
    geocode in the ID column.
  """
  out_dfs = []
  df = df.reset_index()
  for constraint_df in constraint_dfs:
    constraint_df[GEOCODE16] = constraint_df[ID].str[-suffix_length:]
    constraint_df = constraint_df[
        constraint_df['query_name'] == 'hhgq_total_lb_con']
    subtree_geocodes = set(constraint_df[ID].str[-suffix_length:])
    mask = df[GEOCODE16].isin(subtree_geocodes)
    out_df = df[mask].copy()
    out_df = pd.merge(out_df, constraint_df[[GEOCODE16, ID]], how='left',
                      on=GEOCODE16)
    out_dfs.append(out_df[[ID, VALUE]])
  return out_dfs


def aggregate_query_values(row: pd.Series) -> np.ndarray:
  """Subroutine to aggregate all queries appropriately."""
  if row['query_name'] == 'detailed_dpq':
    return np.array(row['value'])

  shape = row['query_shape']
  effective_shape = shape
  if row['query_name'] == 'hhinstlevels_dpq':
    effective_shape = [8, 1, 1, 1]

  values = (np.reshape(row['value'], [8, 2, 2, 63])
            .sum(axis=tuple(np.flatnonzero(np.array(effective_shape) == 1)))
            .reshape(-1))

  if row['query_name'] == 'hhinstlevels_dpq':
    values = np.array([values[0],
                       values[1] + values[2] + values[3] + values[4],
                       values[5] + values[6] + values[7]])
  return values


def generate_single_subtree_ground_truth(
    subtree: str,
    df: pd.DataFrame,
    nmf_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    out_constraint_io: io.AbstractBlockHierarchicalIO,
) -> None:
  """Generate ground truth for a subtree (AIAN or non-AIAN region of state).

  Writes the ground truth values and constraint data to the output IO objects.

  Args:
    subtree: The subtree code, as a length-3 string in constants.FOLDER_IDS.
    df: The aggregated microdata format data for the subtree, with columns
      including ID and VALUE.
    nmf_io: The input IO object for the NMF data.
    constraint_io: The input IO object for the constraint data.
    out_io: The output IO object for the ground truth data.
    out_constraint_io: The output IO object for the constraint data.

  Returns:
    The overall histogram for the subtree.
  """
  nmf_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
  out_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
  constraint_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
  out_constraint_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree

  for level in reversed(constants.LEVELS):
    try:
      constraint_df = constraint_io.read(level, None)
      out_constraint_io.write(level, constants.CONSTRAINT_FNAME, constraint_df)
    except FileNotFoundError:
      pass

    try:
      nmf_df = nmf_io.read(level, None)
    except FileNotFoundError:
      continue
    df[GEOCODE_PREFIX] = df[ID].str[:len(nmf_df[ID].iloc[0])]
    df = df.groupby(GEOCODE_PREFIX).sum()
    df = df.reset_index()
    nmf_df = nmf_df.drop(VALUE, axis=1)
    out_df = pd.merge(nmf_df, df[[GEOCODE_PREFIX, VALUE]], how='left',
                      left_on=ID, right_on=GEOCODE_PREFIX)

    # Fill missing values with zeros.
    out_df[VALUE] = out_df[VALUE].apply(
        lambda x: [0] * 2016 if np.all(pd.isna(x)) else x
    )

    out_df[VALUE] = out_df.apply(aggregate_query_values, axis=1)
    out_df = out_df.drop(GEOCODE_PREFIX, axis=1)
    out_io.write(level, constants.GROUND_TRUTH_FNAME, out_df)


def generate_state_ground_truth(
    state: str,
    df: pd.DataFrame,
    nmf_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    out_constraint_io: io.AbstractBlockHierarchicalIO,
    subtree_patterns: Sequence[str] = constants.SUBTREE_GEOCODE_PATTERNS,
) -> None:
  """Generate ground truth for a single state.

  Args:
    state: The state code, as a length-2 string between '01' and '56'.
    df: The aggregated microdata format data for the state, with columns
      including ID and VALUE.
    nmf_io: The input IO object for the NMF data.
    constraint_io: The input IO object for the constraint data.
    out_io: The output IO object for the ground truth data.
    out_constraint_io: The output IO object for the constraint data.
    subtree_patterns: The patterns for the geocodes of the subtrees.

  Returns:
    The overall histogram for the state.
  """
  subtrees = set(x % state for x in subtree_patterns).intersection(
      constants.FOLDER_IDS
  )
  if len(subtrees) < 0 or len(subtrees) > 2:
    raise ValueError(
        f'Expected 1 or 2 subtrees for state {state}, got {len(subtrees)}.')

  constraint_dfs = []
  for subtree in subtrees:
    constraint_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
    constraint_dfs.append(
        constraint_io.read(level=constants.LEAF_LEVEL, filename=None)
    )

  subtree_dfs = split_df_by_constraint_dfs(df, constraint_dfs, suffix_length=16)

  for subtree, subtree_df in zip(subtrees, subtree_dfs):
    generate_single_subtree_ground_truth(
        subtree, subtree_df, nmf_io, constraint_io, out_io, out_constraint_io)


def generate_us_total_ground_truth(
    us_total: np.ndarray,
    nmf_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    out_constraint_io: io.AbstractBlockHierarchicalIO,
) -> None:
  """Write the US-level ground truth."""
  df = nmf_io.read(constants.ROOT_LEVEL, None)
  df[VALUE] = [us_total] * len(df)
  df[VALUE] = df.apply(aggregate_query_values, axis=1)
  out_io.write(constants.ROOT_LEVEL, constants.GROUND_TRUTH_FNAME, df)

  constraint_df = constraint_io.read(constants.ROOT_LEVEL, None)
  out_constraint_io.write(level=constants.ROOT_LEVEL,
                          filename=constants.CONSTRAINT_FNAME,
                          df=constraint_df)


