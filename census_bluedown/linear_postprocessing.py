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

"""Best unbiased linear estimator for histograms on blocks of estimates.

This generalizes the algorithm described in this paper to vectors of queries:
https://ceur-ws.org/Vol-3556/adkdd23-dawson-optimizing-ceur-paper.pdf

The input and output data for this algorithm consists of separate DataFrames for
each level of the hierarchy. The input DataFrames for the block-level processing
method contains one row for each combination of block ID and query shape, and
consists of columns ID, QUERY_SHAPE, VARIANCE, and VALUE, with types string,
tuple[int], numpy.ndarray, and numpy.ndarray, respectively. The input and output
DataFrames for the bottom-up and top-down processing methods, as well as the
output DataFrame for the block-level processing method, contain a single row for
each geo ID and consist of columns ID and ESTIMATE, with types string and
estimate.Estimate, respectively.
"""

import pandas as pd
from census_bluedown import block
from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import regression


ID = constants.ID
ID_PREFIX = constants.ID_PREFIX
ESTIMATE = constants.ESTIMATE
QUERY_SHAPE = constants.QUERY_SHAPE
VARIANCE = constants.VARIANCE
VALUE = constants.VALUE


def _process_df_block(
    block_shape: block.BlockShape,
    df: pd.DataFrame,
) -> estimate.Estimate:
  """Postprocesses the block in the input dataframe."""
  query_shapes = tuple(df[QUERY_SHAPE])
  query_variances = tuple(df[VARIANCE])
  query_values = tuple(df[VALUE])

  return estimate.Estimate(*regression.process_queries(
      block_shape, query_shapes, query_variances, query_values
  ))


def process_blocks(
    block_shape: block.BlockShape,
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Postprocess each block in the input dataframe.

  For each ID, compute the best linear unbiased estimate for the corresponding
  block given only the queries on that block.

  Args:
    block_shape: The BlockShape object describing the block structure.
    df: A DataFrame with columns ID, QUERY_SHAPE, VARIANCE, VALUE.

  Returns:
    A DataFrame with columns ID and ESTIMATE, with one row for each block ID.
    The ESTIMATE column contains the best linear unbiased estimate for the
    block given only the queries on that block.
  """
  return df.groupby(ID).apply(
      lambda x: _process_df_block(block_shape=block_shape, df=x)
  ).reset_index(name=ESTIMATE)  # pyrefly: ignore[no-matching-overload]


def _aggregate_child_estimates(
    children_df: pd.DataFrame,
    parents_df: pd.DataFrame,
    parent_suffix: str,
    child_suffix: str,
) -> pd.DataFrame:
  """Aggregate the child estimates by parent ID.

  Args:
    children_df: A DataFrame with columns ID and ESTIMATE.
    parents_df: A DataFrame with columns ID and ESTIMATE.
    parent_suffix: The suffix to append to the parent estimate column in the
      output DataFrame.
    child_suffix: The suffix to append to the child estimate column in the
      output DataFrame.

  Returns:
    A DataFrame with columns ID, ESTIMATE + parent_suffix, and
    ESTIMATE + child_suffix. The ESTIMATE + parent_suffix column contains the
    parent estimate. The ESTIMATE + child_suffix column contains the sum of the
    child estimates of the children of the parent.
  """
  # Parent ID is a fixed-length prefix of the child ID.
  parent_id_length = len(parents_df[ID][0])
  children_df[ID_PREFIX] = children_df[ID].str[:parent_id_length]
  children_grouped = (
      children_df[[ID_PREFIX, ESTIMATE]]
      .groupby(ID_PREFIX)  # Group by parent ID
      .agg('sum')          # Aggregate child estimates per parent
      .reset_index()
      .rename(columns={ID_PREFIX: ID}))

  return pd.merge(parents_df, children_grouped, how='outer', on=ID,
                  suffixes=(parent_suffix, child_suffix))


def bottom_up_step(
    block_shape: block.BlockShape,
    children_lower_df: pd.DataFrame,
    parents_input_df: pd.DataFrame,
) -> pd.DataFrame:
  """Bottom up step of linear postprocessing algorithm.

  This step takes the lower inclusive estimates of the children and the initial
  estimates of the parents and outputs the lower inclusive estimates of the
  parents. The lower inclusive estimate is the best linear unbiased estimate
  given only the queries on a block and its descendant blocks. It is computed by
  taking (1) the input estimate of a block and (2) the sum of the optimal lower
  inclusive estimates of its children, and taking the optimal combination of
  these two estimates.

  Args:
    block_shape: The BlockShape object describing the block structure.
    children_lower_df: A DataFrame with columns ID and ESTIMATE containing lower
      inclusive estimates of the children.
    parents_input_df: A DataFrame with columns ID and ESTIMATE containing
      initial estimates of the parents.

  Returns:
    A DataFrame with columns ID and ESTIMATE. The ESTIMATE column contains the
    inclusive lower estimates of the parents, which are the best linear unbiased
    estimates given only the queries on a block and its descendants.
  """
  # Aggregate child estimates by parent ID.
  parent_suffix = '_parent'
  child_lower_sum_suffix = '_child_lower_sum'
  parents_df = _aggregate_child_estimates(
      children_df=children_lower_df, parents_df=parents_input_df,
      parent_suffix=parent_suffix, child_suffix=child_lower_sum_suffix,
  )

  # Combine aggregated child estimates with parent estimates of the same ID.
  lower_estimate_column = 'lower_estimate'
  parents_df[lower_estimate_column] = parents_df.apply(
      lambda x: regression.combine(
          block_shape,
          x[ESTIMATE + child_lower_sum_suffix],
          x[ESTIMATE + parent_suffix]
      ),
      axis=1
    )
  parents_lower_df = (parents_df[[ID, lower_estimate_column]]
                      .rename(columns={lower_estimate_column: ESTIMATE}))
  return parents_lower_df


def _compute_upper_exclusive_estimate(
    children_df: pd.DataFrame,
    parents_df: pd.DataFrame,
    parent_estimate_column: str,
    child_lower_sum_estimate_column: str,
    lower_estimate_column: str,
    upper_exclusive_estimate_column: str,
) -> pd.DataFrame:
  """Compute the upper exclusive estimate for each child.

  The input DataFrame children_df consists of two columns, ID and ESTIMATE. The
  input DataFrame parents_df consists of three columns, ID,
  parent_estimate_column, and child_lower_sum_estimate_column. The output
  DataFrame consists of three columns, ID, lower_estimate_column (which is equal
  to the ESTIMATE column of the input children_df), and
  upper_exclusive_estimate_column, which is computed by taking the parent
  estimate and subtracting the sum of the lower estimates of the siblings.
  This sum of the lower estimates of the siblings is in turn computed by taking
  the sum of lower estimates of all children of the parent and subtracting the
  lower estimate of the given child.

  Args:
    children_df: A DataFrame with columns ID and ESTIMATE.
    parents_df: A DataFrame with columns ID, parent_estimate_column, and
      child_lower_sum_estimate_column.
    parent_estimate_column: The name of the column in parents_df containing the
      parent estimate.
    child_lower_sum_estimate_column: The name of the column in parents_df
      containing the sum of the lower estimates of the siblings.
    lower_estimate_column: The name of the column in the output DataFrame
      containing the estimate from the input children_df.
    upper_exclusive_estimate_column: The name of the column in the output
      DataFrame containing the computed upper exclusive estimate.

  Returns:
    A DataFrame with columns ID, lower_estimate_column, and
    upper_exclusive_estimate_column.
  """
  parent_id_length = len(parents_df[ID][0])
  children_df[ID_PREFIX] = children_df[ID].str[:parent_id_length]
  parents_df.rename(columns={ID: ID_PREFIX}, inplace=True)
  children_df = pd.merge(
      children_df,
      parents_df[[ID_PREFIX,
                  parent_estimate_column,
                  child_lower_sum_estimate_column]],
      how='inner',
      on=ID_PREFIX,
  )
  children_df.rename(columns={ESTIMATE: lower_estimate_column}, inplace=True)
  # parent upper estimate plus estimate of node minus estimate of (node plus
  # siblings) using overloaded operator % to handle subtraction of covariance
  # correctly
  children_df[upper_exclusive_estimate_column] = (
      children_df[parent_estimate_column] -
      (children_df[child_lower_sum_estimate_column]
       % children_df[lower_estimate_column])
  )
  children_df.drop(columns=[parent_estimate_column,
                            child_lower_sum_estimate_column,
                            ID_PREFIX],
                   inplace=True)
  return children_df


def _compute_combined_estimate(
    block_shape: block.BlockShape,
    children_df: pd.DataFrame,
    lower_estimate_column: str,
    upper_exclusive_estimate_column: str,
) -> pd.DataFrame:
  """Compute the combined estimate for each child.

  The input DataFrame children_df consists of three columns, ID,
  lower_estimate_column, and upper_exclusive_estimate_column. The output
  DataFrame consists of two columns, ID and ESTIMATE. The ESTIMATE column in the
  output DataFrame is computed by taking the optimal combination of the lower
  estimate and the upper exclusive estimate.

  Args:
    block_shape: The BlockShape object describing the block structure.
    children_df: A DataFrame with columns ID, lower_estimate_column, and
      upper_exclusive_estimate_column.
    lower_estimate_column: The name of the column in children_df containing the
      lower estimate.
    upper_exclusive_estimate_column: The name of the column in children_df
      containing the upper exclusive estimate.

  Returns:
    A DataFrame with columns ID and ESTIMATE. The ESTIMATE column contains the
    combined estimate.
  """
  combined_estimate_column = 'combined_estimate'
  children_df[combined_estimate_column] = children_df.apply(
      lambda x: regression.combine(
          block_shape,
          x[lower_estimate_column],
          x[upper_exclusive_estimate_column]
      ),
      axis=1
  )

  # Rename columns
  children_combined_df = children_df[[ID, combined_estimate_column]].copy(
      deep=False)
  children_combined_df.rename(columns={combined_estimate_column: ESTIMATE},
                              inplace=True)
  return children_combined_df


def _compute_upper_estimate(
    block_shape: block.BlockShape,
    children_df: pd.DataFrame,
    children_input_df: pd.DataFrame,
    upper_exclusive_estimate_column: str,
) -> pd.DataFrame:
  """Compute the upper inclusive estimate for each child.

  The input DataFrame children_df consists of two columns, ID and
  upper_exclusive_estimate_column. The input DataFrame children_input_df
  consists of two columns, ID and ESTIMATE. The output DataFrame consists of two
  columns, ID and ESTIMATE. The ESTIMATE column in the output DataFrame is
  computed by taking the optimal combination of the input estimate and the
  upper exclusive estimate.

  Args:
    block_shape: The BlockShape object describing the block structure.
    children_df: A DataFrame with columns ID, upper_exclusive_estimate_column.
    children_input_df: A DataFrame with columns ID and ESTIMATE.
    upper_exclusive_estimate_column: The name of the column in children_df
      containing the upper exclusive estimate.

  Returns:
    A DataFrame with columns ID and ESTIMATE. The ESTIMATE column contains the
    upper inclusive estimate.
  """
  input_estimate_column = 'input_estimate'
  upper_estimate_column = 'upper_estimate'
  children_df = pd.merge(children_df, children_input_df, how='outer', on=ID,
                         copy=False)
  children_df.rename(columns={ESTIMATE: input_estimate_column}, inplace=True)

  children_df[upper_estimate_column] = children_df.apply(
      lambda x: regression.combine(
          block_shape,
          x[input_estimate_column],
          x[upper_exclusive_estimate_column]
      ),
      axis=1
  )
  children_df.drop(columns=input_estimate_column, inplace=True)

  # Rename columns
  children_upper_df = children_df[[ID, upper_estimate_column]].copy(deep=False)
  children_upper_df.rename(columns={upper_estimate_column: ESTIMATE},
                           inplace=True)
  return children_upper_df


def top_down_step(
    block_shape: block.BlockShape,
    children_input_df: pd.DataFrame,
    children_lower_df: pd.DataFrame,
    parents_upper_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Top down step of linear postprocessing algorithm.

  This step takes the initial and lower inclusive estimates of the children and
  the upper inclusive estimates of the parents and outputs the upper inclusive
  and combined estimates of the children.

  As an intermediate step, it computes the upper exclusive estimate of a block.
  This is the best linear unbiased estimate of a block given all queries to all
  blocks except for queries to the block itself or its descendants. It is
  computed by taking the upper estimate of the parent and subtracting the sum of
  the lower estimates of the siblings.

  The upper inclusive estimate is the best linear unbiased estimate of a block
  given all queries to all blocks except for queries to the block's descendants
  (but including queries to the block itself). It is computed by taking (1) the
  input estimate of a block and (2) the upper exclusive estimate of the block,
  and taking the optimal combination of these two estimates.

  The combined estimate is the best linear unbiased estimates given all queries
  to all blocks. It is computed by taking (1) the lower estimate of a block and
  (2) the upper exclusive estimate of the block, and taking the optimal
  combination of these two estimates.

  Args:
    block_shape: The BlockShape object describing the block structure.
    children_input_df: A DataFrame with columns ID and ESTIMATE containing input
      estimates of the children.
    children_lower_df: A DataFrame with columns ID and ESTIMATE containing lower
      inclusive estimates of the children.
    parents_upper_df: A DataFrame with columns ID and ESTIMATE containing upper
      inclusive estimates of the parents.
  Returns:
    children_upper_df: A DataFrame with columns ID and ESTIMATE containing upper
      inclusive estimates of the children.
    children_combined_df: A DataFrame with columns ID and ESTIMATE containing
      combined (best linear unbiased) estimates of the children.
  """
  # Aggregate child estimates by parent ID.
  parent_suffix = '_parent'
  child_lower_sum_suffix = '_child_lower_sum'
  parents_df = _aggregate_child_estimates(
      children_df=children_lower_df, parents_df=parents_upper_df,
      parent_suffix=parent_suffix, child_suffix=child_lower_sum_suffix,
  )

  # Compute upper exclusive estimate for each child
  upper_exclusive_estimate_column = 'upper_exclusive_estimate'
  lower_estimate_column = 'lower_estimate'
  children_df = _compute_upper_exclusive_estimate(
      children_df=children_lower_df,
      parents_df=parents_df,
      parent_estimate_column=ESTIMATE + parent_suffix,
      child_lower_sum_estimate_column=ESTIMATE + child_lower_sum_suffix,
      lower_estimate_column=lower_estimate_column,
      upper_exclusive_estimate_column=upper_exclusive_estimate_column,
  )

  # Combine lower estimate and upper exclusive estimate to get combined estimate
  children_combined_df = _compute_combined_estimate(
      block_shape=block_shape,
      children_df=children_df,
      lower_estimate_column=lower_estimate_column,
      upper_exclusive_estimate_column=upper_exclusive_estimate_column,
  )
  children_df.drop(columns=lower_estimate_column, inplace=True)

  # Combine input estimate and upper exclusive estimate to get upper estimate
  children_upper_df = _compute_upper_estimate(
      block_shape=block_shape,
      children_df=children_df,
      children_input_df=children_input_df,
      upper_exclusive_estimate_column=upper_exclusive_estimate_column,
  )

  return (children_upper_df, children_combined_df)
