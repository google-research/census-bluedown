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

"""Best linear unbiased estimator for histograms on hierarchical blocks of estimates.

This module computes the best linear unbiased estimate (BLUE) for histograms
partitioned across a tree hierarchy of geographic blocks. The hierarchy is
represented by string geocodes in the ID column, where each parent's ID is a
fixed-length prefix of its children's IDs. The algorithm is described in
https://arxiv.org/abs/2603.10099, which is also the reference for the lemma,
equation, and algorithm numbers cited throughout this package.

The input and output data for this algorithm consists of separate DataFrames for
each level of the hierarchy. The input DataFrames for the block-level processing
method contain one row for each combination of block ID and query shape, and
consist of columns ID, QUERY_SHAPE, VARIANCE, and VALUE, with types string,
tuple[int], float, and numpy.ndarray, respectively. The input and output
DataFrames for the bottom-up and top-down processing methods, as well as the
output DataFrame for the block-level processing method, contain a single row for
each geo ID and consist of columns ID and ESTIMATE, with types string and
estimate.Estimate, respectively.

The algorithm proceeds in three main stages:
1. Block-level processing (process_blocks): Computes an initial estimate for
   each individual block from its set of noisy query measurements using
   generalized least squares (GLS) regression.
2. Bottom-up step (bottom_up_step): Propagates estimates up the tree hierarchy.
   For each parent block, the lower inclusive estimate is computed by
   combining its own input estimate with the sum of the lower inclusive
   estimates of its children.
3. Top-down step (top_down_step): Propagates estimates down the tree hierarchy.
   For each child block, it computes the upper exclusive estimate by
   subtracting the sum of sibling estimates from the parent's upper inclusive
   estimate. It then computes the final combined estimate (the global BLUE
   given all queries in the hierarchy) by combining the lower and upper
   exclusive estimates, and computes the upper inclusive estimate to pass down
   to descendant blocks.
"""

import pandas as pd
from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import query_encoding
from census_bluedown import regression

ID = constants.ID
ID_PREFIX = constants.ID_PREFIX
ESTIMATE = constants.ESTIMATE
QUERY_SHAPE = constants.QUERY_SHAPE
VARIANCE = constants.VARIANCE
VALUE = constants.VALUE


def _process_df_block(
    df: pd.DataFrame,
) -> estimate.Estimate:
  """Computes the initial block estimate from its noisy queries via GLS."""
  query_shapes = tuple(df[QUERY_SHAPE])
  query_variances = tuple(df[VARIANCE])
  query_values = tuple(df[VALUE])

  workload, cov, y = query_encoding.prepare_succinct_inputs(
      query_shapes, query_variances, query_values
  )

  return estimate.Estimate(*regression.compute_gls_estimate(workload, cov, y))


def process_blocks(
    df: pd.DataFrame,
) -> pd.DataFrame:
  """Postprocess each block in the input dataframe.

  For each ID, compute the best linear unbiased estimate for the corresponding
  block given only the queries on that block.

  Args:
    df: A DataFrame with columns ID, QUERY_SHAPE, VARIANCE, and VALUE.

  Returns:
    A DataFrame with columns ID and ESTIMATE, with one row for each block ID.
    The ESTIMATE column contains the block GLS estimate for the block given
    only the queries on that block.
  """
  return (
      df.groupby(ID)
      .apply(_process_df_block)
      .reset_index(name=ESTIMATE)  # pyrefly: ignore[no-matching-overload]
  )


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
  parent_id_length = len(parents_df[ID].iloc[0])
  children_grouped = (
      children_df.assign(**{ID: children_df[ID].str[:parent_id_length]})
      .groupby(ID, as_index=False)[ESTIMATE]  # Group by parent ID
      .sum()  # Aggregate child estimates per parent
  )

  return pd.merge(
      parents_df,
      children_grouped,
      how="outer",
      on=ID,
      suffixes=(parent_suffix, child_suffix),
  )


def bottom_up_step(
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
    children_lower_df: A DataFrame with columns ID and ESTIMATE containing lower
      inclusive estimates of the children.
    parents_input_df: A DataFrame with columns ID and ESTIMATE containing
      initial estimates of the parents.

  Returns:
    A DataFrame with columns ID and ESTIMATE. The ESTIMATE column contains the
    lower inclusive estimates of the parents, which are the best linear unbiased
    estimates given only the queries on a block and its descendants.
  """
  # Aggregate child estimates by parent ID.
  parent_suffix = "_parent"
  child_lower_sum_suffix = "_child_lower_sum"
  parents_df = _aggregate_child_estimates(
      children_df=children_lower_df,
      parents_df=parents_input_df,
      parent_suffix=parent_suffix,
      child_suffix=child_lower_sum_suffix,
  )

  # Combine aggregated child estimates with parent estimates of the same ID.
  lower_estimate = parents_df.apply(
      lambda x: regression.combine(
          x[ESTIMATE + child_lower_sum_suffix], x[ESTIMATE + parent_suffix]
      ),
      axis=1,
  )
  return pd.DataFrame({
      ID: parents_df[ID],
      ESTIMATE: lower_estimate,
  })


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
      containing the sum of the lower estimates of all children of the parent.
    lower_estimate_column: The name of the column in the output DataFrame
      containing the estimate from the input children_df.
    upper_exclusive_estimate_column: The name of the column in the output
      DataFrame containing the computed upper exclusive estimate.

  Returns:
    A DataFrame with columns ID, lower_estimate_column, and
    upper_exclusive_estimate_column.
  """
  parent_id_length = len(parents_df[ID].iloc[0])
  # Slice parents_df with ID renamed to ID_PREFIX for merging.
  parents_slice = parents_df[[
      ID,
      parent_estimate_column,
      child_lower_sum_estimate_column,
  ]].rename(columns={ID: ID_PREFIX})

  # Add ID_PREFIX to children_df for merging.
  children_with_prefix = children_df.assign(
      **{ID_PREFIX: children_df[ID].str[:parent_id_length]}
  )

  merged = pd.merge(
      children_with_prefix,
      parents_slice,
      how="inner",
      on=ID_PREFIX,
  )

  # parent upper estimate plus estimate of node minus estimate of (node plus
  # siblings) using overloaded operator % to handle subtraction of covariance
  # correctly
  upper_exclusive = merged[parent_estimate_column] - (
      merged[child_lower_sum_estimate_column] % merged[ESTIMATE]
  )

  return pd.DataFrame({
      ID: merged[ID],
      lower_estimate_column: merged[ESTIMATE],
      upper_exclusive_estimate_column: upper_exclusive,
  })


def _compute_combined_estimate(
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
  combined_estimate = children_df.apply(
      lambda x: regression.combine(
          x[lower_estimate_column], x[upper_exclusive_estimate_column]
      ),
      axis=1,
  )
  return pd.DataFrame({
      ID: children_df[ID],
      ESTIMATE: combined_estimate,
  })


def _compute_upper_estimate(
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
    children_df: A DataFrame with columns ID and
      upper_exclusive_estimate_column.
    children_input_df: A DataFrame with columns ID and ESTIMATE.
    upper_exclusive_estimate_column: The name of the column in children_df
      containing the upper exclusive estimate.

  Returns:
    A DataFrame with columns ID and ESTIMATE. The ESTIMATE column contains the
    upper inclusive estimate.
  """
  merged = pd.merge(
      children_df[[ID, upper_exclusive_estimate_column]],
      children_input_df[[ID, ESTIMATE]],
      how="outer",
      on=ID,
  )
  upper_estimate = merged.apply(
      lambda x: regression.combine(
          x[ESTIMATE], x[upper_exclusive_estimate_column]
      ),
      axis=1,
  )
  return pd.DataFrame({
      ID: merged[ID],
      ESTIMATE: upper_estimate,
  })


def top_down_step(
    children_input_df: pd.DataFrame,
    children_lower_df: pd.DataFrame,
    parents_upper_df: pd.DataFrame,
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

  The combined estimate is the best linear unbiased estimate given all queries
  to all blocks. It is computed by taking (1) the lower estimate of a block and
  (2) the upper exclusive estimate of the block, and taking the optimal
  combination of these two estimates.

  Args:
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
  parent_suffix = "_parent"
  child_lower_sum_suffix = "_child_lower_sum"
  parents_df = _aggregate_child_estimates(
      children_df=children_lower_df,
      parents_df=parents_upper_df,
      parent_suffix=parent_suffix,
      child_suffix=child_lower_sum_suffix,
  )

  # Compute upper exclusive estimate for each child
  upper_exclusive_estimate_column = "upper_exclusive_estimate"
  lower_estimate_column = "lower_estimate"
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
      children_df=children_df,
      lower_estimate_column=lower_estimate_column,
      upper_exclusive_estimate_column=upper_exclusive_estimate_column,
  )

  # Combine input estimate and upper exclusive estimate to get upper estimate
  children_upper_df = _compute_upper_estimate(
      children_df=children_df,
      children_input_df=children_input_df,
      upper_exclusive_estimate_column=upper_exclusive_estimate_column,
  )

  return (children_upper_df, children_combined_df)
