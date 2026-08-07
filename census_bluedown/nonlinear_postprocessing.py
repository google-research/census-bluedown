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

"""Optimization pass for histograms on blocks of estimates."""

import logging

import pandas as pd
from census_bluedown import block
from census_bluedown import constants
from census_bluedown import nonlinear_solver


ID = constants.ID
ESTIMATE = constants.ESTIMATE
VALUE = constants.VALUE
ID_PREFIX = constants.ID_PREFIX
QUERY_NAME = constants.QUERY_NAME
QUERY_SHAPE = constants.QUERY_SHAPE
VARIANCE = constants.VARIANCE
FIRST_FEATURE_UB_CONSTRAINT = constants.FIRST_FEATURE_UB_CONSTRAINT
FIRST_FEATURE_LB_CONSTRAINT = constants.FIRST_FEATURE_LB_CONSTRAINT


def optimization_top_down_step(
    block_shape: block.BlockShape,
    children_level: str,
    children_lower_df: pd.DataFrame,
    parents_optimization_df: pd.DataFrame,
    children_constraint_df: pd.DataFrame,
) -> pd.DataFrame:
  """Top down optimization step for linear postprocessing.

  Args:
    block_shape: The BlockShape object describing the block structure.
    children_level: The level of the children.
    children_lower_df: A DataFrame with columns ID and ESTIMATE containing lower
      inclusive estimates of the children.
    parents_optimization_df: A DataFrame with columns ID and VALUE containing
      post-optimization estimates of the parents.
    children_constraint_df: A DataFrame with columns ID, QUERY_NAME and VALUE
      containing constraints on the children.
  Returns:
    A DataFrame with columns ID and VALUE. The VALUE column contains the
    post-optimization estimates of the children.
  """
  parent_id_length = len(parents_optimization_df[ID][0])
  children_lower_df[ID_PREFIX] = children_lower_df[ID].str[:parent_id_length]
  children_grouped = children_lower_df.groupby(ID_PREFIX).agg(list)
  parents_optimization_df = parents_optimization_df.set_index(ID)
  children_constraint_df = children_constraint_df.set_index(ID)
  ub_mask = children_constraint_df[QUERY_NAME] == FIRST_FEATURE_UB_CONSTRAINT
  lb_mask = children_constraint_df[QUERY_NAME] == FIRST_FEATURE_LB_CONSTRAINT
  children_all_upper_bounds = {
      row_id: row[VALUE]
      for row_id, row in children_constraint_df[ub_mask].iterrows()}
  children_all_lower_bounds = {
      row_id: row[VALUE]
      for row_id, row in children_constraint_df[lb_mask].iterrows()}

  if children_level in constants.SINGLE_PASS_LEVELS:
    objective_function_types = (constants.ALTERNATE_PASSES
                                if constants.USE_ALTERNATE_PASSES
                                else constants.FULL_PASSES)
  elif children_level in constants.MULTI_PASS_LEVELS:
    objective_function_types = (constants.ALTERNATE_PASSES
                                if constants.USE_ALTERNATE_PASSES
                                else constants.ALL_PASSES)
  else:
    raise ValueError('Unspecified optimization passes for level '
                     f'{children_level}')

  optimized_children_dict = {}
  for i, (parent_geocode, row) in enumerate(children_grouped.iterrows()):
    logging.info('Running optimization for node %d of %d at level %s',
                 i, len(children_grouped), children_level)
    optimized_children_dict.update(
        nonlinear_solver.run_node_optimization_passes(
            block_shape=block_shape,
            objective_function_types=objective_function_types,
            parent_geocode=parent_geocode,
            child_geocodes=row[ID],
            parent_val=parents_optimization_df[VALUE][parent_geocode],
            child_ests={x: y for x, y in zip(row[ID], row[ESTIMATE])},
            children_upper_bound_dict={x: children_all_upper_bounds[x]
                                       for x in row[ID]},
            children_lower_bound_dict={x: children_all_lower_bounds[x]
                                       for x in row[ID]},
            subtree_total_dict=None
        )
    )
  return pd.DataFrame(optimized_children_dict.items(), columns=[ID, VALUE])


def baseline_census_top_down_step(
    block_shape: block.BlockShape,
    children_level: str,
    children_input_df: pd.DataFrame,
    parents_baseline_df: pd.DataFrame,
    children_constraint_df: pd.DataFrame,
) -> pd.DataFrame:
  """Top down optimization step for linear postprocessing.

  Args:
    block_shape: The BlockShape object describing the block structure.
    children_level: The level of the children.
    children_input_df: A DataFrame with columns ID, QUERY_SHAPE, VALUE and
      VARIANCE containing input estimates of the children.
    parents_baseline_df: A DataFrame with columns ID and VALUE containing
      post-baseline-optimization estimates of the parents.
    children_constraint_df: A DataFrame with columns ID, QUERY_NAME and VALUE
      containing constraints on the children.
  Returns:
    A DataFrame with columns ID and VALUE. The VALUE column contains the
    post-optimization estimates of the children.
  """
  parent_id_length = max([len(x) for x in parents_baseline_df[ID]])
  children_grouped = children_input_df.groupby(ID).agg(list)
  children_grouped[ID_PREFIX] = children_grouped.index.str[:parent_id_length]
  children_grouped = children_grouped.reset_index()
  children_grouped = children_grouped.groupby(ID_PREFIX).agg(list)
  parents_baseline_df = parents_baseline_df.set_index(ID)
  children_constraint_df = children_constraint_df.set_index(ID)
  ub_mask = children_constraint_df[QUERY_NAME] == FIRST_FEATURE_UB_CONSTRAINT
  lb_mask = children_constraint_df[QUERY_NAME] == FIRST_FEATURE_LB_CONSTRAINT
  children_all_upper_bounds = {
      row_id: row[VALUE]
      for row_id, row in children_constraint_df[ub_mask].iterrows()}
  children_all_lower_bounds = {
      row_id: row[VALUE]
      for row_id, row in children_constraint_df[lb_mask].iterrows()}

  if children_level in constants.SINGLE_PASS_LEVELS:
    objective_function_types = (constants.ALTERNATE_PASSES
                                if constants.USE_ALTERNATE_PASSES
                                else constants.FULL_PASSES)
  elif children_level in constants.MULTI_PASS_LEVELS:
    objective_function_types = (constants.ALTERNATE_PASSES
                                if constants.USE_ALTERNATE_PASSES
                                else constants.ALL_PASSES)
  else:
    raise ValueError('Unspecified optimization passes for level '
                     f'{children_level}')

  optimized_children_dict = {}
  for i, (parent_geocode, row) in enumerate(children_grouped.iterrows()):
    logging.info('Running optimization for node %d/%d at level %s, id %s',
                 i, len(children_grouped), children_level, parent_geocode)

    # Parent geocode may be from the previous level or from an earlier level, in
    # which case it may be shorter than the parent_id_length.
    while (parent_geocode not in parents_baseline_df[VALUE] and
           len(parent_geocode) > 1):
      parent_geocode = parent_geocode[:-1]
    optimized_children_dict.update(
        nonlinear_solver.run_node_baseline_passes(
            block_shape=block_shape,
            objective_function_types=objective_function_types,
            parent_geocode=parent_geocode,
            child_geocodes=row[ID],
            parent_val=parents_baseline_df[VALUE][parent_geocode],
            child_ests={
                id: {
                    tuple(x): (y, z)
                    for x, y, z in zip(
                        row[QUERY_SHAPE][i], row[VALUE][i], row[VARIANCE][i]
                    )
                }
                for i, id in enumerate(row[ID])
            },
            children_upper_bound_dict={
                x: children_all_upper_bounds[x] for x in row[ID]
            },
            children_lower_bound_dict={
                x: children_all_lower_bounds[x] for x in row[ID]
            },
            subtree_total_dict=None,
        )
    )

    # Include in the output parents with no children, since they may have
    # children in lower levels.
    children_grouped_keys = children_grouped.index
    for geocode in parents_baseline_df.index:
      if geocode not in children_grouped_keys:
        optimized_children_dict[geocode] = parents_baseline_df[VALUE][geocode]

  return pd.DataFrame(optimized_children_dict.items(), columns=[ID, VALUE])
