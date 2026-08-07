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

"""Postprocessing routines for the root level."""

from collections.abc import Sequence

from absl import logging

import numpy as np
import pandas as pd
from census_bluedown import block
from census_bluedown import constants
from census_bluedown import constrain
from census_bluedown import estimate
from census_bluedown import io
from census_bluedown import linear_postprocessing
from census_bluedown import nonlinear_solver
from census_bluedown import root_optimization


ID = constants.ID
VALUE = constants.VALUE
VARIANCE = constants.VARIANCE
ESTIMATE = constants.ESTIMATE
QUERY_NAME = constants.QUERY_NAME
QUERY_SHAPE = constants.QUERY_SHAPE
FIRST_FEATURE_UB_CONSTRAINT = constants.FIRST_FEATURE_UB_CONSTRAINT
FIRST_FEATURE_LB_CONSTRAINT = constants.FIRST_FEATURE_LB_CONSTRAINT

LEVELS = constants.LEVELS
ROOT_LEVEL = constants.ROOT_LEVEL
FOLDER_IDS = constants.FOLDER_IDS
SUBTREE_GEOCODE_PATTERNS = constants.SUBTREE_GEOCODE_PATTERNS
SUBTREE_FOLDER_PATTERN = constants.SUBTREE_FOLDER_PATTERN

NMF_FNAME = constants.NMF_FNAME
CONSTRAINT_FNAME = constants.CONSTRAINT_FNAME
SUBTREE_TOTALS_FNAME = constants.SUBTREE_TOTALS_FNAME
BLOCK_EST_FNAME = constants.BLOCK_EST_FNAME
LOWER_EST_FNAME = constants.LOWER_EST_FNAME
OPTIMIZED_EST_FNAME = constants.OPTIMIZED_EST_FNAME
BASELINE_EST_FNAME = constants.BASELINE_EST_FNAME


def root_input_pass(
    in_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
    constraint_out_io: io.AbstractBlockHierarchicalIO | None,
) -> pd.DataFrame:
  """Input pass for root level of block hierarchical postprocessing algorithm.

  Postprocesses the root block by itself, including constraints, without
  considering the rest of the tree.

  Args:
    in_io: IO object for reading the DataFrame for the input Noisy
      Measurement File (NMF).
    constraint_io: IO object for reading the constraint DataFrame
    out_io: The IO object for writing the output DataFrame.
    block_shape: The BlockShape object describing the block structure.
    constraint_out_io: The IO object for writing the constraint DataFrame. If
      None, the constraint DataFrame is not copied.

  Returns:
    A DataFrame with columns ID and ESTIMATE consisting of the processed root
    block.
  """
  input_df = in_io.read(level=ROOT_LEVEL, filename=NMF_FNAME)
  constraint_df = constraint_io.read(
      level=ROOT_LEVEL,
      filename=CONSTRAINT_FNAME)

  processed_df = linear_postprocessing.process_blocks(
      block_shape=block_shape,
      df=input_df)

  if constraint_df is not None and constants.CONSTRAIN_BOTTOM_UP:
    processed_df = constrain.constrain_blocks(
        block_shape=block_shape,
        df=processed_df,
        constraint_df=constraint_df)
    if constraint_out_io is not None:
      constraint_out_io.write(level=ROOT_LEVEL,
                              filename=CONSTRAINT_FNAME,
                              df=constraint_df)
  out_io.write(level=ROOT_LEVEL,
               filename=BLOCK_EST_FNAME,
               df=processed_df)
  return processed_df


def read_subtree_folder_lower_estimate(
    subtree_io: io.AbstractBlockHierarchicalIO,
) -> pd.DataFrame:
  """Read subtree estimate for specified subtree_id from file.

  Try to read the lower estimate from the top level of the subtree. If there is
  no lower estimate at that level, read the lower estimate from the top level
  at which it exists. If no lower estimate exists (which is the case if there is
  only one nonempty level in the subtree), instead read the block estimate from
  the top level at which it exists (which should be the unique nonempty level).
  If neither lower nor block estimate exists, raise a ValueError.

  Args:
    subtree_io: The IO object for reading lower estimates from the subtree.

  Returns:
    A DataFrame with columns ID and ESTIMATE consisting of the lower estimate of
    the subtree.
  """

  # Return lower estimate from top level at which it exists.
  for level in LEVELS:
    if subtree_io.exists(level=level, filename=LOWER_EST_FNAME):
      return subtree_io.read(level=level, filename=LOWER_EST_FNAME)

  # If no lower estimate exists, return block estimate from top level at which
  # it exists (which should be the unique level at which it exists).
  for level in LEVELS:
    if subtree_io.exists(level=level, filename=BLOCK_EST_FNAME):
      return subtree_io.read(level=level, filename=BLOCK_EST_FNAME)

  raise ValueError(f'No estimate found for folder {subtree_io.subtree_folder}.')


def read_subtree_lower_estimates(
    subtree_io: io.AbstractBlockHierarchicalIO,
    subtree_id_list: Sequence[str],
    subtree_geocode_patterns: Sequence[str],
) -> dict[str, list[pd.DataFrame]]:
  """Read subtree estimates from files.

  For each subtree ID in subtree_id_list, read the lower estimates from all
  folders corresponding to that subtree, as specified by subtree_patterns.
  In the census setting, there may be one or two folders corresponding to a
  single subtree ID since some of the states are split into AIAN and non-AIAN
  areas, so each list will have one or two elements.

  Args:
    subtree_io: The IO object for reading lower estimates from the subtree.
    subtree_id_list: The list of subtree IDs to read estimates for.
    subtree_geocode_patterns: String formats for geocodes with given subtree ID.

  Returns:
    A dictionary mapping subtree IDs to lists of DataFrames with columns ID and
    ESTIMATE consisting of the lower estimates of the subtrees.
  """
  subtree_estimate_dict = {}

  for subtree_id in subtree_id_list:
    subtree_estimates = []

    for pattern in subtree_geocode_patterns:
      folder_id = pattern % subtree_id
      if folder_id in FOLDER_IDS:
        subtree_io.set_subtree_folder(SUBTREE_FOLDER_PATTERN % folder_id)
        subtree_estimates.append(read_subtree_folder_lower_estimate(subtree_io))

    subtree_estimate_dict[subtree_id] = subtree_estimates
  return subtree_estimate_dict


def read_subtree_constraints(
    constraint_io: io.AbstractBlockHierarchicalIO,
    subtree_id_list: Sequence[str],
) ->  tuple[dict[str, Sequence[int]], dict[str, Sequence[int]]]:
  """Read State-level constraints from files.

  Args:
    constraint_io: The IO object for reading constraints from the subtree.
    subtree_id_list: The list of subtree IDs to read constraints for.

  Returns:
    A tuple with two elements. The first element is a dictionary mapping subtree
    IDs to upper bound constraints, and the second element is a dictionary
    mapping subtree IDs to lower bound constraints.
  """
  subtree_upper_bound_dict = {}
  subtree_lower_bound_dict = {}
  for subtree_id in subtree_id_list:
    constraint_io.set_subtree_folder(SUBTREE_FOLDER_PATTERN % subtree_id)
    constraint_df = constraint_io.read(
        level=LEVELS[0],
        filename=CONSTRAINT_FNAME)
    ub_mask = constraint_df[QUERY_NAME] == FIRST_FEATURE_UB_CONSTRAINT
    lb_mask = constraint_df[QUERY_NAME] == FIRST_FEATURE_LB_CONSTRAINT
    subtree_upper_bound_dict[subtree_id] = tuple(
        constraint_df[ub_mask][VALUE].item())
    subtree_lower_bound_dict[subtree_id] = tuple(
        constraint_df[lb_mask][VALUE].item())
  return subtree_upper_bound_dict, subtree_lower_bound_dict


def write_processed_subtree_estimates(
    processed_io: io.AbstractBlockHierarchicalIO,
    processed_estimates_dict: dict[str, Sequence[int]],
    subtree_id_list: Sequence[str],
    filename: str
):
  """Write processed subtree estimates to files."""
  for subtree_id in subtree_id_list:
    processed_io.set_subtree_folder(SUBTREE_FOLDER_PATTERN % subtree_id)
    output_df = pd.DataFrame({
        ID: subtree_id,
        VALUE: [np.array(processed_estimates_dict[subtree_id])]})
    processed_io.write(
        level=LEVELS[0],
        filename=filename,
        df=output_df)


def subtree_constrained_lower_estimates(
    subtree_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
    subtree_totals: pd.DataFrame,
    subtree_geocode_patterns: Sequence[str] = SUBTREE_GEOCODE_PATTERNS
) -> pd.DataFrame:
  """Compute constrained lower estimates for combined subtrees.

  For each subtree ID in subtree_totals, read and aggregate the lower estimate
  or estimates corresponding to that subtree ID, apply the subtree total
  constraint, and store the result in a DataFrame. In the census setting, this
  corresponds to combining AIAN on- and off-spine areas, and applying the
  statewide total constraint.

  Args:
    subtree_io: The IO object for reading lower estimates from the subtree.
    block_shape: The BlockShape object describing the block structure.
    subtree_totals: A DataFrame with columns ID and VALUE containing the
      statewide totals for each subtree.
    subtree_geocode_patterns: String formats for geocodes with given subtree ID.

  Returns:
    A DataFrame with columns ID and ESTIMATE consisting of the constrained lower
    estimates of each subtree.
  """
  lower_estimates = []
  subtree_ids = subtree_totals[ID]
  subtree_totals = subtree_totals.set_index(ID)
  subtree_estimate_dict = read_subtree_lower_estimates(
      subtree_io, subtree_ids, subtree_geocode_patterns)

  for subtree_id in subtree_ids:
    subtree_est = sum([x[ESTIMATE][0]
                       for x in subtree_estimate_dict[subtree_id]])

    # Apply statewide total constraint and append to list
    if constants.CONSTRAIN_BOTTOM_UP:
      lower_estimates.append(
          constrain.apply_sum_constraint(
              block_shape=block_shape,
              est=subtree_est,  # pyrefly: ignore[bad-argument-type]
              total=subtree_totals.loc[subtree_id][VALUE]))
    else:
      lower_estimates.append(subtree_est)

  subtree_totals = subtree_totals.reset_index()
  return pd.DataFrame({
      ID: subtree_totals[ID],
      ESTIMATE: lower_estimates})


def root_bottom_up_passes(
    nmf_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    subtree_totals_io: io.AbstractBlockHierarchicalIO,
    processed_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
    constraint_out_io: io.AbstractBlockHierarchicalIO | None,
):
  """Process root level for block hierarchical postprocessing algorithm.

  This consists of running the following procedures:
  1) Process the input block to produce the block estimate of the root.
  2) Run the bottom-up step, including the statewide total constraints, to
  produce the lower estimate of the root.

  Args:
    nmf_io: The IO object for reading the root input Noisy Measurement File.
    constraint_io: The IO object for reading the root constraint DataFrame.
    subtree_totals_io: The IO object for reading the statewide totals for each
      subtree.
    processed_io: The IO object for reading the subtree lower estimate
      DataFrames and writing the output DataFrames.
    block_shape: The BlockShape object describing the block structure.
    constraint_out_io: The IO object for writing the constraint DataFrame. If
      None, the constraint DataFrame is not copied.
  """
  # Initial block postprocessing of root
  root_input_df = root_input_pass(
      in_io=nmf_io,
      constraint_io=constraint_io,
      out_io=processed_io,
      block_shape=block_shape,
      constraint_out_io=constraint_out_io)

  # Bottom-up step, including statewide total constraint
  subtree_totals = subtree_totals_io.read(
      level=ROOT_LEVEL,
      filename=SUBTREE_TOTALS_FNAME)
  subtree_constrained_lower_df = subtree_constrained_lower_estimates(
      processed_io, block_shape, subtree_totals)
  processed_root = linear_postprocessing.bottom_up_step(
      block_shape=block_shape,
      children_lower_df=subtree_constrained_lower_df,
      parents_input_df=root_input_df)
  processed_io.write(
      level=ROOT_LEVEL,
      filename=LOWER_EST_FNAME,
      df=processed_root)


def root_top_down_pass(
    in_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    subtree_totals_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
):
  """Process root level for block hierarchical postprocessing algorithm."""
  subtree_totals = subtree_totals_io.read(
      level=ROOT_LEVEL,
      filename=SUBTREE_TOTALS_FNAME)
  constraint_df = constraint_io.read(
      level=ROOT_LEVEL,
      filename=CONSTRAINT_FNAME)
  root_df = in_io.read(level=ROOT_LEVEL, filename=LOWER_EST_FNAME)
  root_est = root_df[ESTIMATE].item()
  subtree_total_dict = dict(zip(subtree_totals[ID], subtree_totals[VALUE]))
  root_total = sum(subtree_totals[VALUE])
  ub_mask = constraint_df[QUERY_NAME] == FIRST_FEATURE_UB_CONSTRAINT
  lb_mask = constraint_df[QUERY_NAME] == FIRST_FEATURE_LB_CONSTRAINT
  root_upper_bounds = constraint_df[ub_mask][VALUE].item()
  root_lower_bounds = constraint_df[lb_mask][VALUE].item()

  # The inequality constraints are likely loose here, so it's tempting to
  # proceed directly from the bottom-up pass to the rounder pass, but to be on
  # the safe side, we do the full pass first.
  logging.info('Running L2 optimization for root node.')
  variables, result = root_optimization.run_root_optimization(
      block_shape=block_shape,
      objective_function_type=constants.ObjectiveFunctionType.FULL,
      root_geocode='',
      root_est=root_est,
      root_upper_bounds=root_upper_bounds,
      root_lower_bounds=root_lower_bounds,
      root_total=root_total
  )
  variable_values = np.array([x if isinstance(x, float)
                              else result.variable_values(x)
                              for x in variables])

  logging.info('Running rounder pass for root node.')
  rounder_variables, rounder_result = root_optimization.run_root_optimization(
      block_shape=block_shape,
      objective_function_type=constants.ObjectiveFunctionType.FULL_ROUNDER,
      root_geocode='',
      root_est=estimate.Estimate(
          val=variable_values,
          cov=np.array(0)),
      root_upper_bounds=root_upper_bounds,
      root_lower_bounds=root_lower_bounds,
      root_total=root_total
  )

  rounder_variable_values = np.array([x if isinstance(x, float)
                                      else rounder_result.variable_values(x)
                                      for x in rounder_variables]).astype(int)
  root_variable_values = (np.floor(variable_values).astype(int)
                          + rounder_variable_values)

  logging.info('Writing root node estimate to file.')
  out_io.write(
      level=ROOT_LEVEL,
      filename=OPTIMIZED_EST_FNAME,
      df=pd.DataFrame({
          ID: [''],
          VALUE: [root_variable_values]}))

  logging.info('Solve top-down step for children of root node')
  subtree_estimates_dict = {}
  children_upper_bound_dict, children_lower_bound_dict = (
      read_subtree_constraints(constraint_io, constants.FOLDER_IDS))

  for subtree_id in constants.FOLDER_IDS:
    in_io.set_subtree_folder(SUBTREE_FOLDER_PATTERN % subtree_id)
    subtree_estimates_dict[subtree_id] = (
        read_subtree_folder_lower_estimate(in_io)[ESTIMATE].item())

  processed_subtree_estimate_dict = (
      nonlinear_solver.run_node_optimization_passes(
          block_shape=block_shape,
          objective_function_types=(
              constants.ALTERNATE_PASSES if constants.USE_ALTERNATE_PASSES
              else constants.ALL_PASSES),
          parent_geocode='',
          child_geocodes=constants.FOLDER_IDS,
          parent_val=root_variable_values,
          child_ests=subtree_estimates_dict,
          children_upper_bound_dict=children_upper_bound_dict,
          children_lower_bound_dict=children_lower_bound_dict,
          subtree_total_dict=subtree_total_dict,
      )
  )

  write_processed_subtree_estimates(
      processed_io=out_io,
      processed_estimates_dict=processed_subtree_estimate_dict,
      subtree_id_list=FOLDER_IDS,
      filename=OPTIMIZED_EST_FNAME)


def root_baseline_census_topdown_pass(
    nmf_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    subtree_totals_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
):
  """Baseline root processing for census TopDown algorithm."""
  subtree_totals = subtree_totals_io.read(
      level=ROOT_LEVEL,
      filename=SUBTREE_TOTALS_FNAME)
  constraint_df = constraint_io.read(
      level=ROOT_LEVEL,
      filename=CONSTRAINT_FNAME)
  root_df = nmf_io.read(level=ROOT_LEVEL, filename=NMF_FNAME)
  root_est_dict = dict(zip(map(tuple, root_df[QUERY_SHAPE]),
                           zip(root_df[VALUE], root_df[VARIANCE])))
  subtree_total_dict = dict(zip(subtree_totals[ID], subtree_totals[VALUE]))
  root_total = sum(subtree_totals[VALUE])
  ub_mask = constraint_df[QUERY_NAME] == FIRST_FEATURE_UB_CONSTRAINT
  lb_mask = constraint_df[QUERY_NAME] == FIRST_FEATURE_LB_CONSTRAINT
  root_upper_bounds = constraint_df[ub_mask][VALUE].item()
  root_lower_bounds = constraint_df[lb_mask][VALUE].item()

  # The inequality constraints are likely loose here, so it's tempting to
  # proceed directly from the bottom-up oass to the rounder pass, but to be on
  # the safe side, we do the full pass first.
  logging.info('Running baseline L2 optimization for root node.')
  variables, result = root_optimization.run_baseline_root_optimization(
      block_shape=block_shape,
      objective_function_type=constants.ObjectiveFunctionType.FULL,
      root_geocode='',
      root_est_dict=root_est_dict,
      root_upper_bounds=root_upper_bounds,
      root_lower_bounds=root_lower_bounds,
      root_total=root_total
  )
  variable_values = np.array([x if isinstance(x, float)
                              else result.variable_values(x)
                              for x in variables])

  logging.info('Running baseline rounder pass for root node.')
  # Run regular rounder pass, since it's the same as baseline rounder pass.
  rounder_variables, rounder_result = (
      root_optimization.run_root_optimization(
          block_shape=block_shape,
          objective_function_type=constants.ObjectiveFunctionType.FULL_ROUNDER,
          root_geocode='',
          root_est=estimate.Estimate(
              val=variable_values,
              cov=np.array(0)),
          root_upper_bounds=root_upper_bounds,
          root_lower_bounds=root_lower_bounds,
          root_total=root_total
      )
  )

  rounder_variable_values = np.array([x if isinstance(x, float)
                                      else rounder_result.variable_values(x)
                                      for x in rounder_variables]).astype(int)

  root_variable_values = (np.floor(variable_values).astype(int)
                          + rounder_variable_values)

  logging.info('Writing root baseline estimate to file.')
  out_io.write(
      level=ROOT_LEVEL,
      filename=BASELINE_EST_FNAME,
      df=pd.DataFrame({
          ID: [''],
          VALUE: [root_variable_values]}))

  logging.info('Solve top-down step for children of root node')
  subtree_inputs_dict = {}
  children_upper_bound_dict, children_lower_bound_dict = (
      read_subtree_constraints(constraint_io, constants.FOLDER_IDS))

  for subtree_id in constants.FOLDER_IDS:
    nmf_io.set_subtree_folder(SUBTREE_FOLDER_PATTERN % subtree_id)
    subtree_df = nmf_io.read(level=LEVELS[0], filename=NMF_FNAME)
    subtree_inputs_dict[subtree_id] = dict(
        zip(map(tuple, subtree_df[QUERY_SHAPE]),
            zip(subtree_df[VALUE], subtree_df[VARIANCE])))

  processed_subtree_estimate_dict = (
      nonlinear_solver.run_node_baseline_passes(
          block_shape=block_shape,
          objective_function_types=(
              constants.ALTERNATE_PASSES if constants.USE_ALTERNATE_PASSES
              else constants.ALL_PASSES),
          parent_geocode='',
          child_geocodes=constants.FOLDER_IDS,
          parent_val=root_variable_values,
          child_ests=subtree_inputs_dict,
          children_upper_bound_dict=children_upper_bound_dict,
          children_lower_bound_dict=children_lower_bound_dict,
          subtree_total_dict=subtree_total_dict,
      )
  )

  write_processed_subtree_estimates(
      processed_io=out_io,
      processed_estimates_dict=processed_subtree_estimate_dict,
      subtree_id_list=FOLDER_IDS,
      filename=BASELINE_EST_FNAME)
