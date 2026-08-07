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

"""Main algorithm passes for the block hierarchical postprocessing algorithm.
"""

from absl import logging

from census_bluedown import block
from census_bluedown import constants
from census_bluedown import constrain
from census_bluedown import io
from census_bluedown import linear_postprocessing
from census_bluedown import nonlinear_postprocessing


LEVELS = constants.LEVELS
NMF_FNAME = constants.NMF_FNAME
CONSTRAINT_FNAME = constants.CONSTRAINT_FNAME
BLOCK_EST_FNAME = constants.BLOCK_EST_FNAME
LOWER_EST_FNAME = constants.LOWER_EST_FNAME
UPPER_EST_FNAME = constants.UPPER_EST_FNAME
COMBINED_EST_FNAME = constants.COMBINED_EST_FNAME
OPTIMIZED_EST_FNAME = constants.OPTIMIZED_EST_FNAME
BASELINE_EST_FNAME = constants.BASELINE_EST_FNAME


def input_pass(
    input_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
    constraint_out_io: io.AbstractBlockHierarchicalIO | None,
):
  """Input pass of the block hierarchical postprocessing algorithm.

  Postprocesses each input block individually in the specified subtree.

  Args:
    input_io: IO object for reading the DataFrame for the input Noisy
      Measurement Files (NMF).
    constraint_io: IO object for reading the constraint DataFrame
    out_io: The IO object for writing the output DataFrame.
    block_shape: The BlockShape object describing the block structure.
    constraint_out_io: The IO object for writing the constraint DataFrame. If
      None, the constraint DataFrame is not copied.
  """
  for level in LEVELS:
    if input_io.exists(level, BLOCK_EST_FNAME):
      input_df = input_io.read(level, filename=constants.NMF_FNAME)
      processed_df = linear_postprocessing.process_blocks(
          block_shape=block_shape,
          df=input_df)
      if (constraint_io.exists(level, CONSTRAINT_FNAME) and
          constants.CONSTRAIN_BOTTOM_UP):
        constraint_df = constraint_io.read(
            level=level,
            filename=CONSTRAINT_FNAME)
        processed_df = constrain.constrain_blocks(
            block_shape=block_shape,
            df=processed_df,
            constraint_df=constraint_df)
        if constraint_out_io is not None:
          constraint_out_io.write(
              level=level, filename=CONSTRAINT_FNAME, df=constraint_df)
      out_io.write(level=level, filename=BLOCK_EST_FNAME, df=processed_df)


def bottom_up_pass(
    in_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
):
  """Bottom up pass of the block hierarchical postprocessing algorithm.

  Perform the bottom up step of the linear postprocessing algorithm for each
  level in the hierarchy on the subtree specified by the input IO object. Handle
  any missing levels by skipping that level. Produces lower estimate files for
  each level that has a block estimate file, except for the leaf level (for
  which the lower estimate is equal to the block estimate).

  Args:
    in_io: The IO object for reading the DataFrame containing the block
      estimates.
    out_io: The IO object for writing the DataFrame containing the lower
      inclusive estimates.
    block_shape: The BlockShape object describing the block structure.
  """
  children_lower_df = None

  for level in reversed(LEVELS):
    if children_lower_df is None:
      if in_io.exists(level, BLOCK_EST_FNAME):
        # At leaf level
        logging.info('Bottom up pass: leaf level %s', level)
        children_lower_df = in_io.read(level, BLOCK_EST_FNAME)
    else:
      if in_io.exists(level, BLOCK_EST_FNAME):
        # At parent level for bottom up step
        logging.info('Bottom up pass: parent level %s', level)
        parents_input_df = in_io.read(level, BLOCK_EST_FNAME)
        processed_df = linear_postprocessing.bottom_up_step(
            block_shape=block_shape,
            children_lower_df=children_lower_df,
            parents_input_df=parents_input_df)
        out_io.write(level=level, filename=LOWER_EST_FNAME, df=processed_df)
        children_lower_df = processed_df


def top_down_pass(
    in_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape,
    subtree_only: bool = False):
  """Top down pass of the block hierarchical postprocessing algorithm.

  Perform the top down step of the linear postprocessing algorithm for each
  level in the hierarchy on the subtree specified by the input IO object. Handle
  any missing levels by skipping that level. Produces upper and combined
  estimate files for each level that has a block estimate file, except for the
  top such level.

  Assumes that a lower estimate has already been produced for each non-leaf
  level. If subtree_only is False, also assumes that the root has already been
  processed and an upper estimate has been produced for the top level in the
  subtree.

  Args:
    in_io: The IO object for reading the DataFrame containing the block and
      lower estimates, and the upper estimate of the root if subtree_only is
      false (i.e. if the root has already been processed and we are propagating
      down to this subtree).
    out_io: The IO object for writing the DataFrame containing the upper
      inclusive estimates and combined estimates.
    block_shape: The BlockShape object describing the block structure.
    subtree_only: Whether to treat the subtree root as the root of the tree.
  """
  parents_upper_df = None

  for level in LEVELS:
    if parents_upper_df is None:
      if not subtree_only and in_io.exists(level, UPPER_EST_FNAME):
        # At top level of subtree (root has already been processed)
        logging.info('Top down pass: top level %s of subtree', level)
        parents_upper_df = in_io.read(level, UPPER_EST_FNAME)
      elif subtree_only and in_io.exists(level, BLOCK_EST_FNAME):
        # At root level of tree (treating subtree root as root of tree)
        # Upper estimate of root is equal to block estimate of root
        logging.info('Top down pass: root level %s', level)
        parents_upper_df = in_io.read(level, BLOCK_EST_FNAME)
    else:
      if in_io.exists(level, BLOCK_EST_FNAME):
        # At child level for top down step
        children_input_df = in_io.read(level, BLOCK_EST_FNAME)
        if in_io.exists(level, LOWER_EST_FNAME):
          logging.info('Top down pass: child level %s', level)
          children_lower_df = in_io.read(level, LOWER_EST_FNAME)
        else:
          # No separate lower estimates found, so must be at leaf level
          # Lower estimate of leaf is equal to block (input) estimate of leaf
          logging.info('Top down pass: leaf level %s', level)
          children_lower_df = children_input_df
        children_upper_df, children_combined_df = (
            linear_postprocessing.top_down_step(
                block_shape=block_shape,
                children_input_df=children_input_df,
                children_lower_df=children_lower_df,
                parents_upper_df=parents_upper_df))
        out_io.write(level=level,
                     filename=UPPER_EST_FNAME,
                     df=children_upper_df)
        out_io.write(level=level,
                     filename=COMBINED_EST_FNAME,
                     df=children_combined_df)
        parents_upper_df = children_upper_df


def optimization_top_down_pass(
    in_io: io.ProcessingFormatIO,
    out_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape):
  """Top down optimization pass of block hierarchical postprocessing algorithm.

  Perform the top down step of the optimization postprocessing algorithm for
  each level in the hierarchy on the subtree specified by the input IO object.
  Handle any missing levels by skipping that level. Produces optimization
  estimate files for each level that has a block estimate file, except for the
  top level.

  Assumes that a lower estimate has already been produced for each non-leaf
  level.

  Args:
    in_io: The IO object for reading the DataFrame containing the block and
      lower estimates, and the optimized estimate of the root.
    out_io: The IO object for writing the DataFrame containing the optimized
      estimates.
    constraint_io: The IO object for reading the constraint DataFrame.
    block_shape: The BlockShape object describing the block structure.
  """
  parents_optimization_df = None

  for level in LEVELS:
    if parents_optimization_df is None:
      if in_io.exists(level, OPTIMIZED_EST_FNAME):
        # At top level of subtree (root has already been processed)
        logging.info('Top down optimization pass: top level %s of subtree',
                     level)
        split_estimates_flag = in_io.split_estimates
        in_io.split_estimates = False
        parents_optimization_df = in_io.read(level, OPTIMIZED_EST_FNAME)
        in_io.split_estimates = split_estimates_flag
    else:
      if in_io.exists(level, BLOCK_EST_FNAME):
        # At child level for top down step
        if in_io.exists(level, LOWER_EST_FNAME):
          logging.info('Top down pass: child level %s', level)
          children_lower_df = in_io.read(level, LOWER_EST_FNAME)
        else:
          # No separate lower estimates found, so must be at leaf level
          # Lower estimate of leaf is equal to block (input) estimate of leaf
          logging.info('Top down pass: leaf level %s', level)
          children_lower_df = in_io.read(level, BLOCK_EST_FNAME)

        constraint_df = constraint_io.read(
            level=level,
            filename=CONSTRAINT_FNAME)

        child_opt_df = nonlinear_postprocessing.optimization_top_down_step(
            block_shape=block_shape,
            children_level=level,
            children_lower_df=children_lower_df,
            parents_optimization_df=parents_optimization_df,
            children_constraint_df=constraint_df)

        out_io.write(level=level,
                     filename=OPTIMIZED_EST_FNAME,
                     df=child_opt_df)
        parents_optimization_df = child_opt_df


def baseline_census_top_down_pass(
    in_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    block_shape: block.BlockShape):
  """Top down optimization pass of block hierarchical postprocessing algorithm.

  Perform the baseline top down postprocessing algorithm for each level in the
  hierarchy on the subtree specified by the input IO object. Handle any missing
  levels by skipping that level. Produces baseline estimate files for each
  level that has a block estimate file, except for the top level.

  Assumes that a lower estimate has already been produced for each non-leaf
  level.

  Args:
    in_io: The IO object for reading the DataFrame containing the input
      estimates, and the baseline estimate of the root.
    out_io: The IO object for writing the DataFrame containing the basline
      optimized estimates.
    constraint_io: The IO object for reading the constraint DataFrame.
    block_shape: The BlockShape object describing the block structure.
  """
  parents_baseline_df = None

  for level in LEVELS:
    if parents_baseline_df is None:
      # Read parent baseline from output directory, not input directory.
      if out_io.exists(level, BASELINE_EST_FNAME):
        # At top level of subtree (root has already been processed)
        logging.info('Top down baseline pass: top level %s of subtree',
                     level)
        parents_baseline_df = out_io.read(level, BASELINE_EST_FNAME)
    else:
      if in_io.exists(level, NMF_FNAME):
        logging.info('Top down pass: child level %s', level)
        children_input_df = in_io.read(level, NMF_FNAME)
        constraint_df = constraint_io.read(
            level=level,
            filename=CONSTRAINT_FNAME)

        child_base_df = nonlinear_postprocessing.baseline_census_top_down_step(
            block_shape=block_shape,
            children_level=level,
            children_input_df=children_input_df,
            parents_baseline_df=parents_baseline_df,
            children_constraint_df=constraint_df)

        out_io.write(level=level,
                     filename=BASELINE_EST_FNAME,
                     df=child_base_df)
        parents_baseline_df = child_base_df
