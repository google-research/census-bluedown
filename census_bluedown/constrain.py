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

"""Add linear equality constraints to block estimate.
"""

from collections.abc import Sequence
import numpy as np
import pandas as pd
from census_bluedown import block
from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import regression


BlockShape = block.BlockShape
Estimate = estimate.Estimate

ID = constants.ID
VALUE = constants.VALUE
ESTIMATE = constants.ESTIMATE
QUERY_NAME = constants.QUERY_NAME
FIRST_FEATURE_UB_CONSTRAINT = constants.FIRST_FEATURE_UB_CONSTRAINT
MULTIFEATURE_ZERO_SEQUENCE = constants.MULTIFEATURE_ZERO_SEQUENCE


def apply_linear_constraint(
    block_shape: BlockShape,
    est: Estimate,
    Q: np.ndarray,  # pylint: disable=invalid-name
    c: np.ndarray
) -> Estimate:
  """Perform restricted least squares regression to add the given constraint.

  Computes the variance-minimizing estimate x of the estimated value est subject
  to the linear equality constraints Q^T @ x  = c. This is also called
  constrained least squares regression. If d = block_shape.compressed_length,
  then Q is a matrix of shape d x k and c is a vector of length k, where k is
  the number of constraints.

  Args:
    block_shape: The BlockShape object describing the block structure
    est: The estimate to constrain
    Q: The linear combinations to constrain, as a matrix of shape d x k where d
      is block_shape.compressed_length and k is the number of constraints
    c: The values assigned to the constrained terms, as a vector of length k.

  Returns:
    The constrained estimate and updated covariance.
  """
  # Let d = block_shape.compressed_length and D = block_shape.length
  cov = est.cov  # shape = (d, d)
  val = est.val  # shape = (D,)

  # Key d x k matrix used to compute transformed estimate and covariance.
  # This formula assumes  that Q.T @ cov @ Q is invertible, i.e. that the
  # constraints are independent from one another and all previous constraints.
  L = cov @ Q @ np.linalg.inv(Q.T @ cov @ Q)  # shape = (d, k). pylint: disable=invalid-name

  # Expressions used to update the estimate. These correspond to the terms
  # L c and L Q^T \beta_{GLS} in the doc linked above, where
  # \beta_{RLS} = \beta_{GLS} + L c - L Q^T \beta_{GLS}
  Lc_term = regression.apply_linear_transformation(  # pylint: disable=invalid-name
      block_shape=block_shape,
      linear_transformation=np.diag(L @ c),
      estimate_value=np.ones(block_shape.length)
  ).flatten()  # shape = (D,)

  LQTbeta_term = regression.apply_linear_transformation(  # pylint: disable=invalid-name
      block_shape=block_shape,
      linear_transformation=L @ Q.T,
      estimate_value=val
  ).flatten()  # shape = (D,)

  new_val = val + (Lc_term - LQTbeta_term)  # shape = (D,)
  new_cov = cov - L @ Q.T @ cov             # shape = (d, d)
  return Estimate(new_val, new_cov)


def sum_constraint(
    block_shape: BlockShape,
    total: int
) -> tuple[np.ndarray, np.ndarray]:
  """Restricted least squares regression to constrain the sum of all values.

  Computes the variance-minimizing estimate x of the estimated value est subject
  to the linear equality constraint that sum(x) = total.

  This constraint is applied to the statewide totals in the census application.

  Args:
    block_shape: The BlockShape object describing the block structure
    total: The value of the sum

  Returns:
    The linear combinations Q and assigned values c corresponding to the added
    constraint Q^T @ x = c. The linear combinations Q are of shape
    (block_shape.compressed_length, 1) and the values c are of shape (1,).
  """
  # For each combination of asymmetric features, the last coordinate for that
  # combination corresponds to the sum over all values of the symmetric features
  constraint = np.zeros(block_shape.compressed_symmetric_length)
  constraint[-1] = 1

  # Take the symmetric feature sum for each combination of asymmetric features
  constraint = np.tile(constraint, block_shape.asymmetric_length)
  q = np.array([constraint])
  return q.T, np.array([total])


def first_feature_zero_constraints(
    block_shape: BlockShape,
    zero_indices: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
  """Restricted least squares regression to constrain variables to zero.

  The variables to constrain to zero are certain values of the first feature,
  as specified by the boolean sequence zero_indices. Computes the variance-
  minimizing estimate x of the estimated value est subject to the linear
  equality constraint that all estimates equal zero for the specified values of
  the first feature.

  This constraint is used in the census application to enforce that for housing
  types with no units in a region, the corresponding estimate is zero. One could
  also write a similar method to apply constraints to other asymmetric features.

  Args:
    block_shape: The BlockShape object describing the block structure
    zero_indices: a sequence of length block_shape.shape[0] indicating which
      values of the first feature should be constrained to zero.

  Returns:
    The linear combinations Q and assigned values c corresponding to the added
    constraints Q^T @ x = c. The linear combinations Q are of shape
    (d, k * m) and the values c are of shape (k,), where k is the number of True
    entries in zero_indices, d = block_shape.compressed_length, and
    m = block_shape.compressed_length // block_shape.shape[0],
  """
  if 0 not in block_shape.asymmetric_features:
    raise ValueError(
        'Attempted to constrain some values of a symmetric feature.'
    )
  if not sum(zero_indices):
    return np.array([]), np.array([])

  # Ensure that boolean indexing will be performed
  zero_indices = list([bool(x) for x in zero_indices])

  expansion_factor = block_shape.compressed_length // block_shape.shape[0]
  q = np.kron(np.diag(zero_indices)[zero_indices],
              np.identity(expansion_factor))
  constraint_c = np.zeros(expansion_factor * sum(zero_indices))

  return q.T, constraint_c


def multifeature_zero_constraints(
    block_shape: BlockShape,
    feature_values: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
  """Restricted least squares regression to constrain variables to zero.

  The features to constrain to zero are specified by a combination of values of
  a prefix of the features, which must be asymmetric features. Computes the
  variance-minimizing estimate x of the estimated value est subject to the
  linear equality constraint that the estimate must equal zero for the specified
  combination of values of a prefix of the sequence of features and all
  possible combinations of the remaining features.

  This constraint is used in the census application to enforce that there are no
  individuals of housing type 3 ("nursing facilities / skilled nursing
  facilities") with voting age value 0 (age < 18). This constraint is applied to
  every region in the census application.

  Args:
    block_shape: The BlockShape object describing the block structure
    feature_values: the asymmetric feature combination to constrain to zero. The
      specified values correspond to a prefix of the features, so that the ith
      entry is the value of the ith feature. The length of the sequence must be
      less than the number of asymmetric features.

  Returns:
    The linear combinations Q and assigned values c corresponding to the added
    constraints Q^T @ x = c. The linear combinations Q are of shape
    (block_shape.compressed_length, k) and the values c are of shape (k,),
    where k is equal to number_of_constraints as computed below.
  """
  if len(feature_values) > block_shape.num_asymmetric_features:
    raise ValueError(
        'Attempted to add asymmetric constraint to symmetric feature.'
    )

  number_of_constraints = block_shape.compressed_length // np.prod(
      block_shape.shape[:len(feature_values)]
  )
  constraint_c = np.zeros(number_of_constraints)

  q = [1]
  for i, feature_value in enumerate(feature_values):
    next_term = np.zeros(block_shape.shape[i])
    next_term[feature_value] = 1
    q = np.kron(q, next_term)
  q = np.kron(q, np.identity(number_of_constraints))  # pyrefly: ignore[bad-argument-type]

  return np.array(q).T, constraint_c


def apply_sum_constraint(
    block_shape: BlockShape,
    est: Estimate,
    total: int
) -> Estimate:
  """Wrapper to apply sum constraint to estimate."""
  q, c = sum_constraint(block_shape, total)
  return apply_linear_constraint(block_shape, est, q, c)


def apply_first_feature_zero_constraints(
    block_shape: BlockShape,
    est: Estimate,
    zero_indices: Sequence[bool],
) -> Estimate:
  """Wrapper to apply first feature zero constraints to estimate."""
  q, c = first_feature_zero_constraints(block_shape, zero_indices)
  if not q.size:
    # If there are no constraints to add, then return the estimate unchanged.
    return est
  return apply_linear_constraint(block_shape, est, q, c)


def apply_multifeature_zero_constraints(
    block_shape: BlockShape,
    est: Estimate,
    already_constrained: bool,
    feature_values: Sequence[int],
) -> Estimate:
  """Wrapper to apply multifeature zero constraints to estimate."""
  if already_constrained:
    # Avoid adding the constraint again if it is already enforced.
    return est
  q, c = multifeature_zero_constraints(block_shape, feature_values)
  return apply_linear_constraint(block_shape, est, q, c)


def constrain_blocks(
    block_shape: block.BlockShape,
    df: pd.DataFrame,
    constraint_df: pd.DataFrame
) -> pd.DataFrame:
  """Apply linear equality constraints to blocks.

  Two types of constraints are applied:
  1. For values of the first feature with an upper bound of zero as specified in
     the rows of constraint_df with QUERY_NAME of FIRST_FEATURE_UB_CONSTRAINT,
     corresponding estimate is constrained to zero.
  2. For the specified combination of values of the prefix of the features given
     by MULTIFEATURE_ZERO_SEQUENCE, the corresponding estimate is constrained to
     zero.

  Args:
    block_shape: The BlockShape object describing the block structure
    df: A dataframe with columns ID and ESTIMATE consisting of estimates before
      constraints are applied.
    constraint_df: A dataframe with columns ID, QUERY_NAME, and VALUE consisting
      of the constraints to apply. Only the constraints with QUERY_NAME equal to
      FIRST_FEATURE_UB_CONSTRAINT are used in this method, and all other rows
      are ignored.

  Returns:
    A dataframe with columns ID and ESTIMATE consisting of estimates after
    constraints are applied.
  """

  # Merge first feature upper bound values with estimate dataframe
  first_feature_ub_constraint_mask = (
      constraint_df[QUERY_NAME] == FIRST_FEATURE_UB_CONSTRAINT
  )
  first_feature_ub_constraints = (
      constraint_df[first_feature_ub_constraint_mask][[ID, VALUE]]
  )
  df = df.merge(first_feature_ub_constraints, how='left', on=ID)

  # Apply first feature zero constraints for first feature values with an upper
  # bound of zero.
  df[ESTIMATE] = df.apply(
      lambda x: apply_first_feature_zero_constraints(
          block_shape=block_shape,
          est=x[ESTIMATE],
          # Boolean vector of values of first feature to constrain to zero
          zero_indices=[y == 0 for y in x[VALUE]]),
      axis=1
  )

  # Apply multifeature zero constraints for specified feature combination,
  # ensuring that we do not constrain again if that value of the first feature
  # was already constrained to zero.
  # This constraint is used in the census application to enforce that there are
  # no individuals of housing type 3 ("nursing facilities / skilled nursing
  # facilities") with voting age value 0 (age < 18). This constraint is applied
  # to every region in the census application.
  df[ESTIMATE] = df.apply(
      lambda x: apply_multifeature_zero_constraints(
          block_shape=block_shape,
          est=x[ESTIMATE],
          # Test if first feature value was already constrained to zero above
          already_constrained=(x[VALUE][MULTIFEATURE_ZERO_SEQUENCE[0]] == 0),
          feature_values=MULTIFEATURE_ZERO_SEQUENCE),
      axis=1
  )

  return df[[ID, ESTIMATE]]
