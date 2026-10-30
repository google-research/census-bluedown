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
import pandas as pd
from census_bluedown import constants
from census_bluedown import estimate


Estimate = estimate.Estimate
Constraint = estimate.Constraint

ID = constants.ID
VALUE = constants.VALUE
ESTIMATE = constants.ESTIMATE
QUERY_NAME = constants.QUERY_NAME
FIRST_FEATURE_UB_CONSTRAINT = constants.FIRST_FEATURE_UB_CONSTRAINT
MULTIFEATURE_ZERO_SEQUENCE = constants.MULTIFEATURE_ZERO_SEQUENCE


def apply_linear_constraint(
    est: Estimate,
    cons: Constraint,
) -> Estimate:
  """Perform restricted least squares regression to add the given constraint.

  Computes the variance-minimizing estimate x of the estimated value est subject
  to the linear equality constraints R @ x = c. This is also called
  constrained least squares regression.

  Args:
    est: The estimate to constrain.
    cons: The Constraint object.

  Returns:
    The constrained estimate and updated covariance.
  """
  return cons.apply(est)


def _get_constraint_class(est: Estimate) -> type[Constraint]:
  if not isinstance(est.cov, estimate.Covariance):
    raise TypeError(f'Expected Covariance, got {type(est.cov).__name__}')
  return est.cov.constraint_class


def apply_sum_constraint(
    est: Estimate,
    total: int,
) -> Estimate:
  """Wrapper to apply sum constraint to estimate.

  Computes the variance-minimizing estimate x of the estimated value est subject
  to the linear equality constraint that sum(x) = total.

  This constraint is applied to the statewide totals in the census application.

  Args:
    est: The estimate to constrain.
    total: The value of the sum.

  Returns:
    The constrained estimate and updated covariance.
  """
  cons_cls = _get_constraint_class(est)
  cons = cons_cls.sum(total)
  return apply_linear_constraint(est, cons)


def apply_first_feature_zero_constraints(
    est: Estimate,
    zero_indices: Sequence[bool],
) -> Estimate:
  """Wrapper to apply first feature zero constraints to estimate.

  The variables to constrain to zero are certain values of the first feature,
  as specified by the boolean sequence zero_indices. Computes the variance-
  minimizing estimate x of the estimated value est subject to the linear
  equality constraint that all estimates equal zero for the specified values of
  the first feature.

  This constraint is used in the census application to enforce that for housing
  types with no units in a region, the corresponding estimate is zero. One could
  also write a similar method to apply constraints to other asymmetric features.

  Args:
    est: The estimate to constrain.
    zero_indices: A sequence indicating which values of the first feature should
      be constrained to zero.

  Returns:
    The constrained estimate and updated covariance.
  """
  cons_cls = _get_constraint_class(est)
  cons = cons_cls.first_feature_zero_constraints(zero_indices)
  return apply_linear_constraint(est, cons)


def apply_multifeature_zero_constraints(
    est: Estimate,
    feature_values: Sequence[int],
) -> Estimate:
  """Wrapper to apply multifeature zero constraints to estimate.

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
    est: The estimate to constrain.
    feature_values: The asymmetric feature combination to constrain to zero. The
      specified values correspond to a prefix of the features, so that the ith
      entry is the value of the ith feature.

  Returns:
    The constrained estimate and updated covariance.
  """
  cons_cls = _get_constraint_class(est)
  cons = cons_cls.multifeature_zero_constraints(feature_values)
  return apply_linear_constraint(est, cons)


def apply_zero_constraints(
    est: Estimate,
    first_feature_zero_indices: Sequence[bool],
    multifeature_zero_sequence: (
        Sequence[int] | None
    ) = MULTIFEATURE_ZERO_SEQUENCE,
) -> Estimate:
  """Applies combined first-feature and multifeature zero constraints in one pass."""
  cons_cls = _get_constraint_class(est)
  cons = cons_cls.zero_constraints(
      first_feature_zero_indices, multifeature_zero_sequence
  )
  return apply_linear_constraint(est, cons)


def constrain_blocks(
    df: pd.DataFrame,
    constraint_df: pd.DataFrame,
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
  first_feature_ub_constraints = constraint_df[
      first_feature_ub_constraint_mask
  ][[ID, VALUE]]
  df = df.merge(first_feature_ub_constraints, how='left', on=ID)

  # Apply combined first-feature and multifeature zero constraints in one pass.
  # In the census application, this constrains housing types with no units to
  # zero and enforces that there are no individuals of housing type 3 ("nursing
  # facilities / skilled nursing facilities") with voting age value 0 (< 18).
  df[ESTIMATE] = df.apply(
      lambda x: apply_zero_constraints(
          est=x[ESTIMATE],
          first_feature_zero_indices=[y == 0 for y in x[VALUE]],
          multifeature_zero_sequence=MULTIFEATURE_ZERO_SEQUENCE,
      ),
      axis=1,
  )

  return df[[ID, ESTIMATE]]
