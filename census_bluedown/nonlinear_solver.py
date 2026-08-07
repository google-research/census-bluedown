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

"""Nonlinear solver for the block hierarchical postprocessing algorithm.
"""

from collections.abc import Sequence
import datetime

from absl import logging

import numpy as np

from census_bluedown import block
from census_bluedown import constants
from census_bluedown import estimate
from ortools.math_opt.python import mathopt


SOLVER = constants.SOLVER

MULTIFEATURE_ZERO_SEQUENCE = constants.MULTIFEATURE_ZERO_SEQUENCE


def do_solve(
    model: mathopt.Model,
    *,
    timeout_minutes: int = 30
) -> mathopt.SolveResult:
  """Solve the given model using Gurobi or SCIP.

  Args:
    model: The model to solve.
    timeout_minutes: The timeout for the solver.

  Returns:
    The result of the solve.

  Raises:
    ValueError: If the solver is not GUROBI_SOLVER or SCIP_SOLVER.
    RuntimeError: If the solver fails to find an optimal solution.
  """
  if SOLVER == constants.SolverType.GUROBI:
    logging.info('Invoking Gurobi solver.')
    time_limit = datetime.timedelta(minutes=timeout_minutes)
    result = mathopt.solve(
        model=model,
        solver_type=mathopt.SolverType.GUROBI,
        params=mathopt.SolveParameters(
            time_limit=time_limit,
            gurobi=mathopt.GurobiParameters({
                'numericFocus': '3',
                'BarHomogeneous': '1'})),
        msg_cb=lambda msgs: mathopt.log_messages(msgs, prefix='[solver] '),
    )
  elif SOLVER == constants.SolverType.SCIP:
    logging.info('Invoking SCIP solver.')
    result = mathopt.solve(model, mathopt.SolverType.GSCIP,
                           params=mathopt.SolveParameters(enable_output=True))
  else:
    raise ValueError(f'Unsupported solver: {SOLVER}')

  if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
    raise RuntimeError(f'model failed to solve: {result.termination}')

  logging.info('Optimal solution returned.')
  return result


def variable_values(
    result: mathopt.SolveResult,
    ests: Sequence[mathopt.Variable | float]
) -> Sequence[float]:
  """Extract variable values from solve result."""
  return [result.variable_values(est)
          if isinstance(est, mathopt.Variable)
          else est
          for est in ests]


def integer_variable_values(
    result: mathopt.SolveResult,
    ests: Sequence[mathopt.Variable | float]
) -> Sequence[float]:
  """Extract integer variable values from solve result."""
  var_vals = variable_values(result, ests)
  rounded_vals = np.rint(var_vals).astype(int)
  if not np.allclose(var_vals, rounded_vals, rtol=0, atol=1e-4):
    raise RuntimeError(f'Variable values {var_vals} are not integers.')
  return list(rounded_vals)


def add_sum_range_constraint(
    model: mathopt.Model,
    variables: Sequence[mathopt.Variable],
    lower_bound: int,
    upper_bound: int,
    offsets: Sequence[int] | None = None
):
  """Add a constraint to the model for the sum of the provided variables.

  If the upper bound is zero, the constraint is that each variable must equal
  zero. If the lower and upper bounds are equal, the constraint is that the sum
  of the variables must equal this value. Otherwise, the constraint is that the
  sum of the variables must be between the lower and upper bounds.

  If offsets are provided, then variables are assumed to be {0, 1}-valued, and
  the constraints apply to the sum of the variables and the offsets. This is
  used for the rounder pass where the offsets are the floored variable values
  from the previous pass.

  Args:
    model: The model to add constraints to.
    variables: The variables to sum.
    lower_bound: The lower bound for the sum.
    upper_bound: The upper bound for the sum.
    offsets: Offsets for each variable to add to the sum, used for the rounder
      pass where the variables are {0, 1}-valued and the offsets are the floored
      variable values from the previous pass.
  """
  if upper_bound == 0:
    # Since the sum of nonnegative variables is zero, each individual variable
    # must be zero. We have already assigned these variables to zero, so we
    # don't need to add an additional constraint.
    return

  offset_sum = 0
  if offsets is not None:
    offset_sum = sum(offsets)
    if any(val < 0 for val in offsets):
      raise ValueError(f'Offsets must be nonnegative and not {offsets}.')
    if offset_sum + len(offsets) < lower_bound:
      raise ValueError('Sum of offsets must be greater than or equal to '
                       f'{lower_bound} minus its length and not {offsets}.')
    if offset_sum > upper_bound:
      raise ValueError('Sum of offsets must be less than or equal to '
                       f'{upper_bound} and not {offsets}.')

  # If offsets are provided, add them to the sum of the variables.
  variable_sum = mathopt.fast_sum(variables) + offset_sum

  # Add upper and lower bounds to the sum of variables.
  if lower_bound == upper_bound:
    model.add_linear_constraint(variable_sum == float(lower_bound))
  else:
    # If offsets are provided, check if upper and lower bound constraints are
    # redundant before adding them.
    if offsets is None or offset_sum < lower_bound:
      model.add_linear_constraint(variable_sum >= float(lower_bound))
    if offsets is None or offset_sum + len(offsets) > upper_bound:
      model.add_linear_constraint(variable_sum <= float(upper_bound))


def add_feature_constraints(
    block_shape: block.BlockShape,
    model: mathopt.Model,
    variables: Sequence[mathopt.Variable],
    lower_bounds: Sequence[int],
    upper_bounds: Sequence[int],
    offsets: Sequence[int] | None = None):
  """Add constraints to the model for the provided variables.

  This function adds the first feature upper and lower bound constraints for the
  specified variables.

  In the census application, this function will be used to add the housing type
  upper and lower bound constraints.  We don't need to add the multifeature zero
  constraint, since these variables are already defined to be zero.

  Args:
    block_shape: The BlockShape object describing the block structure.
    model: The model to add constraints to.
    variables: The variables to constrain.
    lower_bounds: The lower bounds for each feature value.
    upper_bounds: The upper bounds for each feature value.
    offsets: Offsets for each variable.

  Raises:
    ValueError if `lower_bounds` or `upper_bounds` have length different
    from `block_shape.shape[0]`.
  """
  if (len(lower_bounds) != block_shape.shape[0] or
      len(upper_bounds) != block_shape.shape[0]):
    raise ValueError('Length of bounds must match size of first feature.')

  values_per_chunk = block_shape.length // block_shape.shape[0]

  for i in range(block_shape.shape[0]):
    next_offsets = (offsets[values_per_chunk * i : values_per_chunk * (i+1)]
                    if offsets is not None else None)
    add_sum_range_constraint(
        model=model,
        variables=variables[values_per_chunk * i : values_per_chunk * (i+1)],
        lower_bound=lower_bounds[i],
        upper_bound=upper_bounds[i],
        offsets=next_offsets
    )


def add_parent_sum_constraints(
    model: mathopt.Model,
    model_variables: dict[str, Sequence[mathopt.Variable]],
    parent_val: Sequence[int],
    child_sum_offsets: Sequence[int] | None = None,
):
  """Add constraints that the sum of children equals parent for each type.

  In the census application, parent_val will be a list of length 2016 consisting
  of the number of people for each combination of attributes. The parameter
  model_variables consists of a list whose length is the number of child nodes
  and whose elements are lists of length 2016, where the inner lists are the
  variables corresponding to each combination of attributes for that child node.

  Args:
    model: The model to add constraints to.
    model_variables: A dictionary mapping child geocodes to lists of variables
      corresponding to the count for each type at that child.
    parent_val: The parent value for each type.
    child_sum_offsets: The offsets to add to the sum of children for each type.
  """
  for i, val in enumerate(parent_val):
    next_offset = 0 if child_sum_offsets is None else child_sum_offsets[i]
    model.add_linear_constraint(
        mathopt.fast_sum([x[i] for x in model_variables.values()]) ==
        float(val - next_offset)
    )


def condition_covariance_matrix(
    block_shape: block.BlockShape,
    covariance_matrix: np.ndarray,
    constrained_total: bool,
    upper_bounds: Sequence[int]
) -> np.ndarray:
  """Condition covariance matrix based on constraints to avoid numerical issues.

  The covariance matrix is PSD, but due to constraints that were already applied
  in the linear pass, the matrix may be singular. This function modifies the
  covariance matrix to make it positive-definite by adding terms in the
  direction of the constraints. This does not affect the optimization solution,
  since the same constraints are enforced in the optimization problem.

  Three constraints are handled, corresponding to the equality constraints that
  may have been applied in the linear pass:
  1. The multifeature zero constraint
  2. The first feature upper bound constraint in the case that the upper bound
      is zero
  3. The total constraint for nodes whose total value is constrained (the root,
      as well as nodes corresponding to the total for a state without an AIAN
      region)

  Args:
    block_shape: The BlockShape object describing the block structure.
    covariance_matrix: The covariance matrix to condition.
    constrained_total: Whether the covariance matrix constrains the total.
    upper_bounds: The upper bounds for each first feature value.

  Returns:
    The conditioned covariance matrix.
  """
  condition_offset = np.zeros_like(covariance_matrix)
  if constrained_total:
    tiled_copies = [x // 2 for x in covariance_matrix.shape]
    condition_offset += (constants.CONDITION_SCALING *
                         np.tile(np.array([[0, 0], [0, 1]]), tiled_copies))

  multifeature_zero_indices = (24, 25, 26, 27)
  if upper_bounds[MULTIFEATURE_ZERO_SEQUENCE[0]] > 0:
    # Ensure multifeature zero constraint directions are non-singular.
    diag_values = np.zeros(block_shape.compressed_length)
    for i in multifeature_zero_indices:
      diag_values[i] = 1
    condition_offset += constants.CONDITION_SCALING * np.diag(diag_values)

  # Ensure upper bound zero constraint directions are non-singular.
  values_per_first_feature = (block_shape.compressed_length //
                              block_shape.shape[0])
  diag_values = [1 if upper_bounds[i // values_per_first_feature] == 0 else 0
                 for i in range(block_shape.compressed_length)]
  condition_offset += constants.CONDITION_SCALING * np.diag(diag_values)

  return covariance_matrix + condition_offset


def generate_full_objective_terms(
    model: mathopt.Model | None,
    block_shape: block.BlockShape,
    geocode: str,
    est: estimate.Estimate,
    variables: Sequence[mathopt.Variable | float],
    first_feature_upper_bounds: Sequence[int],
    constrained_total: bool
) -> Sequence[mathopt.QuadraticTypes]:
  """Generate objective terms for the model.

  Uses compressed precision matrix to generate objective terms for the model.
  The sum of the objective terms is given by the formula (y - b)^T @ P @ (y - b)
  where y is the estimate vector and b is the variable vector. The precision
  matrix P is the inverse of the covariance matrix of the estimate.

  If model is None, then the variables are assumed to be floats rather than
  mathopt variables, and the function will simply return a list of floats and
  will not add any partial sum dummy variables or constraints to the model. This
  is used for testing.

  Args:
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block, used for naming the partial sum dummy
      variables.
    est: The estimate of the block.
    variables: The variables for the block.
    first_feature_upper_bounds: The upper bounds for each first feature value.
    constrained_total: Whether the total value is constrained in the input
      estimate.

  Returns:
    A list of quadratic expressions or floats, depending on whether a model is
    provided.
  """
  if block_shape.num_features != block_shape.num_asymmetric_features +  1:
    raise ValueError('Current implementation of generate_full_objective_terms'
                     'requires there to be a single symmetric feature.')

  # Split the variables and estimate values into sublists corresponding to each
  # possible value of the asymmetric features.
  symmetric_feature_length = block_shape.uncompressed_symmetric_length
  variable_sublists = [
      variables[i:i+symmetric_feature_length]
      for i in range(0, block_shape.length, symmetric_feature_length)
  ]
  estimate_sublists = [
      est.val[i:i+symmetric_feature_length]
      for i in range(0, block_shape.length, symmetric_feature_length)
  ]

  # Generate partial sums of the estimate and variable sublists.
  estimate_partial_sums = [sum(x) for x in estimate_sublists]

  if model is None:
    # If model is None (which is used for testing purposes), then the variables
    # are floats rather than mathopt variables and we don't need to generate
    # partial sum dummy variables or constraints.
    partial_sum_variables = [
        sum(x) for x in variable_sublists
    ]
  else:
    # Instead of generating mathopt.LinearExpressions for sums of variables,
    # generate partial sum dummy variables corresponding to the sum over
    # possible values of the symmetric feature for each combination of values of
    # the asymmetric features.
    partial_sum_variables = [
        model.add_variable(lb=0, name=f'x_{geocode}_partial_sum_{i}')
        for i in range(block_shape.asymmetric_length)
    ]
    for i in range(block_shape.asymmetric_length):
      model.add_linear_constraint(
          partial_sum_variables[i] == mathopt.fast_sum(variable_sublists[i]))

  # Differences between variables and estimates for each asymmetric feature.
  # Can't use np.array because it doesn't support mathopt.linearExpressions
  differences = [
      [x - y for x, y in zip(variable_sublists[i], estimate_sublists[i])]
      for i in range(block_shape.asymmetric_length)
  ]

  # Generate objective function terms for each combination of values of the
  # asymmetric features.
  if constants.CONSTRAIN_BOTTOM_UP:
    # If bottom-up constraints are used, then the covariance matrix is singular
    # and we need to condition it to avoid numerical issues and ensure PSD.
    cov_matrix = condition_covariance_matrix(
        block_shape=block_shape,
        covariance_matrix=est.cov,
        constrained_total=constrained_total,
        upper_bounds=first_feature_upper_bounds)
  else:
    cov_matrix = est.cov
  compressed_precision_matrix = np.linalg.inv(cov_matrix)

  objective_terms = []
  for asym_part_1 in range(block_shape.asymmetric_length):
    for asym_part_2 in range(asym_part_1 + 1):
      # Generate coefficients for the objective terms corresponding to the
      # current combination of values of the asymmetric features.
      submatrix = compressed_precision_matrix[
          2*asym_part_1:2*(asym_part_1+1),
          2*asym_part_2:2*(asym_part_2+1)
      ]
      first_coefficient = sum(submatrix[0])
      second_coefficient = sum(submatrix[1])

      # Double the off-diagonal coefficients, since the precision matrix is
      # symmetric and we're only considering the lower triangular part.
      if asym_part_1 != asym_part_2:
        first_coefficient *= 2
        second_coefficient *= 2

      # First objective term is first row of submatrix, times dot product of
      # subvectors of quadratic form inputs
      # Can't use np.dot because it doesn't support mathopt.LinearExpression
      inner_product_terms = [x * y for x, y in zip(differences[asym_part_1],
                                                   differences[asym_part_2])]
      if model is None:
        inner_product = sum(inner_product_terms)
      else:
        inner_product = mathopt.fast_sum(inner_product_terms)
      first_term = first_coefficient * inner_product
      objective_terms.append(first_term)

      # Second objective term is second row of submatrix, times product of sum
      # of entries of subvectors of quadratic form inputs
      sum_difference_1 = (partial_sum_variables[asym_part_1]
                          - estimate_partial_sums[asym_part_1])
      sum_difference_2 = (partial_sum_variables[asym_part_2]
                          - estimate_partial_sums[asym_part_2])
      second_term = second_coefficient * sum_difference_1 * sum_difference_2
      objective_terms.append(second_term)
  return objective_terms


def generate_total_only_objective_terms(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    est: estimate.Estimate,
    variables: Sequence[mathopt.Variable | float],
) -> Sequence[mathopt.QuadraticTypes]:
  """Generate objective term for total-only optimization.

  Uses compressed precision matrix to generate a single objective term
  corresponding to the total query. The coefficient of this term is the inverse
  of the variance of the total query.

  Args:
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block, used for naming the sum dummy variable.
    est: The estimate of the block.
    variables: The variables for the block.

  Returns:
    A list containing either a single mathopt.QuadraticProduct or a float,
    depending on whether the argument variables consists of mathopt.Variables
    or floats.
  """
  if block_shape.num_features != block_shape.num_asymmetric_features +  1:
    raise ValueError('Implementation of generate_total_only_objective_terms'
                     'requires there to be a single symmetric feature.')
  # Compute sum of entries of submatrix of covariance matrix that correspond
  # to summing over the symmetric feature
  variance_of_total = est.cov[1::2, 1::2].sum()

  # Generate objective function term for block total only
  # Use mathopt.fast_sum in the actual run when variables are mathopt.Variables,
  # but for tests on floats, use sum()
  if isinstance(variables[0], mathopt.Variable):
    sum_variable = model.add_variable(lb=0, name=f'x_{geocode}_sum')
    model.add_linear_constraint(
        sum_variable == mathopt.fast_sum(variables))
    linear_difference = sum_variable - sum(est.val)
  else:
    linear_difference = sum(variables) - sum(est.val)
  return [linear_difference * linear_difference / variance_of_total]


def query_index_constrained_to_zero(
    index: int,
    first_feature_upper_bounds: Sequence[int],
    query: tuple[int, ...],
) -> bool:
  """Check if the index of a constrained feature in a query is zero."""
  # if not split by housing type, can't be constrained
  if query[0] == 1:
    return False
  # if split by housing type but not voting age, constraint determined by
  # housing type alone
  elif query[0] == 8 and query[1] == 1:
    indices_per_housing_type = np.prod(query[2:])
    return first_feature_upper_bounds[index // indices_per_housing_type] == 0  # pyrefly: ignore[bad-index]
  # For detailed query, constraint determined by housing type and voting age
  elif query == (8, 2, 2, 63):
    return (index >= 756 and index < 882) or query_index_constrained_to_zero(
        index // 252, first_feature_upper_bounds, (8, 1, 1, 1)
    )
  else:
    raise ValueError(f'Unsupported query shape {query}')


def generate_rounder_detailed_query_objective_terms(
    values: Sequence[float],
    floored_values: Sequence[float],
    variables: Sequence[mathopt.Variable],
    first_feature_upper_bounds: Sequence[int],
) -> Sequence[mathopt.LinearTypes]:
  """Generate objective terms for the rounder optimization problem for a query.

  For each variable corresponding to the Detailed query, generate an objective
  term equal to the absolute difference between the rounded and unrounded value.
  This is done using the trick that for a {0, 1}-valued variable x and [0, 1]-
  valued variable y, we can write |x - y| = (1 - 2 * y) * x + y. In the rounding
  pass, y corresponds to the fractional part of the detailed query, and x
  corresponds to whether the query is rounded up or down.

  Args:
    values: the unrounded estimated value for each variable
    floored_values: the floored estimated value for each variable
    variables: the variables for the block
    first_feature_upper_bounds: the upper bounds on the first feature for each
      geocode

  Returns:
    A list of mathopt Variables corresponding to the objective terms.
  """
  objective_terms = []
  diffs = np.asarray(values) - np.asarray(floored_values)

  for i, (diff, var) in enumerate(
      zip(diffs, variables)
  ):
    if query_index_constrained_to_zero(i, first_feature_upper_bounds,
                                       constants.DETAILED_QUERY_SHAPE):
      continue
    objective_terms.append((1 - 2 * diff) * var)  # or + diff

  return objective_terms


def generate_rounder_single_query_objective_terms(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    values: Sequence[float],
    floored_values: Sequence[float],
    variables: Sequence[mathopt.Variable],
    first_feature_upper_bounds: Sequence[int],
    query: tuple[int, ...],
) -> Sequence[mathopt.LinearTypes]:
  """Generate objective terms for the rounder optimization problem for a query.

  For each partial sum variable corresponding to the query, generate an
  objective term equal to the absolute difference between the rounded and
  unrounded value of the partial sum. This is done using the trick that for a
  {0, 1}-valued variable x and [0, 1]-valued variable y, we can write |x - y| =
  (1 - 2 * y) * x + y. In the rounding pass, y corresponds to the fractional
  part of a query, and x corresponds to whether the query is rounded up or down.
  This approach assumes that the query value from the L2 pass is replaced with
  either its floored value or one more than its floored value, so that the
  variable x is either 0 or 1.

  Args:
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block, used for naming the dummy variables.
    values: the unrounded estimated value for each variable
    floored_values: the floored estimated value for each variable
    variables: the variables for the block
    first_feature_upper_bounds: the upper bounds on the first feature for each
      geocode
    query: the shape of the query

  Returns:
    A list of mathopt Variables corresponding to the objective terms.
  """
  if query == constants.DETAILED_QUERY_SHAPE:
    return generate_rounder_detailed_query_objective_terms(
        values=values,
        floored_values=floored_values,
        variables=variables,
        first_feature_upper_bounds=first_feature_upper_bounds,
    )

  aggregated_variables = mathopt_aggregate_to_query_shape(
      block_shape=block_shape,
      values=variables,
      target_shape=query)

  diffs = np.asarray(values) - np.asarray(floored_values)
  part_sum_diffs = aggregate_to_query_shape(diffs, query)

  objective_terms = []
  for i, (diff, aggvar) in enumerate(zip(part_sum_diffs, aggregated_variables)):
    if query_index_constrained_to_zero(i, first_feature_upper_bounds, query):
      continue
    dummy_var = model.add_variable(lb=0, name=f'x_{geocode}_{query}_abs_{i}')
    model.add_linear_constraint(dummy_var >= np.rint(diff) - aggvar)
    model.add_linear_constraint(dummy_var >= aggvar - np.rint(diff))
    objective_terms.append(dummy_var)

  return objective_terms


def generate_rounder_full_objective_terms(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    values: Sequence[float],
    floored_values: Sequence[float],
    variables: Sequence[mathopt.Variable],
    first_feature_upper_bounds: Sequence[int],
) -> Sequence[mathopt.LinearTypes]:
  """Generate objective terms for the rounder full optimization problem.

  Returns objective terms equal to the absolute difference between each rounded
  and unrounded value.

  Arguments:
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block, used for naming the dummy variables.
    values: the unrounded estimated value for each variable
    floored_values: the floored estimated value for each variable
    variables: the variables for the block
    first_feature_upper_bounds: the upper bounds on the first feature for each
      geocode

  Returns:
    A list of mathopt Variables corresponding to the objective terms.
  """
  if len(values) != len(floored_values):
    raise ValueError('Length of values and floored_values must match.')
  if len(values) != len(variables):
    raise ValueError('Length of values and variables must match.')

  objective_terms = []
  for query in constants.CENSUS_ROUNDER_SECOND_PASS_QUERY_SHAPES:
    objective_terms.extend(generate_rounder_single_query_objective_terms(
        model=model,
        block_shape=block_shape,
        geocode=geocode,
        values=values,
        floored_values=floored_values,
        variables=variables,
        first_feature_upper_bounds=first_feature_upper_bounds,
        query=query,
    ))

  return objective_terms


def generate_rounder_total_only_objective_terms(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    values: Sequence[float],
    floored_values: Sequence[float],
    variables: Sequence[mathopt.Variable],
    first_feature_upper_bounds: Sequence[int],
) -> Sequence[mathopt.LinearTypes]:
  """Generate objective term for the rounder total only optimization problem.

  Returns a single objective term equal to the absolute difference between the
  rounded and unrounded total values.

  Arguments:
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block, used for naming the dummy variable.
    values: the unrounded estimated value for each variable
    floored_values: the floored estimated value for each variable
    variables: the variables for the block
    first_feature_upper_bounds: the upper bounds on the first feature for each
      geocode

  Returns:
    A list containing a single mathopt.Variable as the term to minimize.
  """
  if len(values) != len(floored_values):
    raise ValueError('Length of values and floored_values must match.')
  if len(values) != len(variables):
    raise ValueError('Length of values and variables must match.')

  return generate_rounder_single_query_objective_terms(
      model=model,
      block_shape=block_shape,
      geocode=geocode,
      values=values,
      floored_values=floored_values,
      variables=variables,
      first_feature_upper_bounds=first_feature_upper_bounds,
      query=constants.TOTAL_QUERY_SHAPE,
  )


def aggregate_to_query_shape(
    values: Sequence[float],
    target_shape: tuple[int, ...],
    initial_shape: tuple[int, ...] = constants.DETAILED_QUERY_SHAPE
) -> Sequence[float]:
  """Aggregate values to the shape of a query.

  If the target shape is the census asymmetric query, then the grouped values
  of the asymmetric feature are aggregated as well.

  Args:
    values: The values to aggregate.
    target_shape: The shape to aggregate to.
    initial_shape: The shape of the values before aggregation.

  Returns:
    A list of floats corresponding to the aggregated values.
  """
  effective_shape = (target_shape
                     if target_shape != constants.CENSUS_ASYMMETRIC_QUERY_SHAPE
                     else (8, 1, 1, 1))
  values = (np.reshape(values, initial_shape)
            .sum(axis=tuple(np.flatnonzero(np.array(effective_shape) == 1)))
            .reshape(-1))
  if target_shape == constants.CENSUS_ASYMMETRIC_QUERY_SHAPE:
    values = np.array([values[0],  # pyrefly: ignore[bad-assignment]
                       values[1] + values[2] + values[3] + values[4],
                       values[5] + values[6] + values[7]])
  return values


def mathopt_aggregate_to_query_shape(
    block_shape: block.BlockShape,
    values: Sequence[mathopt.Variable],
    target_shape: tuple[int, ...],
    initial_shape: tuple[int, ...] = constants.DETAILED_QUERY_SHAPE
) -> Sequence[mathopt.Variable | mathopt.QuadraticTypes]:
  """Use mathopt.fast_sum to aggregate values to the shape of a query.

  The aggregation is performed by summing over the dimensions of the initial
  shape that are not sliced in the target shape. If the target shape is an
  asymmetric query, then the grouped values of the asymmetric feature are
  aggregated as well.

  Args:
    block_shape: The BlockShape object describing the block structure, used to
      specify the structure of the asymmetric query
    values: The values to aggregate.
    target_shape: The shape to aggregate to.
    initial_shape: The shape of the values before aggregation.

  Returns:
    A list of objects corresponding to the aggregated values.
  """
  values = np.reshape(values, initial_shape)  # pyrefly: ignore[no-matching-overload]
  for i in range(len(initial_shape) - 1, -1, -1):
    if target_shape[i] == 1:
      values = np.apply_along_axis(mathopt.fast_sum, i, values)  # pyrefly: ignore[no-matching-overload]
  values = np.reshape(values, -1)  # pyrefly: ignore[no-matching-overload]

  # If the target shape is an asymmetric query, aggregate the grouped values of
  # the asymmetric feature for this query
  if target_shape in block_shape.asymmetric_query_dict:
    asym_query = block_shape.asymmetric_query_dict[target_shape]
    # Aggregate the partial sums corresponding to the asymmetric query's
    # partition of feature values for the sliced feature.
    grouped_values = np.split(values, np.cumsum(asym_query.partition)[:-1])
    values = [mathopt.fast_sum(x) for x in grouped_values]  # pyrefly: ignore[bad-assignment]
  return values


def baseline_single_query_objective_term(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    variables: Sequence[mathopt.Variable],
    query: tuple[int, ...],
    estimate_value: Sequence[float],
    estimate_variance: float,
) -> Sequence[mathopt.QuadraticTypes]:
  """Generate objective term for a single query for baseline L2 optimization.

  A single query is a tuple of integers corresponding to the dimensions of the
  query, corresponding to a partial rollup of some of the features. (For any
  asymmetric queries defined in block_shape, some values of a particular feature
  may be grouped together.) The objective term is the sum of the squared
  differences between the estimate values and the partial sums of the variables
  corresponding to the rollup for that query.

  Args:
    model: The model the objective term will be added to, used for defining
      partial sum variables.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block, used for naming the partial sum variables
    variables: The variables for the block.
    query: The shape of the query.
    estimate_value: The estimates value of the query.
    estimate_variance: The variance of the estimate.

  Returns:
    A list of quadratic expressions whose sum is the objective term
    corresponding to this query.
  """
  if query == constants.DETAILED_QUERY_SHAPE:
    partial_sum_variables = variables
  else:
    variables = mathopt_aggregate_to_query_shape(block_shape, variables, query)  # pyrefly: ignore[bad-assignment]
    partial_sum_variables = [
        model.add_variable(lb=0, name=f'x_{geocode}_{query}_partial_sum_{i}')
        for i in range(len(variables))
    ]
    for x, y in zip(variables, partial_sum_variables):
      model.add_linear_constraint(x == y)

  return [(x - y) * (x - y) / estimate_variance
          for x, y in zip(partial_sum_variables, estimate_value)]


def generate_objective_terms(
    objective_function_type: constants.ObjectiveFunctionType,
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    est: estimate.Estimate,
    variables: Sequence[mathopt.Variable | float],
    floored_values: np.ndarray | None = None,
    first_feature_upper_bounds: Sequence[int] | None = None,
    constrained_total: bool = False,
) -> Sequence[mathopt.QuadraticTypes]:
  """Wrapper to generate objective terms for the model.

  This function calls the appropriate function to generate objective terms
  depending on the value of the objective function type. It can either generate
  an objective function term for the full optimization problem or an objective
  function term for the total-only optimization problem.

  Args:
    objective_function_type: The type of objective function to generate.
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block.
    est: The estimate of the block.
    variables: The variables for the block.
    floored_values: The floored values for the block, if this is a rounder pass.
      This argument could be removed but avoids recomputing the floored values.
    first_feature_upper_bounds: The upper bounds for the first feature of each
      type.
    constrained_total: Whether the total pass values are constrained in the
      input estimate.

  Returns:
    A list of quadratic expressions or floats, depending on whether a model is
    provided.
  """
  match objective_function_type:
    case constants.ObjectiveFunctionType.FULL:
      return generate_full_objective_terms(
          model=model,
          block_shape=block_shape,
          geocode=geocode,
          est=est,
          variables=variables,
          first_feature_upper_bounds=first_feature_upper_bounds,  # pyrefly: ignore[bad-argument-type]
          constrained_total=constrained_total,
      )
    case constants.ObjectiveFunctionType.TOTAL_ONLY:
      return generate_total_only_objective_terms(
          model=model,
          block_shape=block_shape,
          geocode=geocode,
          est=est,
          variables=variables,
      )
    case constants.ObjectiveFunctionType.FULL_ROUNDER:
      return generate_rounder_full_objective_terms(
          model=model,
          block_shape=block_shape,
          geocode=geocode,
          values=est.val,  # pyrefly: ignore[bad-argument-type]
          floored_values=floored_values,  # pyrefly: ignore[bad-argument-type]
          variables=variables,  # pyrefly: ignore[bad-argument-type]
          first_feature_upper_bounds=first_feature_upper_bounds,  # pyrefly: ignore[bad-argument-type]
      )
    case constants.ObjectiveFunctionType.TOTAL_ONLY_ROUNDER:
      return generate_rounder_total_only_objective_terms(
          model=model,
          block_shape=block_shape,
          geocode=geocode,
          values=est.val,  # pyrefly: ignore[bad-argument-type]
          floored_values=floored_values,  # pyrefly: ignore[bad-argument-type]
          variables=variables,  # pyrefly: ignore[bad-argument-type]
          first_feature_upper_bounds=first_feature_upper_bounds,  # pyrefly: ignore[bad-argument-type]
      )
    case _:
      raise ValueError(
          f'Unsupported objective function type: {objective_function_type}'
      )


def baseline_objective_terms(
    objective_function_type: constants.ObjectiveFunctionType,
    model: mathopt.Model,
    block_shape: block.BlockShape,
    geocode: str,
    est_dict: dict[tuple[int, ...], tuple[np.ndarray, float]],
    variables: Sequence[mathopt.Variable],
) -> Sequence[mathopt.QuadraticTypes]:
  """Wrapper to generate objective terms for the baseline L2 optimization.

  This function calls the appropriate function to generate objective terms
  depending on the value of the objective function type. It can either generate
  an objective function term for the full baseline optimization problem or an
  objective function term for the total-only baseline optimization problem.

  Args:
    objective_function_type: The type of objective function to generate.
    model: The model to generate objective terms for.
    block_shape: The BlockShape object describing the block structure.
    geocode: The geocode of the block.
    est_dict: a dictionary mapping query shapes to tuples. The first element of
      the tuple consists of the estimate values for the query. The second
      element of the tuple consists of the variance of the estimate.
    variables: The variables for the block.

  Returns:
    A list of quadratic expressions whose sum is the objective term
    corresponding to this query.
  """
  match objective_function_type:
    case constants.ObjectiveFunctionType.FULL:
      objective_terms = []
      for query in est_dict:
        objective_terms.extend(baseline_single_query_objective_term(
            model=model,
            block_shape=block_shape,
            geocode=geocode,
            variables=variables,
            query=query,
            estimate_value=est_dict[query][0],  # pyrefly: ignore[bad-argument-type]
            estimate_variance=est_dict[query][1],
        ))
      return objective_terms
    case constants.ObjectiveFunctionType.TOTAL_ONLY:
      total_shape = constants.TOTAL_QUERY_SHAPE
      return baseline_single_query_objective_term(
          model=model,
          block_shape=block_shape,
          geocode=geocode,
          variables=variables,
          query=total_shape,
          estimate_value=est_dict[total_shape][0],  # pyrefly: ignore[bad-argument-type]
          estimate_variance=est_dict[total_shape][1],
      )
    case _:
      raise ValueError(
          f'Unsupported objective function type: {objective_function_type}'
      )


def index_constrained_to_zero(
    block_shape: block.BlockShape,
    variable_index: int,
    first_feature_upper_bounds: Sequence[int],
) -> bool:
  """Determine if variable should be constrained to zero.

  The variables that should be constrained to zero are those satisfying the
  multifeature zero constraints and those for which the first feature upper
  bound is zero.

  In the census application, the multifeature zero constraint is the nursing
  facility constraint for each child node in the optimization problem,
  constraining every variable with housing type 3 and voting age type 0 to be
  zero. This corresponds to the index range [756, 882).

  Args:
    block_shape: The BlockShape object describing the block structure.
    variable_index: The index of the variable.
    first_feature_upper_bounds: The upper bounds for each first feature type.

  Returns:
    True if the variable should be constrained to zero, False otherwise.
  """
  # Check first feature upper bound
  first_feature_value = variable_index // (block_shape.length //
                                           block_shape.shape[0])
  if first_feature_upper_bounds[first_feature_value] == 0:
    return True

  # Check multifeature zero constraint
  values_per_combination = block_shape.length
  first_value = 0

  for i, feature_value in enumerate(MULTIFEATURE_ZERO_SEQUENCE):
    values_per_combination //= block_shape.shape[i]
    first_value += values_per_combination * feature_value

  return first_value <= variable_index < first_value + values_per_combination


def add_variable(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    objective_function_type: constants.ObjectiveFunctionType,
    upper_bound: int,
    first_feature_upper_bounds: Sequence[int],
    geocode: str,
    variable_index: int,
) -> mathopt.Variable | float:
  """Add a variable to the model.

  The variable can be either a non-negative float or a {0,1}-valued integer
  variable, depending on the objective function type.

  Args:
    model: The model to add the variable to.
    block_shape: The BlockShape object describing the block structure.
    objective_function_type: The type of objective function used for the model,
      which determines whether variables are floats or {0, 1}-valued integers.
    upper_bound: The upper bound of the variable, ignored for integer variables.
    first_feature_upper_bounds: The upper bound of the first feature type. This
      is used to constrain the variable to zero if the first feature upper bound
      is zero.
    geocode: The geocode of the block, used for naming the variable.
    variable_index: The index of the variable, used for naming the variable.

  Returns:
    A mathopt.Variable corresponding to the variable added to the model.

  Raises:
    ValueError: If the objective function is neither L2 nor rounder pass.
  """
  if index_constrained_to_zero(
      block_shape=block_shape,
      variable_index=variable_index,
      first_feature_upper_bounds=first_feature_upper_bounds):
    return 0.0

  if objective_function_type in constants.ROUNDER_PASSES:
    # Use integer variables for the rounder pass.

    # As an alternative to using {0, 1}-valued variables with an offset, we
    # could use {offset, offset + 1}-valued variables with no offset. This
    # would make it simpler to apply the constraints, but it is unclear how
    # it would affect the performance of the solver. The current formulation
    # is consistent with the formulation of the optimization problem in the
    # TopDown paper.
    return model.add_binary_variable(name='x_'+geocode+'_'+str(variable_index))

  return model.add_variable(
      lb=0, ub=upper_bound, name='x_'+geocode+'_'+str(variable_index)
    )


def add_variables(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    objective_function_type: constants.ObjectiveFunctionType,
    upper_bounds: int | Sequence[int],
    first_feature_upper_bounds: Sequence[int],
    geocode: str,
    num_variables: int,
) -> Sequence[mathopt.Variable | float]:
  """Wrapper to add multiple variables to the model."""
  if isinstance(upper_bounds, int):
    upper_bounds = [upper_bounds] * num_variables
  return [
      add_variable(
          model=model,
          block_shape=block_shape,
          objective_function_type=objective_function_type,
          upper_bound=upper_bounds[i],
          first_feature_upper_bounds=first_feature_upper_bounds,
          geocode=geocode,
          variable_index=i,
      )
      for i in range(num_variables)
  ]


def generate_and_constrain_node_variables(
    model: mathopt.Model,
    block_shape: block.BlockShape,
    objective_function_type: constants.ObjectiveFunctionType,
    num_variables: int,
    type_upper_bounds: Sequence[int],
    first_feature_upper_bounds: Sequence[int],
    first_feature_lower_bounds: Sequence[int],
    offsets: None | np.ndarray,
    geocode: str,
    total_pass_values: None | dict[str, Sequence[float]],
) -> Sequence[mathopt.Variable | float]:
  """Add (child) node variables to the model and add constraints to them.

  The constraints added to the node are as follows:
  1. The first feature upper and lower bound constraints are respected.
  2. The multifeature zero sequence constraints are added if not redundant. That
      is, if the upper bound for housing type 3 was zero, then these variables
      have already been constrained to zero; otherwise, they will be constrained
      to zero anyway.
  3. If total pass values are provided, then the sum of the variables for the
      node must equal the sum of the total pass values for the node.

  Args:
    model: The model to add the variables and constraints to.
    block_shape: The BlockShape object describing the block structure.
    objective_function_type: The type of objective function used for the model,
      which determines whether variables are floats or {0, 1}-valued integers.
    num_variables: The number of types / variables per node.
    type_upper_bounds: The parent value for each type.
    first_feature_upper_bounds: The upper bounds for each first feature type.
    first_feature_lower_bounds: The lower bounds for each first feature type.
    offsets: The offsets for each variable. Offsets are only provided in the
      rounder pass. If provided, they are equal to the floored values from the
      previous pass.
    geocode: The geocode of the block, used for naming the variables and
      indexing into total_pass_values.
    total_pass_values: A dictionary mapping (child) geocodes to a list of floats
      corresponding to the count for each type for that node. If provided, then
      the sum of the variables for the node must equal the sum of the total
      pass values for that node. If a single integer is provided as the
      dictionary value instead of a list, then that value is assumed to be the
      sum of the total pass values for all types for that node.

  Returns:
    A list of variables added to the model, corresponding to the count for each
    type at this node.
  """
  row_variables = add_variables(
      model=model,
      block_shape=block_shape,
      objective_function_type=objective_function_type,
      upper_bounds=type_upper_bounds,
      first_feature_upper_bounds=first_feature_upper_bounds,
      geocode=geocode,
      num_variables=num_variables
  )

  add_feature_constraints(
      block_shape=block_shape,
      model=model,
      variables=row_variables,  # pyrefly: ignore[bad-argument-type]
      lower_bounds=first_feature_lower_bounds,
      upper_bounds=first_feature_upper_bounds,
      offsets=offsets,  # pyrefly: ignore[bad-argument-type]
  )
  if total_pass_values is not None and geocode in total_pass_values:
    offset_sum = 0
    if offsets is not None and not geocode:
      offset_sum = sum(offsets)
    model.add_linear_constraint(
        mathopt.fast_sum(row_variables)
        == float(sum(total_pass_values[geocode]) - offset_sum)
    )

  return row_variables


def add_subtree_total_constraints(
    model: mathopt.Model,
    model_variables: dict[str, Sequence[mathopt.Variable]],
    subtree_total_dict: dict[str, float],
    offset_sum_dict: None | dict[str, int],
    subtree_patterns: Sequence[str] = constants.SUBTREE_GEOCODE_PATTERNS,
):
  """Add constraints to ensure that the subtree totals are respected.

  These constraints ensure that the sum of the variables for the AIAN and
  non-AIAN (if present) subtrees for a given state equals the total for that
  state (including the offset if provided).

  Args:
    model: The model to add the constraints to.
    model_variables: A dictionary mapping child geocodes to the variables for
      that child (which should correspond to a top-level subtree).
    subtree_total_dict: A dictionary mapping state-level geocodes to totals.
    offset_sum_dict: A dictionary mapping subtree geocodes to offsets,
      specifying the amount to add to the sum of the variables for that subtree
      to account for the floored values from the previous pass if this is a
      rounder pass.
    subtree_patterns: String formats for subtree geocodes with given subtree
      state-level geocode.
  """
  for geocode in subtree_total_dict:
    constraint_variables = []
    offset = 0

    # Iterate over both AIAN and non-AIAN regions for this state, if present.
    for pattern in subtree_patterns:
      subtree_geocode = pattern % geocode
      if subtree_geocode in model_variables:
        constraint_variables.extend(model_variables[subtree_geocode])
        if offset_sum_dict is not None:
          offset += offset_sum_dict[subtree_geocode]
    if not constraint_variables:
      raise ValueError(f'No variables found for subtree geocode {geocode}.')
    model.add_linear_constraint(
        mathopt.fast_sum(constraint_variables) + offset
        == subtree_total_dict[geocode]
    )


def run_node_optimization(
    block_shape: block.BlockShape,
    objective_function_type: constants.ObjectiveFunctionType,
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, estimate.Estimate],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    total_pass_values: None | dict[str, Sequence[float]] = None,
    subtree_total_dict: None | dict[str, int] = None,
) -> tuple[dict[str, Sequence[mathopt.Variable | float]], mathopt.SolveResult]:
  """Solve optimization problem for block hierarchical postprocessing.

  Initializes a model and adds variables and constraints to it. Computes
  the objective function determined by the parameter objective_function_type.
  Solves the model and returns the variables and the solve result.

  The constraints added are as follows:
  Constraints (1-3) as described in the documentation for
  generate_and_constrain_node_variables are added for each child node.
  The following additional constraint is added:
  4. The sum of the variables for each type across all children equals the
      parent value for that type.

  Args:
    block_shape: The BlockShape object describing the block structure.
    objective_function_type: The type of objective function to optimize.
    parent_geocode: The geocode of the parent node.
    child_geocodes: The geocodes of the child nodes.
    parent_val: The parent value for each type.
    child_ests: A dictionary mapping child geocodes to estimates of child nodes.
    children_upper_bound_dict: A dictionary mapping child geocodes to upper
      bounds for each first feature type.
    children_lower_bound_dict: A dictionary mapping child geocodes to lower
      bounds for each first featuretype.
    total_pass_values: A dictionary mapping child geocodes to a list of floats
      corresponding to the count for each type for that child. If provided, then
      the sum of the variables for each child produced in this pass must equal
      the sum of the variable values for that child from the total pass.
    subtree_total_dict: A dictionary mapping state-level geocodes to totals. If
      provided, then the sum of the variables for the AIAN and non-AIAN subtrees
      for that state must equal the total.

  Returns:
    A tuple containing a dictionary and the solve result. The dictionary maps
    child geocodes to a list of variables corresponding to the count for each
    type for that child.
  """
  logging.info(
      'Performing %s optimization for parent node \'%s\' with %d children.',
      objective_function_type.name,
      parent_geocode,
      len(child_geocodes),
  )

  model = mathopt.Model(name=f'{parent_geocode}')
  objective_terms = []

  model_variables = {}
  offset_sum_dict = None
  floored_child_est_sum = (
      np.zeros(block_shape.length)
      if objective_function_type in constants.ROUNDER_PASSES
      else None
  )

  # Add variables, constraints and objective terms for each child node.
  for geocode in child_geocodes:
    child_est = child_ests[geocode]
    floored_child_est = None
    if objective_function_type in constants.ROUNDER_PASSES:
      floored_child_est = np.floor(child_est.val).astype(int)
      floored_child_est_sum += floored_child_est
      if offset_sum_dict is None:
        offset_sum_dict = {}
      offset_sum_dict[geocode] = sum(floored_child_est)

    row_variables = generate_and_constrain_node_variables(
        model=model,
        block_shape=block_shape,
        objective_function_type=objective_function_type,
        num_variables=len(child_est.val),
        type_upper_bounds=parent_val,
        first_feature_upper_bounds=children_upper_bound_dict[geocode],
        first_feature_lower_bounds=children_lower_bound_dict[geocode],
        offsets=floored_child_est,
        geocode=geocode,
        total_pass_values=total_pass_values,
    )

    objective_terms.extend(generate_objective_terms(
        objective_function_type=objective_function_type,
        model=model,
        block_shape=block_shape,
        geocode=geocode,
        est=child_est,
        variables=row_variables,
        floored_values=floored_child_est,
        first_feature_upper_bounds=children_upper_bound_dict[geocode],
        constrained_total=False  # Total has only been constrained in bottom-up
                                 # pass for the root node; for statewide total
                                 # constraints, the per-state constraint is
                                 # added during bottom-up root processing.
    ))
    model_variables[geocode] = row_variables

  logging.info('Adding parent sum constraints')
  add_parent_sum_constraints(
      model=model,
      model_variables=model_variables,
      parent_val=parent_val,
      child_sum_offsets=floored_child_est_sum,  # pyrefly: ignore[bad-argument-type]
  )

  if subtree_total_dict is not None:
    logging.info('Adding subtree total constraints')
    add_subtree_total_constraints(
        model=model,
        model_variables=model_variables,
        subtree_total_dict=subtree_total_dict,  # pyrefly: ignore[bad-argument-type]
        offset_sum_dict=offset_sum_dict,
    )

  logging.info('Minimizing objective function (%d terms)', len(objective_terms))
  model.minimize(0)
  for i, term in enumerate(objective_terms):
    if i % 5000 == 0:
      logging.info('Adding term %d', i)
    model.objective.add(term)
  result = do_solve(model)

  return model_variables, result


def run_baseline_node_optimization(
    block_shape: block.BlockShape,
    objective_function_type: constants.ObjectiveFunctionType,
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, dict[tuple[int, ...], tuple[np.ndarray, float]]],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    total_pass_values: None | dict[str, Sequence[float]] = None,
    subtree_total_dict: None | dict[str, float] = None,
) -> tuple[dict[str, Sequence[mathopt.Variable | float]], mathopt.SolveResult]:
  """Solve baseline optimization problem for block hierarchical postprocessing.

  Initializes a model and adds variables and constraints to it. Computes
  the objective function determined by the parameter objective_function_type.
  Solves the model and returns the variables and the solve result.

  The constraints added are as follows:
  Constraints (1-3) as described in the documentation for
  generate_and_constrain_node_variables are added for each child node.
  The following additional constraint is added:
  4. The sum of the variables for each type across all children equals the
      parent value for that type.

  Args:
    block_shape: The BlockShape object describing the block structure.
    objective_function_type: The type of objective function to optimize.
    parent_geocode: The geocode of the parent node.
    child_geocodes: The geocodes of the child nodes.
    parent_val: The parent value for each type.
    child_ests: A dictionary mapping child geocodes to query dictionaries for
      each child node. Each query dictionary maps query shapes to tuples. The
      first element of the tuple consists of the estimate values for the query.
      The second element of the tuple consists of the variance of the estimate.
    children_upper_bound_dict: A dictionary mapping child geocodes to upper
      bounds for each first feature type.
    children_lower_bound_dict: A dictionary mapping child geocodes to lower
      bounds for each first feature type.
    total_pass_values: A dictionary mapping child geocodes to a list of floats
      corresponding to the count for each type for that child. If provided, then
      the sum of the variables for each child produced in this pass must equal
      the sum of the variable values for that child from the total pass.
    subtree_total_dict: A dictionary mapping state-level geocodes to totals. If
      provided, then the sum of the variables for the AIAN and non-AIAN subtrees
      for that state must equal the total.

  Returns:
    A tuple containing a dictionary and the solve result. The dictionary maps
    child geocodes to a list of variables corresponding to the count for each
    type for that child.
  """
  logging.info(
      'Performing %s baseline pass for parent node %s with %d children.',
      objective_function_type.name,
      parent_geocode,
      len(child_geocodes),
  )
  if objective_function_type in constants.ROUNDER_PASSES:
    raise ValueError('Use method run_node_optimization for baseline rounder '
                     'passes.')
  elif objective_function_type not in constants.L2_PASSES:
    raise ValueError('Objective function must be an L2 pass.')

  model = mathopt.Model(name=f'{parent_geocode}')
  objective_terms = []
  model_variables = {}

  # Add variables, constraints and objective terms for each child node.
  for geocode in child_geocodes:
    child_est_dict = child_ests[geocode]

    row_variables = generate_and_constrain_node_variables(
        model=model,
        block_shape=block_shape,
        objective_function_type=objective_function_type,
        num_variables=len(child_est_dict[constants.DETAILED_QUERY_SHAPE][0]),
        type_upper_bounds=parent_val,
        first_feature_upper_bounds=children_upper_bound_dict[geocode],
        first_feature_lower_bounds=children_lower_bound_dict[geocode],
        offsets=None,
        geocode=geocode,
        total_pass_values=total_pass_values
    )

    objective_terms.extend(baseline_objective_terms(
        objective_function_type=objective_function_type,
        model=model,
        block_shape=block_shape,
        geocode=geocode,
        est_dict=child_est_dict,
        variables=row_variables,  # pyrefly: ignore[bad-argument-type]
    ))
    model_variables[geocode] = row_variables

  add_parent_sum_constraints(
      model=model,
      model_variables=model_variables,
      parent_val=parent_val,
      child_sum_offsets=None,
  )

  if subtree_total_dict is not None:
    add_subtree_total_constraints(
        model=model,
        model_variables=model_variables,
        subtree_total_dict=subtree_total_dict,
        offset_sum_dict=None,
    )

  model.minimize(mathopt.fast_sum(objective_terms))
  result = do_solve(model)
  return model_variables, result


def run_node_optimization_passes(
    block_shape: block.BlockShape,
    objective_function_types: Sequence[constants.ObjectiveFunctionType],
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, estimate.Estimate],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    subtree_total_dict: None | dict[str, int] = None
) -> dict[str, Sequence[int]]:
  """Run optimization passes for block hierarchical postprocessing.

  Args:
    block_shape: The BlockShape object describing the block structure.
    objective_function_types: The list of objective function types to run.
      Must contain at least one L2 pass and one rounder pass.
    parent_geocode: The geocode of the parent node.
    child_geocodes: The geocodes of the child nodes.
    parent_val: The parent value for each type.
    child_ests: A dictionary mapping child geocodes to estimates of child nodes.
    children_upper_bound_dict: A dictionary mapping child geocodes to upper
      bounds for each first feature type.
    children_lower_bound_dict: A dictionary mapping child geocodes to lower
      bounds for each first feature type.
    subtree_total_dict: A dictionary mapping state-level geocodes to totals. If
      provided, then the sum of the variables for the AIAN and non-AIAN subtrees
      for that state must equal the total.

  Returns:
    A dictionary mapping child geocodes to estimates of child nodes.
  """
  if len(child_geocodes) == 1:
    # No need to run optimization passes for a node with a single child.
    return {geocode: parent_val for geocode in child_geocodes}

  total_pass_values = None
  l2_pass_ests, rounder_pass_ests = None, None
  for objective_function_type in objective_function_types:
    logging.info('Running pass %s', objective_function_type.name)

    # If this is a rounder pass, retrieve the child estimates from previous pass
    if objective_function_type in constants.ROUNDER_PASSES:
      if l2_pass_ests is None:
        raise ValueError('Cannot run rounder pass without previous L2 pass')
      child_ests = {
          geocode: estimate.Estimate(
              val=l2_pass_ests[geocode],  # pyrefly: ignore[bad-argument-type]
              cov=np.array(0))
          for geocode in child_geocodes
      }

    model_variables, result = run_node_optimization(
        block_shape=block_shape,
        objective_function_type=objective_function_type,
        parent_geocode=parent_geocode,
        child_geocodes=child_geocodes,
        parent_val=parent_val,
        child_ests=child_ests,
        children_upper_bound_dict=children_upper_bound_dict,
        children_lower_bound_dict=children_lower_bound_dict,
        total_pass_values=total_pass_values,
        subtree_total_dict=subtree_total_dict,
    )

    # Set total_pass_values if we have just completed a total pass.
    if objective_function_type in constants.TOTAL_ONLY_PASSES:
      if objective_function_type in constants.L2_PASSES:
        total_pass_values = {
            geocode: variable_values(result, ests)
            for geocode, ests in model_variables.items()
        }
      else:  # objective_function_type in constants.ROUNDER_PASSES:
        total_pass_values = {
            geocode: integer_variable_values(result, ests)
            for geocode, ests in model_variables.items()
        }
    else:
      total_pass_values = None

    # Update last pass estimates.
    if objective_function_type in constants.L2_PASSES:
      l2_pass_ests = {x: variable_values(result, model_variables[x])
                      for x in child_geocodes}
    elif objective_function_type in constants.ROUNDER_PASSES:
      rounder_pass_ests = {
          x: (np.floor(l2_pass_ests[x]).astype(int) +  # pyrefly: ignore[unsupported-operation]
              integer_variable_values(result, model_variables[x]))
          for x in child_geocodes}

  if rounder_pass_ests is None:
    if constants.USE_ALTERNATE_PASSES and rounder_pass_ests is None:
      logging.info('No rounder pass run; returning L2 pass estimates')
      rounder_pass_ests = {x: np.array(l2_pass_ests[x])  # pyrefly: ignore[unsupported-operation]
                           for x in child_geocodes}
    else:
      raise ValueError('No model variables or result returned.')
  return rounder_pass_ests  # pyrefly: ignore[bad-return]


def run_node_baseline_passes(
    block_shape: block.BlockShape,
    objective_function_types: Sequence[constants.ObjectiveFunctionType],
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, dict[tuple[int, ...], tuple[np.ndarray, float]]],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    subtree_total_dict: None | dict[str, int] = None,
) -> dict[str, Sequence[int]]:
  """Run optimization passes for block hierarchical postprocessing.

  Args:
    block_shape: The BlockShape object describing the block structure.
    objective_function_types: The list of objective function types to run.
      Must contain at least one L2 pass and one rounder pass.
    parent_geocode: The geocode of the parent node.
    child_geocodes: The geocodes of the child nodes.
    parent_val: The parent value for each type.
    child_ests: A dictionary mapping child geocodes to query dictionaries for
      each child node. Each query dictionary maps query shapes to tuples. The
      first element of the tuple consists of the estimate values for the query.
      The second element of the tuple consists of the variance of the estimate.
    children_upper_bound_dict: A dictionary mapping child geocodes to upper
      bounds for each first feature type.
    children_lower_bound_dict: A dictionary mapping child geocodes to lower
      bounds for each first feature type.
    subtree_total_dict: A dictionary mapping state-level geocodes to totals. If
      provided, then the sum of the variables for the AIAN and non-AIAN subtrees
      for that state must equal the total.

  Returns:
    A dictionary mapping child geocodes to estimates of child nodes.
  """
  total_pass_values = None
  l2_pass_ests, rounder_pass_ests = None, None
  for objective_function_type in objective_function_types:

    if objective_function_type in constants.L2_PASSES:
      model_variables, result = run_baseline_node_optimization(
          block_shape=block_shape,
          objective_function_type=objective_function_type,
          parent_geocode=parent_geocode,
          child_geocodes=child_geocodes,
          parent_val=parent_val,
          child_ests=child_ests,
          children_upper_bound_dict=children_upper_bound_dict,
          children_lower_bound_dict=children_lower_bound_dict,
          total_pass_values=total_pass_values,
          subtree_total_dict=subtree_total_dict,  # pyrefly: ignore[bad-argument-type]
      )
    elif objective_function_type in constants.ROUNDER_PASSES:
      # If this is a rounder pass, retrieve the child estimates from the
      # previous pass.
      if l2_pass_ests is None:
        raise ValueError('Cannot run rounder pass without previous L2 pass')
      child_ests = {  # pyrefly: ignore[bad-assignment]
          geocode: estimate.Estimate(
              val=l2_pass_ests[geocode],  # pyrefly: ignore[bad-argument-type]
              cov=np.array(0))
          for geocode in child_geocodes
      }
      # Run regular rounder pass, since it's the same as baseline rounder pass.
      model_variables, result = run_node_optimization(
          block_shape=block_shape,
          objective_function_type=objective_function_type,
          parent_geocode=parent_geocode,
          child_geocodes=child_geocodes,
          parent_val=parent_val,
          child_ests=child_ests,  # pyrefly: ignore[bad-argument-type]
          children_upper_bound_dict=children_upper_bound_dict,
          children_lower_bound_dict=children_lower_bound_dict,
          total_pass_values=total_pass_values,
          subtree_total_dict=subtree_total_dict,
      )
    else:
      raise ValueError('Objective function must be an L2 or rounder pass.')

    # Set total_pass_values if we have just completed a total pass.
    if objective_function_type in constants.TOTAL_ONLY_PASSES:
      if objective_function_type in constants.L2_PASSES:
        total_pass_values = {
            geocode: variable_values(result, ests)
            for geocode, ests in model_variables.items()
        }
      else:  # objective_function_type in constants.ROUNDER_PASSES:
        total_pass_values = {
            geocode: integer_variable_values(result, ests)
            for geocode, ests in model_variables.items()
        }
    else:
      total_pass_values = None

    # Update last pass estimates.
    if objective_function_type in constants.L2_PASSES:
      l2_pass_ests = {x: variable_values(result, model_variables[x])
                      for x in child_geocodes}
    elif objective_function_type in constants.ROUNDER_PASSES:
      rounder_pass_ests = {
          x: (np.floor(l2_pass_ests[x]).astype(int) +  # pyrefly: ignore[unsupported-operation]
              integer_variable_values(result, model_variables[x]))
          for x in child_geocodes}

  if rounder_pass_ests is None:
    if constants.USE_ALTERNATE_PASSES and rounder_pass_ests is None:
      logging.info('No rounder pass run; returning L2 pass estimates')
      rounder_pass_ests = {x: np.array(l2_pass_ests[x])  # pyrefly: ignore[unsupported-operation]
                           for x in child_geocodes}
    else:
      raise ValueError('Must run a rounder pass.')
  return rounder_pass_ests  # pyrefly: ignore[bad-return]
