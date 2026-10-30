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

from collections.abc import Mapping, Sequence
import datetime
from absl import logging
import numpy as np
from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import query_encoding
from census_bluedown import succinct
from ortools.math_opt.python import mathopt


SOLVER = constants.SOLVER
ALPHA_VALUES = constants.ALPHA_VALUES
MULTIFEATURE_ZERO_SEQUENCE = constants.MULTIFEATURE_ZERO_SEQUENCE

DETAILED_SHAPE = constants.DETAILED_QUERY_SHAPE  # (8, 2, 2, 63)
ASYM_LEN = constants.COEFFICIENT_DIM  # 32
SYM_LEN = constants.PROJECTION_DIM  # 63
TOTAL_LEN = constants.TOTAL_LEN  # 2016


def do_solve(
    model: mathopt.Model,
    *,
    timeout_minutes: int = 30,
    solve_parameters: mathopt.SolveParameters | None = None,
    use_simplex: bool = False,
) -> mathopt.SolveResult:
  """Solve the given model using Gurobi or SCIP.

  Args:
    model: The model to solve.
    timeout_minutes: The timeout for the solver.
    solve_parameters: Optional SolveParameters to pass to the solver.
    use_simplex: If True, use Gurobi dual simplex instead of barrier.

  Returns:
    The result of the solve.

  Raises:
    ValueError: If the solver is not GUROBI_SOLVER or SCIP_SOLVER.
    RuntimeError: If the solver fails to find an optimal solution.
  """
  if SOLVER == constants.SolverType.GUROBI:
    if use_simplex:
      logging.info('Invoking Gurobi dual simplex solver.')
      gurobi_params = {'Method': '1', 'numericFocus': '3'}
    else:
      logging.info('Invoking Gurobi solver.')
      gurobi_params = {'numericFocus': '3', 'BarHomogeneous': '1'}
    time_limit = datetime.timedelta(minutes=timeout_minutes)
    result = mathopt.solve(
        model=model,
        solver_type=mathopt.SolverType.GUROBI,
        params=mathopt.SolveParameters(
            time_limit=time_limit,
            gurobi=mathopt.GurobiParameters(gurobi_params)),
        msg_cb=lambda msgs: mathopt.log_messages(msgs, prefix='[solver] '),
    )
  elif SOLVER == constants.SolverType.SCIP:
    logging.info('Invoking SCIP solver.')
    params = solve_parameters or mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, mathopt.SolverType.GSCIP, params=params)
  else:
    raise ValueError(f'Unsupported solver: {SOLVER}')

  if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
    reason_name = getattr(
        result.termination.reason, 'name', str(result.termination.reason)
    )
    raise RuntimeError(
        f'model failed to solve ({reason_name}): {result.termination}'
    )

  logging.info('Optimal solution returned.')
  return result


def variable_values(
    result: mathopt.SolveResult,
    ests: Sequence[mathopt.Variable | float],
) -> Sequence[float]:
  """Extract nonnegative variable values from solve result."""
  var_vals = [
      result.variable_values(est) if isinstance(est, mathopt.Variable) else est
      for est in ests
  ]
  if any(val < -1e-4 for val in var_vals):
    raise RuntimeError(f'Variable values {var_vals} are negative.')
  return [max(0.0, float(val)) for val in var_vals]


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
    model: The model to add constraints to.
    variables: The variables to constrain.
    lower_bounds: The lower bounds for each feature value.
    upper_bounds: The upper bounds for each feature value.
    offsets: Offsets for each variable.

  Raises:
    ValueError if `lower_bounds` or `upper_bounds` have length different
    from `DETAILED_SHAPE[0]` (which is 8).
  """
  if (
      len(lower_bounds) != DETAILED_SHAPE[0]
      or len(upper_bounds) != DETAILED_SHAPE[0]
  ):
    raise ValueError('Length of bounds must match size of first feature.')

  values_per_chunk = TOTAL_LEN // DETAILED_SHAPE[0]

  for i in range(DETAILED_SHAPE[0]):
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


def generate_succinct_full_objective_terms(
    model: mathopt.Model,
    geocode: str,
    est: estimate.Estimate,
    variables: Sequence[mathopt.Variable | float],
    cons: succinct.SuccinctConstraint,
    alpha: float = 50.0,
) -> Sequence[mathopt.QuadraticTypes]:
  """Generate full-pass objective terms using succinct representation.

  Uses the succinct precision matrix P = (cov + alpha R^T R)^-1 to generate
  objective terms for the model. The sum of the objective terms is given by the
  formula (z - z_hat)^T P (z - z_hat), where z_hat is the estimate vector and z
  is the variable vector. Let A ⊆ {0, ..., ASYM_LEN - 1} (`active_rows`) be the
  set of active asymmetric row indices (those containing `mathopt.Variable`s;
  the remaining rows were fixed to 0.0 by `add_variable` for cells constrained
  to zero, and `cons` encodes the same zero constraints for the alpha
  regularization). Restricting to the rows in A and using the Cholesky
  decomposition P_AA = L L^T, this quadratic form is expressed as y^T y where
  y = L^T (z_A - z_hat_A), reducing the objective to SYM_LEN * |A| diagonal
  squared terms.

  Args:
    model: The model to generate objective terms for.
    geocode: The geocode of the block, used for naming the auxiliary variables.
    est: The estimate of the block, whose covariance must be a
      SuccinctCovariance.
    variables: The variables for the block. Entries constrained to zero are
      floats (0.0) rather than variables.
    cons: The local SuccinctConstraint for the block.
    alpha: Conditioning parameter for the local constraints.

  Returns:
    A list of quadratic expressions.

  Raises:
    TypeError: If est.cov is not a SuccinctCovariance.
    ValueError: If an entry in an inactive row differs from est.val by more
      than 1e-4.
  """
  if not isinstance(est.cov, succinct.SuccinctCovariance):
    raise TypeError('Expected SuccinctCovariance')

  prec = (est.cov + cons.gram() * alpha).inv()
  diff = np.asarray(variables) - est.val
  diffs = np.reshape(diff, (ASYM_LEN, SYM_LEN))
  active_rows = [
      a
      for a in range(ASYM_LEN)
      if any(isinstance(v, mathopt.LinearBase) for v in diffs[a])
  ]
  inactive_rows = [a for a in range(ASYM_LEN) if a not in active_rows]
  if (np.abs(diffs[inactive_rows].astype(float)) > 1e-4).any():
    raise ValueError('Inactive rows must have zero difference from est.val.')
  diffs_active = diffs[active_rows]
  num_active = len(active_rows)

  # Instead of generating dense mathopt.LinearExpressions for row means over the
  # symmetric feature, introduce |A| auxiliary variables representing the mean
  # difference across the symmetric feature for each active asymmetric feature
  # combination. This keeps the subsequent linear constraints sparse.
  delta_bar = [
      model.add_variable(name=f'delta_bar_{geocode}_{row}')
      for row in active_rows
  ]
  for k in range(num_active):
    model.add_linear_constraint(
        delta_bar[k] == mathopt.fast_sum(diffs_active[k, :]) / SYM_LEN
    )

  # Write P_AA = C_0 ⊗ P_0 + C_1 ⊗ P_1 and let L_0, L_1 be the Cholesky factors
  # of C_0, C_1, so that L = L_0 ⊗ P_0 + L_1 ⊗ P_1 satisfies P_AA = L L^T (this
  # is what SuccinctCovariance.cholesky returns). View the difference vector
  # z_A - z_hat_A as the |A| x SYM_LEN matrix D = diffs_active, with row a
  # indexing active asymmetric feature combinations and column s the symmetric
  # feature. By the vec trick, y = L^T (z_A - z_hat_A) viewed the same way is
  #   Y = L_0^T D P_0 + L_1^T D P_1.
  # P_1 = J/d averages each row, so (D P_1)[a, s] = delta_bar[a], and
  # D P_0 = D - D P_1. Therefore, with k indexing the asymmetric row of y,
  #   Y[k, s] = Σ_a L_0[a, k] D[a, s]
  #           + Σ_a (L_1[a, k] - L_0[a, k]) delta_bar[a],
  # where both sums run over a >= k because L_0, L_1 are lower triangular.
  # The second sum (mean_term[k]) does not depend on s, so it is a single
  # auxiliary variable shared by the SYM_LEN constraints defining y[k, :].
  cholesky_matrix = prec.principal_submatrix(active_rows).cholesky()
  l0, l1 = cholesky_matrix.coefficient_0, cholesky_matrix.coefficient_1
  l_diff = l1 - l0
  objective_terms = []
  for k, row in enumerate(active_rows):
    mean_term = model.add_variable(name=f'mean_term_{geocode}_{row}')
    model.add_linear_constraint(
        mean_term
        == mathopt.fast_sum(
            l_diff[a, k] * delta_bar[a] for a in range(k, num_active)
        )
    )
    for s in range(SYM_LEN):
      y_var = model.add_variable(name=f'y_{geocode}_{row * SYM_LEN + s}')
      expr = (
          mathopt.fast_sum(
              l0[a, k] * diffs_active[a, s] for a in range(k, num_active)
          )
          + mean_term
      )
      model.add_linear_constraint(y_var == expr)
      objective_terms.append(y_var * y_var)

  return objective_terms


def generate_succinct_total_only_objective_terms(
    model: mathopt.Model,
    geocode: str,
    est: estimate.Estimate,
    variables: Sequence[mathopt.Variable | float],
) -> Sequence[mathopt.QuadraticTypes]:
  """Generate objective term for total-only optimization using succinct representation.

  Args:
    model: The model to generate objective terms for.
    geocode: The geocode of the block, used for naming the sum dummy variable.
    est: The estimate of the block, whose covariance must be a
      SuccinctCovariance.
    variables: The variables for the block.

  Returns:
    A list containing a single quadratic expression.

  Raises:
    TypeError: If est.cov is not a SuccinctCovariance.
  """
  if not isinstance(est.cov, succinct.SuccinctCovariance):
    raise TypeError('Expected SuccinctCovariance')

  # Compute variance of the total count by summing over the all-ones vector:
  # 1^T (C_0 \otimes P_0 + C_1 \otimes P_1) 1 = SYM_LEN * sum(C_1)
  variance_of_total = SYM_LEN * est.cov.coefficient_1.sum()

  sum_variable = model.add_variable(lb=0, name=f'x_{geocode}_sum')
  model.add_linear_constraint(sum_variable == mathopt.fast_sum(variables))
  linear_difference = sum_variable - sum(est.val)
  return [linear_difference * linear_difference / variance_of_total]


def query_index_constrained_to_zero(
    index: int,
    first_feature_upper_bounds: Sequence[int],
    query: tuple[int, ...] = DETAILED_SHAPE,
) -> bool:
  """Check if the index of a constrained feature in a query is zero."""
  if query[0] == 1:
    return False
  if query[0] != DETAILED_SHAPE[0] or query[1] not in (1, DETAILED_SHAPE[1]):
    raise ValueError(f'Unsupported query shape {query}')

  coords = tuple(map(int, np.unravel_index(index, query)))
  if first_feature_upper_bounds[coords[0]] == 0:
    return True
  k = len(MULTIFEATURE_ZERO_SEQUENCE)
  return (
      query[:k] == DETAILED_SHAPE[:k]
      and coords[:k] == MULTIFEATURE_ZERO_SEQUENCE
  )


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
    values: The unrounded estimated value for each variable.
    floored_values: The floored estimated value for each variable.
    variables: The variables for the block.
    first_feature_upper_bounds: The upper bounds on the first feature for each
      geocode.

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
    geocode: The geocode of the block, used for naming the dummy variables.
    values: The unrounded estimated value for each variable.
    floored_values: The floored estimated value for each variable.
    variables: The variables for the block.
    first_feature_upper_bounds: The upper bounds on the first feature for each
      geocode.
    query: The shape of the query.

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
      values=variables,
      target_shape=query)

  diffs = np.asarray(values) - np.asarray(floored_values)
  part_sum_diffs = query_encoding.project_to_query_shape(diffs, query)

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
    geocode: str,
    values: Sequence[float],
    floored_values: Sequence[float],
    variables: Sequence[mathopt.Variable],
    first_feature_upper_bounds: Sequence[int],
) -> Sequence[mathopt.LinearTypes]:
  """Generate objective terms for the rounder full optimization problem.

  Returns objective terms equal to the absolute difference between each rounded
  and unrounded value.

  Args:
    model: The model to generate objective terms for.
    geocode: The geocode of the block, used for naming the dummy variables.
    values: The unrounded estimated value for each variable.
    floored_values: The floored estimated value for each variable.
    variables: The variables for the block.
    first_feature_upper_bounds: The upper bounds on the first feature for each
      geocode.

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
    geocode: str,
    values: Sequence[float],
    floored_values: Sequence[float],
    variables: Sequence[mathopt.Variable],
    first_feature_upper_bounds: Sequence[int],
) -> Sequence[mathopt.LinearTypes]:
  """Generate objective term for the rounder total only optimization problem.

  Returns a single objective term equal to the absolute difference between the
  rounded and unrounded total values.

  Args:
    model: The model to generate objective terms for.
    geocode: The geocode of the block, used for naming the dummy variable.
    values: The unrounded estimated value for each variable.
    floored_values: The floored estimated value for each variable.
    variables: The variables for the block.
    first_feature_upper_bounds: The upper bounds on the first feature for each
      geocode.

  Returns:
    A list containing a single mathopt.Variable as the term to minimize.
  """
  if len(values) != len(floored_values):
    raise ValueError('Length of values and floored_values must match.')
  if len(values) != len(variables):
    raise ValueError('Length of values and variables must match.')

  return generate_rounder_single_query_objective_terms(
      model=model,
      geocode=geocode,
      values=values,
      floored_values=floored_values,
      variables=variables,
      first_feature_upper_bounds=first_feature_upper_bounds,
      query=constants.TOTAL_QUERY_SHAPE,
  )


def mathopt_aggregate_to_query_shape(
    values: Sequence[mathopt.Variable | float],
    target_shape: tuple[int, ...],
) -> Sequence[mathopt.QuadraticTypes]:
  """Use mathopt.fast_sum to aggregate values to the shape of a query.

  Args:
    values: The values to aggregate, indexed by the detailed histogram.
    target_shape: The shape of the query to aggregate to.

  Returns:
    A list with one linear expression for each cell of the query.
  """
  value_array = np.asarray(values, dtype=object)
  return [
      mathopt.fast_sum(value_array[members])
      for members in query_encoding.query_cell_members(target_shape)
  ]


def baseline_single_query_objective_term(
    model: mathopt.Model,
    geocode: str,
    variables: Sequence[mathopt.Variable | float],
    query: tuple[int, ...],
    estimate_value: Sequence[float] | np.ndarray,
    estimate_variance: float,
) -> Sequence[mathopt.QuadraticTypes]:
  """Generate objective term for a single query for baseline L2 optimization.

  A single query is a tuple of integers corresponding to the dimensions of the
  query, corresponding to a partial rollup of some of the features. (For any
  asymmetric queries, some values of a particular feature may be grouped
  together.) The objective term is the sum of the squared differences between
  the estimate values and the partial sums of the variables corresponding to the
  rollup for that query.

  Args:
    model: The model the objective term will be added to, used for defining
      partial sum variables.
    geocode: The geocode of the block, used for naming the partial sum
      variables.
    variables: The variables for the block.
    query: The shape of the query.
    estimate_value: The estimate values of the query.
    estimate_variance: The variance of the estimate.

  Returns:
    A list of quadratic expressions whose sum is the objective term
    corresponding to this query.
  """
  if query == constants.DETAILED_QUERY_SHAPE:
    partial_sum_variables = variables
  else:
    agg_vars = mathopt_aggregate_to_query_shape(variables, query)
    partial_sum_variables = [
        model.add_variable(lb=0, name=f'x_{geocode}_{query}_partial_sum_{i}')
        for i in range(len(agg_vars))
    ]
    for x, y in zip(agg_vars, partial_sum_variables):
      model.add_linear_constraint(x == y)

  return [(x - y) * (x - y) / estimate_variance
          for x, y in zip(partial_sum_variables, estimate_value)]


def baseline_objective_terms(
    objective_function_type: constants.ObjectiveFunctionType,
    model: mathopt.Model,
    geocode: str,
    est_dict: dict[tuple[int, ...], tuple[np.ndarray, float]],
    variables: Sequence[mathopt.Variable | float],
) -> Sequence[mathopt.QuadraticTypes]:
  """Wrapper to generate objective terms for the baseline L2 optimization.

  This function calls the appropriate function to generate objective terms
  depending on the value of the objective function type. It can either generate
  an objective function term for the full baseline optimization problem or an
  objective function term for the total-only baseline optimization problem.

  Args:
    objective_function_type: The type of objective function to generate.
    model: The model to generate objective terms for.
    geocode: The geocode of the block.
    est_dict: A dictionary mapping query shapes to tuples. The first element of
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
        objective_terms.extend(
            baseline_single_query_objective_term(
                model=model,
                geocode=geocode,
                variables=variables,
                query=query,
                estimate_value=est_dict[query][0],
                estimate_variance=est_dict[query][1],
            )
        )
      return objective_terms
    case constants.ObjectiveFunctionType.TOTAL_ONLY:
      total_shape = constants.TOTAL_QUERY_SHAPE
      return baseline_single_query_objective_term(
          model=model,
          geocode=geocode,
          variables=variables,
          query=total_shape,
          estimate_value=est_dict[total_shape][0],
          estimate_variance=est_dict[total_shape][1],
      )
    case _:
      raise ValueError(
          f'Unsupported objective function type: {objective_function_type}'
      )


def generate_objective_terms(
    objective_function_type: constants.ObjectiveFunctionType,
    model: mathopt.Model,
    geocode: str,
    est: estimate.Estimate,
    variables: Sequence[mathopt.Variable | float],
    cons: succinct.SuccinctConstraint,
    floored_values: np.ndarray | None = None,
    first_feature_upper_bounds: Sequence[int] | None = None,
    alpha: float = ALPHA_VALUES[0],
) -> Sequence[mathopt.QuadraticTypes]:
  """Wrapper to generate objective terms for the model.

  This function calls the appropriate function to generate objective terms
  depending on the value of the objective function type. It can either generate
  an objective function term for the full optimization problem or an objective
  function term for the total-only optimization problem.

  Args:
    objective_function_type: The type of objective function to generate.
    model: The model to generate objective terms for.
    geocode: The geocode of the block.
    est: The estimate of the block.
    variables: The variables for the block.
    cons: The local SuccinctConstraint for the block.
    floored_values: The floored values for the block, if this is a rounder pass.
      This argument could be removed but avoids recomputing the floored values.
    first_feature_upper_bounds: The upper bounds for the first feature of each
      type.
    alpha: Conditioning parameter for the local constraints in the full pass.

  Returns:
    A list of quadratic expressions.
  """
  match objective_function_type:
    case constants.ObjectiveFunctionType.FULL:
      return generate_succinct_full_objective_terms(
          model=model,
          geocode=geocode,
          est=est,
          variables=variables,
          cons=cons,
          alpha=alpha,
      )
    case constants.ObjectiveFunctionType.TOTAL_ONLY:
      return generate_succinct_total_only_objective_terms(
          model=model,
          geocode=geocode,
          est=est,
          variables=variables,
      )
    case constants.ObjectiveFunctionType.FULL_ROUNDER:
      return generate_rounder_full_objective_terms(
          model=model,
          geocode=geocode,
          values=est.val,  # pyrefly: ignore[bad-argument-type]
          floored_values=floored_values,  # pyrefly: ignore[bad-argument-type]
          variables=variables,  # pyrefly: ignore[bad-argument-type]
          first_feature_upper_bounds=first_feature_upper_bounds,  # pyrefly: ignore[bad-argument-type]
      )
    case constants.ObjectiveFunctionType.TOTAL_ONLY_ROUNDER:
      return generate_rounder_total_only_objective_terms(
          model=model,
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


def index_constrained_to_zero(
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
    variable_index: The index of the variable.
    first_feature_upper_bounds: The upper bounds for each first feature type.

  Returns:
    True if the variable should be constrained to zero, False otherwise.
  """
  return query_index_constrained_to_zero(
      variable_index, first_feature_upper_bounds, DETAILED_SHAPE
  )


def add_variable(
    model: mathopt.Model,
    objective_function_type: constants.ObjectiveFunctionType,
    upper_bound: int,
    first_feature_upper_bounds: Sequence[int],
    geocode: str,
    variable_index: int,
) -> mathopt.Variable | float:
  """Add a variable to the model.

  If the variable is constrained to zero, returns 0.0 instead of adding a
  variable to the model. Otherwise, adds either a float or {0, 1}-valued integer
  variable, depending on the objective function type.

  Args:
    model: The model to add the variable to.
    objective_function_type: The type of objective function used for the model,
      which determines whether variables are floats or {0, 1}-valued integers.
    upper_bound: The upper bound of the variable, ignored for integer variables.
    first_feature_upper_bounds: The upper bound of the first feature type. This
      is used to constrain the variable to zero if the first feature upper bound
      is zero.
    geocode: The geocode of the block, used for naming the variable.
    variable_index: The index of the variable, used for naming the variable.

  Returns:
    A mathopt.Variable corresponding to the variable added to the model, or 0.0
    if the variable is constrained to zero.
  """
  if index_constrained_to_zero(
      variable_index=variable_index,
      first_feature_upper_bounds=first_feature_upper_bounds):
    return 0.0

  if objective_function_type in constants.ROUNDER_PASSES:
    # Use integer variables for the rounder pass.
    #
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
    objective_function_type: constants.ObjectiveFunctionType,
    upper_bounds: int | Sequence[int] | np.ndarray,
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
          objective_function_type=objective_function_type,
          upper_bound=int(upper_bounds[i]),
          first_feature_upper_bounds=first_feature_upper_bounds,
          geocode=geocode,
          variable_index=i,
      )
      for i in range(num_variables)
  ]


def generate_and_constrain_node_variables(
    model: mathopt.Model,
    objective_function_type: constants.ObjectiveFunctionType,
    num_variables: int,
    type_upper_bounds: Sequence[int] | np.ndarray,
    first_feature_upper_bounds: Sequence[int],
    first_feature_lower_bounds: Sequence[int],
    offsets: None | np.ndarray,
    geocode: str,
    total_pass_values: None | Mapping[str, Sequence[float] | np.ndarray],
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
      the sum of the variables for the node must equal the sum of the total pass
      values for that node. If a single integer is provided as the dictionary
      value instead of a list, then that value is assumed to be the sum of the
      total pass values for all types for that node.

  Returns:
    A list of variables added to the model, corresponding to the count for each
    type at this node.
  """
  row_variables = add_variables(
      model=model,
      objective_function_type=objective_function_type,
      upper_bounds=type_upper_bounds,
      first_feature_upper_bounds=first_feature_upper_bounds,
      geocode=geocode,
      num_variables=num_variables
  )

  add_feature_constraints(
      model=model,
      variables=row_variables,  # pyrefly: ignore[bad-argument-type]
      lower_bounds=first_feature_lower_bounds,
      upper_bounds=first_feature_upper_bounds,
      offsets=offsets,  # pyrefly: ignore[bad-argument-type]
  )
  if total_pass_values is not None and geocode in total_pass_values:
    offset_sum = 0 if offsets is None else sum(offsets)
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
    objective_function_type: constants.ObjectiveFunctionType,
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, estimate.Estimate],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    total_pass_values: None | Mapping[str, Sequence[float] | np.ndarray] = None,
    subtree_total_dict: None | dict[str, int] = None,
    solve_parameters: mathopt.SolveParameters | None = None,
    alpha_values: Sequence[float] = ALPHA_VALUES,
) -> tuple[dict[str, Sequence[mathopt.Variable | float]], mathopt.SolveResult]:
  """Solve optimization problem for block hierarchical postprocessing.

  Initializes a model and adds variables and constraints to it. Computes
  the objective function determined by the parameter objective_function_type.
  Solves the model and returns the variables and the solve result. If solving
  fails to find an optimal solution for the FULL pass, retries immediately over
  the sequence of candidate alpha values in `alpha_values`. If all alpha values
  fail (e.g. due to barrier imprecision), retries the last model with dual
  simplex as a final fallback.

  The constraints added are as follows:
  Constraints (1-3) as described in the documentation for
  generate_and_constrain_node_variables are added for each child node.
  The following additional constraint is added:
  4. The sum of the variables for each type across all children equals the
     parent value for that type.

  Args:
    objective_function_type: The type of objective function to optimize.
    parent_geocode: The geocode of the parent node.
    child_geocodes: The geocodes of the child nodes.
    parent_val: The parent value for each type.
    child_ests: A dictionary mapping child geocodes to estimates of child nodes.
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
    solve_parameters: Optional SolveParameters to pass to the solver.
    alpha_values: Sequence of candidate alpha conditioning parameters to try
      when optimizing the FULL pass.

  Returns:
    A tuple containing a dictionary and the solve result. The dictionary maps
    child geocodes to a list of variables corresponding to the count for each
    type for that child.

  Raises:
    ValueError: If `alpha_values` is empty.
    RuntimeError: If optimization fails for all candidate alpha values.
  """
  if not alpha_values:
    raise ValueError('alpha_values must be non-empty.')

  logging.info(
      'Performing %s optimization for parent node \'%s\' with %d children.',
      objective_function_type.name,
      parent_geocode,
      len(child_geocodes),
  )

  candidate_alphas = (
      alpha_values
      if objective_function_type == constants.ObjectiveFunctionType.FULL
      else (alpha_values[0],)
  )

  failures = []
  last_error = None

  for alpha in candidate_alphas:
    model = mathopt.Model(name=f'{parent_geocode}')
    objective_terms = []

    model_variables = {}
    offset_sum_dict = None
    floored_child_est_sum = (
        np.zeros(TOTAL_LEN)
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
          objective_function_type=objective_function_type,
          num_variables=len(child_est.val),
          type_upper_bounds=parent_val,
          first_feature_upper_bounds=children_upper_bound_dict[geocode],
          first_feature_lower_bounds=children_lower_bound_dict[geocode],
          offsets=floored_child_est,
          geocode=geocode,
          total_pass_values=total_pass_values,
      )

      # Construct local constraints R_v for the child.
      child_cons = succinct.SuccinctConstraint.zero_constraints(
          first_feature_zero_indices=[
              y == 0 for y in children_upper_bound_dict[geocode]
          ],
          multifeature_zero_sequence=MULTIFEATURE_ZERO_SEQUENCE,
      )

      objective_terms.extend(
          generate_objective_terms(
              objective_function_type=objective_function_type,
              model=model,
              geocode=geocode,
              est=child_est,
              variables=row_variables,
              cons=child_cons,
              floored_values=floored_child_est,
              first_feature_upper_bounds=children_upper_bound_dict[geocode],
              alpha=alpha,
          )
      )
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

    logging.info(
        'Minimizing objective function (%d terms)', len(objective_terms)
    )
    model.minimize(mathopt.fast_sum(objective_terms))

    try:
      result = do_solve(model, solve_parameters=solve_parameters)
      return model_variables, result
    except RuntimeError as err:
      last_error = err
      failures.append(f'alpha={alpha}: {err}')
      logging.warning(
          "Optimization failed for parent node '%s' (%s) with alpha=%s: %s",
          parent_geocode,
          objective_function_type.name,
          alpha,
          err,
      )

  # All alpha values exhausted with barrier; retry the last model with dual
  # simplex.
  if SOLVER == constants.SolverType.GUROBI:
    logging.warning(
        'All alpha values exhausted for parent node %r (%s). '
        'Retrying last model with dual simplex.',
        parent_geocode,
        objective_function_type.name,
    )
    try:
      result = do_solve(
          model,  # pyrefly: ignore[unbound-name]
          use_simplex=True,
      )
      logging.info('Dual simplex retry succeeded.')
      return model_variables, result  # pyrefly: ignore[unbound-name]
    except RuntimeError as err:
      last_error = err
      failures.append(f'simplex retry: {err}')

  raise RuntimeError(
      f'Optimization failed for parent node {parent_geocode!r} '
      f'({objective_function_type.name}) across all alpha values '
      f'{list(candidate_alphas)}: {"; ".join(failures)}'
  ) from last_error


def run_baseline_node_optimization(
    objective_function_type: constants.ObjectiveFunctionType,
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, dict[tuple[int, ...], tuple[np.ndarray, float]]],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    total_pass_values: None | Mapping[str, Sequence[float] | np.ndarray] = None,
    subtree_total_dict: None | dict[str, float] = None,
    solve_parameters: mathopt.SolveParameters | None = None,
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
    solve_parameters: Optional SolveParameters to pass to the solver.

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
        objective_function_type=objective_function_type,
        num_variables=len(child_est_dict[constants.DETAILED_QUERY_SHAPE][0]),
        type_upper_bounds=parent_val,
        first_feature_upper_bounds=children_upper_bound_dict[geocode],
        first_feature_lower_bounds=children_lower_bound_dict[geocode],
        offsets=None,
        geocode=geocode,
        total_pass_values=total_pass_values
    )

    objective_terms.extend(
        baseline_objective_terms(
            objective_function_type=objective_function_type,
            model=model,
            geocode=geocode,
            est_dict=child_est_dict,
            variables=row_variables,  # pyrefly: ignore[bad-argument-type]
        )
    )
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
  result = do_solve(model, solve_parameters=solve_parameters)
  return model_variables, result


def run_node_optimization_passes(
    objective_function_types: Sequence[constants.ObjectiveFunctionType],
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, estimate.Estimate],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    subtree_total_dict: None | dict[str, int] = None,
    solve_parameters: mathopt.SolveParameters | None = None,
) -> dict[str, Sequence[int]]:
  """Run optimization passes for block hierarchical postprocessing.

  Args:
    objective_function_types: The list of objective function types to run. Must
      contain at least one L2 pass and one rounder pass.
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
    solve_parameters: Optional SolveParameters to pass to the solver.

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

    # If this is a rounder pass, retrieve the child estimates from previous
    # pass.
    if objective_function_type in constants.ROUNDER_PASSES:
      if l2_pass_ests is None:
        raise ValueError('Cannot run rounder pass without previous L2 pass')
      child_ests = {
          geocode: estimate.Estimate(
              val=l2_pass_ests[geocode],  # pyrefly: ignore[bad-argument-type]
              # dummy cov, not used in rounder
              cov=succinct.SuccinctCovariance.identity(),
          )
          for geocode in child_geocodes
      }

    model_variables, result = run_node_optimization(
        objective_function_type=objective_function_type,
        parent_geocode=parent_geocode,
        child_geocodes=child_geocodes,
        parent_val=parent_val,
        child_ests=child_ests,
        children_upper_bound_dict=children_upper_bound_dict,
        children_lower_bound_dict=children_lower_bound_dict,
        total_pass_values=total_pass_values,
        subtree_total_dict=subtree_total_dict,
        solve_parameters=solve_parameters,
    )

    # Update last pass estimates and set total_pass_values if we have just
    # completed a total pass.
    if objective_function_type in constants.L2_PASSES:
      l2_pass_ests = {
          x: variable_values(result, model_variables[x]) for x in child_geocodes
      }
      total_pass_values = (
          l2_pass_ests
          if objective_function_type in constants.TOTAL_ONLY_PASSES
          else None
      )
    elif objective_function_type in constants.ROUNDER_PASSES:
      rounder_pass_ests = {
          x: (
              np.floor(l2_pass_ests[x]).astype(int)  # pyrefly: ignore[unsupported-operation]
              + integer_variable_values(result, model_variables[x])
          )
          for x in child_geocodes
      }
      total_pass_values = (
          rounder_pass_ests
          if objective_function_type in constants.TOTAL_ONLY_PASSES
          else None
      )

  if rounder_pass_ests is None:
    raise ValueError('No model variables or result returned.')
  return rounder_pass_ests  # pyrefly: ignore[bad-return]


def run_node_baseline_passes(
    objective_function_types: Sequence[constants.ObjectiveFunctionType],
    parent_geocode: str,
    child_geocodes: Sequence[str],
    parent_val: Sequence[int],
    child_ests: dict[str, dict[tuple[int, ...], tuple[np.ndarray, float]]],
    children_upper_bound_dict: dict[str, Sequence[int]],
    children_lower_bound_dict: dict[str, Sequence[int]],
    subtree_total_dict: None | dict[str, int] = None,
    solve_parameters: mathopt.SolveParameters | None = None,
) -> dict[str, Sequence[int]]:
  """Run baseline optimization passes for block hierarchical postprocessing.

  Args:
    objective_function_types: The list of objective function types to run. Must
      contain at least one L2 pass and one rounder pass.
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
    solve_parameters: Optional SolveParameters to pass to the solver.

  Returns:
    A dictionary mapping child geocodes to estimates of child nodes.
  """
  total_pass_values = None
  l2_pass_ests, rounder_pass_ests = None, None
  for objective_function_type in objective_function_types:
    logging.info('Running baseline pass %s', objective_function_type.name)

    if objective_function_type in constants.ROUNDER_PASSES:
      # If this is a rounder pass, retrieve the child estimates from the
      # previous pass and run regular rounder pass since it is the same as the
      # baseline rounder pass.
      if l2_pass_ests is None:
        raise ValueError('Cannot run rounder pass without previous L2 pass')
      child_ests_non_baseline = {
          geocode: estimate.Estimate(
              val=l2_pass_ests[geocode],  # pyrefly: ignore[bad-argument-type]
              cov=succinct.SuccinctCovariance.identity(),
          )
          for geocode in child_geocodes
      }
      model_variables, result = run_node_optimization(
          objective_function_type=objective_function_type,
          parent_geocode=parent_geocode,
          child_geocodes=child_geocodes,
          parent_val=parent_val,
          child_ests=child_ests_non_baseline,
          children_upper_bound_dict=children_upper_bound_dict,
          children_lower_bound_dict=children_lower_bound_dict,
          total_pass_values=total_pass_values,
          subtree_total_dict=subtree_total_dict,
          solve_parameters=solve_parameters,
      )
    else:
      model_variables, result = run_baseline_node_optimization(
          objective_function_type=objective_function_type,
          parent_geocode=parent_geocode,
          child_geocodes=child_geocodes,
          parent_val=parent_val,
          child_ests=child_ests,
          children_upper_bound_dict=children_upper_bound_dict,
          children_lower_bound_dict=children_lower_bound_dict,
          total_pass_values=total_pass_values,
          subtree_total_dict=subtree_total_dict,  # pyrefly: ignore[bad-argument-type]
          solve_parameters=solve_parameters,
      )

    # Update last pass estimates and set total_pass_values if we have just
    # completed a total pass.
    if objective_function_type in constants.L2_PASSES:
      l2_pass_ests = {
          x: variable_values(result, model_variables[x]) for x in child_geocodes
      }
      total_pass_values = (
          l2_pass_ests
          if objective_function_type in constants.TOTAL_ONLY_PASSES
          else None
      )
    elif objective_function_type in constants.ROUNDER_PASSES:
      rounder_pass_ests = {
          x: (
              np.floor(l2_pass_ests[x]).astype(int)  # pyrefly: ignore[unsupported-operation]
              + integer_variable_values(result, model_variables[x])
          )
          for x in child_geocodes
      }
      total_pass_values = (
          rounder_pass_ests
          if objective_function_type in constants.TOTAL_ONLY_PASSES
          else None
      )

  if rounder_pass_ests is None:
    raise ValueError('No model variables or result returned.')
  return rounder_pass_ests  # pyrefly: ignore[bad-return]
