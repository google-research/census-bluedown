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

"""Root optimization for block hierarchical postprocessing algorithms.

Contains both baseline and new optimization functions for the root node and its
children.
"""

from collections.abc import Sequence
from absl import logging
import numpy as np
from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import nonlinear_solver
from census_bluedown import succinct
from ortools.math_opt.python import mathopt

DETAILED_SHAPE = constants.DETAILED_QUERY_SHAPE
TOTAL_LEN = constants.TOTAL_LEN


def run_root_optimization(
    objective_function_type: constants.ObjectiveFunctionType,
    root_geocode: str,
    root_est: estimate.Estimate,
    root_upper_bounds: Sequence[int],
    root_lower_bounds: Sequence[int],
    root_total: float,
    alpha_values: Sequence[float] = constants.ALPHA_VALUES,
) -> tuple[Sequence[mathopt.Variable | float], mathopt.SolveResult]:
  """Solve root optimization problem for block hierarchical postprocessing.

  Initializes a model and adds variables and constraints to it. Computes
  the objective function determined by the parameter objective_function_type.
  Solves the model and returns the variables and the solve result. If solving
  fails to find an optimal solution for the FULL pass, retries immediately over
  the sequence of candidate alpha values in `alpha_values`.

  The constraints added for the root node are as follows:
  1. The first feature upper and lower bound constraints are respected.
  2. The multifeature zero sequence constraints are added if not redundant. That
     is, if the upper bound for housing type 3 was zero, then these variables
     have already been constrained to zero; otherwise, they will be constrained
     to zero anyway.
  3. The sum of all variables must equal the root_total.

  Args:
    objective_function_type: The type of objective function to optimize.
    root_geocode: The geocode of the root node.
    root_est: The estimate of the root node.
    root_upper_bounds: Upper bounds for each first feature type.
    root_lower_bounds: Lower bounds for each first feature type.
    root_total: The total count for the root node.
    alpha_values: Sequence of candidate alpha conditioning parameters to try
      when optimizing the FULL pass.

  Returns:
    A tuple containing a list of variables and the solve result. The list of
    variables corresponds to the count for each type at the root node.

  Raises:
    ValueError: If `objective_function_type` is a total-only pass or
      `alpha_values` is empty.
    RuntimeError: If optimization fails for all candidate alpha values.
  """
  if objective_function_type in constants.TOTAL_ONLY_PASSES:
    raise ValueError('Total only passes should not be used for the root node.')
  if not alpha_values:
    raise ValueError('alpha_values must be non-empty.')

  logging.info(
      'Performing %s optimization for root node %s',
      objective_function_type.name,
      root_geocode
  )

  floored_est = None
  if objective_function_type in constants.ROUNDER_PASSES:
    floored_est = np.floor(root_est.val).astype(int)

  # Construct local constraints R_v for the root node, including the total
  # sum constraint since the total count is fixed for the root.
  cons_zeros = succinct.SuccinctConstraint.zero_constraints(
      first_feature_zero_indices=[y == 0 for y in root_upper_bounds],
      multifeature_zero_sequence=nonlinear_solver.MULTIFEATURE_ZERO_SEQUENCE,
  )
  root_cons = succinct.SuccinctConstraint.combine(
      [cons_zeros, succinct.SuccinctConstraint.sum(root_total)]
  )

  candidate_alphas = (
      alpha_values
      if objective_function_type == constants.ObjectiveFunctionType.FULL
      else (alpha_values[0],)
  )

  failures = []
  last_error = None

  for alpha in candidate_alphas:
    model = mathopt.Model(name=f'{root_geocode}')

    root_variables = nonlinear_solver.generate_and_constrain_node_variables(
        model=model,
        objective_function_type=objective_function_type,
        num_variables=len(root_est.val),
        # For root, use the same type upper bound for each type with the same
        # first feature value.
        type_upper_bounds=np.repeat(
            root_upper_bounds, TOTAL_LEN // DETAILED_SHAPE[0]
        ),
        first_feature_upper_bounds=root_upper_bounds,
        first_feature_lower_bounds=root_lower_bounds,
        offsets=floored_est,
        geocode=root_geocode,
        total_pass_values={root_geocode: (root_total,)},
    )

    objective_terms = nonlinear_solver.generate_objective_terms(
        objective_function_type=objective_function_type,
        model=model,
        geocode=root_geocode,
        est=root_est,
        variables=root_variables,
        cons=root_cons,
        floored_values=floored_est,
        first_feature_upper_bounds=root_upper_bounds,
        alpha=alpha,
    )

    objective_term_sum = mathopt.fast_sum(objective_terms)
    model.minimize(objective_term_sum)
    try:
      result = nonlinear_solver.do_solve(model)
      return root_variables, result
    except RuntimeError as err:
      last_error = err
      failures.append(f'alpha={alpha}: {err}')
      logging.warning(
          "Optimization failed for root node '%s' (%s) with alpha=%s: %s",
          root_geocode,
          objective_function_type.name,
          alpha,
          err,
      )

  raise RuntimeError(
      f'Optimization failed for root node {root_geocode!r} '
      f'({objective_function_type.name}) across all alpha values '
      f'{list(candidate_alphas)}: {"; ".join(failures)}'
  ) from last_error


def run_baseline_root_optimization(
    objective_function_type: constants.ObjectiveFunctionType,
    root_geocode: str,
    root_est_dict: dict[tuple[int, ...], tuple[np.ndarray, float]],
    root_upper_bounds: Sequence[int],
    root_lower_bounds: Sequence[int],
    root_total: float
) -> tuple[Sequence[mathopt.Variable | float], mathopt.SolveResult]:
  """Solve baseline root optimization for block hierarchical postprocessing.

  Initializes a model and adds variables and constraints to it. Computes
  the objective function determined by the parameter objective_function_type.
  Solves the model and returns the variables and the solve result.

  The constraints added for the root node are as follows:
  1. The first feature upper and lower bound constraints are respected.
  2. The multifeature zero sequence constraints are added if not redundant. That
     is, if the upper bound for housing type 3 was zero, then these variables
     have already been constrained to zero; otherwise, they will be constrained
     to zero anyway.
  3. The sum of all variables must equal the root_total.

  Args:
    objective_function_type: The type of objective function to optimize.
    root_geocode: The geocode of the root node.
    root_est_dict: A dictionary mapping query shapes to tuples of estimate
      values and variances.
    root_upper_bounds: Upper bounds for each first feature type.
    root_lower_bounds: Lower bounds for each first feature type.
    root_total: The total count for the root node.

  Returns:
    A tuple containing a list of variables and the solve result. The list of
    variables corresponds to the count for each type at the root node.
  """
  if objective_function_type in constants.TOTAL_ONLY_PASSES:
    raise ValueError('Total only passes should not be used for the root node.')
  if objective_function_type in constants.ROUNDER_PASSES:
    raise ValueError('Use method run_root_optimization for baseline rounder '
                     'passes.')

  logging.info(
      'Performing %s baseline pass for root node %s.',
      objective_function_type.name,
      root_geocode
  )

  model = mathopt.Model(name=f'{root_geocode}')

  root_variables = nonlinear_solver.generate_and_constrain_node_variables(
      model=model,
      objective_function_type=objective_function_type,
      num_variables=len(root_est_dict[constants.DETAILED_QUERY_SHAPE][0]),
      # For root, use the same type upper bound for each type with the same
      # first feature value.
      type_upper_bounds=np.repeat(
          root_upper_bounds, TOTAL_LEN // DETAILED_SHAPE[0]
      ),
      first_feature_upper_bounds=root_upper_bounds,
      first_feature_lower_bounds=root_lower_bounds,
      offsets=None,
      geocode=root_geocode,
      total_pass_values={root_geocode: (root_total,)},
  )

  objective_terms = nonlinear_solver.baseline_objective_terms(
      objective_function_type=objective_function_type,
      model=model,
      geocode=root_geocode,
      est_dict=root_est_dict,
      variables=root_variables,
  )

  model.minimize(mathopt.fast_sum(objective_terms))
  result = nonlinear_solver.do_solve(model)
  return root_variables, result
