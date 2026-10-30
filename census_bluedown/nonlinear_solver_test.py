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

# pylint: disable=invalid-name

import collections
from collections.abc import Sequence
from unittest import mock
import numpy as np
from census_bluedown import constants
from census_bluedown import estimate
from census_bluedown import nonlinear_solver
from census_bluedown import succinct
from absl.testing import absltest
from absl.testing import parameterized
from ortools.math_opt.python import mathopt

Estimate = estimate.Estimate


class NonlinearSolverTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self.m = succinct.SuccinctCovariance.coefficient_dim  # 32
    self.d = succinct.SuccinctCovariance.projection_dim  # 63
    self.n = self.m * self.d  # 2016

    self.I_d = np.eye(self.d)
    self.J_d = np.ones((self.d, self.d))
    self.P_0 = self.I_d - self.J_d / self.d
    self.P_1 = self.J_d / self.d

  def _materialize_covariance(
      self, cov: succinct.SuccinctCovariance
  ) -> np.ndarray:
    return np.kron(cov.coefficient_0, self.P_0) + np.kron(
        cov.coefficient_1, self.P_1
    )

  def _materialize_constraints(
      self, cons: succinct.SuccinctConstraint
  ) -> np.ndarray:
    part1 = np.kron(cons.r1, self.I_d)
    part2 = np.kron(cons.r2, np.ones((1, self.d)))
    return np.vstack([part1, part2])

  def _random_succinct_covariance(self) -> succinct.SuccinctCovariance:
    a0 = np.random.randn(self.m, self.m)
    c0 = a0 @ a0.T + np.eye(self.m)
    a1 = np.random.randn(self.m, self.m)
    c1 = a1 @ a1.T + np.eye(self.m)
    return succinct.SuccinctCovariance(c0, c1)

  @mock.patch.object(nonlinear_solver, "add_sum_range_constraint")
  def test_add_feature_constraints(self, add_sum_range_constraint_mock):
    model = mock.Mock()
    variables = [mock.Mock() for _ in range(2016)]
    lower_bounds = list(range(8))
    upper_bounds = list(range(2, 10))

    nonlinear_solver.add_feature_constraints(
        model=model,
        variables=variables,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    add_sum_range_constraint_mock.assert_has_calls([
        mock.call(
            model=model,
            variables=variables[0:252],
            lower_bound=0,
            upper_bound=2,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[252:504],
            lower_bound=1,
            upper_bound=3,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[504:756],
            lower_bound=2,
            upper_bound=4,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[756:1008],
            lower_bound=3,
            upper_bound=5,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[1008:1260],
            lower_bound=4,
            upper_bound=6,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[1260:1512],
            lower_bound=5,
            upper_bound=7,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[1512:1764],
            lower_bound=6,
            upper_bound=8,
            offsets=None,
        ),
        mock.call(
            model=model,
            variables=variables[1764:2016],
            lower_bound=7,
            upper_bound=9,
            offsets=None,
        ),
    ])

  @mock.patch.object(mathopt, "fast_sum")
  def test_add_parent_sum_constraints(self, mock_fast_sum):
    model = mock.Mock()
    child_geocodes = ["child1", "child2", "child3"]
    model_variables = {
        "child1": [mock.Mock() for _ in range(2016)],
        "child2": [mock.Mock() for _ in range(2016)],
        "child3": [mock.Mock() for _ in range(2016)],
    }

    parent_val = list(range(2016))
    equality_expressions = [mock.Mock() for _ in range(2016)]
    mock_sums = [mock.MagicMock() for _ in range(2016)]

    mock_fast_sum.side_effect = mock_sums
    for i in range(2016):
      mock_sums[i].__eq__.return_value = equality_expressions[i]  # pyrefly: ignore[missing-attribute]

    nonlinear_solver.add_parent_sum_constraints(
        model=model,
        model_variables=model_variables,  # pyrefly: ignore[bad-argument-type]
        parent_val=parent_val,
    )

    model.add_linear_constraint.assert_has_calls(
        [mock.call(x) for x in equality_expressions]
    )
    mock_fast_sum.assert_has_calls([
        mock.call([model_variables[j][i] for j in child_geocodes])
        for i in range(2016)
    ])
    for i in range(2016):
      mock_sums[i].__eq__.assert_called_once_with(float(i))  # pyrefly: ignore[missing-attribute]

  def test_generate_succinct_full_objective_terms_mathopt_equivalence(self):
    cov = self._random_succinct_covariance()
    val = np.random.randn(self.n)
    est = Estimate(val, cov)

    zero_indices = [True, False, True, False, False, False, False, False]
    cons = succinct.SuccinctConstraint.first_feature_zero_constraints(
        zero_indices
    )
    vars_float = np.random.randn(self.n)

    # 1. Dense evaluation
    C = self._materialize_covariance(cov)
    R = self._materialize_constraints(cons)  # shape (k, 2016)
    P_inv = C + 50.0 * R.T @ R
    P = np.linalg.inv(P_inv)
    delta = vars_float - val
    expected_val = delta.T @ P @ delta

    # 2. MathOpt model evaluation (fixing decision variables to vars_float)
    model = mathopt.Model(name="test_equiv")
    model_vars = [
        model.add_variable(lb=float(v), ub=float(v), name=f"x_{i}")
        for i, v in enumerate(vars_float)
    ]
    terms = nonlinear_solver.generate_succinct_full_objective_terms(
        model=model,
        geocode="test",
        est=est,
        variables=model_vars,
        cons=cons,
        alpha=50.0,
    )
    # With all 32 asymmetric rows active: 2016 squared objective terms, and
    # 2016 (y) + 32 (delta_bar) + 32 (mean_term) = 2080 auxiliary variables and
    # equality constraints, on top of the 2016 decision variables.
    self.assertLen(terms, 2016)
    self.assertEqual(model.get_num_variables(), self.n + 2080)
    self.assertEqual(model.get_num_linear_constraints(), 2080)
    model.minimize(mathopt.fast_sum(terms))
    result = mathopt.solve(model, mathopt.SolverType.GSCIP)
    self.assertEqual(
        result.termination.reason, mathopt.TerminationReason.OPTIMAL
    )
    self.assertAlmostEqual(
        result.objective_value(),
        float(expected_val),
        places=4,
    )

  def test_generate_succinct_full_objective_terms_sparse_active_rows(self):
    cov = self._random_succinct_covariance()
    val = np.random.randn(self.n)

    # Zero out all housing types except 1 and 5 (non-leading, non-contiguous
    # active rows: [4, 5, 6, 7, 20, 21, 22, 23], k = 8 active rows).
    zero_indices = [True, False, True, True, True, False, True, True]
    cons = succinct.SuccinctConstraint.zero_constraints(
        first_feature_zero_indices=zero_indices,
        multifeature_zero_sequence=constants.MULTIFEATURE_ZERO_SEQUENCE,
    )
    est = cons.apply(Estimate(val, cov))
    active_rows = [4, 5, 6, 7, 20, 21, 22, 23]
    k_len = len(active_rows)

    vars_float = np.zeros(self.n)
    for a in active_rows:
      vars_float[a * self.d : (a + 1) * self.d] = np.random.randn(self.d)

    # 1. Dense evaluation at fixed point
    self.assertIsInstance(est.cov, succinct.SuccinctCovariance)
    C = self._materialize_covariance(est.cov)
    R = self._materialize_constraints(cons)
    P = np.linalg.inv(C + 50.0 * R.T @ R)
    delta = vars_float - est.val
    expected_val = delta.T @ P @ delta

    # 2. MathOpt evaluation with float-fixed inactive rows
    model = mathopt.Model(name="test_sparse_fixed")
    mixed_vars: list[mathopt.Variable | float] = []
    for a in range(self.m):
      for s in range(self.d):
        idx = a * self.d + s
        if a in active_rows:
          v = float(vars_float[idx])
          mixed_vars.append(model.add_variable(lb=v, ub=v, name=f"x_{idx}"))
        else:
          mixed_vars.append(0.0)

    terms = nonlinear_solver.generate_succinct_full_objective_terms(
        model=model,
        geocode="test",
        est=est,
        variables=mixed_vars,
        cons=cons,
        alpha=50.0,
    )
    # Decision variables: k_len * 63; Auxiliary: k_len * 65.
    self.assertEqual(model.get_num_variables(), k_len * self.d + k_len * 65)
    self.assertEqual(model.get_num_linear_constraints(), k_len * 65)

    model.minimize(mathopt.fast_sum(terms))
    result = mathopt.solve(model, mathopt.SolverType.GSCIP)
    self.assertEqual(
        result.termination.reason, mathopt.TerminationReason.OPTIMAL
    )
    self.assertAlmostEqual(
        result.objective_value(),
        float(expected_val),
        places=4,
    )

    # Verify ValueError when inactive rows have nonzero est.val.
    with self.assertRaisesRegex(ValueError, "Inactive rows must have zero"):
      nonlinear_solver.generate_succinct_full_objective_terms(
          model=mathopt.Model(name="test_unprojected"),
          geocode="test",
          est=Estimate(val, cov),
          variables=mixed_vars,
          cons=cons,
          alpha=50.0,
      )

  def test_generate_succinct_total_only_objective_terms(self):
    cov = self._random_succinct_covariance()
    val = np.random.randn(self.n)
    est = Estimate(val, cov)

    # Test mixed list starting with 0.0 followed by mathopt.Variables fixed to
    # nonnegative values.
    vars_float = np.abs(np.random.randn(self.n))
    vars_float[:252] = 0.0

    model = mathopt.Model(name="test_total_only")
    mixed_vars: list[mathopt.Variable | float] = [0.0] * 252 + [
        model.add_variable(
            lb=float(vars_float[i]), ub=float(vars_float[i]), name=f"v_{i}"
        )
        for i in range(252, self.n)
    ]
    terms = nonlinear_solver.generate_succinct_total_only_objective_terms(
        model=model,
        geocode="test",
        est=est,
        variables=mixed_vars,
    )
    self.assertLen(terms, 1)
    self.assertIsInstance(terms[0], mathopt.QuadraticBase)
    self.assertEqual(model.get_num_variables(), self.n - 252 + 1)

    # Expected: (sum(vars) - sum(val))^2 / var_total
    var_total = self.d * cov.coefficient_1.sum()
    expected_val = (sum(vars_float) - sum(val)) ** 2 / var_total

    model.minimize(mathopt.fast_sum(terms))
    result = mathopt.solve(model, mathopt.SolverType.GSCIP)
    self.assertEqual(
        result.termination.reason, mathopt.TerminationReason.OPTIMAL
    )
    self.assertAlmostEqual(
        result.objective_value(), float(expected_val), places=4
    )

  @mock.patch.object(nonlinear_solver, "do_solve")
  def test_run_node_optimization_model_construction(self, mock_do_solve):
    cov1 = self._random_succinct_covariance()
    cov2 = self._random_succinct_covariance()
    val1 = np.random.randn(self.n)
    val2 = np.random.randn(self.n)

    # Ensure they are positive to avoid bounds
    val1 = np.abs(val1) * 10 + 5
    val2 = np.abs(val2) * 10 + 5

    parent_val = np.rint(val1 + val2 + np.random.randn(self.n) * 2).astype(int)
    parent_val = np.maximum(parent_val, 10)

    # Zero out constrained indices to ensure feasibility and consistency
    ub = [1000000] * 8
    for i in range(self.n):
      if nonlinear_solver.index_constrained_to_zero(i, ub):
        val1[i] = 0
        val2[i] = 0
        parent_val[i] = 0

    est1 = Estimate(val1, cov1)
    est2 = Estimate(val2, cov2)

    child_ests = {"c1": est1, "c2": est2}
    children_ub: dict[str, Sequence[int]] = {"c1": ub, "c2": ub}
    children_lb: dict[str, Sequence[int]] = {"c1": [0] * 8, "c2": [0] * 8}

    # Mock do_solve to return a dummy result
    mock_result = mock.Mock(spec=mathopt.SolveResult)
    mock_result.variable_values.return_value = collections.defaultdict(float)
    mock_do_solve.return_value = mock_result

    _, _ = nonlinear_solver.run_node_optimization(
        objective_function_type=constants.ObjectiveFunctionType.FULL,
        parent_geocode="p",
        child_geocodes=["c1", "c2"],
        parent_val=parent_val,  # pyrefly: ignore[bad-argument-type]
        child_ests=child_ests,
        children_upper_bound_dict=children_ub,
        children_lower_bound_dict=children_lb,
    )

    # Verify do_solve was called
    mock_do_solve.assert_called_once()

    # Extract the model passed to do_solve
    model = mock_do_solve.call_args[0][0]

    # With ub = [1000000] * 8, MULTIFEATURE_ZERO_SEQUENCE constrains 2
    # asymmetric rows (126 cells) to zero, leaving 30 active rows (1890 cells).
    # Variables: 1890 * 2 (c1, c2 decision vars)
    #            + (1890 + 30 + 30) * 2 (aux y, delta_bar and mean_term for c1,
    #            c2) = 7680
    self.assertEqual(model.get_num_variables(), 7680)

    # Constraints:
    # - Parent sum: 2016
    # - Feature constraints: 8 * 2 (lb, ub) * 2 (children) = 32
    # - Auxiliary constraints: (1890 + 30 + 30) * 2 = 3900
    # Total: 2016 + 32 + 3900 = 5948
    self.assertEqual(model.get_num_linear_constraints(), 5948)

    # Verify objective terms
    # We expect 1890 * 2 = 3780 quadratic terms (y^2 for both children)
    quad_terms = list(model.quadratic_objective_terms())
    self.assertLen(quad_terms, 3780)
    for term in quad_terms:
      self.assertEqual(term.key.first_var, term.key.second_var)  # must be y^2
      self.assertEqual(term.coefficient, 1.0)

  @mock.patch.object(nonlinear_solver, "generate_objective_terms")
  @mock.patch.object(nonlinear_solver, "do_solve")
  def test_run_node_optimization_retries_alpha_on_failure(
      self, mock_do_solve, mock_gen_terms
  ):
    cov1 = self._random_succinct_covariance()
    cov2 = self._random_succinct_covariance()
    val1 = np.ones(self.n) * 5
    val2 = np.ones(self.n) * 5
    parent_val = [10] * self.n
    ub = [1000000] * 8

    mock_gen_terms.return_value = []
    mock_result = mock.Mock(spec=mathopt.SolveResult)
    mock_do_solve.side_effect = [
        RuntimeError("model failed to solve (NUMERICAL_ERROR)"),
        RuntimeError("model failed to solve (IMPRECISE)"),
        mock_result,
    ]

    _, result = nonlinear_solver.run_node_optimization(
        objective_function_type=constants.ObjectiveFunctionType.FULL,
        parent_geocode="p",
        child_geocodes=["c1", "c2"],
        parent_val=parent_val,
        child_ests={"c1": Estimate(val1, cov1), "c2": Estimate(val2, cov2)},
        children_upper_bound_dict={"c1": ub, "c2": ub},
        children_lower_bound_dict={"c1": [0] * 8, "c2": [0] * 8},
    )

    self.assertIs(result, mock_result)
    self.assertEqual(mock_do_solve.call_count, 3)
    tried_alphas = [
        call.kwargs["alpha"] for call in mock_gen_terms.call_args_list[::2]
    ]
    self.assertEqual(tried_alphas, [10, 50, 5])

    # If all barrier alphas fail, falls back to dual simplex on the last model.
    mock_do_solve.reset_mock()
    mock_do_solve.side_effect = [
        RuntimeError("model failed to solve (IMPRECISE)")
    ] * len(constants.ALPHA_VALUES) + [mock_result]

    _, result = nonlinear_solver.run_node_optimization(
        objective_function_type=constants.ObjectiveFunctionType.FULL,
        parent_geocode="p",
        child_geocodes=["c1", "c2"],
        parent_val=parent_val,
        child_ests={"c1": Estimate(val1, cov1), "c2": Estimate(val2, cov2)},
        children_upper_bound_dict={"c1": ub, "c2": ub},
        children_lower_bound_dict={"c1": [0] * 8, "c2": [0] * 8},
    )

    self.assertIs(result, mock_result)
    self.assertLen(
        mock_do_solve.call_args_list, len(constants.ALPHA_VALUES) + 1
    )
    self.assertTrue(mock_do_solve.call_args_list[-1].kwargs.get("use_simplex"))

  @mock.patch.object(nonlinear_solver, "generate_objective_terms")
  @mock.patch.object(nonlinear_solver, "do_solve")
  def test_run_node_optimization_raises_when_all_alphas_fail(
      self, mock_do_solve, mock_gen_terms
  ):
    cov1 = self._random_succinct_covariance()
    cov2 = self._random_succinct_covariance()
    val1 = np.ones(self.n) * 5
    val2 = np.ones(self.n) * 5
    parent_val = [10] * self.n
    ub = [1000000] * 8

    mock_gen_terms.return_value = []
    mock_do_solve.side_effect = RuntimeError(
        "model failed to solve (NUMERICAL_ERROR)"
    )

    with self.assertRaisesRegex(
        RuntimeError,
        r"Optimization failed for parent node 'p' \(FULL\) across all alpha "
        r"values \[10, 50, 5, 100, 1, 500, 0\.1\].*NUMERICAL_ERROR",
    ):
      nonlinear_solver.run_node_optimization(
          objective_function_type=constants.ObjectiveFunctionType.FULL,
          parent_geocode="p",
          child_geocodes=["c1", "c2"],
          parent_val=parent_val,
          child_ests={"c1": Estimate(val1, cov1), "c2": Estimate(val2, cov2)},
          children_upper_bound_dict={"c1": ub, "c2": ub},
          children_lower_bound_dict={"c1": [0] * 8, "c2": [0] * 8},
      )

    # All alpha values tried with barrier, then one simplex retry.
    self.assertLen(
        mock_do_solve.call_args_list, len(constants.ALPHA_VALUES) + 1
    )
    self.assertTrue(mock_do_solve.call_args_list[-1].kwargs.get("use_simplex"))
    tried_alphas = [
        call.kwargs["alpha"] for call in mock_gen_terms.call_args_list[::2]
    ]
    self.assertEqual(tried_alphas, list(constants.ALPHA_VALUES))

  def test_variable_values_clamps_small_negative_solver_noise(self):
    model = mathopt.Model(name="test")
    v_pos = model.add_variable(lb=0, name="v_pos")
    v_noise = model.add_variable(lb=0, name="v_noise")
    v_neg = model.add_variable(name="v_neg")

    raw_vals = {v_pos: 22.0, v_noise: -6.17e-12, v_neg: -0.5}
    mock_result = mock.Mock(spec=mathopt.SolveResult)
    mock_result.variable_values.side_effect = lambda v: raw_vals[v]

    vals = nonlinear_solver.variable_values(mock_result, [v_pos, v_noise, 0.0])
    self.assertEqual(vals, [22.0, 0.0, 0.0])
    self.assertEqual(
        list(np.floor(vals).astype(int)),
        [22, 0, 0],
    )

    with self.assertRaisesRegex(RuntimeError, "are negative"):
      nonlinear_solver.variable_values(mock_result, [v_pos, v_neg])


if __name__ == "__main__":
  absltest.main()
