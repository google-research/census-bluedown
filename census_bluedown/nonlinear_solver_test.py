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

from unittest import mock

import numpy as np

from census_bluedown import block
from census_bluedown import estimate
from census_bluedown import nonlinear_solver
from absl.testing import absltest
from absl.testing import parameterized
from ortools.math_opt.python import mathopt


class NonlinearSolverTest(parameterized.TestCase):
  @mock.patch.object(nonlinear_solver, 'add_sum_range_constraint')
  def test_add_feature_constraints(
      self,
      add_sum_range_constraint_mock):
    block_shape = block.BlockShape(
        shape=(8, 2, 2, 63),
        num_asymmetric_features=3)
    model = mock.Mock()
    variables = [mock.Mock() for _ in range(2016)]
    lower_bounds = list(range(8))
    upper_bounds = list(range(2, 10))

    nonlinear_solver.add_feature_constraints(
        block_shape=block_shape,
        model=model,
        variables=variables,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds
    )

    add_sum_range_constraint_mock.assert_has_calls(
        [mock.call(model=model, variables=variables[0:252],
                   lower_bound=0, upper_bound=2, offsets=None),
         mock.call(model=model, variables=variables[252:504],
                   lower_bound=1, upper_bound=3, offsets=None),
         mock.call(model=model, variables=variables[504:756],
                   lower_bound=2, upper_bound=4, offsets=None),
         mock.call(model=model, variables=variables[756:1008],
                   lower_bound=3, upper_bound=5, offsets=None),
         mock.call(model=model, variables=variables[1008:1260],
                   lower_bound=4, upper_bound=6, offsets=None),
         mock.call(model=model, variables=variables[1260:1512],
                   lower_bound=5, upper_bound=7, offsets=None),
         mock.call(model=model, variables=variables[1512:1764],
                   lower_bound=6, upper_bound=8, offsets=None),
         mock.call(model=model, variables=variables[1764:2016],
                   lower_bound=7, upper_bound=9, offsets=None)]
    )

  @mock.patch.object(mathopt, 'fast_sum')
  def test_add_parent_sum_constraints(
      self,
      mock_fast_sum
  ):
    model = mock.Mock()
    child_geocodes = ['child1', 'child2', 'child3']
    model_variables = {
        'child1': [mock.Mock() for _ in range(2016)],
        'child2': [mock.Mock() for _ in range(2016)],
        'child3': [mock.Mock() for _ in range(2016)]
    }

    parent_val = list(range(2016))
    equality_expressions = [mock.Mock() for _ in range(2016)]
    mock_sums = [mock.MagicMock() for _ in range(2016)]

    mock_fast_sum.side_effect = mock_sums
    for i in range(2016):
      mock_sums[i].__eq__.return_value = equality_expressions[i]

    nonlinear_solver.add_parent_sum_constraints(
        model=model,
        model_variables=model_variables,  # pyrefly: ignore[bad-argument-type]
        parent_val=parent_val
    )

    model.add_linear_constraint.assert_has_calls(
        [mock.call(x) for x in equality_expressions]
    )
    mock_fast_sum.assert_has_calls(
        [mock.call([model_variables[j][i] for j in child_geocodes])
         for i in range(2016)]
    )
    for i in range(2016):
      mock_sums[i].__eq__.assert_called_once_with(float(i))  # pyrefly: ignore[missing-attribute]

  def _symmetric_compression_matrix(self, symmetric_feature_values):
    matrix = [[0]*symmetric_feature_values, [1]*symmetric_feature_values]
    matrix[0][0] = 1
    return np.asarray(matrix)

  def _get_compressed_estimate(
      self,
      block_shape: block.BlockShape,
      estimate_values: np.ndarray,
      uncompressed_cov: np.ndarray
    ):
    compression_matrix = np.kron(
        np.eye(block_shape.asymmetric_length),
        self._symmetric_compression_matrix(block_shape.shape[-1]),
    )

    return estimate.Estimate(
        val=estimate_values,
        cov=compression_matrix @ uncompressed_cov @ compression_matrix.T
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='two_features',
          block_shape=block.BlockShape(
              shape=(2, 3),
              num_asymmetric_features=1
          ),
          estimate_values=np.array([1, 2, 3, 4, 5, 6]),
          uncompressed_cov=np.array(
              [[27, 16, 16, 36, 1, 1],
               [16, 27, 16, 1, 36, 1],
               [16, 16, 27, 1, 1, 36],
               [36, 1, 1, 9, 2, 2],
               [1, 36, 1, 2, 9, 2],
               [1, 1, 36, 2, 2, 9]]),
          variable_values=np.array([5, 3, 1, 6, 7, 4]),
      ),
      dict(
          testcase_name='singular_covariance',
          block_shape=block.BlockShape(
              shape=(2, 3),
              num_asymmetric_features=1
          ),
          estimate_values=np.array([1, 2, 3, 4, 5, 6]),
          uncompressed_cov=np.array(
              [[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 5, 1, 1],
               [0, 0, 0, 1, 5, 1],
               [0, 0, 0, 1, 1, 5]]),
          variable_values=np.array([1, 2, 3, 7, 3, 5]),
      ),
      dict(
          testcase_name='three_features',
          block_shape=block.BlockShape(
              shape=(2, 2, 2),
              num_asymmetric_features=2
          ),
          estimate_values=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
          uncompressed_cov=np.array(
              [[5, 3, 2, 1, 4, 1, 1, 1],
               [3, 5, 1, 2, 1, 4, 1, 1],
               [2, 1, 5, 3, 2, 0, 4, 1],
               [1, 2, 3, 5, 0, 2, 1, 4],
               [4, 1, 2, 0, 8, 2, 2, 2],
               [1, 4, 0, 2, 2, 8, 2, 2],
               [1, 1, 4, 1, 2, 2, 4, 3],
               [1, 1, 1, 4, 2, 2, 3, 4]]),
          variable_values=np.array([1, 2, 3, 7, 3, 5, 4, 6]),
      ),
  )
  def test_generate_total_only_objective_terms(
      self,
      block_shape,
      estimate_values,
      uncompressed_cov,
      variable_values
  ):
    est = self._get_compressed_estimate(
        block_shape=block_shape,
        estimate_values=estimate_values,
        uncompressed_cov=uncompressed_cov
    )

    computed_list = nonlinear_solver.generate_total_only_objective_terms(
        model=mock.Mock(),
        block_shape=block_shape,
        geocode='test',
        est=est,
        variables=variable_values
    )

    difference_vector = np.array(est.val) - np.array(variable_values)
    total_var = uncompressed_cov.sum()
    expected_sum = np.sum(difference_vector)**2 / total_var

    self.assertAlmostEqual(sum(computed_list), expected_sum)


if __name__ == '__main__':
  absltest.main()
