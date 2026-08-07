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

from collections.abc import Iterator, Sequence
import numpy as np
from census_bluedown import block
from census_bluedown import estimate
from census_bluedown import regression
from absl.testing import absltest
from absl.testing import parameterized


class RegressionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="asymmetric_feature",
           shape=(2,),
           num_asymmetric_features=1,
           query_shapes=((2,),),
           query_variances=(1.5,),
           expected_combination=np.array([[1.0, 0.0],
                                          [0.0, 1.0]]),
           expected_covariance=np.array([[1.5, 0.0],
                                         [0.0, 1.5]])),
      dict(testcase_name="symmetric_feature",
           shape=(3,),
           num_asymmetric_features=0,
           query_shapes=((3,),),
           query_variances=(1.5,),
           expected_combination=np.array([[1.0, 0.0],
                                          [0.0, 1.0]]),
           expected_covariance=np.array([[1.5, 1.5],
                                         [1.5, 4.5]])),
      dict(testcase_name="multiple_features",
           shape=(2, 3),
           num_asymmetric_features=1,
           query_shapes=((2, 3), (2, 1), (1, 3), (1, 1)),
           query_variances=(1.0, 1.0, 1.0, 1.0),
           expected_combination=np.array(
               [[2/3, -1/6, -1/3, 1/12, 1/6, -1/12, 1/3, -1/12, 1/12],
                [0, 1/6, 0, -1/12, 1/2, -1/4, 0, 1/12, 1/4],
                [-1/3, 1/12, 2/3, -1/6, -1/12, 1/6, 1/3, -1/12, 1/12],
                [0, -1/12, 0, 1/6, -1/4, 1/2, 0, 1/12, 1/4]]
           ),
           expected_covariance=np.array(
               [[1/2, 1/6, -1/4, -1/12],
                [1/6, 1/2, -1/12, -1/4],
                [-1/4, -1/12, 1/2, 1/6],
                [-1/12, -1/4, 1/6, 1/2]]
           )),
  )
  def test_regression_combination(
      self,
      shape: block.QueryShape,
      num_asymmetric_features: int,
      query_shapes: Sequence[block.QueryShape],
      query_variances: Sequence[float],
      expected_combination: np.ndarray,
      expected_covariance: np.ndarray
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    linear_combination, covariance = regression.regression_combination(
        block_shape, query_shapes, query_variances)
    self.assertEqual(
        linear_combination.T.shape,
        block_shape.design_matrix(query_shapes).shape)
    self.assertEqual(
        covariance.shape,
        (block_shape.compressed_length, block_shape.compressed_length))
    np.testing.assert_allclose(linear_combination, expected_combination)
    np.testing.assert_allclose(covariance, expected_covariance)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           shape=(2,),
           num_asymmetric_features=1,
           asymmetric_queries=(),
           query_shape=(2,),
           query_value=np.array([1.0, 2.0]),
           linear_combination_iterator=iter(np.array([[1.0, 0.0],
                                                      [0.0, 1.0]])),
           expected=np.array([1.0, 2.0])),
      dict(testcase_name="slice_symmetric",
           shape=(2, 3),
           num_asymmetric_features=1,
           asymmetric_queries=(),
           query_shape=(1, 3),
           query_value=np.array([[1.0, 2.0, 3.0]]),
           linear_combination_iterator=iter(np.array([[2.0, 1.0],
                                                      [1.0, 3.0]])),
           # The aggregated query values are [[1.0, 2.0, 3.0]] and [[6.0]]
           # and the updates to estimate_update are
           # [[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]] and
           # [[6.0, 6.0, 6.0], [18.0, 18.0, 18.0]]
           expected=np.array([[8.0, 10.0, 12.0],
                              [19.0, 20.0, 21.0]])),
      dict(testcase_name="slice_both",
           shape=(2, 3),
           num_asymmetric_features=1,
           asymmetric_queries=(),
           query_shape=(2, 3),
           query_value=np.array([[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]),
           linear_combination_iterator=iter(
               np.array([[2.0, 1.0],
                         [1.0, 3.0],
                         [2.0, 2.0],
                         [1.0, 4.0]])),
           # The aggregated query values are [[1.0, 2.0, 3.0]], [[6.0]],
           # [[4.0, 5.0, 6.0]], [[15.0]], and the updates to estimate_update are
           # [[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]],
           # [[6.0, 6.0, 6.0], [18.0, 18.0, 18.0]],
           # [[8.0, 10.0, 12.0], [8.0, 10.0, 12.0]], and
           # [[15.0, 15.0, 15.0], [60.0, 60.0, 60.0]]
           expected=np.array([[31.0, 35.0, 39.0],
                              [87.0, 90.0, 93.0]])),
      dict(testcase_name="multiple_features",
           shape=(2, 2, 2, 2),
           num_asymmetric_features=2,
           asymmetric_queries=(),
           query_shape=(1, 2, 2, 1),
           query_value=np.array([[4.0, 3.0],
                                 [2.0, 1.0]]),
           linear_combination_iterator=iter(
               np.array([[1.0, 2.0, 0.0, 0.0],
                         [0.0, 0.0, 2.0, 1.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [1.0, 3.0, 3.0, 1.0]])),
           # The aggregated query values are [[[[4.0], [3.0]]]], [[[[7.0]]]],
           # [[[[2.0], [1.0]]]], [[[[3.0]]]], and the updates to estimate_update
           # are
           # [[[[4], [3]], [[8], [6]]], [[[0], [0]], [[0], [0]]]],
           # [[[[0], [0]], [[0], [0]]], [[[14], [14]], [[7], [7]]]],
           # [[[[2], [1]], [[2], [1]]], [[[2], [1]], [[2], [1]]]], and
           # [[[[3], [3]], [[9], [9]]], [[[9], [9]], [[3], [3]]]]
           expected=np.array(
               [[[[9.0, 9.0], [7.0, 7.0]],
                 [[19.0, 19.0], [16.0, 16.0]]],
                [[[25.0, 25.0], [24.0, 24.0]],
                 [[12.0, 12.0], [11.0, 11.0]]]])),
      dict(testcase_name="asymmetric_query",
           shape=(6,),
           num_asymmetric_features=1,
           asymmetric_queries=(block.AsymmetricQuery(
               num_features=1,
               sliced_feature=0,
               partition=(1, 3, 2)
           ),),
           query_shape=(3,),
           query_value=np.array([1.0, 2.0, 3.0]),
           linear_combination_iterator=iter(
               np.array([[4.0, 3.0, 2.0, 1.0, 2.0, 3.0],
                         [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
           ),
           # The aggregated query values are [1.0], [2.0], [3.0], and the
           # updates to estimate_update are
           # [4.0, 3.0, 2.0, 1.0, 2.0, 3.0],
           # [2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
           # [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
           expected=np.array([6.0, 3.0, 2.0, 3.0, 2.0, 6.0]))
  )
  def test_query_contribution(
      self,
      shape: block.QueryShape,
      num_asymmetric_features: int,
      asymmetric_queries: tuple[block.AsymmetricQuery, ...],
      query_shape: block.QueryShape,
      query_value: np.ndarray,
      linear_combination_iterator: Iterator[np.ndarray],
      expected: np.ndarray
  ):
    block_shape = block.BlockShape(shape,
                                   num_asymmetric_features,
                                   asymmetric_queries)
    computed = regression.query_contribution(
        block_shape, query_shape, query_value, linear_combination_iterator)
    np.testing.assert_allclose(computed, expected)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           shape=(1,),
           num_asymmetric_features=1,
           linear_transformation=np.array([[2.0]]),
           initial_estimate=np.array([1.0]),
           expected=np.array([2.0])),
      dict(testcase_name="two_features",
           shape=(2, 3),
           num_asymmetric_features=1,
           linear_transformation=np.array(
               [[2.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 3.0, 1.0],
                [2.0, 0.0, 2.0, 1.0],
                [1.0, 0.0, 4.0, 0.0]]).T,
           initial_estimate=np.array([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]]),
           # Same as slice_both test case for regression.update_estimate
           expected=np.array([[31.0, 35.0, 39.0],
                              [87.0, 90.0, 93.0]])),
  )
  def test_apply_linear_transformation(
      self,
      shape: block.QueryShape,
      num_asymmetric_features: int,
      linear_transformation: np.ndarray,
      initial_estimate: np.ndarray,
      expected: np.ndarray
  ):
    block_shape = block.BlockShape(shape, num_asymmetric_features)
    computed = regression.apply_linear_transformation(
        block_shape, linear_transformation, initial_estimate)
    np.testing.assert_allclose(computed, expected)
    if block_shape.num_features - block_shape.num_asymmetric_features == 1:
      computed_alternate = regression.apply_linear_transformation_alternate(
          block_shape, linear_transformation, initial_estimate)
      np.testing.assert_allclose(computed_alternate, expected)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           block_shape=block.BlockShape(shape=(1,), num_asymmetric_features=1),
           query_shapes=((1,),),
           query_variances=(2.0,),
           query_values=([3.0],),
           expected_estimate_value=np.array([3.0]),
           expected_covariance=np.array([[2.0]])),
      dict(testcase_name="asymmetric_feature",
           block_shape=block.BlockShape(shape=(3,), num_asymmetric_features=1),
           query_shapes=((3,), (1,)),
           query_variances=(2.0, 3.0),
           query_values=([1.0, 2.0, 3.0], [6.0]),
           expected_estimate_value=np.array([1.0, 2.0, 3.0]),
           expected_covariance=np.array([[14/9, -4/9, -4/9],
                                         [-4/9, 14/9, -4/9],
                                         [-4/9, -4/9, 14/9]])),
      dict(testcase_name="symmetric_feature_inexact_queries",
           block_shape=block.BlockShape(shape=(3,), num_asymmetric_features=0),
           query_shapes=((3,), (1,)),
           query_variances=(2.0, 3.0),
           query_values=([1.0, 2.0, 3.0], [7.0]),
           expected_estimate_value=np.array([11/9, 20/9, 29/9]),
           expected_covariance=np.array([[14/9, 2/3],
                                         [2/3, 2]])),
      dict(testcase_name="multiple_features",
           block_shape=block.BlockShape(shape=(2, 3),
                                        num_asymmetric_features=1),
           query_shapes=((2, 3), (2, 1), (1, 3), (1, 1)),
           query_variances=(1.0, 1.0, 1.0, 1.0),
           query_values=([1, 1, 1, 3, 3, 3],
                         [3, 9],
                         [2, 3, 4],
                         [15]),
           expected_estimate_value=np.array([5/6, 7/6, 3/2, 17/6, 19/6, 7/2]),
           expected_covariance=np.array([[1/2, 1/6, -1/4, -1/12],
                                         [1/6, 1/2, -1/12, -1/4],
                                         [-1/4, -1/12, 1/2, 1/6],
                                         [-1/12, -1/4, 1/6, 1/2]])),
  )
  def test_process_queries(
      self,
      block_shape: block.BlockShape,
      query_shapes: tuple[block.QueryShape, ...],
      query_variances: tuple[float, ...],
      query_values: tuple[list[int], ...],
      expected_estimate_value: np.ndarray,
      expected_covariance: np.ndarray
  ):
    computed_value, computed_covariance = regression.process_queries(
        block_shape, query_shapes, query_variances, query_values
    )
    np.testing.assert_allclose(computed_value, expected_estimate_value)
    np.testing.assert_allclose(computed_covariance, expected_covariance)

  @parameterized.named_parameters(
      dict(testcase_name="simple",
           block_shape=block.BlockShape(shape=(1,), num_asymmetric_features=1),
           estimate_1=estimate.Estimate(val=np.array([1.0]),
                                        cov=np.array([[2.0]])),
           estimate_2=estimate.Estimate(val=np.array([3.0]),
                                        cov=np.array([[4.0]])),
           expected=estimate.Estimate(val=np.array([5/3]),
                                      cov=np.array([[4/3]]))),
      dict(testcase_name="singular_est",
           block_shape=block.BlockShape(shape=(1,), num_asymmetric_features=1),
           estimate_1=estimate.Estimate(val=np.array([1.0]),
                                        cov=np.array([[0.0]])),
           estimate_2=estimate.Estimate(val=np.array([3.0]),
                                        cov=np.array([[4.0]])),
           expected=estimate.Estimate(val=np.array([1.0]),
                                      cov=np.array([[0.0]]))),
      dict(testcase_name="both_singular",
           block_shape=block.BlockShape(shape=(1,), num_asymmetric_features=1),
           estimate_1=estimate.Estimate(val=np.array([3.0]),
                                        cov=np.array([[0.0]])),
           estimate_2=estimate.Estimate(val=np.array([3.0]),
                                        cov=np.array([[0.0]])),
           expected=estimate.Estimate(val=np.array([3.0]),
                                      cov=np.array([[0.0]]))),
      dict(testcase_name="asymmetric_feature",
           block_shape=block.BlockShape(shape=(3,), num_asymmetric_features=1),
           estimate_1=estimate.Estimate(
               val=np.array([1.0, 2.0, 3.0]),
               cov=np.array([[2.0, 1.0, 0.0],
                             [1.0, 2.0, 0.0],
                             [0.0, 0.0, 0.0]])),
           estimate_2=estimate.Estimate(
               val=np.array([2.0, 3.0, 4.0]),
               cov=np.array([[2.0, 1.0, 1.0],
                             [1.0, 2.0, 1.0],
                             [1.0, 1.0, 2.0]])),
           expected=estimate.Estimate(
               val=np.array([1.3, 2.3, 3.0]),
               cov=np.array([[0.85, 0.35, 0.0],
                             [0.35, 0.85, 0.0],
                             [0.0, 0.0, 0.0]]))),
      dict(testcase_name="simple_singular_covariances",
           block_shape=block.BlockShape(shape=(3,), num_asymmetric_features=1),
           estimate_1=estimate.Estimate(
               val=np.array([1.0, 2.0, 3.0]),
               cov=np.array([[2.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])),
           estimate_2=estimate.Estimate(
               val=np.array([2.0, 3.0, 3.0]),
               cov=np.array([[2.0, 1.0, 0.0],
                             [1.0, 2.0, 0.0],
                             [0.0, 0.0, 0.0]])),
           expected=estimate.Estimate(
               val=np.array([9/7, 2, 3]),
               cov=np.array([[6/7, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]]))),
  )
  def test_combine_estimates(
      self,
      block_shape: block.BlockShape,
      estimate_1: estimate.Estimate,
      estimate_2: estimate.Estimate,
      expected: estimate.Estimate,
  ):
    computed_1 = regression.combine_estimates(
        block_shape, estimate_1, estimate_2)
    computed_2 = regression.combine_estimates(
        block_shape, estimate_2, estimate_1)
    np.testing.assert_allclose(computed_1.val, expected.val)
    np.testing.assert_allclose(computed_1.cov, expected.cov, atol=1e-20)
    np.testing.assert_allclose(computed_2.val, expected.val)
    np.testing.assert_allclose(computed_2.cov, expected.cov, atol=1e-20)

  def test_combine(self):
    block_shape = block.BlockShape(shape=(1,), num_asymmetric_features=1)
    est = estimate.Estimate(val=np.array([1.0]), cov=np.array([[2.0]]))

    # Both estimates belong to class estimate.Estimate
    computed_estimate = regression.combine(block_shape, est, est)
    np.testing.assert_allclose(computed_estimate.val, np.array([1.0]))
    np.testing.assert_allclose(computed_estimate.cov, np.array([[1.0]]))

    # First estimate is np.nan
    computed_estimate = regression.combine(block_shape, np.nan, est)
    np.testing.assert_allclose(computed_estimate.val, est.val)
    np.testing.assert_allclose(computed_estimate.cov, est.cov)

    # Second estimate is np.nan
    computed_estimate = regression.combine(block_shape, est, np.nan)
    np.testing.assert_allclose(computed_estimate.val, est.val)
    np.testing.assert_allclose(computed_estimate.cov, est.cov)

    # Both estimates are np.nan
    with self.assertRaises(ValueError):
      regression.combine(block_shape, np.nan, np.nan)


if __name__ == "__main__":
  absltest.main()
