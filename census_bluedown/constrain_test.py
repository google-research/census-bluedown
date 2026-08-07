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

from collections.abc import Sequence
import numpy as np
import pandas as pd
from census_bluedown import block
from census_bluedown import constants
from census_bluedown import constrain
from census_bluedown import estimate
from absl.testing import absltest
from absl.testing import parameterized


BlockShape = block.BlockShape
Estimate = estimate.Estimate

ID = constants.ID
ESTIMATE = constants.ESTIMATE
VALUE = constants.VALUE
QUERY_NAME = constants.QUERY_NAME
FIRST_FEATURE_UB_CONSTRAINT = constants.FIRST_FEATURE_UB_CONSTRAINT
FIRST_FEATURE_LB_CONSTRAINT = constants.FIRST_FEATURE_LB_CONSTRAINT


class ConstrainTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="simple_case",
          block_shape=BlockShape(
              shape=(3,),
              num_asymmetric_features=1
          ),
          est=estimate.Estimate(
              val=np.array([1, 3, 5]),
              cov=np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])),
          Q=np.array([[1, 1, 0]]).T,
          c=np.array([2]),
          expected=estimate.Estimate(
              val=np.array([0, 2, 5]),
              cov=np.array([[0.5, -0.5, 0],
                            [-0.5, 0.5, 0],
                            [0, 0, 1]]))),
      dict(
          testcase_name="two_asymmetric_features",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=2
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])),
          Q=np.array([[1, 1, 0, 0]]).T,  # Constrain sum of first two values
          c=np.array([2]),
          expected=estimate.Estimate(
              val=np.array([0.5, 1.5, 3, 4]),
              cov=np.array([[0.5, -0.5, 0, 0],
                            [-0.5, 0.5, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]))),
      dict(
          testcase_name="one_asymmetric_one_symmetric",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 1, 0, 0],  # Uncorrelated estimates of each
                            [1, 2, 0, 0],  # uncompressed value
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]])),
          Q=np.array([[0, 1, 0, 0]]).T,  # Constrain sum of first two values
          c=np.array([2]),
          expected=estimate.Estimate(
              val=np.array([0.5, 1.5, 3, 4]),
              cov=np.array([[0.5, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]]))),
      dict(
          testcase_name="constrain_individual_values",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 1, 0, 0],  # Uncorrelated estimates of each
                            [1, 2, 0, 0],  # uncompressed value
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]])),
          Q=np.array([[1, 0, 0, 0],      # Constrain first two values and their
                      [0, 1, 0, 0]]).T,  # sum.
          c=np.array([-1, -2]),
          expected=estimate.Estimate(
              val=np.array([-1, -1, 3, 4]),
              cov=np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]]))),
      dict(
          testcase_name="two_asymmetric_correlated",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=2
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[2, .5, 1, 0],
                            [.5, 2, 0, 1],
                            [1, 0, 4, .5],
                            [0, 1, .5, 4]])),
          Q=np.array([[1, 1, 0, 0]]).T,  # Constrain sum of first two values
          c=np.array([5]),
          expected=estimate.Estimate(
              val=np.array([2, 3, 3.4, 4.4]),
              cov=np.array([[0.75, -0.75, 0.5, -0.5],
                            [-0.75, 0.75, -0.5, 0.5],
                            [0.5, -0.5, 3.8, 0.3],
                            [-0.5, 0.5, 0.3, 3.8]]))),
      dict(
          # The same example but compressing the second feature
          testcase_name="one_asymmetric_one_symmetric_correlated",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[2, 2.5, 1, 1],
                            [2.5, 5, 1, 2],
                            [1, 1, 4, 4.5],
                            [1, 2, 4.5, 9]])),
          Q=np.array([[0, 1, 0, 0]]).T,  # Constrain sum of first two values
          c=np.array([5]),
          expected=estimate.Estimate(
              val=np.array([2, 3, 3.4, 4.4]),
              # Can compute covariance by transforming the covariance of the
              # uncompressed computation (test "two_asymmetric_corrleated")
              # via the formula L @ cov @ L.T using the compression linear
              # transformation L = np.array([[1, 0, 0, 0],
              #                              [1, 1, 0, 0],
              #                              [0, 0, 1, 0],
              #                              [0, 0, 1, 1]])
              cov=np.array([[0.75, 0, 0.5, 0],
                            [0, 0, 0, 0],
                            [0.5, 0, 3.8, 4.1],
                            [0, 0, 4.1, 8.2]]))),
      dict(
          testcase_name="three_features",
          block_shape=BlockShape(
              shape=(2, 3, 2),
              num_asymmetric_features=3
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4, 3, 4, 5, 6, 4, 3, 2, 1]),
              cov=np.array([
                  [8, 2, 3, 1, 3, 1, 1, 0, .5, 0, .5, 0],
                  [2, 8, 1, 3, 1, 3, 0, 1, 0, .5, 0, .5],
                  [3, 1, 8, 2, 3, 1, .5, 0, 1, 0, .5, 0],
                  [1, 3, 2, 8, 1, 3, 0, .5, 0, 1, 0, .5],
                  [3, 1, 3, 1, 8, 2, .5, 0, .5, 0, 1, 0],
                  [1, 3, 1, 3, 2, 8, 0, .5, 0, .5, 0, 1],
                  [1, 0, .5, 0, .5, 0, 4, -1, 2, 0, 2, 0],
                  [0, 1, 0, .5, 0, .5, -1, 4, 0, 2, 0, 2],
                  [.5, 0, 1, 0, .5, 0, 2, 0, 4, -1, 2, 0],
                  [0, .5, 0, 1, 0, .5, 0, 2, -1, 4, 0, 2],
                  [.5, 0, .5, 0, 1, 0, 2, 0, 2, 0, 4, -1],
                  [0, .5, 0, .5, 0, 1, 0, 2, 0, 2, -1, 4]
                  ])),
          # Constrain sum of first six values
          Q=np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]).T,
          c=np.array([10]),
          expected=estimate.Estimate(
              val=np.array(
                  [-1/6, 5/6, 11/6, 17/6, 11/6, 17/6,
                   4.87037, 5.87037, 3.87037, 2.87037, 1.87037, 0.87037]
                  ),
              cov=np.array(
                  [[5.0, -1.0, 0.0, -2.0, 0.0, -2.0,
                    2/3, -1/3, 1/6, -1/3, 1/6, -1/3],
                   [-1.0, 5.0, -2.0, 0.0, -2.0, 0.0,
                    -1/3, 2/3, -1/3, 1/6, -1/3, 1/6],
                   [0.0, -2.0, 5.0, -1.0, 0.0, -2.0,
                    1/6, -1/3, 2/3, -1/3, 1/6, -1/3],
                   [-2.0, 0.0, -1.0, 5.0, -2.0, 0.0,
                    -1/3, 1/6, -1/3, 2/3, -1/3, 1/6],
                   [0.0, -2.0, 0.0, -2.0, 5.0, -1.0,
                    1/6, -1/3, 1/6, -1/3, 2/3, -1/3],
                   [-2.0, 0.0, -2.0, 0.0, -1.0, 5.0,
                    -1/3, 1/6, -1/3, 1/6, -1/3, 2/3],
                   [2/3, -1/3, 1/6, -1/3, 1/6, -1/3,
                    107/27, -28/27, 53/27, -1/27, 53/27, -1/27],
                   [-1/3, 2/3, -1/3, 1/6, -1/3, 1/6,
                    -28/27, 107/27, -1/27, 53/27, -1/27, 53/27],
                   [1/6, -1/3, 2/3, -1/3, 1/6, -1/3,
                    53/27, -1/27, 107/27, -28/27, 53/27, -1/27],
                   [-1/3, 1/6, -1/3, 2/3, -1/3, 1/6,
                    -1/27, 53/27, -28/27, 107/27, -1/27, 53/27],
                   [1/6, -1/3, 1/6, -1/3, 2/3, -1/3,
                    53/27, -1/27, 53/27, -1/27, 107/27, -28/27],
                   [-1/3, 1/6, -1/3, 1/6, -1/3, 2/3,
                    -1/27, 53/27, -1/27, 53/27, -28/27, 107/27]]))),
      dict(
          testcase_name="three_features_two_symmetric",
          # The same example but compressing the second and third features
          block_shape=BlockShape(
              shape=(2, 3, 2),
              num_asymmetric_features=1
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4, 3, 4, 5, 6, 4, 3, 2, 1]),
              # Can compute covariances by transforming the covariances of the
              # uncompressed computation (test "three_features")
              # via the formula L @ cov @ L.T using the compression linear
              # transformation L = np.array([
              #   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              #   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              #   [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              #   [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              #   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              #   [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              #   [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
              #   [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
              cov=np.array([
                  [8, 10, 14, 18, 1, 1, 2, 2],
                  [10, 20, 18, 36, 1, 2, 2, 4],
                  [14, 18, 42, 54, 2, 2, 6, 6],
                  [18, 36, 54, 108, 2, 4, 6, 12],
                  [1, 1, 2, 2, 4, 3, 8, 7],
                  [1, 2, 2, 4, 3, 6, 7, 14],
                  [2, 2, 6, 6, 8, 7, 24, 21],
                  [2, 4, 6, 12, 7, 14, 21, 42]])),
          # Constrain sum of first six values
          Q=np.array([[0, 0, 0, 1, 0, 0, 0, 0]]).T,
          c=np.array([10]),
          expected=estimate.Estimate(
              val=np.array(
                  [-1/6, 5/6, 11/6, 17/6, 11/6, 17/6,
                   4.87037, 5.87037, 3.87037, 2.87037, 1.87037, 0.87037]
                  ),
              cov=np.array(
                  [[5.0, 4.0, 5.0, 0.0, 2/3, 1/3, 1.0, 0.0],
                   [4.0, 8.0, 0.0, 0.0, 1/3, 2/3, 0.0, 0.0],
                   [5.0, 0.0, 15.0, 0.0, 1.0, 0.0, 3.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [2/3, 1/3, 1.0, 0.0, 107/27, 79/27, 71/9, 61/9],
                   [1/3, 2/3, 0.0, 0.0, 79/27, 158/27, 61/9, 122/9],
                   [1.0, 0.0, 3.0, 0.0, 71/9, 61/9, 71/3, 61/3],
                   [0.0, 0.0, 0.0, 0.0, 61/9, 122/9, 61/3, 122/3]]))),
      dict(
          testcase_name="sum_one_symmetric_feature",
          block_shape=BlockShape(
              shape=(3,),
              num_asymmetric_features=0
          ),
          est=estimate.Estimate(
              val=np.array([1, 2, 3]),
              cov=np.array([[2, 4],
                            [4, 12]])),
          Q=np.array([[0, 1]]).T,
          c=np.array([3]),
          expected=estimate.Estimate(
              val=np.array([0, 1, 2]),
              cov=np.array([[2/3, 0],
                            [0, 0]]))),
      dict(
          testcase_name="sum_two_features",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 1, 0, 0],  # Uncorrelated estimates of each
                            [1, 2, 0, 0],  # uncompressed value
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]])),
          Q=np.array([[0, 1, 0, 1]]).T,
          c=np.array([6]),
          expected=Estimate(
              val=np.array([0, 1, 2, 3]),
              cov=np.array([[0.75, 0.5, -0.25, -0.5],
                            [0.5, 1, -0.5, -1],
                            [-0.25, -0.5, 0.75, 0.5],
                            [-0.5, -1, 0.5, 1]]))),
      dict(
          testcase_name="sum_pseudoinverse_objective_counterexample",
          block_shape=BlockShape(
              shape=(2,),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([0, 1]),
              cov=np.array([[0, 0],
                            [0, 2]])),
          Q=np.array([[1, 1]]).T,
          c=np.array([0]),
          expected=Estimate(
              val=np.array([0, 0]),
              cov=np.array([[0, 0],
                            [0, 0]]))),
      dict(
          testcase_name="sum_singular_covariance",
          block_shape=BlockShape(
              shape=(4,),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([1, 0, 0, 3]),
              cov=np.array([[2, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 2]])),
          Q=np.array([[1, 1, 1, 1]]).T,
          c=np.array([6]),
          expected=Estimate(
              val=np.array([2, 0, 0, 4]),
              cov=np.array([[1, 0, 0, -1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-1, 0, 0, 1]]))),
      dict(
          testcase_name="sum_singular_covariance_nonzero_fixed_values",
          block_shape=BlockShape(
              shape=(4,),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[2, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 2]])),
          Q=np.array([[1, 1, 1, 1]]).T,
          c=np.array([8]),
          expected=Estimate(
              val=np.array([0, 2, 3, 3]),
              cov=np.array([[1, 0, 0, -1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-1, 0, 0, 1]]))),
      dict(
          testcase_name="sum_singular_covariance_with_compression",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[2, 2, 0, 0],
                            [2, 4, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])),
          Q=np.array([[0, 1, 0, 1]]).T,
          c=np.array([8]),
          expected=Estimate(
              val=np.array([0, 1, 3, 4]),
              cov=np.array([[1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]))),
      dict(
          testcase_name="first_feature_zero_constraint_two_features",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 1, 0, 0],  # Uncorrelated estimates of each
                            [1, 2, 0, 0],  # uncompressed value
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]])),
          Q=np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).T,
          c=np.array([0, 0]),
          expected=Estimate(
              val=np.array([0, 0, 3, 4]),
              cov=np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]]))),
      dict(
          testcase_name="first_feature_zero_constraint_corr_no_compression",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=2),
          est=Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 0, .2, .2],
                            [0, 1, .2, .2],
                            [.2, .2, 1, 0],
                            [.2, .2, 0, 1]])),
          Q=np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).T,
          c=np.array([0, 0]),
          expected=Estimate(
              val=np.array([0, 0, 2.4, 3.4]),
              cov=np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, .92, -0.08],
                            [0, 0, -0.08, .92]]))),
      dict(
          testcase_name="first_feature_zero_constraint_correlated",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          est=Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 1, .2, .4],
                            [1, 2, .4, .8],
                            [.2, .4, 1, 1],
                            [.4, .8, 1, 2]])),
          Q=np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).T,
          c=np.array([0, 0]),
          expected=Estimate(
              val=np.array([0, 0, 2.4, 3.4]),
              cov=np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, .92, .84],
                            [0, 0, .84, 1.68]]))),
      dict(
          testcase_name="multifeature_zero_combination_of_all",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=2),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4, 5, 6]),
              cov=np.identity(6)),
          Q=np.array([[0, 0, 0, 1, 0, 0]]).T,
          c=np.array([0]),
          expected=estimate.Estimate(
              val=np.array([1, 2, 3, 0, 5, 6]),
              cov=np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]]))),
      dict(
          testcase_name="multifeature_zero_with_symmetric_feature",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 4]),
              cov=np.array([[1, 1, 0, 0],
                            [1, 2, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 1, 2]])),
          Q=np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1]]).T,
          c=np.array([0, 0]),
          expected=estimate.Estimate(
              val=np.array([1, 2, 0, 0]),
              cov=np.array([[1, 1, 0, 0],
                            [1, 2, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]]))),
      dict(
          testcase_name="multifeature_zero_correlations",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=2),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 2, 3, 4]),
              cov=np.array([[2, 0, 0, 1, 1, 1],
                            [0, 2, 0, 1, 1, 1],
                            [0, 0, 2, 1, 1, 1],
                            [1, 1, 1, 4, 2, 2],
                            [1, 1, 1, 2, 4, 2],
                            [1, 1, 1, 2, 2, 4]])),
          Q=np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T,
          c=np.array([0, 0, 0]),
          expected=estimate.Estimate(
              val=np.array([0, 0, 0, -1, 0, 1]),
              cov=np.array([[0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 2.5, 0.5, 0.5],
                            [0, 0, 0, 0.5, 2.5, 0.5],
                            [0, 0, 0, 0.5, 0.5, 2.5]]))),
      dict(
          testcase_name="multifeature_zero_correlations_with_symmetric_feature",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=1),
          est=estimate.Estimate(
              val=np.array([1, 2, 3, 2, 3, 4]),
              cov=np.array([[2, 2, 1, 3],
                            [2, 6, 3, 9],
                            [1, 3, 4, 8],
                            [3, 9, 8, 24]])),
          Q=np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]]).T,
          c=np.array([0, 0]),
          expected=estimate.Estimate(
              val=np.array([0, 0, 0, -1, 0, 1]),
              cov=np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 2.5, 3.5],
                            [0, 0, 3.5, 10.5]]))),
  )
  def test_apply_linear_constraint(
      self,
      block_shape: BlockShape,
      est: Estimate,
      Q: np.ndarray,  # pylint: disable=invalid-name
      c: np.ndarray,
      expected: Estimate):
    computed = constrain.apply_linear_constraint(
        block_shape,
        est,
        Q,
        c)
    np.testing.assert_allclose(computed.val, expected.val, atol=1e-5)
    np.testing.assert_allclose(computed.cov, expected.cov, atol=1e-5)

  @parameterized.named_parameters(
      dict(
          testcase_name="one_symmetric_feature",
          block_shape=BlockShape(
              shape=(3,),
              num_asymmetric_features=0
          ),
          total=3,
          expected_Q=np.array([[0, 1]]).T,
          expected_c=np.array([3])),
      dict(
          testcase_name="one_asymmetric_feature",
          block_shape=BlockShape(
              shape=(4,),
              num_asymmetric_features=1),
          total=6,
          expected_Q=np.array([[1, 1, 1, 1]]).T,
          expected_c=np.array([6])),
      dict(
          testcase_name="two_features",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          total=6,
          expected_Q=np.array([[0, 1, 0, 1]]).T,
          expected_c=np.array([6])),
      dict(
          testcase_name="three_features_two_symmetric",
          block_shape=BlockShape(
              shape=(2, 3, 2),
              num_asymmetric_features=1),
          total=10,
          expected_Q=np.array([[0, 0, 0, 1, 0, 0, 0, 1]]).T,
          expected_c=np.array([10])),
  )
  def test_sum_constraint(
      self,
      block_shape: BlockShape,
      total: int,
      expected_Q: np.ndarray,  # pylint: disable=invalid-name
      expected_c: np.ndarray):
    Q, c = constrain.sum_constraint(  # pylint: disable=invalid-name
        block_shape=block_shape,
        total=total)
    np.testing.assert_allclose(Q, expected_Q)
    np.testing.assert_allclose(c, expected_c)

  @parameterized.named_parameters(
      dict(
          testcase_name="two_features",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          zero_indices=(True, False),
          expected_Q=np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]]).T,
          expected_c=np.array([0, 0])),
      dict(
          testcase_name="two_asymmetric_features",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=2),
          zero_indices=(True, False),
          expected_Q=np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0]]).T,
          expected_c=np.array([0, 0, 0])),
  )
  def test_first_feature_zero_constraints(
      self,
      block_shape: BlockShape,
      zero_indices: Sequence[bool],
      expected_Q: np.ndarray,  # pylint: disable=invalid-name
      expected_c: np.ndarray):
    Q, c = constrain.first_feature_zero_constraints(  # pylint: disable=invalid-name
        block_shape=block_shape,
        zero_indices=zero_indices)
    np.testing.assert_allclose(Q, expected_Q)
    np.testing.assert_allclose(c, expected_c)

  @parameterized.named_parameters(
      dict(
          testcase_name="combination_of_all",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=2),
          feature_values=(1, 0),
          expected_Q=np.array([[0, 0, 0, 1, 0, 0]]).T,
          expected_c=np.array([0])),
      dict(
          testcase_name="asymmetric_only_constraint_on_first_feature",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=2),
          feature_values=(0,),
          expected_Q=np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0]]).T,
          expected_c=np.array([0, 0, 0])),
      dict(
          testcase_name="with_symmetric_feature_constrain_first_value",
          block_shape=BlockShape(
              shape=(2, 3),
              num_asymmetric_features=1),
          feature_values=(0,),
          expected_Q=np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]]).T,
          expected_c=np.array([0, 0])),
      dict(
          testcase_name="with_symmetric_feature_constrain_second_value",
          block_shape=BlockShape(
              shape=(2, 2),
              num_asymmetric_features=1),
          feature_values=(1,),
          expected_Q=np.array([[0, 0, 1, 0],
                               [0, 0, 0, 1]]).T,
          expected_c=np.array([0, 0])),
  )
  def test_apply_multifeature_zero_constraints(
      self,
      block_shape: BlockShape,
      feature_values: tuple[int, ...],
      expected_Q: np.ndarray,  # pylint: disable=invalid-name
      expected_c: np.ndarray):
    Q, c = constrain.multifeature_zero_constraints(  # pylint: disable=invalid-name
        block_shape=block_shape,
        feature_values=feature_values)
    np.testing.assert_allclose(Q, expected_Q)
    np.testing.assert_allclose(c, expected_c)

  def test_constrain_blocks(self):
    block_shape = BlockShape(
        shape=(4, 2),
        num_asymmetric_features=2)
    df = pd.DataFrame({
        ID: ["0", "1", "2"],
        ESTIMATE: [
            Estimate(val=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                     cov=np.identity(8)),
            Estimate(val=np.array([2, 3, 4, 5, 6, 7, 8, 9]),
                     cov=np.identity(8)),
            Estimate(val=np.array([3, 4, 5, 6, 7, 8, 9, 10]),
                     cov=np.identity(8))]})
    constraint_df = pd.DataFrame({
        ID: ["0", "0", "0", "1", "1", "1", "2", "2", "2"],
        QUERY_NAME: 3 * [FIRST_FEATURE_UB_CONSTRAINT,
                         FIRST_FEATURE_LB_CONSTRAINT,
                         "multifeature_zero_constraint"],
        VALUE: [[1, 2, 3, 4], None, None,
                [0, 0, 5, 5], None, None,
                [0, 0, 5, 0], None, None]})

    computed = constrain.constrain_blocks(
        block_shape=block_shape,
        df=df,
        constraint_df=constraint_df)
    cov1 = np.identity(8)
    cov1[6, 6] = 0
    cov2 = np.copy(cov1)
    for i in range(4):
      cov2[i, i] = 0
    cov3 = np.copy(cov2)
    cov3[7, 7] = 0
    expected = pd.DataFrame({
        ID: ["0", "1", "2"],
        ESTIMATE: [
            Estimate(val=np.array([1, 2, 3, 4, 5, 6, 0, 8]), cov=cov1),
            Estimate(val=np.array([0, 0, 0, 0, 6, 7, 0, 9]), cov=cov2),
            Estimate(val=np.array([0, 0, 0, 0, 7, 8, 0, 0]), cov=cov3)]})
    self.assertEqual(computed.columns.tolist(), expected.columns.tolist())
    self.assertEqual(computed[ID].tolist(), expected[ID].tolist())
    for x, y in zip(computed[ESTIMATE].tolist(), expected[ESTIMATE].tolist()):
      np.testing.assert_allclose(x.val, y.val)
      np.testing.assert_allclose(x.cov, y.cov)


if __name__ == "__main__":
  absltest.main()
