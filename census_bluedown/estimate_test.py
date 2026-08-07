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

import numpy as np
from census_bluedown import estimate
from absl.testing import absltest
from absl.testing import parameterized


class EstimateTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("add_zero", [1], [1], [0], [1], [1], [2]),
      ("add_one", [5], [5], [1], [2], [6], [7]),
      ("add_vector", [1, 2], [[1, 0], [0, 1]], [0, 5], [[2, 1], [1, 2]],
       [1, 7], [[3, 1], [1, 3]]))
  def test_add(self, val_1, cov_1, val_2, cov_2, expected_val, expected_cov):
    est_1 = estimate.Estimate(np.array(val_1), np.array(cov_1))
    est_2 = estimate.Estimate(np.array(val_2), np.array(cov_2))
    est_sum = est_1 + est_2
    np.testing.assert_allclose(est_sum.val, np.array(expected_val))
    np.testing.assert_allclose(est_sum.cov, np.array(expected_cov))

  @parameterized.named_parameters(
      ("add_zero", [1], [1], 0, [1]),
      ("add_zero_to_vector", [1, 2], [[2, 1], [1, 2]], 0, [1, 2]),
      ("add_one", [1], [1], 1, [2]),
      ("add_one_to_vector", [1, 2], [[2, 1], [1, 2]], 1, [2, 3]),
      ("add_float_to_vector", [1, 2], [[2, 1], [1, 2]], 1.5, [2.5, 3.5])
      )
  def test_radd(self, val, cov, shift, expected_val):
    est = estimate.Estimate(np.array(val), np.array(cov))
    est_sum = shift + est
    np.testing.assert_allclose(est_sum.val, np.array(expected_val))
    np.testing.assert_allclose(est_sum.cov, np.array(cov))

  @parameterized.named_parameters(
      ("sub_zero", [1], [1], [0], [1], [1], [2]),
      ("sub_one", [5], [5], [1], [2], [4], [7]),
      ("sub_vector", [1, 2], [[1, 0], [0, 1]], [0, 5], [[2, 1], [1, 2]],
       [1, -3], [[3, 1], [1, 3]]))
  def test_sub(self, val_1, cov_1, val_2, cov_2, expected_val, expected_cov):
    est_1 = estimate.Estimate(np.array(val_1), np.array(cov_1))
    est_2 = estimate.Estimate(np.array(val_2), np.array(cov_2))
    est_diff = est_1 - est_2
    np.testing.assert_allclose(est_diff.val, np.array(expected_val))
    np.testing.assert_allclose(est_diff.cov, np.array(expected_cov))

  @parameterized.named_parameters(
      ("sum_scalars", [0], [1], [1], [1], [2], [4]),
      ("sum_vectors", [1, 1], [[1, 0], [0, 1]], [1, 2], [[2, 1], [1, 2]],
       [1, 3], [[3, 1], [1, 3]]))
  def test_two_out_of_three_summation(self, val_1, cov_1, val_2, cov_2,
                                      val_3, cov_3):
    est_1 = estimate.Estimate(np.array(val_1), np.array(cov_1))
    est_2 = estimate.Estimate(np.array(val_2), np.array(cov_2))
    est_3 = estimate.Estimate(np.array(val_3), np.array(cov_3))
    est_sum123 = est_1 + est_2 + est_3
    est_sum12 = est_1 + est_2
    est_sum13 = est_1 + est_3
    est_sum23 = est_2 + est_3
    est_sum123_minus_1 = est_sum123 % est_1
    est_sum123_minus_2 = est_sum123 % est_2
    est_sum123_minus_3 = est_sum123 % est_3

    np.testing.assert_allclose(est_sum12.val, est_sum123_minus_3.val)
    np.testing.assert_allclose(est_sum12.cov, est_sum123_minus_3.cov)
    np.testing.assert_allclose(est_sum13.val, est_sum123_minus_2.val)
    np.testing.assert_allclose(est_sum13.cov, est_sum123_minus_2.cov)
    np.testing.assert_allclose(est_sum23.val, est_sum123_minus_1.val)
    np.testing.assert_allclose(est_sum23.cov, est_sum123_minus_1.cov)


if __name__ == "__main__":
  absltest.main()
