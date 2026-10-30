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


class SmallDenseConstraint(estimate.DenseConstraint):
  """DenseConstraint over a small (2, 2, 3) feature layout for testing."""

  feature_shapes = (2, 2, 3)
  dim = 12
  first_feature_stride = 6


class DenseConstraintTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self.n = SmallDenseConstraint.dim
    a = np.random.randn(self.n, self.n)
    self.cov = a @ a.T + np.eye(self.n)
    self.val = np.random.randn(self.n)
    self.est = estimate.Estimate(self.val, self.cov)

  def test_empty(self):
    cons = SmallDenseConstraint.empty()
    self.assertTrue(cons.is_empty)
    self.assertEqual(cons.R.shape, (0, self.n))
    self.assertIs(cons.apply(self.est), self.est)

  def test_sum(self):
    cons = SmallDenseConstraint.sum(7.0)
    self.assertFalse(cons.is_empty)
    np.testing.assert_allclose(cons.R, np.ones((1, self.n)))
    np.testing.assert_allclose(cons.c, [7.0])

  @parameterized.named_parameters(
      ("first_feature_only", [True, False], None, [0, 1, 2, 3, 4, 5]),
      ("multifeature_only", [False, False], (1, 0), [6, 7, 8]),
      ("disjoint", [True, False], (1, 1), [0, 1, 2, 3, 4, 5, 9, 10, 11]),
      ("overlapping", [True, False], (0, 1), [0, 1, 2, 3, 4, 5]),
  )
  def test_zero_constraints(self, first_feature, multifeature, expected):
    cons = SmallDenseConstraint.zero_constraints(first_feature, multifeature)
    np.testing.assert_allclose(cons.R, np.eye(self.n)[expected])
    np.testing.assert_allclose(cons.c, np.zeros(len(expected)))

  def test_zero_constraints_none_is_empty(self):
    self.assertTrue(
        SmallDenseConstraint.zero_constraints([False, False]).is_empty
    )

  def test_apply_satisfies_constraint(self):
    cons = SmallDenseConstraint.zero_constraints([True, False], (1, 0))
    new_est = cons.apply(self.est)
    np.testing.assert_allclose(cons.R @ new_est.val, cons.c, atol=1e-8)
    np.testing.assert_allclose(
        cons.R @ new_est.cov, np.zeros_like(cons.R), atol=1e-8
    )

  def test_apply_matches_rls_formula(self):
    cons = SmallDenseConstraint.sum(7.0)
    new_est = cons.apply(self.est)
    R, c = cons.R, cons.c  # pylint: disable=invalid-name
    L = self.cov @ R.T @ np.linalg.pinv(R @ self.cov @ R.T)  # pylint: disable=invalid-name
    np.testing.assert_allclose(
        new_est.val, self.val - L @ (R @ self.val - c), atol=1e-8
    )
    np.testing.assert_allclose(
        new_est.cov, self.cov - L @ R @ self.cov, atol=1e-8
    )

  def test_apply_dense_covariance(self):
    est = estimate.Estimate(self.val, estimate.DenseCovariance(self.cov))
    cons = SmallDenseConstraint.sum(7.0)
    new_est = cons.apply(est)
    self.assertIsInstance(new_est.cov, estimate.DenseCovariance)
    np.testing.assert_allclose(
        new_est.cov.matrix, cons.apply(self.est).cov, atol=1e-8
    )

  def test_apply_is_idempotent(self):
    cons = SmallDenseConstraint.zero_constraints([True, False], (1, 0))
    once = cons.apply(self.est)
    twice = cons.apply(once)
    np.testing.assert_allclose(twice.val, once.val, atol=1e-8)
    np.testing.assert_allclose(twice.cov, once.cov, atol=1e-8)

  def test_apply_is_order_invariant(self):
    # Overlapping constraints: both fix entries 3, 4, 5 to zero; the sum
    # constraint overlaps with every entry.
    cons_1 = SmallDenseConstraint.zero_constraints([True, False])
    cons_2 = SmallDenseConstraint.zero_constraints([False, False], (0, 1))
    cons_3 = SmallDenseConstraint.sum(7.0)
    stacked = SmallDenseConstraint(
        np.vstack([cons_1.R, cons_2.R, cons_3.R]),
        np.concatenate([cons_1.c, cons_2.c, cons_3.c]),
    )
    expected = stacked.apply(self.est)

    for order in ((cons_1, cons_2, cons_3), (cons_3, cons_2, cons_1)):
      actual = self.est
      for cons in order:
        actual = cons.apply(actual)
      np.testing.assert_allclose(actual.val, expected.val, atol=1e-8)
      np.testing.assert_allclose(actual.cov, expected.cov, atol=1e-8)

  def test_estimate_constrain_delegates_to_apply(self):
    cons = SmallDenseConstraint.sum(7.0)
    expected = cons.apply(self.est)
    actual = self.est.constrain(cons)
    np.testing.assert_allclose(actual.val, expected.val)
    np.testing.assert_allclose(actual.cov, expected.cov)


if __name__ == "__main__":
  absltest.main()
