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

import numpy as np
from census_bluedown import estimate
from census_bluedown import query_encoding
from census_bluedown import regression
from census_bluedown import succinct
from absl.testing import absltest
from absl.testing import parameterized


class RegressionTest(parameterized.TestCase):

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

  def _materialize_covariance(self, cov: object) -> np.ndarray:
    assert isinstance(cov, succinct.SuccinctCovariance)
    return np.kron(cov.coefficient_0, self.P_0) + np.kron(
        cov.coefficient_1, self.P_1
    )

  def _materialize_input_covariance(
      self, cov: succinct.SuccinctInputCovariance
  ) -> np.ndarray:
    m1 = len(cov.diagonal_1)
    m2 = len(cov.diagonal_2)
    part1 = np.kron(np.diag(cov.diagonal_1), self.I_d)
    part2 = np.diag(cov.diagonal_2)
    res = np.zeros((m1 * self.d + m2, m1 * self.d + m2))
    res[: m1 * self.d, : m1 * self.d] = part1
    res[m1 * self.d :, m1 * self.d :] = part2
    return res

  def _materialize_workload(
      self, workload: succinct.SuccinctWorkload
  ) -> np.ndarray:
    part1 = np.kron(workload.w1, self.I_d)
    part2 = np.kron(workload.w2, np.ones((1, self.d)))
    return np.vstack([part1, part2])

  def test_compute_gls_estimate_vs_dense(self):
    query_shapes = [
        (8, 2, 2, 63),  # detailed (individual)
        (1, 1, 1, 63),  # CENRACE-like (individual)
        (8, 1, 1, 1),  # hhgq-like (aggregate)
        (3, 1, 1, 1),  # HHINSTLEVELS (aggregate)
        (1, 1, 1, 1),  # total (aggregate)
    ]
    # Use small positive variances to ensure invertibility
    query_variances = [0.5, 1.2, 0.8, 1.5, 2.0]
    query_values = [
        np.random.randn(int(np.prod(shape))) for shape in query_shapes
    ]

    # Run succinct compute_gls_estimate
    workload, cov, y = query_encoding.prepare_succinct_inputs(
        query_shapes, query_variances, query_values
    )
    x_succinct, cov_succinct = regression.compute_gls_estimate(workload, cov, y)
    W = self._materialize_workload(workload)
    Sigma = self._materialize_input_covariance(cov)
    Sigma_inv = np.linalg.inv(Sigma)

    # GLS formula: cov = (W^T Sigma^-1 W)^-1
    # val = cov @ W^T Sigma^-1 y
    dense_cov_inv = W.T @ Sigma_inv @ W
    dense_cov = np.linalg.pinv(dense_cov_inv, hermitian=True)
    dense_val = dense_cov @ W.T @ Sigma_inv @ y

    # Verify covariance
    np.testing.assert_allclose(
        self._materialize_covariance(cov_succinct),
        dense_cov,
        rtol=0,
        atol=1e-8,
    )

    # Verify value
    np.testing.assert_allclose(x_succinct, dense_val, rtol=0, atol=1e-8)

  def test_combine_estimates_vs_dense(self):
    # Generate two random succinct estimates
    a0_1 = np.random.randn(self.m, self.m)
    c0_1 = a0_1 @ a0_1.T + np.eye(self.m)
    a1_1 = np.random.randn(self.m, self.m)
    c1_1 = a1_1 @ a1_1.T + np.eye(self.m)
    cov_1 = succinct.SuccinctCovariance(c0_1, c1_1)
    val_1 = np.random.randn(self.n)
    est_1 = estimate.Estimate(val_1, cov_1)

    a0_2 = np.random.randn(self.m, self.m)
    c0_2 = a0_2 @ a0_2.T + np.eye(self.m)
    a1_2 = np.random.randn(self.m, self.m)
    c1_2 = a1_2 @ a1_2.T + np.eye(self.m)
    cov_2 = succinct.SuccinctCovariance(c0_2, c1_2)
    val_2 = np.random.randn(self.n)
    est_2 = estimate.Estimate(val_2, cov_2)

    # Run succinct combine_estimates
    est_combined = regression.combine_estimates(est_1, est_2)

    # Materialize and run dense combine
    C1 = self._materialize_covariance(cov_1)
    C2 = self._materialize_covariance(cov_2)
    I = np.eye(self.n)
    A = C2 @ np.linalg.pinv(C1 + C2, hermitian=True)
    B = I - A
    dense_val = A @ val_1 + B @ val_2
    dense_cov = B @ C2
    # ensure symmetry
    dense_cov = 0.5 * (dense_cov + dense_cov.T)

    # Verify value
    np.testing.assert_allclose(est_combined.val, dense_val, rtol=0, atol=1e-8)

    # Verify covariance
    np.testing.assert_allclose(
        self._materialize_covariance(est_combined.cov),
        dense_cov,
        rtol=0,
        atol=1e-8,
    )

  def test_combine(self):
    # Test combine with np.nan
    a0 = np.random.randn(self.m, self.m)
    c0 = a0 @ a0.T + np.eye(self.m)
    a1 = np.random.randn(self.m, self.m)
    c1 = a1 @ a1.T + np.eye(self.m)
    cov = succinct.SuccinctCovariance(c0, c1)
    val = np.random.randn(self.n)
    est = estimate.Estimate(val, cov)

    # Both estimates are valid
    computed = regression.combine(est, est)
    # should be same as combine_estimates(est, est)
    expected = regression.combine_estimates(est, est)
    np.testing.assert_allclose(computed.val, expected.val)
    np.testing.assert_allclose(
        self._materialize_covariance(computed.cov),
        self._materialize_covariance(expected.cov),
    )

    # First is nan
    computed = regression.combine(np.nan, est)
    np.testing.assert_allclose(computed.val, est.val)
    np.testing.assert_allclose(
        self._materialize_covariance(computed.cov),
        self._materialize_covariance(est.cov),
    )

    # Second is nan
    computed = regression.combine(est, np.nan)
    np.testing.assert_allclose(computed.val, est.val)
    np.testing.assert_allclose(
        self._materialize_covariance(computed.cov),
        self._materialize_covariance(est.cov),
    )

    # Both nan
    with self.assertRaises(ValueError):
      regression.combine(np.nan, np.nan)


if __name__ == "__main__":
  absltest.main()
