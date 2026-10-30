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
from census_bluedown import succinct
from absl.testing import absltest
from absl.testing import parameterized


class SuccinctTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self.m = succinct.SuccinctCovariance.coefficient_dim
    self.d = succinct.SuccinctCovariance.projection_dim
    self.n = self.m * self.d

    # Helpful matrices for materialization
    self.I_m = np.eye(self.m)
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

  def _materialize_input_covariance(
      self, cov: succinct.SuccinctInputCovariance
  ) -> np.ndarray:
    m1 = len(cov.diagonal_1)
    m2 = len(cov.diagonal_2)
    part1 = np.kron(np.diag(cov.diagonal_1), self.I_d)
    part2 = np.diag(cov.diagonal_2)
    # block diagonal
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

  def _materialize_constraints(
      self, cons: succinct.SuccinctConstraint
  ) -> np.ndarray:
    part1 = np.kron(cons.r1, self.I_d)
    part2 = np.kron(cons.r2, np.ones((1, self.d)))
    return np.vstack([part1, part2])

  def test_covariance_identity(self):
    cov = succinct.SuccinctCovariance.identity()
    mat = self._materialize_covariance(cov)
    np.testing.assert_allclose(mat, np.eye(self.n))

  def test_covariance_add(self):
    c0_1 = np.random.randn(self.m, self.m)
    c1_1 = np.random.randn(self.m, self.m)
    c0_2 = np.random.randn(self.m, self.m)
    c1_2 = np.random.randn(self.m, self.m)

    cov_1 = succinct.SuccinctCovariance(c0_1, c1_1)
    cov_2 = succinct.SuccinctCovariance(c0_2, c1_2)

    cov_sum = cov_1 + cov_2
    mat_1 = self._materialize_covariance(cov_1)
    mat_2 = self._materialize_covariance(cov_2)

    np.testing.assert_allclose(
        self._materialize_covariance(cov_sum), mat_1 + mat_2
    )

  def test_covariance_sub(self):
    c0_1 = np.random.randn(self.m, self.m)
    c1_1 = np.random.randn(self.m, self.m)
    c0_2 = np.random.randn(self.m, self.m)
    c1_2 = np.random.randn(self.m, self.m)

    cov_1 = succinct.SuccinctCovariance(c0_1, c1_1)
    cov_2 = succinct.SuccinctCovariance(c0_2, c1_2)

    cov_diff = cov_1 - cov_2
    mat_1 = self._materialize_covariance(cov_1)
    mat_2 = self._materialize_covariance(cov_2)

    np.testing.assert_allclose(
        self._materialize_covariance(cov_diff), mat_1 - mat_2
    )

  def test_covariance_neg(self):
    c0 = np.random.randn(self.m, self.m)
    c1 = np.random.randn(self.m, self.m)

    cov = succinct.SuccinctCovariance(c0, c1)
    cov_neg = -cov
    mat = self._materialize_covariance(cov)

    np.testing.assert_allclose(self._materialize_covariance(cov_neg), -mat)

  def test_covariance_matmul_succinct(self):
    c0_1 = np.random.randn(self.m, self.m)
    c1_1 = np.random.randn(self.m, self.m)
    c0_2 = np.random.randn(self.m, self.m)
    c1_2 = np.random.randn(self.m, self.m)

    cov_1 = succinct.SuccinctCovariance(c0_1, c1_1)
    cov_2 = succinct.SuccinctCovariance(c0_2, c1_2)

    cov_prod = cov_1 @ cov_2
    mat_1 = self._materialize_covariance(cov_1)
    mat_2 = self._materialize_covariance(cov_2)

    np.testing.assert_allclose(
        self._materialize_covariance(cov_prod), mat_1 @ mat_2
    )

  def test_covariance_matmul_vector(self):
    c0 = np.random.randn(self.m, self.m)
    c1 = np.random.randn(self.m, self.m)
    cov = succinct.SuccinctCovariance(c0, c1)
    mat = self._materialize_covariance(cov)

    x = np.random.randn(self.n)
    res_succinct = cov @ x
    res_dense = mat @ x

    np.testing.assert_allclose(res_succinct, res_dense)

  def test_covariance_pinv(self):
    # Use symmetric positive definite matrices to ensure invertibility for test
    a0 = np.random.randn(self.m, self.m)
    c0 = a0 @ a0.T + np.eye(self.m)
    a1 = np.random.randn(self.m, self.m)
    c1 = a1 @ a1.T + np.eye(self.m)

    cov = succinct.SuccinctCovariance(c0, c1)
    cov_inv = cov.pinv()
    mat = self._materialize_covariance(cov)
    mat_inv = np.linalg.pinv(mat, hermitian=True)

    np.testing.assert_allclose(
        self._materialize_covariance(cov_inv), mat_inv, rtol=0, atol=1e-8
    )

  def test_input_covariance_inv(self):
    m1, m2 = 10, 5
    diag1 = np.random.rand(m1) + 0.1  # ensure non-zero
    diag2 = np.random.rand(m2) + 0.1
    cov = succinct.SuccinctInputCovariance(diag1, diag2)
    cov_inv = cov.inv()

    mat = self._materialize_input_covariance(cov)
    mat_inv = np.linalg.inv(mat)

    np.testing.assert_allclose(
        self._materialize_input_covariance(cov_inv), mat_inv
    )

  def test_input_covariance_multiply(self):
    m1, m2 = 10, 5
    diag1 = np.random.rand(m1)
    diag2 = np.random.rand(m2)
    cov = succinct.SuccinctInputCovariance(diag1, diag2)
    mat = self._materialize_input_covariance(cov)

    y = np.random.randn(m1 * self.d + m2)
    res_succinct = cov.multiply(y)
    res_dense = mat @ y

    np.testing.assert_allclose(res_succinct, res_dense)

  def test_workload_multiply(self):
    m1, m2 = 10, 5
    w1 = np.random.randn(m1, self.m)
    w2 = np.random.randn(m2, self.m)
    workload = succinct.SuccinctWorkload(w1, w2)
    mat = self._materialize_workload(workload)

    x = np.random.randn(self.n)
    res_succinct = workload.multiply(x)
    res_dense = mat @ x

    np.testing.assert_allclose(res_succinct, res_dense)

  def test_workload_multiply_transpose(self):
    m1, m2 = 10, 5
    w1 = np.random.randn(m1, self.m)
    w2 = np.random.randn(m2, self.m)
    workload = succinct.SuccinctWorkload(w1, w2)
    mat = self._materialize_workload(workload)

    y = np.random.randn(m1 * self.d + m2)
    res_succinct = workload.multiply_transpose(y)
    res_dense = mat.T @ y

    np.testing.assert_allclose(res_succinct, res_dense)

  def test_constraints_multiply(self):
    k1, k2 = 3, 2
    r1 = np.random.randn(k1, self.m)
    r2 = np.random.randn(k2, self.m)
    # c1 and c2 are not used in multiply, but needed for constructor
    c1 = np.random.randn(k1 * self.d)
    c2 = np.random.randn(k2)
    cons = succinct.SuccinctConstraint(r1, r2, c1, c2)
    mat = self._materialize_constraints(cons)

    x = np.random.randn(self.n)
    res_succinct = cons.multiply(x)
    res_dense = mat @ x

    np.testing.assert_allclose(res_succinct, res_dense)

  def test_covariance_scalar_mul(self):
    c0 = np.random.randn(self.m, self.m)
    c1 = np.random.randn(self.m, self.m)
    cov = succinct.SuccinctCovariance(c0, c1)
    mat = self._materialize_covariance(cov)

    np.testing.assert_allclose(
        self._materialize_covariance(cov * 2.5), mat * 2.5
    )
    np.testing.assert_allclose(
        self._materialize_covariance(2.5 * cov), mat * 2.5
    )

  def test_covariance_inv_and_cholesky(self):
    a0 = np.random.randn(self.m, self.m)
    c0 = a0 @ a0.T + np.eye(self.m)
    a1 = np.random.randn(self.m, self.m)
    c1 = a1 @ a1.T + np.eye(self.m)
    cov = succinct.SuccinctCovariance(c0, c1)

    cov_inv = cov.inv()
    mat = self._materialize_covariance(cov)
    np.testing.assert_allclose(
        self._materialize_covariance(cov_inv),
        np.linalg.inv(mat),
        rtol=0,
        atol=1e-8,
    )

    l = cov.cholesky()
    self.assertIsInstance(l, succinct.SuccinctCovariance)
    np.testing.assert_allclose(l.coefficient_0 @ l.coefficient_0.T, c0)
    np.testing.assert_allclose(l.coefficient_1 @ l.coefficient_1.T, c1)
    np.testing.assert_allclose(
        self._materialize_covariance(l @ l.T), mat, rtol=0, atol=1e-8
    )

  def test_covariance_principal_submatrix(self):
    c0 = np.random.randn(self.m, self.m)
    c1 = np.random.randn(self.m, self.m)
    cov = succinct.SuccinctCovariance(c0, c1)
    rows = [1, 4, 5, 20]
    entries = np.concatenate(
        [np.arange(a * self.d, (a + 1) * self.d) for a in rows]
    )

    sub = cov.principal_submatrix(rows)
    dense_sub = self._materialize_covariance(cov)[np.ix_(entries, entries)]
    np.testing.assert_allclose(self._materialize_covariance(sub), dense_sub)

  def test_constraints_gram(self):
    k1, k2 = 3, 2
    r1 = np.random.randn(k1, self.m)
    r2 = np.random.randn(k2, self.m)
    c1 = np.random.randn(k1 * self.d)
    c2 = np.random.randn(k2)
    cons = succinct.SuccinctConstraint(r1, r2, c1, c2)

    gram_cov = cons.gram()
    mat = self._materialize_constraints(cons)
    np.testing.assert_allclose(
        self._materialize_covariance(gram_cov), mat.T @ mat
    )

  def test_covariance_flatten_roundtrip(self):
    c0 = np.random.randn(self.m, self.m)
    c1 = np.random.randn(self.m, self.m)
    cov = succinct.SuccinctCovariance(c0, c1)

    flattened = cov.flatten()
    self.assertEqual(flattened.shape, (2 * self.m * self.m,))

    reconstructed = succinct.SuccinctCovariance.from_flattened(flattened)
    np.testing.assert_allclose(reconstructed.coefficient_0, c0)
    np.testing.assert_allclose(reconstructed.coefficient_1, c1)


if __name__ == "__main__":
  absltest.main()
