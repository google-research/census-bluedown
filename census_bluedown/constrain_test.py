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
import pandas as pd
from census_bluedown import constrain
from census_bluedown import estimate
from census_bluedown import succinct
from absl.testing import absltest
from absl.testing import parameterized

Estimate = estimate.Estimate
ID = constrain.ID
ESTIMATE = constrain.ESTIMATE
VALUE = constrain.VALUE
QUERY_NAME = constrain.QUERY_NAME
FIRST_FEATURE_UB_CONSTRAINT = constrain.FIRST_FEATURE_UB_CONSTRAINT
MULTIFEATURE_ZERO_SEQUENCE = constrain.MULTIFEATURE_ZERO_SEQUENCE


class ConstrainTest(parameterized.TestCase):

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

  def _assert_matches_dense_rls(
      self, est: Estimate, cons: succinct.SuccinctConstraint
  ) -> None:
    """Checks apply_linear_constraint against dense Algorithm 1."""
    computed = constrain.apply_linear_constraint(est, cons)

    C = self._materialize_covariance(est.cov)
    R = self._materialize_constraints(cons)
    c = np.concatenate([cons.c1, cons.c2])
    L = C @ R.T @ np.linalg.pinv(R @ C @ R.T, hermitian=True)
    expected_val = est.val - L @ (R @ est.val - c)
    expected_cov = C - L @ R @ C

    np.testing.assert_allclose(computed.val, expected_val, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        self._materialize_covariance(computed.cov),
        expected_cov,
        rtol=0,
        atol=1e-8,
    )

  def test_apply_linear_constraint_vs_dense(self):
    cov = self._random_succinct_covariance()
    est = Estimate(np.random.randn(self.n), cov)

    # 1. Sum constraint (aggregate only)
    self._assert_matches_dense_rls(est, succinct.SuccinctConstraint.sum(100.0))

    # 2. First feature zero constraints (individual only)
    # Constrain housing types 0 and 2 to zero
    zero_indices = [True, False, True, False, False, False, False, False]
    self._assert_matches_dense_rls(
        est,
        succinct.SuccinctConstraint.first_feature_zero_constraints(
            zero_indices
        ),
    )

    # 3. Multifeature zero constraints (individual only, prefix)
    # HHGQ = 3, VOTINGAGE = 0
    self._assert_matches_dense_rls(
        est, succinct.SuccinctConstraint.multifeature_zero_constraints((3, 0))
    )

  def test_sum_constraint(self):
    cons = succinct.SuccinctConstraint.sum(100.0)
    R = self._materialize_constraints(cons)
    # R should be all-ones row vector of size 2016
    np.testing.assert_allclose(R, np.ones((1, self.n)))
    np.testing.assert_allclose(cons.c2, [100.0])

  def test_first_feature_zero_constraints(self):
    zero_indices = [True, False, True, False, False, False, False, False]
    cons = succinct.SuccinctConstraint.first_feature_zero_constraints(
        zero_indices
    )
    R = self._materialize_constraints(cons)

    # We expect 2 * 4 = 8 constraints in asymmetric space.
    # Stacked R1 is shape 8 x 32.
    # Materialized R is shape 8 * 63 x 2016 = 504 x 2016.
    self.assertEqual(R.shape, (504, 2016))

    # Verify that it selects the correct elements.
    # The elements constrained to zero should be:
    # HHGQ=0 (indices 0 to 251) and HHGQ=2 (indices 504 to 755)
    # Let's check that if we multiply R by a vector, it extracts these elements.
    x = np.arange(self.n)
    res = R @ x

    expected_indices = []
    for i in [0, 2]:
      for j in range(252):
        expected_indices.append(i * 252 + j)
    np.testing.assert_allclose(res, x[expected_indices])

  def test_multifeature_zero_constraints(self):
    # For HHGQ=3, VOTINGAGE=0, HISPANIC can be 0 or 1, corresponding to
    # asymmetric indices 12 and 13 (3 * 4 + 0 * 2 + {0, 1}), each spanning
    # 63 symmetric elements.
    # So we constrain indices [756..818] and [819..881], which is the
    # contiguous range [756..881].

    cons = succinct.SuccinctConstraint.multifeature_zero_constraints((3, 0))
    R = self._materialize_constraints(cons)

    self.assertEqual(R.shape, (126, 2016))

    x = np.arange(self.n)
    res = R @ x
    expected_indices = list(range(756, 882))
    np.testing.assert_allclose(res, x[expected_indices])

  def test_constrain_blocks(self):
    cov = self._random_succinct_covariance()
    val_1 = np.random.randn(self.n)
    val_2 = np.random.randn(self.n)
    df = pd.DataFrame({
        ID: ["0100", "0200"],
        ESTIMATE: [Estimate(val_1, cov), Estimate(val_2, cov)],
    })

    # Constraint df:
    # 0100: first feature UB is [1, 2, 3, 4, 5, 6, 7, 8] (no zeros)
    # 0200: first feature UB is [0, 2, 3, 4, 5, 6, 7, 8] (HHGQ=0 is zero)
    # Both: multifeature zero constraint (HHGQ=3, VOTINGAGE=0)
    constraint_df = pd.DataFrame({
        ID: ["0100", "0200"],
        QUERY_NAME: [FIRST_FEATURE_UB_CONSTRAINT, FIRST_FEATURE_UB_CONSTRAINT],
        VALUE: [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 2, 3, 4, 5, 6, 7, 8],
        ],
    })

    computed = constrain.constrain_blocks(df, constraint_df)

    # Verify 0100: only multifeature constraint applied
    est_0100_expected = constrain.apply_multifeature_zero_constraints(
        Estimate(val_1, cov), MULTIFEATURE_ZERO_SEQUENCE
    )
    np.testing.assert_allclose(
        computed[computed[ID] == "0100"][ESTIMATE].values[0].val,
        est_0100_expected.val,
        rtol=0,
        atol=1e-8,
    )

    # Verify 0200: both first feature and multifeature constraints applied
    # Since HHGQ=0 is constrained to zero, and multifeature is HHGQ=3,
    # VOTINGAGE=0. They don't overlap, so both should be applied.
    est_0200_expected = constrain.apply_first_feature_zero_constraints(
        Estimate(val_2, cov), [True] + [False] * 7
    )
    est_0200_expected = constrain.apply_multifeature_zero_constraints(
        est_0200_expected, MULTIFEATURE_ZERO_SEQUENCE
    )
    np.testing.assert_allclose(
        computed[computed[ID] == "0200"][ESTIMATE].values[0].val,
        est_0200_expected.val,
        rtol=0,
        atol=1e-8,
    )


if __name__ == "__main__":
  absltest.main()
