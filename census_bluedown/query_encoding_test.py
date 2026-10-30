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
from census_bluedown import query_encoding
from census_bluedown import succinct
from absl.testing import absltest
from absl.testing import parameterized


class QueryEncodingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)
    self.m = succinct.SuccinctCovariance.coefficient_dim  # 32
    self.d = succinct.SuccinctCovariance.projection_dim  # 63
    self.n = self.m * self.d  # 2016

  @parameterized.named_parameters(
      dict(testcase_name="total", query_shape=(1, 1, 1, 1)),
      dict(testcase_name="voting_age", query_shape=(1, 2, 1, 1)),
      dict(testcase_name="cenrace", query_shape=(1, 1, 1, 63)),
      dict(testcase_name="hispanic_x_cenrace", query_shape=(1, 1, 2, 63)),
      dict(testcase_name="detailed", query_shape=(8, 2, 2, 63)),
      dict(testcase_name="housing_type", query_shape=(8, 1, 1, 1)),
      dict(testcase_name="hhinstlevels", query_shape=(3, 1, 1, 1)),
      dict(testcase_name="hhinstlevels_x_cenrace", query_shape=(3, 1, 1, 63)),
  )
  def test_query_projections_agree(self, query_shape):
    values = np.random.randint(0, 100, size=(4, self.n))
    projected = query_encoding.project_to_query_shape(values, query_shape)
    self.assertEqual(projected.shape, (4, np.prod(query_shape)))

    # Matches the design matrix, which is W_asym (x) I_63 or W_asym (x) 1^T.
    w_asym = query_encoding.asymmetric_design_matrix(query_shape)
    w_sym = np.eye(self.d) if query_shape[3] == self.d else np.ones((1, self.d))
    np.testing.assert_array_equal(
        projected, values @ np.kron(w_asym, w_sym).T.astype(int)
    )

    # Matches the query cell members.
    members = query_encoding.query_cell_members(query_shape)
    np.testing.assert_array_equal(
        projected, np.stack([values[:, m].sum(axis=1) for m in members], axis=1)
    )

    # A single histogram is projected the same way as a batch.
    np.testing.assert_array_equal(
        query_encoding.project_to_query_shape(values[0], query_shape),
        projected[0],
    )

  def test_project_to_query_shape_hhinstlevels(self):
    values = np.arange(self.n).reshape(8, -1)
    projected = query_encoding.project_to_query_shape(
        values.reshape(-1), (3, 1, 1, 1)
    )
    row_sums = values.sum(axis=1)
    np.testing.assert_array_equal(
        projected, [row_sums[0], row_sums[1:5].sum(), row_sums[5:].sum()]
    )

  @parameterized.named_parameters(
      dict(testcase_name="wrong_length", query_shape=(8, 2, 2)),
      dict(testcase_name="partial_slice", query_shape=(1, 1, 1, 7)),
      dict(testcase_name="grouping_not_first", query_shape=(1, 1, 1, 3)),
  )
  def test_unsupported_query_shape_raises(self, query_shape):
    with self.assertRaises(ValueError):
      query_encoding.feature_groups(query_shape)

  def test_asymmetric_design_matrix(self):
    # Test detailed shape
    w = query_encoding.asymmetric_design_matrix((8, 2, 2, 63))
    np.testing.assert_allclose(w, np.eye(32))

    # Test total shape
    w = query_encoding.asymmetric_design_matrix((1, 1, 1, 1))
    np.testing.assert_allclose(w, np.ones((1, 32)))

    # Test HHINSTLEVELS shape
    w = query_encoding.asymmetric_design_matrix((3, 1, 1, 1))
    self.assertEqual(w.shape, (3, 32))
    # verify row sums
    np.testing.assert_allclose(np.sum(w, axis=1), [4.0, 16.0, 12.0])

  def test_prepare_succinct_inputs(self):
    query_shapes = [
        (8, 2, 2, 63),  # detailed (individual)
        (1, 1, 1, 63),  # CENRACE-like (individual)
        (8, 1, 1, 1),  # hhgq-like (aggregate)
        (1, 1, 1, 1),  # total (aggregate)
    ]
    query_variances = [0.5, 1.2, 0.8, 2.0]
    query_values = [
        np.random.randn(int(np.prod(shape))) for shape in query_shapes
    ]

    workload, cov, y = query_encoding.prepare_succinct_inputs(
        query_shapes, query_variances, query_values
    )

    # Verify workload shape
    # w1 should have shape (32 + 1, 32) = (33, 32)
    # w2 should have shape (8 + 1, 32) = (9, 32)
    self.assertEqual(workload.w1.shape, (33, 32))
    self.assertEqual(workload.w2.shape, (9, 32))

    # Verify input covariance shape
    # diagonal_1 should have size 33
    # diagonal_2 should have size 9
    self.assertEqual(cov.diagonal_1.shape, (33,))
    self.assertEqual(cov.diagonal_2.shape, (9,))

    # Verify y shape
    # y1 should have size 33 * 63 = 2079
    # y2 should have size 9
    # total size: 2088
    self.assertEqual(y.shape, (2088,))

    # Verify contents of y
    y_expected = np.concatenate([
        query_values[0].flatten(),
        query_values[1].flatten(),
        query_values[2].flatten(),
        query_values[3].flatten(),
    ])
    np.testing.assert_allclose(y, y_expected)

    # Verify variances are correctly repeated
    expected_diag1 = np.concatenate([
        np.full(32, 0.5),
        np.full(1, 1.2),
    ])
    expected_diag2 = np.concatenate([
        np.full(8, 0.8),
        np.full(1, 2.0),
    ])
    np.testing.assert_allclose(cov.diagonal_1, expected_diag1)
    np.testing.assert_allclose(cov.diagonal_2, expected_diag2)


if __name__ == "__main__":
  absltest.main()
