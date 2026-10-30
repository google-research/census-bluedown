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
from census_bluedown import estimate
from census_bluedown import linear_postprocessing
from census_bluedown import query_encoding
from census_bluedown import regression
from census_bluedown import succinct
from absl.testing import absltest
from absl.testing import parameterized

Estimate = estimate.Estimate
ID = linear_postprocessing.ID
ESTIMATE = linear_postprocessing.ESTIMATE
QUERY_SHAPE = linear_postprocessing.QUERY_SHAPE
VARIANCE = linear_postprocessing.VARIANCE
VALUE = linear_postprocessing.VALUE


class ParameterizedEstimateDFTestCase(parameterized.TestCase):
  """Parameterized test cases involving data frames of Estimate objects."""

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

  def assert_pandas_frames_of_estimates_equal(
      self,
      computed: pd.DataFrame,
      expected: pd.DataFrame,
      estimate_columns: tuple[str, ...] = (ESTIMATE,),
  ):
    """Asserts that two Pandas DataFrames of Estimate objects are equal.

    The data frames must consist of an ID column and one or more columns of
    Estimate objects with column names specified by estimate_columns. The test
    checks that the two data frames have the same columns and that each value of
    each column of Estimate objects has approximately the same value and
    covariance matrix.

    Arguments:
      computed: The computed data frame.
      expected: The expected data frame.
      estimate_columns: The column names of the columns of Estimate objects.
    """
    self.assertEqual(computed.columns.tolist(), expected.columns.tolist())
    self.assertEqual(computed.columns.tolist(), [ID] + list(estimate_columns))
    self.assertEqual(computed[ID].tolist(), expected[ID].tolist())
    merged = pd.merge(
        computed,
        expected,
        on=ID,
        how="outer",
        suffixes=("_computed", "_expected"),
    )
    for row in merged.itertuples():
      for col in estimate_columns:
        computed_est = getattr(row, col + "_computed")
        expected_est = getattr(row, col + "_expected")
        np.testing.assert_allclose(
            computed_est.val, expected_est.val, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            self._materialize_covariance(computed_est.cov),
            self._materialize_covariance(expected_est.cov),
            rtol=0,
            atol=1e-8,
        )


class LinearPostprocessingTest(ParameterizedEstimateDFTestCase):

  def _random_succinct_covariance(self) -> succinct.SuccinctCovariance:
    a0 = np.random.randn(self.m, self.m)
    c0 = a0 @ a0.T + np.eye(self.m)
    a1 = np.random.randn(self.m, self.m)
    c1 = a1 @ a1.T + np.eye(self.m)
    return succinct.SuccinctCovariance(c0, c1)

  def _random_estimate(self) -> Estimate:
    return Estimate(np.random.randn(self.n), self._random_succinct_covariance())

  def test_process_blocks(self):
    # Two blocks, each has a detailed and a total query
    df = pd.DataFrame({
        ID: ["0100", "0100", "0200", "0200"],
        QUERY_SHAPE: [
            (8, 2, 2, 63),
            (1, 1, 1, 1),
            (8, 2, 2, 63),
            (1, 1, 1, 1),
        ],
        VARIANCE: [1.0, 2.0, 1.5, 3.0],
        VALUE: [
            np.random.randn(2016),
            np.random.randn(1),
            np.random.randn(2016),
            np.random.randn(1),
        ],
    })

    computed = linear_postprocessing.process_blocks(df)

    # Compute expected manually for each block
    df_0100 = df[df[ID] == "0100"]
    workload_0100, cov_0100, y_0100 = query_encoding.prepare_succinct_inputs(
        tuple(df_0100[QUERY_SHAPE]),
        tuple(df_0100[VARIANCE]),
        tuple(df_0100[VALUE]),
    )
    val_0100, cov_0100 = regression.compute_gls_estimate(
        workload_0100, cov_0100, y_0100
    )
    est_0100 = Estimate(val_0100, cov_0100)

    df_0200 = df[df[ID] == "0200"]
    workload_0200, cov_0200, y_0200 = query_encoding.prepare_succinct_inputs(
        tuple(df_0200[QUERY_SHAPE]),
        tuple(df_0200[VARIANCE]),
        tuple(df_0200[VALUE]),
    )
    val_0200, cov_0200 = regression.compute_gls_estimate(
        workload_0200, cov_0200, y_0200
    )
    est_0200 = Estimate(val_0200, cov_0200)

    expected = pd.DataFrame({
        ID: ["0100", "0200"],
        ESTIMATE: [est_0100, est_0200],
    })

    self.assert_pandas_frames_of_estimates_equal(computed, expected)

  def test_aggregate_child_estimates(self):
    est_0100 = self._random_estimate()
    est_0101 = self._random_estimate()
    est_0200 = self._random_estimate()
    children_df = pd.DataFrame({
        ID: ["0100", "0101", "0200"],
        ESTIMATE: [est_0100, est_0101, est_0200],
    })

    est_p01 = self._random_estimate()
    est_p02 = self._random_estimate()
    parents_df = pd.DataFrame({
        ID: ["01", "02"],
        ESTIMATE: [est_p01, est_p02],
    })

    parent_suffix = "_parent"
    child_suffix = "_child"

    computed = linear_postprocessing._aggregate_child_estimates(
        children_df=children_df,
        parents_df=parents_df,
        parent_suffix=parent_suffix,
        child_suffix=child_suffix,
    )

    expected_child_01 = est_0100 + est_0101
    expected_child_02 = est_0200

    expected = pd.DataFrame({
        ID: ["01", "02"],
        ESTIMATE + parent_suffix: [est_p01, est_p02],
        ESTIMATE + child_suffix: [expected_child_01, expected_child_02],
    })

    self.assert_pandas_frames_of_estimates_equal(
        computed=computed,
        expected=expected,
        estimate_columns=(ESTIMATE + parent_suffix, ESTIMATE + child_suffix),
    )

  def test_bottom_up_step(self):
    est_0100 = self._random_estimate()
    est_0101 = self._random_estimate()
    children_lower_df = pd.DataFrame({
        ID: ["0100", "0101"],
        ESTIMATE: [est_0100, est_0101],
    })

    est_p01 = self._random_estimate()
    parents_input_df = pd.DataFrame({
        ID: ["01"],
        ESTIMATE: [est_p01],
    })

    computed = linear_postprocessing.bottom_up_step(
        children_lower_df=children_lower_df,
        parents_input_df=parents_input_df,
    )

    child_sum = est_0100 + est_0101
    expected_combined = regression.combine(child_sum, est_p01)
    expected = pd.DataFrame({
        ID: ["01"],
        ESTIMATE: [expected_combined],
    })

    self.assert_pandas_frames_of_estimates_equal(computed, expected)

  def test_compute_upper_exclusive_estimate(self):
    parent_estimate_column = "parent_estimate"
    child_lower_sum_estimate_column = "child_lower_sum_estimate"
    lower_estimate_column = "lower_estimate"
    upper_exclusive_estimate_column = "upper_exclusive_estimate"

    est_0100 = self._random_estimate()
    est_0101 = self._random_estimate()
    est_0200 = self._random_estimate()
    children_df = pd.DataFrame({
        ID: ["0100", "0101", "0200"],
        ESTIMATE: [est_0100, est_0101, est_0200],
    })

    est_p01 = self._random_estimate()
    est_p02 = self._random_estimate()
    parents_df = pd.DataFrame({
        ID: ["01", "02"],
        parent_estimate_column: [est_p01, est_p02],
        child_lower_sum_estimate_column: [
            est_0100 + est_0101,
            est_0200,
        ],
    })

    computed = linear_postprocessing._compute_upper_exclusive_estimate(
        children_df=children_df,
        parents_df=parents_df,
        parent_estimate_column=parent_estimate_column,
        child_lower_sum_estimate_column=child_lower_sum_estimate_column,
        lower_estimate_column=lower_estimate_column,
        upper_exclusive_estimate_column=upper_exclusive_estimate_column,
    )

    # Expected:
    # 0100: parent_01 - (child_sum_01 % child_0100) = parent_01 - child_0101
    # 0101: parent_01 - (child_sum_01 % child_0101) = parent_01 - child_0100
    # 0200: parent_02 - (child_sum_02 % child_0200) = parent_02 - 0
    # Note: child_sum % child = child_sum - child (values & cov subtraction)
    # Actually, in estimate.py, __mod__ is:
    # val = self.val - other.val
    # cov = self.cov - other.cov
    # So:
    # 0100: parent_01 - (child_sum_01 - child_0100) = parent_01 - child_0101
    # For cov: parent_cov_01 + (child_sum_cov_01 - child_cov_0100)
    #          = parent_cov_01 + child_cov_0101
    # Wait, parent_01 - sibling is:
    # val = parent_val - sibling_val
    # cov = parent_cov + sibling_cov
    # Let's check how __sub__ and __mod__ are used:
    # parent - (child_sum % child)
    # child_sum % child = (child_sum_val - child_val, child_sum_cov - child_cov)
    #                   = (sibling_val, sibling_cov)
    # parent - (sibling) = (parent_val - sibling_val, parent_cov + sibling_cov)
    # Yes, this is correct.

    sibling_0100 = est_0101
    sibling_0101 = est_0100
    # sibling_0200 is empty estimate (subtraction of child_0200 from itself)
    # Actually est_0200 % est_0200 = Estimate(0, 0)
    # So sibling_0200 is Estimate(0, 0)
    # Wait, does Estimate support subtraction resulting in 0?
    # est_0200 % est_0200 = Estimate(val - val, cov - cov)
    # val - val = 0.
    # cov - cov = SuccinctCovariance(0, 0).
    # parent_p02 - Estimate(0, 0) = (p02_val - 0, p02_cov + 0) = parent_p02.

    zero_cov = succinct.SuccinctCovariance(
        np.zeros((32, 32)), np.zeros((32, 32))
    )
    zero_est = Estimate(np.zeros(self.n), zero_cov)

    expected = pd.DataFrame({
        ID: ["0100", "0101", "0200"],
        lower_estimate_column: [est_0100, est_0101, est_0200],
        upper_exclusive_estimate_column: [
            est_p01 - sibling_0100,
            est_p01 - sibling_0101,
            est_p02 - zero_est,
        ],
    })

    self.assert_pandas_frames_of_estimates_equal(
        computed,
        expected,
        estimate_columns=(
            lower_estimate_column,
            upper_exclusive_estimate_column,
        ),
    )

  def test_top_down_step(self):
    est_c0100_in = self._random_estimate()
    est_c0101_in = self._random_estimate()
    children_input_df = pd.DataFrame({
        ID: ["0100", "0101"],
        ESTIMATE: [est_c0100_in, est_c0101_in],
    })

    est_c0100_low = self._random_estimate()
    est_c0101_low = self._random_estimate()
    children_lower_df = pd.DataFrame({
        ID: ["0100", "0101"],
        ESTIMATE: [est_c0100_low, est_c0101_low],
    })

    est_p01_up = self._random_estimate()
    parents_upper_df = pd.DataFrame({
        ID: ["01"],
        ESTIMATE: [est_p01_up],
    })

    computed_upper, computed_combined = linear_postprocessing.top_down_step(
        children_input_df=children_input_df,
        children_lower_df=children_lower_df,
        parents_upper_df=parents_upper_df,
    )

    # Expected:
    # parents_df has child_lower_sum = est_c0100_low + est_c0101_low
    # upper_exclusive_0100 = est_p01_up - (child_lower_sum % est_c0100_low)
    #                      = est_p01_up - est_c0101_low
    # upper_exclusive_0101 = est_p01_up - (child_lower_sum % est_c0101_low)
    #                      = est_p01_up - est_c0100_low
    #
    # combined_0100 = combine(est_c0100_low, upper_exclusive_0100)
    # combined_0101 = combine(est_c0101_low, upper_exclusive_0101)
    #
    # upper_0100 = combine(est_c0100_in, upper_exclusive_0100)
    # upper_0101 = combine(est_c0101_in, upper_exclusive_0101)

    child_sum = est_c0100_low + est_c0101_low
    excl_0100 = est_p01_up - (child_sum % est_c0100_low)
    excl_0101 = est_p01_up - (child_sum % est_c0101_low)

    expected_combined = pd.DataFrame({
        ID: ["0100", "0101"],
        ESTIMATE: [
            regression.combine(est_c0100_low, excl_0100),
            regression.combine(est_c0101_low, excl_0101),
        ],
    })

    expected_upper = pd.DataFrame({
        ID: ["0100", "0101"],
        ESTIMATE: [
            regression.combine(est_c0100_in, excl_0100),
            regression.combine(est_c0101_in, excl_0101),
        ],
    })

    self.assert_pandas_frames_of_estimates_equal(
        computed_combined, expected_combined
    )
    self.assert_pandas_frames_of_estimates_equal(computed_upper, expected_upper)


if __name__ == "__main__":
  absltest.main()
