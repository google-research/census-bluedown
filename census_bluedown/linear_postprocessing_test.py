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
import pandas as pd

from census_bluedown import block
from census_bluedown import estimate
from census_bluedown import linear_postprocessing
from absl.testing import absltest
from absl.testing import parameterized


Estimate = estimate.Estimate
ID = linear_postprocessing.ID
ESTIMATE = linear_postprocessing.ESTIMATE
QUERY_SHAPE = linear_postprocessing.QUERY_SHAPE
VARIANCE = linear_postprocessing.VARIANCE
VALUE = linear_postprocessing.VALUE


class ParameterizedEstimateDFTestCase(parameterized.TestCase):
  """Parameterized test cases invovling data frames of Estimate objects."""

  def assert_pandas_frames_of_estimates_equal(
      self,
      computed: pd.DataFrame,
      expected: pd.DataFrame,
      estimate_columns: tuple[str, ...] = (ESTIMATE,)
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
    merged = pd.merge(computed, expected, on=ID, how='outer',
                      suffixes=('_computed', '_expected'))
    for row in merged.itertuples():
      for col in estimate_columns:
        np.testing.assert_allclose(
            getattr(row, col + '_computed').val,
            getattr(row, col + '_expected').val
        )
        np.testing.assert_allclose(
            getattr(row, col + '_computed').cov,
            getattr(row, col + '_expected').cov
        )


class LinearPostprocessingTest(ParameterizedEstimateDFTestCase):

  def test_process_blocks(self):
    block_shape = block.BlockShape((2,), 1)
    df = pd.DataFrame({
        ID: ['0100', '0100', '0200', '0200'],
        QUERY_SHAPE: [(2,), (1,), (2,), (1,)],
        VARIANCE: [1, 2, 3, 4],
        VALUE: [np.array([1, 2]),
                np.array([3]),
                np.array([4, 5]),
                np.array([6])]
    })
    computed = linear_postprocessing.process_blocks(
        block_shape=block_shape,
        df=df,
    )
    expected = pd.DataFrame({
        ID: ['0100', '0200'],
        ESTIMATE: [Estimate(np.array([1, 2]),
                            np.array([[3/4, -1/4],
                                      [-1/4, 3/4]])),
                   Estimate(np.array([3.1, 4.1]),
                            np.array([[2.1, -0.9],
                                      [-0.9, 2.1]]))],
    })
    self.assert_pandas_frames_of_estimates_equal(computed, expected)

  def test_aggregate_child_estimates(self):
    children_df = pd.DataFrame({
        ID: ['0100', '0101', '0200'],
        ESTIMATE: [Estimate(np.array([1, 2]), np.eye(2)),
                   Estimate(np.array([2, 3]), np.ones((2, 2))),
                   Estimate(np.array([3, 4]), 2 * np.eye(2))]
    })
    parents_df = pd.DataFrame({
        ID: ['01', '02'],
        ESTIMATE: [Estimate(np.array([3, 4]), np.eye(2)),
                   Estimate(np.array([5, 6]), np.eye(2))],
    })
    parent_suffix = '_parent'
    child_suffix = '_child'

    computed = linear_postprocessing._aggregate_child_estimates(
        children_df=children_df, parents_df=parents_df,
        parent_suffix=parent_suffix, child_suffix=child_suffix,
    )
    expected = pd.DataFrame({
        ID: ['01', '02'],
        ESTIMATE + parent_suffix: [Estimate(np.array([3, 4]), np.eye(2)),
                                   Estimate(np.array([5, 6]), np.eye(2))],
        ESTIMATE + child_suffix: [Estimate(np.array([3, 5]),
                                           np.array([[2, 1], [1, 2]])),
                                  Estimate(np.array([3, 4]), 2 * np.eye(2))]
    })

    self.assert_pandas_frames_of_estimates_equal(
        computed=computed, expected=expected,
        estimate_columns=(ESTIMATE + parent_suffix, ESTIMATE + child_suffix)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='simple',
          block_shape=block.BlockShape((2,), 1),
          children_lower_df=pd.DataFrame({
              ID: ['0100', '0101'],
              ESTIMATE: [Estimate(np.array([1, 2]), np.eye(2)),
                         Estimate(np.array([0, 0]), np.eye(2))],
          }),
          parents_input_df=pd.DataFrame({
              ID: ['01'],
              ESTIMATE: [Estimate(np.array([3, 4]), 2 * np.eye(2))],
          }),
          expected=pd.DataFrame({
              ID: ['01'],
              ESTIMATE: [Estimate(np.array([2, 3]), np.eye(2))],
          }),
      ),
      dict(
          testcase_name='multiple_parents',
          block_shape=block.BlockShape((1,), 1),
          children_lower_df=pd.DataFrame({
              ID: ['0100', '0101', '0200'],
              ESTIMATE: [Estimate(np.array([1]), np.array([[2]])),
                         Estimate(np.array([3]), np.array([[2]])),
                         Estimate(np.array([3]), np.array([[2]]))],
          }),
          parents_input_df=pd.DataFrame({
              ID: ['01', '02'],
              ESTIMATE: [Estimate(np.array([4]), np.array([[1]])),
                         Estimate(np.array([4]), np.array([[1]]))],
          }),
          expected=pd.DataFrame({
              ID: ['01', '02'],
              ESTIMATE: [Estimate(np.array([4]), np.array([[4/5]])),
                         Estimate(np.array([11/3]), np.array([[2/3]]))],
          }),
      ),
  )
  def test_bottom_up_step(
      self,
      block_shape: block.BlockShape,
      children_lower_df: pd.DataFrame,
      parents_input_df: pd.DataFrame,
      expected: pd.DataFrame,
  ):
    computed = linear_postprocessing.bottom_up_step(
        block_shape=block_shape,
        children_lower_df=children_lower_df,
        parents_input_df=parents_input_df
    )
    self.assert_pandas_frames_of_estimates_equal(computed, expected)

  def test_compute_upper_exclusive_estimate(self):
    parent_estimate_column = 'parent_estimate'
    child_lower_sum_estimate_column = 'child_lower_sum_estimate'
    lower_estimate_column = 'lower_estimate'
    upper_exclusive_estimate_column = 'upper_exclusive_estimate'
    children_df = pd.DataFrame({
        ID: ['0100', '0101', '0200'],
        ESTIMATE: [
            Estimate(np.array([1, 2]), np.eye(2)),
            Estimate(np.array([2, 3]), np.ones((2, 2))),
            Estimate(np.array([3, 4]), 2 * np.eye(2))],
    })
    parents_df = pd.DataFrame({
        ID: ['01', '02'],
        parent_estimate_column: [
            Estimate(np.array([3, 4]), 2 * np.eye(2)),
            Estimate(np.array([5, 6]), np.eye(2))],
        child_lower_sum_estimate_column: [
            Estimate(np.array([3, 5]), np.array([[2, 1], [1, 2]])),
            Estimate(np.array([3, 4]), 2 * np.eye(2))],
    })
    computed = linear_postprocessing._compute_upper_exclusive_estimate(
        children_df=children_df, parents_df=parents_df,
        parent_estimate_column=parent_estimate_column,
        child_lower_sum_estimate_column=child_lower_sum_estimate_column,
        lower_estimate_column=lower_estimate_column,
        upper_exclusive_estimate_column=upper_exclusive_estimate_column,
    )
    expected = pd.DataFrame({
        ID: ['0100', '0101', '0200'],
        lower_estimate_column: [
            Estimate(np.array([1, 2]), np.eye(2)),
            Estimate(np.array([2, 3]), np.ones((2, 2))),
            Estimate(np.array([3, 4]), 2 * np.eye(2))],
        upper_exclusive_estimate_column: [
            Estimate(np.array([1, 1]), np.array([[3, 1], [1, 3]])),
            Estimate(np.array([2, 2]), 3 * np.eye(2)),
            Estimate(np.array([5, 6]), np.eye(2))],
    })
    self.assert_pandas_frames_of_estimates_equal(
        computed,
        expected,
        estimate_columns=(lower_estimate_column,
                          upper_exclusive_estimate_column))

  @parameterized.named_parameters(
      dict(
          testcase_name='simple',
          block_shape=block.BlockShape((2,), 1),
          children_input_df=pd.DataFrame({
              ID: ['0100', '0101'],
              ESTIMATE: [Estimate(np.array([1, 2]), np.eye(2)),
                         Estimate(np.array([0, 0]), np.eye(2))],
          }),
          children_lower_df=pd.DataFrame({
              ID: ['0100', '0101'],
              ESTIMATE: [Estimate(np.array([2, 3]), np.eye(2)),
                         Estimate(np.array([0, 1]), np.eye(2))],
          }),
          parents_upper_df=pd.DataFrame({
              ID: ['01'],
              ESTIMATE: [Estimate(np.array([3, 4]), 2 * np.eye(2))],
          }),
          expected_upper=pd.DataFrame({
              ID: ['0100', '0101'],
              ESTIMATE: [Estimate(np.array([3/2, 9/4]), 0.75 * np.eye(2)),
                         Estimate(np.array([1/4, 1/4]), 0.75 * np.eye(2))],
          }),
          expected_combined=pd.DataFrame({
              ID: ['0100', '0101'],
              ESTIMATE: [Estimate(np.array([9/4, 3]), 0.75 * np.eye(2)),
                         Estimate(np.array([1/4, 1]), 0.75 *np.eye(2))],
          })
      ),
      dict(
          testcase_name='multiple_parents',
          block_shape=block.BlockShape((1,), 1),
          children_input_df=pd.DataFrame({
              ID: ['0100', '0101', '0200'],
              ESTIMATE: [Estimate(np.array([0]), np.array([[1]])),
                         Estimate(np.array([1]), np.array([[2]])),
                         Estimate(np.array([2]), np.array([[3]]))],
          }),
          children_lower_df=pd.DataFrame({
              ID: ['0100', '0101', '0200'],
              ESTIMATE: [Estimate(np.array([1]), np.array([[2]])),
                         Estimate(np.array([3]), np.array([[2]])),
                         Estimate(np.array([3]), np.array([[2]]))],
          }),
          parents_upper_df=pd.DataFrame({
              ID: ['01', '02'],
              ESTIMATE: [Estimate(np.array([4]), np.array([[1]])),
                         Estimate(np.array([4]), np.array([[1]]))],
          }),
          expected_upper=pd.DataFrame({
              ID: ['0100', '0101', '0200'],
              ESTIMATE: [Estimate(np.array([1/4]), np.array([[3/4]])),
                         Estimate(np.array([9/5]), np.array([[6/5]])),
                         Estimate(np.array([7/2]), np.array([[3/4]]))],
          }),
          expected_combined=pd.DataFrame({
              ID: ['0100', '0101', '0200'],
              ESTIMATE: [Estimate(np.array([1]), np.array([[6/5]])),
                         Estimate(np.array([3]), np.array([[6/5]])),
                         Estimate(np.array([11/3]), np.array([[2/3]]))],
          }),
      )
  )
  def test_top_down_step(
      self,
      block_shape: block.BlockShape,
      children_input_df: pd.DataFrame,
      children_lower_df: pd.DataFrame,
      parents_upper_df: pd.DataFrame,
      expected_upper: pd.DataFrame,
      expected_combined: pd.DataFrame,
  ):
    computed_upper, computed_combined = linear_postprocessing.top_down_step(
        block_shape=block_shape,
        children_input_df=children_input_df,
        children_lower_df=children_lower_df,
        parents_upper_df=parents_upper_df,
    )
    self.assert_pandas_frames_of_estimates_equal(
        computed_upper,
        expected_upper
    )
    self.assert_pandas_frames_of_estimates_equal(
        computed_combined,
        expected_combined
    )


if __name__ == '__main__':
  absltest.main()
