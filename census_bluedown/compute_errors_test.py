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
from census_bluedown import compute_errors
from absl.testing import absltest
from absl.testing import parameterized


class ComputeErrorsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='two_races',
          num_races=2,
          expected_matrix=[
              [1, 0],
              [0, 1],
              [1, 1],
          ],
      ),
      dict(
          testcase_name='three_races',
          num_races=3,
          expected_matrix=[
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [1, 1, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1],
          ],
      ),
  )
  def test_race_counts_combined_transformation_matrix(
      self, num_races: int, expected_matrix: list[list[int]]
  ):
    actual_matrix = compute_errors.race_counts_combined_transformation_matrix(
        num_races=num_races
    )
    np.testing.assert_array_equal(actual_matrix, np.array(expected_matrix))

  @parameterized.named_parameters(
      dict(
          testcase_name='total',
          query_name='TOTAL',
          expected_length=1,
      ),
      dict(
          testcase_name='voting_age',
          query_name='VOTINGAGE',
          expected_length=2,
      ),
      dict(
          testcase_name='hispanic',
          query_name='HISPANIC',
          expected_length=2,
      ),
      dict(
          testcase_name='cenrace',
          query_name='CENRACE',
          expected_length=63,
      ),
      dict(
          testcase_name='hispanic_x_cenrace',
          query_name='HISPANICxCENRACE',
          expected_length=126,
      ),
      dict(
          testcase_name='detailed',
          query_name='DETAILED',
          expected_length=2016,
      ),
      dict(
          testcase_name='housing_type',
          query_name='HOUSING_TYPE',
          expected_length=8,
      ),
      dict(
          testcase_name='age_x_hisp_x_cenrace',
          query_name='AGExHISPxCENRACE',
          expected_length=252,
      ),
  )
  def test_query_lambdas(self, query_name: str, expected_length: int):
    ones_input = np.ones(2016, dtype=int)
    projected = compute_errors.QUERY_LAMBDAS[query_name](ones_input)
    np.testing.assert_array_equal(
        projected, np.full(expected_length, 2016 // expected_length)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='block_level',
          level='Block',
          expected_base_diffs=[
              np.array([2, -2]),
              np.array([5, -2]),
              np.array([-2, 5]),
          ],
          expected_opt_diffs=[
              np.array([1, 1]),
              np.array([1, -1]),
              np.array([1, 2]),
          ],
      ),
      dict(
          testcase_name='county_level',
          level='County',
          expected_base_diffs=[
              np.array([7, -4]),
              np.array([-2, 5]),
          ],
          expected_opt_diffs=[
              np.array([2, 0]),
              np.array([1, 2]),
          ],
      ),
      dict(
          testcase_name='state_level',
          level='State',
          expected_base_diffs=[
              np.array([5, 1]),
          ],
          expected_opt_diffs=[
              np.array([3, 2]),
          ],
      ),
  )
  def test_get_diffs(
      self,
      level: str,
      expected_base_diffs: list[np.ndarray],
      expected_opt_diffs: list[np.ndarray],
  ):
    ground_truth_df = pd.DataFrame({
        'geocode': [
            '0100102010010011',
            '0100102010010012',
            '0100203010010011',
        ],
        'value': [[10, 20], [30, 40], [50, 60]],
    })
    base_df = pd.DataFrame({
        'geocode': [
            '0100102010010011',
            '0100102010010012',
            '0100203010010011',
        ],
        'value': [[12, 18], [35, 38], [48, 65]],
    })
    opt_df = pd.DataFrame({
        'geocode': [
            '0100102010010011',
            '0100102010010012',
            '0100203010010011',
        ],
        'value': [[11, 21], [31, 39], [51, 62]],
    })

    base_diffs_df, opt_diffs_df = compute_errors.get_diffs(
        ground_truth_df=ground_truth_df,
        base_df=base_df,
        opt_df=opt_df,
        level=level,
    )

    self.assertLen(base_diffs_df, len(expected_base_diffs))
    self.assertLen(opt_diffs_df, len(expected_opt_diffs))

    for actual, expected in zip(base_diffs_df['diffs'], expected_base_diffs):
      np.testing.assert_array_equal(actual, expected)

    for actual, expected in zip(opt_diffs_df['diffs'], expected_opt_diffs):
      np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
