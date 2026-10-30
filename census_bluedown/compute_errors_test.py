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
from census_bluedown import constants
from absl.testing import absltest
from absl.testing import parameterized


def _block_id(geocode16: str, prefix: str = '0' * 15) -> str:
  """Returns a 31-character block ID: a TopDown prefix plus the geocode16."""
  return prefix + geocode16


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

  def test_query_errors(self):
    # Two regions; region 0 is off by +2 in cell (3, 1, 0, 62) and region 1 is
    # off by -1 in cell (0, 0, 1, 5), so total absolute errors are 2 and 1 for
    # every query (the projection sums each cell into exactly one entry), and
    # the mean across regions is 1.5.
    diffs = np.zeros((2, constants.TOTAL_LEN))
    diffs[
        0, np.ravel_multi_index((3, 1, 0, 62), constants.DETAILED_QUERY_SHAPE)
    ] = 2
    diffs[
        1, np.ravel_multi_index((0, 0, 1, 5), constants.DETAILED_QUERY_SHAPE)
    ] = -1
    errs = compute_errors.query_errors(diffs)
    self.assertCountEqual(errs.keys(), compute_errors.QUERY_DICT.keys())
    for query, err in errs.items():
      self.assertAlmostEqual(err, 1.5, msg=query)

    # Opposite-sign errors in different cells cancel only when projected into
    # the same entry: region 0 has +2 and -2 in two race cells of the same
    # housing type, so TOTAL and HOUSING_TYPE see 0 but CENRACE sees 4.
    diffs = np.zeros((1, constants.TOTAL_LEN))
    diffs[
        0, np.ravel_multi_index((3, 1, 0, 62), constants.DETAILED_QUERY_SHAPE)
    ] = 2
    diffs[
        0, np.ravel_multi_index((3, 1, 0, 5), constants.DETAILED_QUERY_SHAPE)
    ] = -2
    errs = compute_errors.query_errors(diffs)
    self.assertEqual(errs['TOTAL'], 0)
    self.assertEqual(errs['HOUSING_TYPE'], 0)
    self.assertEqual(errs['CENRACE'], 4)
    self.assertEqual(errs['DETAILED'], 4)

  def test_alternate_query_errors(self):
    # One region with a single error of +1 at detailed cell (housing type 3,
    # voting age 1, non-Hispanic, race code 62 = all six races).
    diffs = np.zeros((1, constants.TOTAL_LEN))
    diffs[
        0, np.ravel_multi_index((3, 1, 0, 62), constants.DETAILED_QUERY_SHAPE)
    ] = 1
    errs = compute_errors.alternate_query_errors(diffs)

    # The flattened (query, entry) names match ALTERNATE_QUERIES in order.
    names = []
    for q, arr in errs.items():
      names.extend(f'{q}_{i}' for i in range(len(arr)))
    self.assertEqual(names, compute_errors.ALTERNATE_QUERIES)

    def one_hot(length: int, index: int) -> np.ndarray:
      out = np.zeros(length)
      out[index] = 1
      return out

    np.testing.assert_array_equal(errs[compute_errors.VOTINGAGE_QUERY], [0, 1])
    np.testing.assert_array_equal(errs[compute_errors.HISPANIC_QUERY], [1, 0])
    np.testing.assert_array_equal(
        errs[compute_errors.HOUSING_TYPE_QUERY], one_hot(8, 3)
    )
    # Race: "two or more races" is the last of 7 alone cells, all 6 races are
    # present in the combined counts, and the number of races is 6 (index 5).
    np.testing.assert_array_equal(
        errs[compute_errors.CENRACE_ALONE_QUERY], one_hot(7, 6)
    )
    np.testing.assert_array_equal(
        errs[compute_errors.CENRACE_COMBINED_QUERY], np.ones(6)
    )
    np.testing.assert_array_equal(
        errs[compute_errors.CENRACE_NUMBER_QUERY], one_hot(6, 5)
    )
    # Crossed with Hispanic (group 0 of 2) and with voting age x Hispanic
    # (group 1 * 2 + 0 = 2 of 4), the same pattern appears within that group.
    np.testing.assert_array_equal(
        errs[compute_errors.HISP_CENRACE_ALONE_QUERY], one_hot(14, 6)
    )
    np.testing.assert_array_equal(
        errs[compute_errors.HISP_CENRACE_COMBINED_QUERY],
        np.concatenate([np.ones(6), np.zeros(6)]),
    )
    np.testing.assert_array_equal(
        errs[compute_errors.HISP_CENRACE_NUMBER_QUERY], one_hot(12, 5)
    )
    np.testing.assert_array_equal(
        errs[compute_errors.VOTINGAGE_HISP_CENRACE_ALONE_QUERY],
        one_hot(28, 2 * 7 + 6),
    )
    np.testing.assert_array_equal(
        errs[compute_errors.VOTINGAGE_HISP_CENRACE_COMBINED_QUERY],
        np.concatenate([np.zeros(12), np.ones(6), np.zeros(6)]),
    )
    np.testing.assert_array_equal(
        errs[compute_errors.VOTINGAGE_HISP_CENRACE_NUMBER_QUERY],
        one_hot(24, 2 * 6 + 5),
    )
    # Housing type 3 is an institutional group quarters type (HHINSTLEVELS
    # bin 1), so it lands in the first of the two GQ cells.
    np.testing.assert_array_equal(errs[compute_errors.GQ_GROUPED_QUERY], [1, 0])

  def test_errors_dataframe(self):
    df = compute_errors._errors_dataframe(
        base_errs={'TOTAL': 1.5, 'CENRACE': 2.0},
        opt_errs={'TOTAL': 0.5, 'CENRACE': 1.0},
        num_regions=4,
    )
    self.assertEqual(list(df.index), ['TOTAL', 'CENRACE'])
    self.assertEqual(list(df.columns), ['baseline', 'optimized', 'n'])
    np.testing.assert_array_equal(df['baseline'], [1.5, 2.0])
    np.testing.assert_array_equal(df['optimized'], [0.5, 1.0])
    np.testing.assert_array_equal(df['n'], [4, 4])

  def test_errors_list_dataframe(self):
    ground_truth = np.zeros((2, constants.TOTAL_LEN), dtype=int)
    ground_truth[:, 0] = [10, 25]

    # Region 0 has +2 and -2 in two race cells of the same housing type, so
    # TOTAL and HOUSING_TYPE have absolute error 0 while CENRACE and DETAILED
    # have absolute error 4, and bias is 0. Region 1 has -3 in one cell.
    base_diffs = np.zeros((2, constants.TOTAL_LEN), dtype=int)
    base_diffs[0, [0, 1]] = [2, -2]
    base_diffs[1, 0] = -3

    opt_diffs = np.zeros((2, constants.TOTAL_LEN), dtype=int)
    opt_diffs[:, 0] = [1, -1]

    df = compute_errors.errors_list_dataframe(
        ground_truth=ground_truth,
        base_diffs=base_diffs,
        opt_diffs=opt_diffs,
    )
    self.assertEqual(list(df.index), list(compute_errors.QUERY_DICT.keys()))
    self.assertEqual(df.loc['TOTAL', 'baseline'], '[0, 3]')
    self.assertEqual(df.loc['HOUSING_TYPE', 'baseline'], '[0, 3]')
    self.assertEqual(df.loc['CENRACE', 'baseline'], '[4, 3]')
    self.assertEqual(df.loc['DETAILED', 'baseline'], '[4, 3]')
    for query in compute_errors.QUERY_DICT:
      self.assertEqual(df.loc[query, 'optimized'], '[1, 1]')
      self.assertEqual(df.loc[query, 'base_bias'], '[0, -3]')
      self.assertEqual(df.loc[query, 'opt_bias'], '[1, -1]')
      self.assertEqual(df.loc[query, 'ground_truth'], '[10, 25]')
      self.assertEqual(df.loc[query, 'n'], 2)

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
  def test_diffs_by_level(
      self,
      level: str,
      expected_base_diffs: list[np.ndarray],
      expected_opt_diffs: list[np.ndarray],
  ):
    geocodes = [
        _block_id('0100102010010011'),
        _block_id('0100102010010012'),
        _block_id('0100203010010011'),
    ]
    ground_truth_df = pd.DataFrame({
        'geocode': geocodes,
        'value': [[10, 20], [30, 40], [50, 60]],
    })
    base_df = pd.DataFrame({
        'geocode': geocodes,
        'value': [[12, 18], [35, 38], [48, 65]],
    })
    opt_df = pd.DataFrame({
        'geocode': geocodes,
        'value': [[11, 21], [31, 39], [51, 62]],
    })

    diffs = {
        level_diffs.level: level_diffs
        for level_diffs in compute_errors.diffs_by_level(
            ground_truth_df=ground_truth_df,
            base_df=base_df,
            opt_df=opt_df,
        )
    }

    self.assertCountEqual(
        diffs.keys(), ['Block', 'Block_Group', 'Tract', 'County', 'State']
    )
    np.testing.assert_array_equal(
        diffs[level].base_diffs, np.stack(expected_base_diffs)
    )
    np.testing.assert_array_equal(
        diffs[level].opt_diffs, np.stack(expected_opt_diffs)
    )

  def test_diffs_by_level_mismatched_geocodes_raises(self):
    ground_truth_df = pd.DataFrame({
        'geocode': [
            _block_id('0100102010010011'),
            _block_id('0100102010010012'),
        ],
        'value': [[10, 20], [30, 40]],
    })
    other_df = pd.DataFrame({
        'geocode': [
            _block_id('0100102010010011'),
            _block_id('0100203010010011'),
        ],
        'value': [[10, 20], [30, 40]],
    })

    with self.assertRaises(ValueError):
      next(
          compute_errors.diffs_by_level(
              ground_truth_df=ground_truth_df,
              base_df=other_df,
              opt_df=ground_truth_df,
          )
      )

  def test_diffs_by_level_wrong_id_length_raises(self):
    # A bare geocode16 without the 15-character TopDown prefix is rejected.
    df = pd.DataFrame({
        'geocode': ['0100102010010011'],
        'value': [[10, 20]],
    })
    with self.assertRaisesRegex(ValueError, 'length'):
      next(compute_errors.diffs_by_level(df, df, df))

  def test_diffs_by_level_duplicate_geocode16_raises(self):
    # The same geocode16 under two different TopDown prefixes (e.g. the AIAN
    # and non-AIAN subtrees) would otherwise be silently summed.
    df = pd.DataFrame({
        'geocode': [
            _block_id('0100102010010011', prefix='0' * 15),
            _block_id('0100102010010011', prefix='1' * 15),
        ],
        'value': [[10, 20], [30, 40]],
    })
    with self.assertRaisesRegex(ValueError, 'unique'):
      next(compute_errors.diffs_by_level(df, df, df))


if __name__ == '__main__':
  absltest.main()
