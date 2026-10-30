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

from census_bluedown import constants
from census_bluedown import ground_truth
from absl.testing import absltest


class GroundTruthTest(absltest.TestCase):

  def test_aggregate_microdata_format(self):
    df = pd.DataFrame({
        'TABBLKST': ['01', '01', '01', '01'],
        'TABBLKCOU': ['001', '001', '001', '001'],
        'TABTRACTCE': ['123456', '123456', '123456', '123456'],
        'TABBLKGRPCE': ['1', '1', '1', '1'],
        'TABBLK': ['1234', '1234', '1234', '9999'],
        'GQTYPE_PL': ['0', '0', '0', '7'],
        'VOTING_AGE': ['1', '1', '1', '2'],
        'CENHISP': ['1', '1', '1', '2'],
        'CENRACE': ['01', '01', '03', '63']})

    df_out = ground_truth.aggregate_microdata_format(df)

    expected_geocodes = ['0100112345611234', '0100112345619999']
    value_1 = np.zeros(2016, dtype=int)
    value_2 = np.zeros(2016, dtype=int)
    value_1[0] = 2
    value_1[2] = 1
    value_2[2015] = 1
    expected_values = [value_1, value_2]
    np.testing.assert_equal(df_out.columns.tolist(), ('value',))
    df_out.reset_index(inplace=True)
    np.testing.assert_equal(df_out.columns.tolist(), ('geocode16', 'value'))
    np.testing.assert_equal(df_out['geocode16'].tolist(), expected_geocodes)
    np.testing.assert_equal(df_out['value'].tolist(), expected_values)

  def test_aggregate_microdata_format_sorts_and_patches_geocodes(self):
    df = pd.DataFrame({
        'TABBLKST': ['45', '01', '45'],
        'TABBLKCOU': ['029', '001', '029'],
        'TABTRACTCE': ['970800', '123456', '970800'],
        'TABBLKGRPCE': ['3', '1', '3'],
        'TABBLK': ['3004', '1234', '3004'],
        'GQTYPE_PL': ['0', '0', '1'],
        'VOTING_AGE': ['1', '2', '1'],
        'CENHISP': ['1', '1', '2'],
        'CENRACE': ['01', '02', '63'],
    })

    df_out = ground_truth.aggregate_microdata_format(df)

    # '4502997080033004' is remapped by the water block patch.
    self.assertEqual(df_out.index.name, 'geocode16')
    self.assertEqual(
        df_out.index.tolist(), ['0100112345611234', '4502999010000001']
    )
    value_1 = np.zeros(2016, dtype=int)
    value_1[(0 * 2 + 1) * 2 * 63 + 1] = 1
    value_2 = np.zeros(2016, dtype=int)
    value_2[0] = 1
    value_2[(1 * 2 * 2 + 1) * 63 + 62] = 1
    np.testing.assert_equal(df_out['value'].tolist(), [value_1, value_2])

  def test_aggregate_microdata_format_out_of_range_raises(self):
    for column, value in [
        ('GQTYPE_PL', '8'),
        ('VOTING_AGE', '0'),
        ('VOTING_AGE', '3'),
        ('CENHISP', '3'),
        ('CENRACE', '00'),
        ('CENRACE', '64'),
    ]:
      with self.subTest(column=column, value=value):
        df = pd.DataFrame({
            'TABBLKST': ['01'],
            'TABBLKCOU': ['001'],
            'TABTRACTCE': ['123456'],
            'TABBLKGRPCE': ['1'],
            'TABBLK': ['1234'],
            'GQTYPE_PL': ['0'],
            'VOTING_AGE': ['1'],
            'CENHISP': ['1'],
            'CENRACE': ['01'],
        })
        df[column] = [value]
        with self.assertRaises(ValueError):
          ground_truth.aggregate_microdata_format(df)

  def test_split_df_by_constraint_dfs(self):
    df = pd.DataFrame({
        'geocode16': ['00', '01', '02', '03', '04', '05'],
        'value': [0, 1, 2, 3, 4, 5]})
    constraint_dfs = [
        pd.DataFrame({'geocode': ['11101', '11102', '11103'],
                      'foo': [None, None, None],
                      'query_name': ['hhgq_total_lb_con']*3}),
        pd.DataFrame({'geocode': ['12300'],
                      'bar': [17],
                      'query_name': ['hhgq_total_lb_con']}),
        pd.DataFrame({'geocode': ['23404', '45605', '23404', '45605'],
                      'baz': [-1, -2, -3, -4],
                      'query_name': ['hhgq_total_lb_con', 'hhgq_total_lb_con',
                                     'a', 'b']})]
    df.set_index('geocode16', inplace=True)

    out_dfs = ground_truth.split_df_by_constraint_dfs(
        df,
        constraint_dfs,
        suffix_length=2)

    expected_values = [[1, 2, 3], [0], [4, 5]]
    expected_geocodes = [['11101', '11102', '11103'],
                         ['12300'],
                         ['23404', '45605']]
    self.assertLen(out_dfs, len(constraint_dfs))
    for out_df, expected_value, expected_geocode in zip(
        out_dfs, expected_values, expected_geocodes):
      np.testing.assert_equal(out_df.columns.tolist(), ('geocode', 'value'))
      np.testing.assert_equal(out_df['geocode'].tolist(), expected_geocode)
      np.testing.assert_equal(out_df['value'].tolist(), expected_value)

  def test_aggregate_histograms_to_ids(self):
    # Three blocks under two tracts ('11' and '12'); tract '13' has no blocks.
    histograms = ground_truth._histograms_by_id(
        pd.DataFrame({
            'geocode': ['1101', '1102', '1201'],
            'value': [[1, 2], [3, 4], [5, 6]],
        })
    )

    tracts, values = ground_truth._aggregate_histograms_to_ids(
        histograms, ['11', '12', '13']
    )
    np.testing.assert_equal(values, [[4, 6], [5, 6], [0, 0]])

    # Aggregating the tract result to the county equals aggregating the blocks
    # directly, since geocode prefixes nest.
    _, from_tracts = ground_truth._aggregate_histograms_to_ids(tracts, ['1'])
    _, from_blocks = ground_truth._aggregate_histograms_to_ids(
        histograms, ['1']
    )
    np.testing.assert_equal(from_tracts, [[9, 12]])
    np.testing.assert_equal(from_blocks, from_tracts)

  def test_aggregate_histograms_to_ids_empty_subtree(self):
    histograms = ground_truth._histograms_by_id(
        pd.DataFrame({
            'geocode': pd.Series([], dtype=str),
            'value': pd.Series([], dtype=object),
        })
    )
    self.assertEqual(histograms.shape, (0, constants.TOTAL_LEN))

    _, values = ground_truth._aggregate_histograms_to_ids(histograms, ['11'])
    np.testing.assert_equal(values, np.zeros((1, constants.TOTAL_LEN)))


if __name__ == '__main__':
  absltest.main()
