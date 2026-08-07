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


if __name__ == '__main__':
  absltest.main()
