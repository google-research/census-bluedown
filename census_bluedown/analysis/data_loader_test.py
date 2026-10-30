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

import os

import tempfile
import numpy as np
import pandas as pd

from census_bluedown.analysis import common
from census_bluedown.analysis import (
    data_loader,
)
from absl.testing import absltest


class DataLoaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.enter_context(tempfile.TemporaryDirectory())

  def test_compute_us_level_replicate_errors(self):
    num_reps = 2
    gt_dir = os.path.join(self.temp_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    gt_file = os.path.join(gt_dir, 'ground_truth.parquet')
    gt_arr = np.ones((8, 2, 2, 63), dtype=np.float64)
    pd.DataFrame({
        'query_name': ['detailed_dpq'],
        'value': [gt_arr.flatten()],
    }).to_parquet(gt_file)

    for rep in range(num_reps):
      rep_dir = os.path.join(self.temp_dir, str(rep), 'US')
      os.makedirs(rep_dir, exist_ok=True)
      opt_arr = np.ones((8, 2, 2, 63), dtype=np.float64) * 1.1
      base_arr = np.ones((8, 2, 2, 63), dtype=np.float64) * 1.5
      pd.DataFrame({'value': [opt_arr.flatten()]}).to_parquet(
          os.path.join(rep_dir, 'optimized_estimate.parquet')
      )
      pd.DataFrame({'value': [base_arr.flatten()]}).to_parquet(
          os.path.join(rep_dir, 'baseline_estimate.parquet')
      )

    us_df = data_loader._compute_us_level_replicate_errors(
        base_dir=self.temp_dir,
        gt_parquet_path=gt_file,
        num_replicates=num_reps,
    )
    self.assertFalse(us_df.empty)
    self.assertEqual(
        list(us_df['query']), list(common.US_QUERIES_AND_SHAPES.keys())
    )
    self.assertLen(us_df, len(common.US_QUERIES_AND_SHAPES))
    self.assertLen(us_df['baseline'].iloc[0], num_reps)
    # With constant tensors the L1 error is the same for every marginal:
    # |1.1 - 1| and |1.5 - 1| summed over all 8*2*2*63 = 2016 cells.
    n_cells = float(np.prod(common.US_DETAILED_SHAPE))
    for _, row in us_df.iterrows():
      for rep in range(num_reps):
        self.assertAlmostEqual(row['optimized'][rep], 0.1 * n_cells, places=6)
        self.assertAlmostEqual(row['baseline'][rep], 0.5 * n_cells, places=6)

  def test_compute_us_level_replicate_errors_keeps_voting_age_in_detailed(
      self,
  ):
    # An error of +1 on voting-age slice 0 and -1 on slice 1 cancels exactly
    # under any marginal that sums out the voting-age axis, so only queries
    # that keep that axis (VOTINGAGE, DETAILED, AGExHISPxCENRACE) see it.
    gt_dir = os.path.join(self.temp_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    gt_file = os.path.join(gt_dir, 'ground_truth.parquet')
    gt_arr = np.zeros(common.US_DETAILED_SHAPE, dtype=np.float64)
    pd.DataFrame({
        'query_name': ['detailed_dpq'],
        'value': [gt_arr.flatten()],
    }).to_parquet(gt_file)

    est = np.zeros(common.US_DETAILED_SHAPE, dtype=np.float64)
    est[:, 0, :, :] = 1.0
    est[:, 1, :, :] = -1.0
    rep_dir = os.path.join(self.temp_dir, '0', 'US')
    os.makedirs(rep_dir, exist_ok=True)
    for fname in ('optimized_estimate.parquet', 'baseline_estimate.parquet'):
      pd.DataFrame({'value': [est.flatten()]}).to_parquet(
          os.path.join(rep_dir, fname)
      )

    us_df = data_loader._compute_us_level_replicate_errors(
        base_dir=self.temp_dir, gt_parquet_path=gt_file, num_replicates=1
    ).set_index('query')

    # Every cell is off by exactly 1, so DETAILED sees the full cell count.
    n_cells = float(np.prod(common.US_DETAILED_SHAPE))
    self.assertEqual(us_df.loc[common.QUERY_DETAILED, 'optimized'], [n_cells])
    # VOTINGAGE sums 1008 same-signed cells into each of its 2 entries.
    self.assertEqual(us_df.loc[common.QUERY_VOTINGAGE, 'optimized'], [n_cells])
    # AGExHISPxCENRACE sums the 8 housing types into each of its 252 entries.
    self.assertEqual(
        us_df.loc[common.QUERY_AGExHISPxCENRACE, 'optimized'],
        [8.0 * 2 * 2 * 63],
    )
    for query in (
        common.QUERY_TOTAL,
        common.QUERY_HOUSING_TYPE,
        common.QUERY_HISPANIC,
        common.QUERY_CENRACE,
        common.QUERY_HISPANICxCENRACE,
    ):
      self.assertEqual(us_df.loc[query, 'optimized'], [0.0], msg=query)

  def test_load_summary_replicate_errors(self):
    num_reps = 2
    for rep in range(num_reps):
      rep_dir = os.path.join(self.temp_dir, str(rep), 'US')
      os.makedirs(rep_dir, exist_ok=True)
      df = pd.DataFrame({
          'level': ['County', 'County'],
          'query': ['TOTAL', 'VOTINGAGE'],
          'baseline': [10.0, 20.0],
          'optimized': [8.0, 15.0],
      })
      df.to_parquet(os.path.join(rep_dir, 'query_errors.parquet'))

    summary_df = data_loader.load_summary_replicate_errors(
        base_dir=self.temp_dir, num_replicates=num_reps
    )
    self.assertFalse(summary_df.empty)
    self.assertIn('baseline', summary_df.columns)
    self.assertIn('optimized', summary_df.columns)

    county_total = summary_df[
        (summary_df['level'] == 'County') & (summary_df['query'] == 'TOTAL')
    ]
    self.assertEqual(county_total['baseline'].iloc[0], [10.0, 10.0])
    self.assertEqual(county_total['optimized'].iloc[0], [8.0, 8.0])

  def test_load_detailed_query_errors(self):
    num_reps = 2
    for rep in range(num_reps):
      rep_dir = os.path.join(self.temp_dir, str(rep), 'US')
      os.makedirs(rep_dir, exist_ok=True)
      df = pd.DataFrame({
          'level': ['County'],
          'query': ['Race_alone_0'],
          'baseline': [5.0],
          'optimized': [4.0],
      })
      df.to_parquet(os.path.join(rep_dir, 'alternate_query_errors.parquet'))

    detailed_df = data_loader.load_detailed_query_errors(
        base_dir=self.temp_dir, num_replicates=num_reps
    )
    self.assertFalse(detailed_df.empty)
    self.assertEqual(detailed_df['baseline'].iloc[0], [5.0, 5.0])

  def test_load_population_binned_data(self):
    num_reps = 1
    state_id = '001'
    level = 'County'
    level_dir = os.path.join(self.temp_dir, '0', f'State={state_id}', level)
    os.makedirs(level_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            'base_bias': ['[10.0, 20.0]'],
            'opt_bias': ['[5.0, 10.0]'],
            'ground_truth': ['[1000.0, 6000.0]'],
        },
        index=['TOTAL'],
    )
    df.to_parquet(os.path.join(level_dir, 'query_errors_list.parquet'))

    binned_df = data_loader.load_population_binned_data(
        base_dir=self.temp_dir,
        levels=[level],
        state_ids=[state_id],
        num_replicates=num_reps,
    )
    # One row per (metric, replicate, bin): 2 metrics x 1 replicate x 5 bins.
    self.assertLen(binned_df, 2 * len(common.BIAS_LABELS[level]))
    bias_df = binned_df[binned_df['metric'] == common.METRIC_BIAS]
    self.assertEqual(list(bias_df['population_bin']), common.BIAS_LABELS[level])
    # Units with population 1000 and 6000 fall in the first two bins.
    self.assertEqual(list(bias_df['topdown_value']), [10.0, 20.0, 0, 0, 0])
    self.assertEqual(list(bias_df['bluedown_value']), [5.0, 10.0, 0, 0, 0])

  def test_load_sliced_query_errors(self):
    num_reps = 1
    state_id = '001'
    level = 'County'
    level_dir = os.path.join(self.temp_dir, '0', f'State={state_id}', level)
    os.makedirs(level_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            'baseline': ['[1.0, 2.0]'],
            'optimized': ['[0.5, 1.5]'],
            'n': [2],
        },
        index=['TOTAL'],
    )
    df.to_parquet(os.path.join(level_dir, 'query_errors_list.parquet'))

    sliced_errs = data_loader.load_sliced_query_errors(
        base_dir=self.temp_dir,
        levels=[level],
        state_ids=[state_id],
        num_replicates=num_reps,
    )
    self.assertIn(level, sliced_errs)
    self.assertEqual(sliced_errs[level].loc['TOTAL', 'baseline'], [1.0, 2.0])
    self.assertEqual(sliced_errs[level].loc['TOTAL', 'optimized'], [0.5, 1.5])

  def test_histogram_query_errors_rounds_and_orders_queries(self):
    # Index is in reverse RIDGELINE_QUERY_ORDER to verify reordering.
    level_df = pd.DataFrame(
        {
            common.COL_OPTIMIZED: [[0.2], [1.4, 1.6, 2.2]],
            common.COL_BASELINE: [[1.1], [0.9, 3.1, 3.4]],
        },
        index=[common.QUERY_VOTINGAGE, common.QUERY_TOTAL],
    )

    hist_df = data_loader.histogram_query_errors(common.LEVEL_COUNTY, level_df)

    expected = pd.DataFrame({
        common.COL_LEVEL: [common.LEVEL_COUNTY] * 5,
        common.COL_QUERY: [
            common.QUERY_TOTAL,
            common.QUERY_TOTAL,
            common.QUERY_TOTAL,
            common.QUERY_VOTINGAGE,
            common.QUERY_VOTINGAGE,
        ],
        common.COL_ERROR_VALUE: [1, 2, 3, 0, 1],
        common.COL_BLUEDOWN_COUNT: [1, 2, 0, 1, 0],
        common.COL_TOPDOWN_COUNT: [1, 0, 2, 0, 1],
    })
    pd.testing.assert_frame_equal(hist_df, expected)

  def test_histogram_query_errors_omits_total_for_trivial_total_levels(self):
    level_df = pd.DataFrame(
        {
            common.COL_OPTIMIZED: [[0.0], [1.2]],
            common.COL_BASELINE: [[0.0], [2.2]],
        },
        index=[common.QUERY_TOTAL, common.QUERY_VOTINGAGE],
    )

    hist_df = data_loader.histogram_query_errors(common.LEVEL_STATE, level_df)

    self.assertEqual(
        list(hist_df[common.COL_QUERY]),
        [
            common.QUERY_VOTINGAGE,
            common.QUERY_VOTINGAGE,
        ],
    )


if __name__ == '__main__':
  absltest.main()
