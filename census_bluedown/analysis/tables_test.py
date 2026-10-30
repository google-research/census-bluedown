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

import pandas as pd
from census_bluedown.analysis import common
from census_bluedown.analysis import tables
from absl.testing import absltest


class ComputeSummaryStatsTest(absltest.TestCase):

  def test_computes_means_stds_and_p_values_in_query_level_order(self):
    summary_df = pd.DataFrame({
        common.COL_LEVEL: [
            common.LEVEL_TRACT,
            common.LEVEL_COUNTY,
        ],
        common.COL_QUERY: [
            common.QUERY_VOTINGAGE,
            common.QUERY_VOTINGAGE,
        ],
        common.COL_OPTIMIZED: [
            [5.0],
            [1.0, 3.0],
        ],
        common.COL_BASELINE: [
            [10.0],
            [11.0, 13.0],
        ],
    })

    stats_df = tables.compute_summary_stats(
        summary_df,
        queries=[common.QUERY_VOTINGAGE],
        levels=[common.LEVEL_COUNTY, common.LEVEL_TRACT],
    )

    self.assertEqual(
        list(stats_df[common.COL_GEOGRAPHIC_LEVEL]),
        [common.LEVEL_COUNTY, common.LEVEL_TRACT],
    )
    self.assertAlmostEqual(stats_df[common.COL_BLUEDOWN_MEAN].iloc[0], 2.0)
    self.assertAlmostEqual(stats_df[common.COL_BLUEDOWN_STD].iloc[0], 1.0)
    self.assertAlmostEqual(stats_df[common.COL_TOPDOWN_MEAN].iloc[0], 12.0)
    self.assertAlmostEqual(stats_df[common.COL_TOPDOWN_STD].iloc[0], 1.0)
    # Single replicate at Tract has no paired t-test p-value.
    self.assertEqual(stats_df[common.COL_P_VALUE].iloc[1], 'N/A')

  def test_formats_small_p_value_below_floor(self):
    summary_df = pd.DataFrame({
        common.COL_LEVEL: [common.LEVEL_COUNTY],
        common.COL_QUERY: [common.QUERY_VOTINGAGE],
        common.COL_OPTIMIZED: [[1.0, 1.1, 0.9, 1.0, 1.0]],
        common.COL_BASELINE: [[100.0, 100.2, 99.8, 100.1, 99.9]],
    })

    stats_df = tables.compute_summary_stats(
        summary_df,
        queries=[common.QUERY_VOTINGAGE],
        levels=[common.LEVEL_COUNTY],
    )

    self.assertEqual(stats_df[common.COL_P_VALUE].iloc[0], '<0.0001')

  def test_identical_replicates_return_not_available_p_value(self):
    summary_df = pd.DataFrame({
        common.COL_LEVEL: [common.LEVEL_COUNTY],
        common.COL_QUERY: [common.QUERY_VOTINGAGE],
        common.COL_OPTIMIZED: [[5.0, 5.0, 5.0]],
        common.COL_BASELINE: [[5.0, 5.0, 5.0]],
    })

    stats_df = tables.compute_summary_stats(
        summary_df,
        queries=[common.QUERY_VOTINGAGE],
        levels=[common.LEVEL_COUNTY],
    )

    self.assertEqual(stats_df[common.COL_P_VALUE].iloc[0], 'N/A')

  def test_zeros_total_for_trivial_total_levels_even_without_total_row(self):
    # State has only VOTINGAGE (as when loaded from State_queries.csv, which
    # omits TOTAL); County has TOTAL with non-zero errors.
    summary_df = pd.DataFrame({
        common.COL_LEVEL: [common.LEVEL_STATE, common.LEVEL_COUNTY],
        common.COL_QUERY: [common.QUERY_VOTINGAGE, common.QUERY_TOTAL],
        common.COL_OPTIMIZED: [[2.0, 4.0], [6.0, 8.0]],
        common.COL_BASELINE: [[20.0, 40.0], [60.0, 80.0]],
    })

    stats_df = tables.compute_summary_stats(
        summary_df,
        queries=[common.QUERY_TOTAL],
        levels=[common.LEVEL_STATE, common.LEVEL_COUNTY],
    )

    expected = pd.DataFrame({
        common.COL_GEOGRAPHIC_LEVEL: [common.LEVEL_STATE, common.LEVEL_COUNTY],
        common.COL_QUERY: [common.QUERY_TOTAL, common.QUERY_TOTAL],
        common.COL_BLUEDOWN_MEAN: [0.0, 7.0],
        common.COL_BLUEDOWN_STD: [0.0, 1.0],
        common.COL_TOPDOWN_MEAN: [0.0, 70.0],
        common.COL_TOPDOWN_STD: [0.0, 10.0],
        common.COL_P_VALUE: ['', '0.0903'],
    })
    pd.testing.assert_frame_equal(stats_df, expected)


_STATS_DF = pd.DataFrame({
    common.COL_GEOGRAPHIC_LEVEL: [common.LEVEL_STATE, common.LEVEL_COUNTY],
    common.COL_QUERY: [common.QUERY_TOTAL, common.QUERY_VOTINGAGE],
    common.COL_BLUEDOWN_MEAN: [0.0, 1.25],
    common.COL_BLUEDOWN_STD: [0.0, 0.5],
    common.COL_TOPDOWN_MEAN: [0.0, 12.3],
    common.COL_TOPDOWN_STD: [0.0, 2.0],
    common.COL_P_VALUE: ['', '<0.0001'],
})

_EXPECTED_LATEX_WITH_P = (
    r'\begin{tabular}{|ll|r@{\hspace{1em}$\pm$\,}r|r@{\hspace{1em}$\pm$\,}r|r|}'
    '\n'
    r'\hline'
    '\n'
    r'Geographic level & Query & \multicolumn{2}{c|}{BlueDown error} &'
    r' \multicolumn{2}{c|}{TopDown error} & p-value \\'
    '\n'
    r'\hline'
    '\n'
    r'State & Total & 0 & 0 & 0 & 0 &  \\'
    '\n'
    r'County & Voting age & 1.25 & 0.50 & 12.30 & 2.00 & $<0.0001$ \\'
    '\n'
    r'\hline'
    '\n'
    r'\end{tabular}'
)

_EXPECTED_LATEX_WITHOUT_P = (
    r'\begin{tabular}{|ll|r@{\hspace{1em}$\pm$\,}r|r@{\hspace{1em}$\pm$\,}r|}'
    '\n'
    r'\hline'
    '\n'
    r'Geographic level & Query & \multicolumn{2}{c|}{BlueDown error} &'
    r' \multicolumn{2}{c|}{TopDown error} \\'
    '\n'
    r'\hline'
    '\n'
    r'State & Total & 0 & 0 & 0 & 0 \\'
    '\n'
    r'County & Voting age & 1.25 & 0.50 & 12.30 & 2.00 \\'
    '\n'
    r'\hline'
    '\n'
    r'\end{tabular}'
)


class RenderLatexTableTest(absltest.TestCase):

  def test_renders_tabular_with_pvalues(self):
    self.assertEqual(
        tables.render_latex_table(_STATS_DF, include_pvalues=True),
        _EXPECTED_LATEX_WITH_P,
    )

  def test_renders_tabular_without_pvalues(self):
    self.assertEqual(
        tables.render_latex_table(_STATS_DF, include_pvalues=False),
        _EXPECTED_LATEX_WITHOUT_P,
    )


if __name__ == '__main__':
  absltest.main()
