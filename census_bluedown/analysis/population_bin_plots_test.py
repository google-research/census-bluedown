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
from census_bluedown.analysis import population_bin_plots
from absl.testing import absltest

# Bins are deliberately listed out of BIAS_LABELS order so the test verifies
# that tick labels follow first appearance in the frame, not sorted order
# (lexicographic sorting would put '<10000' before '<5000').
_BINNED_DF = pd.DataFrame(
    [
        (common.LEVEL_COUNTY, common.METRIC_BIAS, 0, '<10000', 0.3, -2.4),
        (common.LEVEL_COUNTY, common.METRIC_BIAS, 1, '<10000', 0.4, -2.5),
        (common.LEVEL_COUNTY, common.METRIC_BIAS, 0, '<5000', -0.5, 1.2),
        (common.LEVEL_COUNTY, common.METRIC_BIAS, 1, '<5000', -0.6, 1.3),
        (common.LEVEL_TRACT, common.METRIC_BIAS, 0, '<500', -0.1, 0.8),
    ],
    columns=[
        common.COL_LEVEL,
        common.COL_METRIC,
        common.COL_REPLICATE,
        common.COL_POPULATION_BIN,
        common.COL_BLUEDOWN_VALUE,
        common.COL_TOPDOWN_VALUE,
    ],
)


class PlotPopulationBinnedTest(absltest.TestCase):

  def test_tick_labels_follow_bin_order_in_frame(self):
    _, ax = population_bin_plots.plot_population_binned(
        _BINNED_DF, common.LEVEL_COUNTY, common.METRIC_BIAS
    )
    self.assertEqual(
        [t.get_text() for t in ax.xaxis.get_ticklabels()],
        ['<10000', '<5000'],
    )
    self.assertEqual(ax.get_xlabel(), 'size of county')

  def test_unsupported_metric_raises(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported metric'):
      population_bin_plots.plot_population_binned(
          _BINNED_DF, common.LEVEL_COUNTY, 'no_such_metric'
      )

  def test_empty_selection_raises(self):
    with self.assertRaisesRegex(ValueError, 'No rows.*Block'):
      population_bin_plots.plot_population_binned(
          _BINNED_DF, common.LEVEL_BLOCK, common.METRIC_BIAS
      )


if __name__ == '__main__':
  absltest.main()
