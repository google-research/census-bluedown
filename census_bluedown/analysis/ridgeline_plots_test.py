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
from census_bluedown.analysis import common
from census_bluedown.analysis import ridgeline_plots
from absl.testing import absltest

_GRID = np.linspace(-50.0, 50.0, 2001)
_GRID_STEP = _GRID[1] - _GRID[0]


class WeightedKdeTest(absltest.TestCase):

  def test_zero_weights_give_zero_density(self):
    density = ridgeline_plots._weighted_kde(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]), _GRID
    )
    np.testing.assert_array_equal(density, np.zeros_like(_GRID))

  def test_positive_weights_integrate_to_one(self):
    density = ridgeline_plots._weighted_kde(
        np.array([0.0, 1.0, 2.0]), np.array([5.0, 2.0, 1.0]), _GRID
    )
    self.assertAlmostEqual(float(np.sum(density) * _GRID_STEP), 1.0, places=3)

  def test_density_peaks_near_heaviest_value(self):
    density = ridgeline_plots._weighted_kde(
        np.array([-10.0, 10.0]), np.array([1.0, 100.0]), _GRID
    )
    self.assertAlmostEqual(float(_GRID[np.argmax(density)]), 10.0, delta=1.0)

  def test_degenerate_single_value_returns_zeros(self):
    # A single distinct value has zero variance, which gaussian_kde rejects.
    density = ridgeline_plots._weighted_kde(
        np.array([3.0]), np.array([7.0]), _GRID
    )
    np.testing.assert_array_equal(density, np.zeros_like(_GRID))


class PlotLevelRidgelineTest(absltest.TestCase):

  def test_one_row_per_query_in_frame_order_with_relabels(self):
    hist_df = pd.DataFrame(
        [
            (common.LEVEL_COUNTY, common.QUERY_VOTINGAGE, 0, 5, 3),
            (common.LEVEL_COUNTY, common.QUERY_VOTINGAGE, 1, 2, 4),
            (common.LEVEL_COUNTY, common.QUERY_TOTAL, 0, 1, 1),
            (common.LEVEL_COUNTY, common.QUERY_TOTAL, 2, 3, 2),
        ],
        columns=[
            common.COL_LEVEL,
            common.COL_QUERY,
            common.COL_ERROR_VALUE,
            common.COL_BLUEDOWN_COUNT,
            common.COL_TOPDOWN_COUNT,
        ],
    )

    _, axes = ridgeline_plots.plot_level_ridgeline(hist_df, common.LEVEL_COUNTY)

    self.assertLen(axes, 2)
    labels = [ax.texts[0].get_text() for ax in axes]
    self.assertEqual(labels, [common.LABEL_VOTINGAGE, common.LABEL_TOTAL])
    self.assertEqual(
        axes[-1].get_xlim(), common.RIDGELINE_X_BOUNDS[common.LEVEL_COUNTY]
    )

  def test_negative_x_bounds_and_empty_groups(self):
    _, axes = ridgeline_plots.plot_ridgeline_dual_series(
        level=common.LEVEL_COUNTY,
        values=[np.array([-10.0, -5.0])],
        weights_a=[np.array([3.0, 2.0])],
        weights_b=[np.array([1.0, 4.0])],
        group_labels=[common.LABEL_TOTAL],
        x_min=-20.0,
        x_max=-2.0,
    )
    self.assertEqual(axes[-1].get_xlim(), (-20.0, -2.0))

    _, empty_axes = ridgeline_plots.plot_ridgeline_dual_series(
        level=common.LEVEL_COUNTY,
        values=[],
        weights_a=[],
        weights_b=[],
        group_labels=[],
    )
    self.assertLen(empty_axes, 1)


if __name__ == '__main__':
  absltest.main()
