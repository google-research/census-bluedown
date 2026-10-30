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

from collections.abc import Sequence
import os

import tempfile
import pandas as pd

from census_bluedown.analysis import common
from census_bluedown.analysis import plot_pipeline
from absl.testing import absltest

_SUMMARY_DF = pd.DataFrame(
    [
        (common.LEVEL_COUNTY, common.QUERY_TOTAL, [0.1, 0.2], [1.0, 2.0]),
        (
            common.LEVEL_COUNTY,
            common.QUERY_HOUSING_TYPE,
            [1.1, 1.2],
            [11.0, 12.0],
        ),
        (common.LEVEL_COUNTY, common.QUERY_VOTINGAGE, [2.1, 2.2], [21.0, 22.0]),
        (common.LEVEL_COUNTY, common.QUERY_HISPANIC, [3.1, 3.2], [31.0, 32.0]),
        (common.LEVEL_COUNTY, common.QUERY_CENRACE, [4.1, 4.2], [41.0, 42.0]),
        (
            common.LEVEL_COUNTY,
            common.QUERY_HISPANICxCENRACE,
            [5.1, 5.2],
            [51.0, 52.0],
        ),
        (common.LEVEL_COUNTY, common.QUERY_DETAILED, [6.1, 6.2], [61.0, 62.0]),
        (
            common.LEVEL_COUNTY,
            common.QUERY_AGExHISPxCENRACE,
            [7.1, 7.2],
            [71.0, 72.0],
        ),
    ],
    columns=[
        common.COL_LEVEL,
        common.COL_QUERY,
        common.COL_OPTIMIZED,
        common.COL_BASELINE,
    ],
)

# standard_detailed_query_specs() defines 4 specs for County (23 query_keys in
# total). To keep the literal test fixture compact, _DETAILED_DF populates only
# the 2-row grouped-GQ spec; _FakeSource.detailed_df returns None for the other
# 3 County specs so plot_pipeline.generate() skips them.
_GROUPED_GQ_SPEC_NAME = (
    'Institutionalized and non-institutionalized GQ population, County'
)

_DETAILED_DF = pd.DataFrame(
    [
        (common.LEVEL_COUNTY, 'GQTYPE_GROUPED_0', [1.5, 2.5], [15.0, 25.0]),
        (common.LEVEL_COUNTY, 'GQTYPE_GROUPED_1', [3.5, 4.5], [35.0, 45.0]),
    ],
    columns=[
        common.COL_LEVEL,
        common.COL_QUERY,
        common.COL_OPTIMIZED,
        common.COL_BASELINE,
    ],
)

_POPULATION_BINNED_DF = pd.DataFrame(
    [
        (common.LEVEL_COUNTY, common.METRIC_BIAS, 0, '<5000', -0.5, 1.2),
        (common.LEVEL_COUNTY, common.METRIC_BIAS, 1, '<5000', 0.3, -2.4),
        (
            common.LEVEL_COUNTY,
            common.METRIC_BINNED_ERROR,
            0,
            '<5000',
            0.5,
            1.2,
        ),
        (
            common.LEVEL_COUNTY,
            common.METRIC_BINNED_ERROR,
            1,
            '<5000',
            0.3,
            2.4,
        ),
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

_RIDGELINE_HIST_DF = pd.DataFrame(
    [
        (common.LEVEL_COUNTY, common.QUERY_TOTAL, 0, 5, 3),
        (common.LEVEL_COUNTY, common.QUERY_TOTAL, 1, 2, 4),
    ],
    columns=[
        common.COL_LEVEL,
        common.COL_QUERY,
        common.COL_ERROR_VALUE,
        common.COL_BLUEDOWN_COUNT,
        common.COL_TOPDOWN_COUNT,
    ],
)


class _FakeSource:
  """In-memory PlotSource backed by literal test DataFrames."""

  def summary_df(self) -> pd.DataFrame | None:
    return _SUMMARY_DF

  def detailed_df(self, spec: common.DetailedQuerySpec) -> pd.DataFrame | None:
    if spec.name != _GROUPED_GQ_SPEC_NAME:
      return None
    return _DETAILED_DF

  def population_binned_df(self, levels: Sequence[str]) -> pd.DataFrame | None:
    if common.LEVEL_COUNTY not in levels:
      return None
    return _POPULATION_BINNED_DF

  def ridgeline_hist(self, level: str) -> pd.DataFrame | None:
    if level != common.LEVEL_COUNTY:
      return None
    return _RIDGELINE_HIST_DF


class PlotPipelineRoundTripTest(absltest.TestCase):

  def test_csv_export_and_csv_source_round_trip(self):
    raw_dir = self.enter_context(tempfile.TemporaryDirectory())
    csv_out_dir = self.enter_context(tempfile.TemporaryDirectory())
    levels = [common.LEVEL_COUNTY]

    plot_pipeline.generate(
        source=_FakeSource(),
        output_dir=raw_dir,
        plot_types={common.FLAG_ALL},
        levels=levels,
        formats=[common.FORMAT_CSV],
    )

    csv_source = plot_pipeline.CsvSource(raw_dir)
    pd.testing.assert_frame_equal(csv_source.summary_df(), _SUMMARY_DF)
    gq_spec = next(
        s
        for s in common.standard_detailed_query_specs()
        if s.name == _GROUPED_GQ_SPEC_NAME
    )
    pd.testing.assert_frame_equal(csv_source.detailed_df(gq_spec), _DETAILED_DF)
    pd.testing.assert_frame_equal(
        csv_source.population_binned_df(levels),
        _POPULATION_BINNED_DF,
    )
    pd.testing.assert_frame_equal(
        csv_source.ridgeline_hist(common.LEVEL_COUNTY),
        _RIDGELINE_HIST_DF,
    )

    # Re-generating tables from CsvSource produces identical LaTeX output.
    plot_pipeline.generate(
        source=csv_source,
        output_dir=csv_out_dir,
        plot_types={common.PLOT_TABLES},
        levels=levels,
        formats=[],
    )
    for index in (1, 2):
      stem = f'{common.SUMMARY_TABLE_STEM.format(index=index)}.tex'
      with open(os.path.join(raw_dir, common.TABLES_DIR, stem)) as f1:
        raw_tex = f1.read()
      with open(os.path.join(csv_out_dir, common.TABLES_DIR, stem)) as f2:
        csv_tex = f2.read()
      self.assertEqual(csv_tex, raw_tex)


if __name__ == '__main__':
  absltest.main()
