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
import textwrap

import pandas as pd

from census_bluedown.analysis import common
from census_bluedown.analysis import csv_utils
from absl.testing import absltest

# One row per (level, query) with list-valued replicate errors. Queries are in
# reverse standard order so that the exporter's reordering is exercised.
_SUMMARY_ROWS = [
    # (query, optimized, baseline)
    (common.QUERY_DETAILED, [6.1, 6.2], [61.0, 62.0]),
    (common.QUERY_HISPANICxCENRACE, [5.1, 5.2], [51.0, 52.0]),
    (common.QUERY_CENRACE, [4.1, 4.2], [41.0, 42.0]),
    (common.QUERY_HISPANIC, [3.1, 3.2], [31.0, 32.0]),
    (common.QUERY_VOTINGAGE, [2.1, 2.2], [21.0, 22.0]),
    (common.QUERY_HOUSING_TYPE, [1.1, 1.2], [11.0, 12.0]),
    (common.QUERY_TOTAL, [0.1, 0.2], [1.0, 2.0]),
]

# The same data as the exporter should write it: one row per replicate, queries
# in standard order, TOTAL first (County) or omitted (State).
_EXPECTED_COUNTY_CSV = textwrap.dedent("""\
    level,query,replicate,bluedown_error,topdown_error
    County,TOTAL,0,0.1,1.0
    County,TOTAL,1,0.2,2.0
    County,HOUSING_TYPE,0,1.1,11.0
    County,HOUSING_TYPE,1,1.2,12.0
    County,VOTINGAGE,0,2.1,21.0
    County,VOTINGAGE,1,2.2,22.0
    County,HISPANIC,0,3.1,31.0
    County,HISPANIC,1,3.2,32.0
    County,CENRACE,0,4.1,41.0
    County,CENRACE,1,4.2,42.0
    County,HISPANICxCENRACE,0,5.1,51.0
    County,HISPANICxCENRACE,1,5.2,52.0
    County,DETAILED,0,6.1,61.0
    County,DETAILED,1,6.2,62.0
    """)

_EXPECTED_STATE_CSV = textwrap.dedent("""\
    level,query,replicate,bluedown_error,topdown_error
    State,HOUSING_TYPE,0,1.1,11.0
    State,HOUSING_TYPE,1,1.2,12.0
    State,VOTINGAGE,0,2.1,21.0
    State,VOTINGAGE,1,2.2,22.0
    State,HISPANIC,0,3.1,31.0
    State,HISPANIC,1,3.2,32.0
    State,CENRACE,0,4.1,41.0
    State,CENRACE,1,4.2,42.0
    State,HISPANICxCENRACE,0,5.1,51.0
    State,HISPANICxCENRACE,1,5.2,52.0
    State,DETAILED,0,6.1,61.0
    State,DETAILED,1,6.2,62.0
    """)


def _summary_df(level: str) -> pd.DataFrame:
  """Builds the replicate summary DataFrame from _SUMMARY_ROWS."""
  queries, optimized, baseline = zip(*_SUMMARY_ROWS)
  return pd.DataFrame({
      common.COL_LEVEL: [level] * len(queries),
      common.COL_QUERY: list(queries),
      common.COL_OPTIMIZED: list(optimized),
      common.COL_BASELINE: list(baseline),
  })


class ExportLevelQueriesCsvTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    temp_dir = self.enter_context(tempfile.TemporaryDirectory())
    self.out_path = os.path.join(temp_dir, 'out.csv')

  def _export(self, level: str) -> str:
    csv_utils.export_level_queries_csv(_summary_df(level), level, self.out_path)
    with open(self.out_path, 'r') as f:
      return f.read()

  def test_county_writes_tidy_rows_in_standard_query_order(self):
    self.assertEqual(self._export(common.LEVEL_COUNTY), _EXPECTED_COUNTY_CSV)

  def test_state_omits_total(self):
    self.assertEqual(self._export(common.LEVEL_STATE), _EXPECTED_STATE_CSV)

  def test_includes_age_hisp_race_query_when_present(self):
    df = pd.concat(
        [
            _summary_df(common.LEVEL_COUNTY),
            pd.DataFrame({
                common.COL_LEVEL: [common.LEVEL_COUNTY],
                common.COL_QUERY: [common.QUERY_AGExHISPxCENRACE],
                common.COL_OPTIMIZED: [[7.1, 7.2]],
                common.COL_BASELINE: [[71.0, 72.0]],
            }),
        ],
        ignore_index=True,
    )
    csv_utils.export_level_queries_csv(df, common.LEVEL_COUNTY, self.out_path)
    with open(self.out_path, 'r') as f:
      expected = (
          _EXPECTED_COUNTY_CSV
          + 'County,AGExHISPxCENRACE,0,7.1,71.0\n'
          + 'County,AGExHISPxCENRACE,1,7.2,72.0\n'
      )
      self.assertEqual(f.read(), expected)

  def test_missing_standard_query_raises(self):
    df = _summary_df(common.LEVEL_COUNTY)
    df = df[df[common.COL_QUERY] != common.QUERY_HISPANIC]
    with self.assertRaisesRegex(ValueError, 'HISPANIC.*County'):
      csv_utils.export_level_queries_csv(df, common.LEVEL_COUNTY, self.out_path)


class LoadLevelQueriesDfTest(absltest.TestCase):

  def test_loads_csv_back_into_summary_frame_in_file_order(self):
    temp_dir = self.enter_context(tempfile.TemporaryDirectory())
    csv_path = os.path.join(temp_dir, 'in.csv')
    with open(csv_path, 'w') as f:
      f.write(_EXPECTED_COUNTY_CSV)

    loaded = csv_utils.load_level_queries_df(csv_path)

    # Rows come back in the CSV's (standard) order, i.e. _SUMMARY_ROWS reversed.
    expected = (
        _summary_df(common.LEVEL_COUNTY).iloc[::-1].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(loaded, expected)


_DETAILED_SPEC = common.DetailedQuerySpec(
    name='Grouped GQ, County',
    level=common.LEVEL_COUNTY,
    measurement_names=('Institutionalized', 'Noninstitutionalized'),
    query_keys=('GQ_0', 'GQ_1'),
    x_label='Grouped GQ population, county',
)

_EXPECTED_DETAILED_CSV = textwrap.dedent("""\
    level,benchmark,measurement,query_key,replicate,bluedown_error,topdown_error
    County,"Grouped GQ, County",Institutionalized,GQ_0,0,1.5,15.0
    County,"Grouped GQ, County",Institutionalized,GQ_0,1,2.5,25.0
    County,"Grouped GQ, County",Noninstitutionalized,GQ_1,0,3.5,35.0
    County,"Grouped GQ, County",Noninstitutionalized,GQ_1,1,4.5,45.0
    """)


def _detailed_df() -> pd.DataFrame:
  """Detailed query errors with keys in reverse spec order."""
  return pd.DataFrame({
      common.COL_LEVEL: [common.LEVEL_COUNTY, common.LEVEL_COUNTY],
      common.COL_QUERY: ['GQ_1', 'GQ_0'],
      common.COL_OPTIMIZED: [[3.5, 4.5], [1.5, 2.5]],
      common.COL_BASELINE: [[35.0, 45.0], [15.0, 25.0]],
  })


class DetailedQueriesCsvTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    temp_dir = self.enter_context(tempfile.TemporaryDirectory())
    self.csv_path = os.path.join(temp_dir, 'detail.csv')

  def test_exports_tidy_rows_in_spec_order(self):
    csv_utils.export_detailed_queries_csv(
        _detailed_df(), _DETAILED_SPEC, self.csv_path
    )
    with open(self.csv_path, 'r') as f:
      self.assertEqual(f.read(), _EXPECTED_DETAILED_CSV)

  def test_loads_csv_back_into_summary_frame_and_spec(self):
    with open(self.csv_path, 'w') as f:
      f.write(_EXPECTED_DETAILED_CSV)

    loaded_df, loaded_spec = csv_utils.load_detailed_queries_csv(self.csv_path)

    expected_df = _detailed_df().iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded_df, expected_df)
    self.assertEqual(
        loaded_spec,
        common.DetailedQuerySpec(
            name='Grouped GQ, County',
            level=common.LEVEL_COUNTY,
            measurement_names=('Institutionalized', 'Noninstitutionalized'),
            query_keys=('GQ_0', 'GQ_1'),
            x_label=None,
        ),
    )


_EXPECTED_COUNTY_BIAS_CSV = textwrap.dedent("""\
    level,metric,replicate,population_bin,bluedown_value,topdown_value
    County,bias,0,<5000,-0.5,1.2
    County,bias,0,<10000,0.3,-2.4
    """)


def _population_binned_df() -> pd.DataFrame:
  """Population-binned frame spanning multiple levels and metrics."""
  return pd.DataFrame({
      common.COL_LEVEL: [
          common.LEVEL_COUNTY,
          common.LEVEL_COUNTY,
          common.LEVEL_COUNTY,
          common.LEVEL_TRACT,
      ],
      common.COL_METRIC: [
          common.METRIC_BIAS,
          common.METRIC_BIAS,
          common.METRIC_BINNED_ERROR,
          common.METRIC_BIAS,
      ],
      common.COL_REPLICATE: [0, 0, 0, 0],
      common.COL_POPULATION_BIN: ['<5000', '<10000', '<5000', '<500'],
      common.COL_BLUEDOWN_VALUE: [-0.5, 0.3, 0.5, -0.1],
      common.COL_TOPDOWN_VALUE: [1.2, -2.4, 1.2, 0.8],
  })


class PopulationBinnedCsvTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    temp_dir = self.enter_context(tempfile.TemporaryDirectory())
    self.csv_path = os.path.join(temp_dir, 'binned.csv')

  def test_exports_only_matching_level_and_metric(self):
    csv_utils.export_population_binned_data_csv(
        _population_binned_df(),
        common.LEVEL_COUNTY,
        common.METRIC_BIAS,
        self.csv_path,
    )
    with open(self.csv_path, 'r') as f:
      self.assertEqual(f.read(), _EXPECTED_COUNTY_BIAS_CSV)

  def test_loads_csv_back_into_tidy_frame(self):
    with open(self.csv_path, 'w') as f:
      f.write(_EXPECTED_COUNTY_BIAS_CSV)

    loaded = csv_utils.load_population_binned_data_csv(self.csv_path)

    expected = _population_binned_df().iloc[:2].reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded, expected)


_EXPECTED_RIDGELINE_CSV = textwrap.dedent("""\
    level,query,error_value,bluedown_count,topdown_count
    County,TOTAL,0,5,1
    County,TOTAL,1,2,4
    County,VOTINGAGE,2,3,6
    """)


def _ridgeline_hist_df() -> pd.DataFrame:
  """Tidy integer-error histogram frame."""
  return pd.DataFrame({
      common.COL_LEVEL: [
          common.LEVEL_COUNTY,
          common.LEVEL_COUNTY,
          common.LEVEL_COUNTY,
      ],
      common.COL_QUERY: [
          common.QUERY_TOTAL,
          common.QUERY_TOTAL,
          common.QUERY_VOTINGAGE,
      ],
      common.COL_ERROR_VALUE: [0, 1, 2],
      common.COL_BLUEDOWN_COUNT: [5, 2, 3],
      common.COL_TOPDOWN_COUNT: [1, 4, 6],
  })


class RidgelineHistogramsCsvTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    temp_dir = self.enter_context(tempfile.TemporaryDirectory())
    self.csv_path = os.path.join(temp_dir, 'ridge.csv')

  def test_exports_tidy_histogram_csv(self):
    csv_utils.export_ridgeline_histograms_csv(
        _ridgeline_hist_df(), self.csv_path
    )
    with open(self.csv_path, 'r') as f:
      self.assertEqual(f.read(), _EXPECTED_RIDGELINE_CSV)

  def test_loads_csv_back_into_histogram_frame(self):
    with open(self.csv_path, 'w') as f:
      f.write(_EXPECTED_RIDGELINE_CSV)

    loaded = csv_utils.load_ridgeline_histograms_csv(self.csv_path)

    pd.testing.assert_frame_equal(loaded, _ridgeline_hist_df())


if __name__ == '__main__':
  absltest.main()
