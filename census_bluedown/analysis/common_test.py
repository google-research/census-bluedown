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
from absl.testing import absltest


class UsQueriesAndShapesTest(absltest.TestCase):

  def test_shapes_are_marginals_of_the_detailed_shape(self):
    # Each axis is either summed out (1) or kept at its full extent; any other
    # value would silently compute the wrong marginal.
    for query, shape in common.US_QUERIES_AND_SHAPES.items():
      self.assertLen(shape, len(common.US_DETAILED_SHAPE), msg=query)
      for dim, full in zip(shape, common.US_DETAILED_SHAPE):
        self.assertIn(dim, (1, full), msg=query)

  def test_detailed_query_keeps_every_axis(self):
    self.assertEqual(
        common.US_QUERIES_AND_SHAPES[common.QUERY_DETAILED],
        common.US_DETAILED_SHAPE,
    )


def _summary_df() -> pd.DataFrame:
  """Two levels, queries deliberately out of STANDARD_LEVEL_QUERIES order."""
  return pd.DataFrame({
      common.COL_LEVEL: [
          common.LEVEL_COUNTY,
          common.LEVEL_COUNTY,
          common.LEVEL_TRACT,
      ],
      common.COL_QUERY: [
          common.QUERY_VOTINGAGE,
          common.QUERY_TOTAL,
          common.QUERY_TOTAL,
      ],
      common.COL_OPTIMIZED: [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0],
      ],
      common.COL_BASELINE: [
          [10.0, 20.0, 30.0],
          [40.0, 50.0, 60.0],
          [70.0, 80.0, 90.0],
      ],
  })


class ResolveLevelsTest(absltest.TestCase):

  def test_all_returns_every_level_in_order(self):
    self.assertEqual(
        common.resolve_levels([common.FLAG_ALL]), list(common.ALL_LEVELS)
    )

  def test_subset_is_returned_in_hierarchical_order(self):
    self.assertEqual(
        common.resolve_levels([common.LEVEL_BLOCK, common.LEVEL_COUNTY]),
        [common.LEVEL_COUNTY, common.LEVEL_BLOCK],
    )

  def test_unknown_levels_are_dropped(self):
    self.assertEqual(
        common.resolve_levels([common.LEVEL_COUNTY, 'Galaxy']),
        [common.LEVEL_COUNTY],
    )


class LevelQueriesTest(absltest.TestCase):

  def test_county_returns_all_standard_queries_with_aligned_labels(self):
    queries, labels = common.level_queries(common.LEVEL_COUNTY)
    self.assertEqual(queries, list(common.STANDARD_LEVEL_QUERIES))
    self.assertEqual(labels, list(common.STANDARD_LEVEL_QUERY_LABELS))

  def test_trivial_total_levels_omit_total_only(self):
    for level in common.LEVELS_WITH_TRIVIAL_TOTAL:
      queries, labels = common.level_queries(level)
      self.assertNotIn(common.QUERY_TOTAL, queries)
      self.assertNotIn(common.LABEL_TOTAL, labels)
      self.assertEqual(
          queries,
          [q for q in common.STANDARD_LEVEL_QUERIES if q != common.QUERY_TOTAL],
      )
      self.assertLen(labels, len(queries))


class ExtractQueryErrorsTest(absltest.TestCase):

  def test_returns_errors_in_requested_order_not_dataframe_order(self):
    optimized, baseline = common.extract_query_errors(
        _summary_df(),
        common.LEVEL_COUNTY,
        [common.QUERY_TOTAL, common.QUERY_VOTINGAGE],
    )
    self.assertEqual(optimized, [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    self.assertEqual(baseline, [[40.0, 50.0, 60.0], [10.0, 20.0, 30.0]])

  def test_ignores_rows_from_other_levels(self):
    optimized, baseline = common.extract_query_errors(
        _summary_df(), common.LEVEL_TRACT, [common.QUERY_TOTAL]
    )
    self.assertEqual(optimized, [[7.0, 8.0, 9.0]])
    self.assertEqual(baseline, [[70.0, 80.0, 90.0]])

  def test_missing_query_raises_naming_query_and_level(self):
    with self.assertRaisesRegex(ValueError, 'VOTINGAGE.*Tract'):
      common.extract_query_errors(
          _summary_df(), common.LEVEL_TRACT, [common.QUERY_VOTINGAGE]
      )


class NormalizeByBaselineMedianTest(absltest.TestCase):

  def test_divides_both_series_by_baseline_median(self):
    opt_norm, base_norm = common.normalize_by_baseline_median(
        optimized=[[1.0, 2.0, 3.0]], baseline=[[2.0, 4.0, 6.0]]
    )
    self.assertEqual(opt_norm, [[0.25, 0.5, 0.75]])
    self.assertEqual(base_norm, [[0.5, 1.0, 1.5]])

  def test_each_query_uses_its_own_baseline_median(self):
    opt_norm, _ = common.normalize_by_baseline_median(
        optimized=[[2.0], [2.0]], baseline=[[2.0], [4.0]]
    )
    self.assertEqual(opt_norm, [[1.0], [0.5]])

  def test_baseline_median_below_one_divides_by_exact_median(self):
    opt_norm, base_norm = common.normalize_by_baseline_median(
        optimized=[[0.1, 0.3]], baseline=[[0.2, 0.8]]
    )
    self.assertAlmostEqual(opt_norm[0][0], 0.2)
    self.assertAlmostEqual(opt_norm[0][1], 0.6)
    self.assertAlmostEqual(base_norm[0][0], 0.4)
    self.assertAlmostEqual(base_norm[0][1], 1.6)

  def test_empty_baseline_raises(self):
    with self.assertRaisesRegex(ValueError, 'empty'):
      common.normalize_by_baseline_median(optimized=[[1.0]], baseline=[[]])

  def test_non_positive_baseline_median_raises(self):
    with self.assertRaisesRegex(ValueError, 'non-positive'):
      common.normalize_by_baseline_median(
          optimized=[[1.0]], baseline=[[0.0, 0.0]]
      )


class DetailedQuerySpecsTest(absltest.TestCase):

  def test_detailed_queries_stem_normalizes_punctuation_and_case(self):
    self.assertEqual(
        common.detailed_queries_stem('Single race, County'),
        'detailed_queries_single_race_county',
    )
    self.assertEqual(
        common.detailed_queries_stem('A/B, C'),
        'detailed_queries_a_b_c',
    )

  def test_standard_specs_have_unique_stems_and_aligned_fields(self):
    specs = common.standard_detailed_query_specs()
    stems = [common.detailed_queries_stem(s.name) for s in specs]
    self.assertLen(set(stems), len(specs))
    for spec in specs:
      self.assertIn(spec.level, common.ALL_LEVELS)
      self.assertLen(spec.measurement_names, len(spec.query_keys))
      self.assertTrue(spec.x_label)


if __name__ == '__main__':
  absltest.main()
