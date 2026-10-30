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

"""Shared plot and table generation pipeline for bluedown analysis."""

from collections.abc import Sequence
import dataclasses
import functools
import os
from typing import Protocol

from absl import logging
import pandas as pd

from census_bluedown.analysis import boxplots
from census_bluedown.analysis import common
from census_bluedown.analysis import csv_utils
from census_bluedown.analysis import population_bin_plots
from census_bluedown.analysis import ridgeline_plots
from census_bluedown.analysis import tables


class PlotSource(Protocol):
  """Provider of DataFrames for a plot pipeline run.

  A return value of None indicates that the data for that plot or section is
  unavailable and should be skipped silently.
  """

  def summary_df(self) -> pd.DataFrame | None:
    """Returns level-query replicate errors."""

  def detailed_df(self, spec: common.DetailedQuerySpec) -> pd.DataFrame | None:
    """Returns replicate errors for one detailed-query spec."""

  def population_binned_df(self, levels: Sequence[str]) -> pd.DataFrame | None:
    """Returns tidy population-binned metric data for the given levels."""

  def ridgeline_hist(self, level: str) -> pd.DataFrame | None:
    """Returns the integer-error histogram for one geographic level."""


@dataclasses.dataclass
class CsvSource:
  """Loads plot inputs from a directory of exported tidy CSV files.

  Attributes:
    input_dir: Root directory containing the per-plot CSV subdirectories.
  """

  input_dir: str

  @functools.cached_property
  def _summary(self) -> pd.DataFrame | None:
    """Reads and concatenates all present level-query CSVs once."""
    boxplots_dir = os.path.join(self.input_dir, common.BOXPLOTS_DIR)
    if not os.path.exists(boxplots_dir):
      return None
    frames = []
    for level in common.ALL_LEVELS:
      stem = common.LEVEL_QUERIES_STEM.format(level=level)
      csv_path = os.path.join(boxplots_dir, f'{stem}.csv')
      if os.path.exists(csv_path):
        frames.append(csv_utils.load_level_queries_df(csv_path))
    if not frames:
      return None
    return pd.concat(frames, ignore_index=True)

  def summary_df(self) -> pd.DataFrame | None:
    """Loads and concatenates all present level-query CSVs."""
    return self._summary

  def detailed_df(self, spec: common.DetailedQuerySpec) -> pd.DataFrame | None:
    """Loads replicate errors for one detailed-query spec from its CSV."""
    stem = common.detailed_queries_stem(spec.name)
    csv_path = os.path.join(
        self.input_dir, common.DETAILED_QUERIES_DIR, f'{stem}.csv'
    )
    if not os.path.exists(csv_path):
      return None
    df, _ = csv_utils.load_detailed_queries_csv(csv_path)
    return df

  def population_binned_df(self, levels: Sequence[str]) -> pd.DataFrame | None:
    """Loads and concatenates population-binned CSVs for the given levels."""
    sources = (
        (common.BIAS_DIR, common.BIAS_STEM),
        (common.ERR_POPULATION_BINS_DIR, common.ERR_POPULATION_BINS_STEM),
    )
    frames = []
    for subdir, stem_template in sources:
      in_dir = os.path.join(self.input_dir, subdir)
      for level in levels:
        stem = stem_template.format(level=level)
        csv_path = os.path.join(in_dir, f'{stem}.csv')
        if os.path.exists(csv_path):
          frames.append(csv_utils.load_population_binned_data_csv(csv_path))
    if not frames:
      return None
    return pd.concat(frames, ignore_index=True)

  def ridgeline_hist(self, level: str) -> pd.DataFrame | None:
    """Loads the ridgeline error histogram CSV for one geographic level."""
    stem = common.RIDGELINE_STEM.format(level=level)
    csv_path = os.path.join(self.input_dir, common.RIDGELINE_DIR, f'{stem}.csv')
    if not os.path.exists(csv_path):
      return None
    return csv_utils.load_ridgeline_histograms_csv(csv_path)


def generate(
    source: PlotSource,
    output_dir: str,
    plot_types: set[str],
    levels: Sequence[str],
    formats: Sequence[str],
) -> None:
  """Runs the requested plot and table generation stages.

  Args:
    source: Provider of input DataFrames (from parquets or CSVs).
    output_dir: Root directory where output subdirectories will be created.
    plot_types: Set of plot type tokens (e.g. {'boxplots'} or {'all'}).
    levels: Geographic levels to include, in hierarchical order.
    formats: Output file formats (e.g. ['pdf', 'png', 'csv']).
  """
  all_plots = common.FLAG_ALL in plot_types
  export_csv = common.FORMAT_CSV in formats

  os.makedirs(output_dir, exist_ok=True)
  logging.info('Output directory: %s', output_dir)

  # 1. Level Boxplots
  if all_plots or common.PLOT_BOXPLOTS in plot_types:
    summary_df = source.summary_df()
    if summary_df is not None:
      boxplots_dir = os.path.join(output_dir, common.BOXPLOTS_DIR)
      os.makedirs(boxplots_dir, exist_ok=True)
      present_levels = set(summary_df[common.COL_LEVEL])
      for level in levels:
        if level not in present_levels:
          continue
        showlegend = level in (common.LEVEL_COUNTY, common.LEVEL_BLOCK)
        stem = common.LEVEL_QUERIES_STEM.format(level=level)
        fig, _ = boxplots.plot_level_queries(
            df=summary_df,
            level=level,
            showlegend=showlegend,
            rotation=30,
        )
        common.save_figure(fig, boxplots_dir, stem, formats)
        if export_csv:
          csv_utils.export_level_queries_csv(
              df=summary_df,
              level=level,
              output_path=os.path.join(boxplots_dir, f'{stem}.csv'),
          )

  # 2. Detailed Summary Queries
  if all_plots or common.PLOT_DETAILED_QUERIES in plot_types:
    detailed_dir = os.path.join(output_dir, common.DETAILED_QUERIES_DIR)
    for spec in common.standard_detailed_query_specs():
      if spec.level not in levels:
        continue
      df = source.detailed_df(spec)
      if df is None:
        continue
      os.makedirs(detailed_dir, exist_ok=True)
      fig, _ = boxplots.plot_detailed_queries(
          df=df,
          spec=spec,
          showlegend=spec.show_legend,
          rotation=30,
          normalize=True,
      )
      stem = common.detailed_queries_stem(spec.name)
      common.save_figure(fig, detailed_dir, stem, formats)
      if export_csv:
        csv_utils.export_detailed_queries_csv(
            df=df,
            spec=spec,
            output_path=os.path.join(detailed_dir, f'{stem}.csv'),
        )

  # 3. Bias & Binned Error Plots
  if (
      all_plots
      or common.PLOT_BIAS in plot_types
      or common.PLOT_BINNED_ERROR in plot_types
  ):
    binned_levels = [
        lvl for lvl in levels if lvl in common.POPULATION_BINNED_LEVELS
    ]
    if binned_levels:
      binned_df = source.population_binned_df(binned_levels)
      if binned_df is not None:
        if all_plots or common.PLOT_BIAS in plot_types:
          bias_dir = os.path.join(output_dir, common.BIAS_DIR)
          os.makedirs(bias_dir, exist_ok=True)
          for lvl in binned_levels:
            has_rows = (
                (binned_df[common.COL_LEVEL] == lvl)
                & (binned_df[common.COL_METRIC] == common.METRIC_BIAS)
            ).any()
            if not has_rows:
              continue
            showlegend = lvl in (common.LEVEL_TRACT, common.LEVEL_BLOCK)
            stem = common.BIAS_STEM.format(level=lvl)
            fig, _ = population_bin_plots.plot_population_binned(
                df=binned_df,
                level=lvl,
                metric=common.METRIC_BIAS,
                showlegend=showlegend,
            )
            common.save_figure(fig, bias_dir, stem, formats)
            if export_csv:
              csv_utils.export_population_binned_data_csv(
                  df=binned_df,
                  level=lvl,
                  metric=common.METRIC_BIAS,
                  output_path=os.path.join(bias_dir, f'{stem}.csv'),
              )

        if all_plots or common.PLOT_BINNED_ERROR in plot_types:
          err_bins_dir = os.path.join(
              output_dir, common.ERR_POPULATION_BINS_DIR
          )
          os.makedirs(err_bins_dir, exist_ok=True)
          for lvl in binned_levels:
            has_rows = (
                (binned_df[common.COL_LEVEL] == lvl)
                & (binned_df[common.COL_METRIC] == common.METRIC_BINNED_ERROR)
            ).any()
            if not has_rows:
              continue
            showlegend = lvl == common.LEVEL_TRACT
            stem = common.ERR_POPULATION_BINS_STEM.format(level=lvl)
            fig, _ = population_bin_plots.plot_population_binned(
                df=binned_df,
                level=lvl,
                metric=common.METRIC_BINNED_ERROR,
                showlegend=showlegend,
            )
            common.save_figure(fig, err_bins_dir, stem, formats)
            if export_csv:
              csv_utils.export_population_binned_data_csv(
                  df=binned_df,
                  level=lvl,
                  metric=common.METRIC_BINNED_ERROR,
                  output_path=os.path.join(err_bins_dir, f'{stem}.csv'),
              )

  # 4. Ridgeline Plots (Error Distributions)
  if all_plots or common.PLOT_RIDGELINE in plot_types:
    ridge_levels = [lvl for lvl in levels if lvl in common.RIDGELINE_LEVELS]
    if ridge_levels:
      ridgeline_dir = os.path.join(output_dir, common.RIDGELINE_DIR)
      os.makedirs(ridgeline_dir, exist_ok=True)
      for lvl in ridge_levels:
        hist_df = source.ridgeline_hist(lvl)
        if hist_df is None:
          continue
        stem = common.RIDGELINE_STEM.format(level=lvl)
        fig, _ = ridgeline_plots.plot_level_ridgeline(hist_df, level=lvl)
        common.save_figure(fig, ridgeline_dir, stem, formats)
        if export_csv:
          csv_utils.export_ridgeline_histograms_csv(
              hist_df, os.path.join(ridgeline_dir, f'{stem}.csv')
          )

  # 5. Summary Tables (LaTeX)
  if all_plots or common.PLOT_TABLES in plot_types:
    summary_df = source.summary_df()
    if summary_df is not None:
      tables_dir = os.path.join(output_dir, common.TABLES_DIR)
      os.makedirs(tables_dir, exist_ok=True)
      logging.info('Generating LaTeX tables...')
      for index, queries in enumerate(common.SUMMARY_TABLE_QUERIES, start=1):
        stem = common.SUMMARY_TABLE_STEM.format(index=index)
        stats_df = tables.compute_summary_stats(summary_df, queries, levels)
        tex_path = os.path.join(tables_dir, f'{stem}.tex')
        with open(tex_path, 'w') as out_f:
          out_f.write(tables.render_latex_table(stats_df))
        logging.info('Wrote %s', tex_path)

  logging.info('All requested plots and tables generated successfully.')
