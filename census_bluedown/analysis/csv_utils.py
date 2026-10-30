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

"""CSV data export routines."""

import pandas as pd

from census_bluedown.analysis import common


def export_level_queries_csv(
    df: pd.DataFrame,
    level: str,
    output_path: str,
) -> None:
  """Writes a tidy CSV of raw replicate errors for the standard level queries.

  Args:
    df: Replicate summary DataFrame containing columns: * 'level' (str):
      Geographic level name. * 'query' (str): Query name. * 'optimized'
      (list[float]): Error values across replicates. * 'baseline' (list[float]):
      Error values across replicates.
    level: The geographic level to export (e.g. 'County'). The TOTAL query is
      omitted for levels in common.LEVELS_WITH_TRIVIAL_TOTAL.
    output_path: Destination file path for the written CSV.

  Raises:
    ValueError: If any standard query is missing for the level.
  """
  queries, _ = common.level_queries(level)
  present_queries = set(df[df[common.COL_LEVEL] == level][common.COL_QUERY])
  if common.QUERY_AGExHISPxCENRACE in present_queries:
    queries.append(common.QUERY_AGExHISPxCENRACE)
  optimized, baseline = common.extract_query_errors(df, level, queries)

  rows = []
  for query, opt_errors, base_errors in zip(queries, optimized, baseline):
    for rep_idx, (opt_err, base_err) in enumerate(zip(opt_errors, base_errors)):
      rows.append({
          common.COL_LEVEL: level,
          common.COL_QUERY: query,
          common.COL_REPLICATE: rep_idx,
          common.COL_BLUEDOWN_ERROR: opt_err,
          common.COL_TOPDOWN_ERROR: base_err,
      })

  out_df = pd.DataFrame(rows)
  with open(output_path, 'w') as out_f:
    out_df.to_csv(out_f, index=False)


def load_level_queries_df(csv_path: str) -> pd.DataFrame:
  """Reads a tidy CSV of replicate errors and reconstructs summary DataFrame.

  Args:
    csv_path: Path to the tidy CSV file containing columns: 'level', 'query',
      'replicate', 'bluedown_error', 'topdown_error'.

  Returns:
    DataFrame grouped by ['level', 'query'] with list-valued columns:
      * 'level' (str): Geographic level name.
      * 'query' (str): Query name.
      * 'optimized' (list[float]): Replicate errors for optimized method.
      * 'baseline' (list[float]): Replicate errors for baseline method.
  """
  with open(csv_path, 'r') as in_f:
    df = pd.read_csv(in_f)

  # sort=False keeps queries in file (= plotting) order.
  return (
      df.groupby([common.COL_LEVEL, common.COL_QUERY], sort=False)
      .agg(
          optimized=(common.COL_BLUEDOWN_ERROR, list),
          baseline=(common.COL_TOPDOWN_ERROR, list),
      )
      .reset_index()
  )


def export_detailed_queries_csv(
    df: pd.DataFrame,
    spec: common.DetailedQuerySpec,
    output_path: str,
) -> None:
  """Builds and writes tidy CSV of raw replicate errors for detailed queries.

  Args:
    df: Replicate summary DataFrame containing columns: * 'level' (str):
      Geographic level name. * 'query' (str): Query name. * 'optimized'
      (list[float]): Error values across replicates. * 'baseline' (list[float]):
      Error values across replicates.
    spec: DetailedQuerySpec identifying the queries, labels, and level to
      export.
    output_path: Destination file path for the written CSV.
  """
  optimized, baseline = common.extract_query_errors(
      df, spec.level, spec.query_keys
  )

  rows = []
  for m_name, q_key, opt_errors, base_errors in zip(
      spec.measurement_names, spec.query_keys, optimized, baseline
  ):
    for rep_idx, (opt_err, base_err) in enumerate(zip(opt_errors, base_errors)):
      rows.append({
          common.COL_LEVEL: spec.level,
          common.COL_BENCHMARK: spec.name,
          common.COL_MEASUREMENT: m_name,
          common.COL_QUERY_KEY: q_key,
          common.COL_REPLICATE: rep_idx,
          common.COL_BLUEDOWN_ERROR: opt_err,
          common.COL_TOPDOWN_ERROR: base_err,
      })

  out_df = pd.DataFrame(rows)
  with open(output_path, 'w') as out_f:
    out_df.to_csv(out_f, index=False)


def load_detailed_queries_csv(
    csv_path: str,
) -> tuple[pd.DataFrame, common.DetailedQuerySpec]:
  """Reads a detailed queries CSV back into a summary DataFrame and its spec.

  Args:
    csv_path: Path to the tidy CSV written by export_detailed_queries_csv,
      containing columns: * 'level' (str): Geographic level name. * 'benchmark'
      (str): Benchmark group name. * 'measurement' (str): Measurement display
      label. * 'query_key' (str): Identifier key for the query. * 'replicate'
      (int): Replicate index. * 'bluedown_error' (float): Raw error for
      optimized method. * 'topdown_error' (float): Raw error for baseline
      method.

  Returns:
    A tuple of (df, spec) where:
      * df: DataFrame grouped by ['level', 'query'] with list-valued columns
        'optimized' and 'baseline', in the same format plot_detailed_queries
        accepts.
      * spec: DetailedQuerySpec reconstructed from the CSV (name = benchmark,
        measurement_names and query_keys in file order). x_label is not stored
        in the CSV and is therefore None.
  """
  with open(csv_path, 'r') as in_f:
    raw = pd.read_csv(in_f)

  df = (
      raw.groupby([common.COL_LEVEL, common.COL_QUERY_KEY], sort=False)
      .agg(
          optimized=(common.COL_BLUEDOWN_ERROR, list),
          baseline=(common.COL_TOPDOWN_ERROR, list),
      )
      .reset_index()
      .rename(columns={common.COL_QUERY_KEY: common.COL_QUERY})
  )

  per_measurement = raw.drop_duplicates(common.COL_QUERY_KEY)
  spec = common.DetailedQuerySpec(
      name=str(raw[common.COL_BENCHMARK].iloc[0]),
      level=str(raw[common.COL_LEVEL].iloc[0]),
      measurement_names=tuple(per_measurement[common.COL_MEASUREMENT]),
      query_keys=tuple(per_measurement[common.COL_QUERY_KEY]),
  )
  return df, spec


def export_population_binned_data_csv(
    df: pd.DataFrame,
    level: str,
    metric: str,
    output_path: str,
) -> None:
  """Writes the rows of one (level, metric) from a population-binned DataFrame.

  Args:
    df: Tidy DataFrame as returned by data_loader.load_population_binned_data,
      with columns 'level', 'metric', 'replicate', 'population_bin',
      'bluedown_value', 'topdown_value'.
    level: Geographic level to select.
    metric: Metric to select ('bias' or 'binned_error').
    output_path: Destination file path for the written CSV.
  """
  subset = df[
      (df[common.COL_LEVEL] == level) & (df[common.COL_METRIC] == metric)
  ]
  with open(output_path, 'w') as out_f:
    subset.to_csv(out_f, index=False)


def load_population_binned_data_csv(csv_path: str) -> pd.DataFrame:
  """Reads a population-binned CSV written by export_population_binned_data_csv.

  Args:
    csv_path: Path to the CSV file.

  Returns:
    Tidy DataFrame with columns 'level', 'metric', 'replicate',
    'population_bin', 'bluedown_value', 'topdown_value', in the same format
    data_loader.load_population_binned_data returns.
  """
  with open(csv_path, 'r') as in_f:
    return pd.read_csv(in_f)


def export_ridgeline_histograms_csv(
    hist_df: pd.DataFrame, output_path: str
) -> None:
  """Writes a ridgeline histogram DataFrame to CSV.

  Args:
    hist_df: Tidy histogram as returned by data_loader.histogram_query_errors,
      with columns 'level', 'query', 'error_value', 'bluedown_count',
      'topdown_count'.
    output_path: Destination file path for the written CSV.
  """
  with open(output_path, 'w') as out_f:
    hist_df.to_csv(out_f, index=False)


def load_ridgeline_histograms_csv(csv_path: str) -> pd.DataFrame:
  """Reads a ridgeline histogram CSV written by export_ridgeline_histograms_csv.

  Args:
    csv_path: Path to the CSV file.

  Returns:
    Tidy DataFrame with columns 'level', 'query', 'error_value',
    'bluedown_count', 'topdown_count', in the same format
    data_loader.histogram_query_errors returns.
  """
  with open(csv_path, 'r') as in_f:
    return pd.read_csv(in_f)
