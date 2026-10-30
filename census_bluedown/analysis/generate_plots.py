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

r"""Main CLI to generate analysis plots and tables from replicate error data.

Replicate i of the postprocessing pipeline is expected under <input_dir>/<i>/
with the same layout that run_pass.py writes (US/query_errors.parquet,
State=<sid>/<level>/query_errors_list.parquet, ...).

Sample usage:
  python3 census_bluedown/analysis/generate_plots.py \
    --input_dir=output/synthetic-processed-replicates \
    --gt_parquet_path=output/synthetic-ground-truth/US.parquet/DPQuery/ground_truth.parquet \
    --output_dir=output/plots \
    --plots=boxplots,detailed_queries,bias,binned_error,ridgeline,tables \
    --formats=pdf,png,csv
"""

from collections.abc import Sequence
import dataclasses
import functools
from absl import app
from absl import flags
from absl import logging
import pandas as pd
from census_bluedown.analysis import common
from census_bluedown.analysis import data_loader
from census_bluedown.analysis import plot_pipeline

_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    'output/synthetic-processed-replicates',
    'Root directory containing replicate output subdirectories (e.g. 0, 1,'
    ' ...).',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    'output/plots',
    'Directory where output figures and tables will be written.',
)
_GT_PARQUET_PATH = flags.DEFINE_string(
    'gt_parquet_path',
    'output/synthetic-ground-truth/US.parquet/DPQuery/ground_truth.parquet',
    'Path to the US ground truth parquet file.',
)
_NUM_REPLICATES = flags.DEFINE_integer(
    'num_replicates', 10, 'Number of replicates to load and aggregate.'
)
_PLOTS = flags.DEFINE_list(
    'plots',
    [common.FLAG_ALL],
    'Comma-separated list of plot types to generate: all, boxplots,'
    ' detailed_queries, bias, binned_error, ridgeline, tables.',
)
_LEVELS = flags.DEFINE_list(
    'levels',
    [common.FLAG_ALL],
    'Comma-separated list of geographic levels to generate: all, '
    + ', '.join(common.ALL_LEVELS)
    + '.',
)
_FORMATS = flags.DEFINE_list(
    'formats',
    [common.FORMAT_PDF],
    'Comma-separated list of output formats: pdf, png, csv.',
)


@dataclasses.dataclass
class ReplicateSource:
  """Loads plot inputs lazily from replicate parquet directories.

  Attributes:
    input_dir: Root directory containing replicate subdirectories.
    num_replicates: Number of replicates to load and aggregate.
    gt_parquet_path: Optional path to the US ground truth parquet file.
  """

  input_dir: str
  num_replicates: int
  gt_parquet_path: str | None = None

  @functools.cached_property
  def _summary(self) -> pd.DataFrame:
    logging.info('Loading replicate summary errors...')
    return data_loader.load_summary_replicate_errors(
        base_dir=self.input_dir,
        num_replicates=self.num_replicates,
        gt_parquet_path=self.gt_parquet_path,
    )

  @functools.cached_property
  def _detailed(self) -> pd.DataFrame:
    logging.info('Loading detailed summary query errors...')
    return data_loader.load_detailed_query_errors(
        base_dir=self.input_dir, num_replicates=self.num_replicates
    )

  def summary_df(self) -> pd.DataFrame | None:
    """Returns level-query replicate errors across all levels."""
    return self._summary

  def detailed_df(self, spec: common.DetailedQuerySpec) -> pd.DataFrame | None:
    """Returns replicate errors for one detailed-query spec."""
    df = self._detailed
    if spec.level not in df[common.COL_LEVEL].values:
      return None
    return df

  def population_binned_df(self, levels: Sequence[str]) -> pd.DataFrame | None:
    """Returns tidy population-binned metric data for the given levels."""
    logging.info('Loading population-binned data for %s...', levels)
    df = data_loader.load_population_binned_data(
        base_dir=self.input_dir,
        levels=levels,
        num_replicates=self.num_replicates,
    )
    return None if df.empty else df

  def ridgeline_hist(self, level: str) -> pd.DataFrame | None:
    """Computes the integer-error histogram for one geographic level."""
    logging.info('Loading sliced errors for %s ridgeline plot...', level)
    sliced = data_loader.load_sliced_query_errors(
        base_dir=self.input_dir,
        levels=[level],
        num_replicates=self.num_replicates,
    )
    if level not in sliced:
      return None
    return data_loader.histogram_query_errors(level, sliced[level])


def main(argv: Sequence[str]) -> None:
  """Executes the plot generation pipeline based on command-line flags.

  Args:
    argv: Positional command-line arguments.
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  source = ReplicateSource(
      input_dir=str(_INPUT_DIR.value),
      num_replicates=_NUM_REPLICATES.value,
      gt_parquet_path=str(_GT_PARQUET_PATH.value),
  )
  plot_pipeline.generate(
      source=source,
      output_dir=str(_OUTPUT_DIR.value),
      plot_types=set(_PLOTS.value),
      levels=common.resolve_levels(_LEVELS.value),
      formats=_FORMATS.value,
  )


if __name__ == '__main__':
  app.run(main)
