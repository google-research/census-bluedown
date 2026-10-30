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

r"""Main CLI to generate analysis plots and tables directly from CSV files.

This binary regenerates publication figures and LaTeX tables purely by reading
tidy CSV data, verifying data completeness and enabling offline/open-source
plotting without requiring CNS or raw replicate parquet access.

Sample usage:
  python3 census_bluedown/analysis/generate_plots_from_csv.py \
    --input_dir=output/plots \
    --output_dir=output/plots_from_csv \
    --plots=boxplots,detailed_queries,bias,binned_error,ridgeline,tables \
    --formats=pdf,png
"""

from collections.abc import Sequence
from absl import app
from absl import flags
from census_bluedown.analysis import common
from census_bluedown.analysis import plot_pipeline

_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    '',
    'Directory containing the exported CSV data files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '',
    'Directory where re-generated figures and tables will be written.',
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
    'Comma-separated list of output formats: pdf, png, csv (csv re-exports the'
    ' input tables beside the plots).',
)


def main(argv: Sequence[str]) -> None:
  """Executes CSV-based plot generation pipeline based on command-line flags.

  Args:
    argv: Positional command-line arguments.
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  plot_pipeline.generate(
      source=plot_pipeline.CsvSource(str(_INPUT_DIR.value)),
      output_dir=str(_OUTPUT_DIR.value),
      plot_types=set(_PLOTS.value),
      levels=common.resolve_levels(_LEVELS.value),
      formats=_FORMATS.value,
  )


if __name__ == '__main__':
  app.run(main)
