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

"""Population-binned metric (signed bias, L1 error) plotting routines."""

import dataclasses
import matplotlib.axes
import matplotlib.figure
import pandas as pd
from census_bluedown.analysis import boxplots
from census_bluedown.analysis import common
from census_bluedown.analysis import style


@dataclasses.dataclass(frozen=True)
class _MetricStyle:
  """Per-metric plot settings.

  Attributes:
    y_label: Text label for the y-axis.
    axhline: Whether to draw a horizontal guide line at y = 0.
    legend_anchor: Default (x, y) coordinates for the legend anchor.
  """

  y_label: str
  axhline: bool
  legend_anchor: tuple[float, float] = style.DEFAULT_LEGEND_ANCHOR


_METRIC_STYLES: dict[str, _MetricStyle] = {
    common.METRIC_BIAS: _MetricStyle(y_label='bias', axhline=True),
    common.METRIC_BINNED_ERROR: _MetricStyle(
        y_label='L1 error', axhline=False, legend_anchor=(0.95, 0.25)
    ),
}


def plot_population_binned(
    df: pd.DataFrame,
    level: str,
    metric: str,
    showlegend: bool = True,
    legend_anchor: tuple[float, float] | None = None,
    condition_names: tuple[str, str] = style.DEFAULT_CONDITION_NAMES,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Plots a population-binned metric as paired boxplots over replicates.

  Args:
    df: Tidy DataFrame as returned by data_loader.load_population_binned_data or
      csv_utils.load_population_binned_data_csv, with columns 'level', 'metric',
      'replicate', 'population_bin', 'bluedown_value', 'topdown_value'.
    level: Geographic level to plot (one of common.POPULATION_BINNED_LEVELS).
    metric: Metric to plot; must be a key of the module's metric style table
      (e.g. common.METRIC_BIAS).
    showlegend: Whether to display the legend.
    legend_anchor: Bounding box coordinates for legend anchor. If None, uses
      style.DEFAULT_LEGEND_ANCHOR, except for the Block level where the legend
      is lowered to avoid the data.
    condition_names: 2-tuple of labels for optimized and baseline conditions.

  Returns:
    A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).

  Raises:
    ValueError: If metric is not supported or no rows match (level, metric).
  """
  if metric not in _METRIC_STYLES:
    raise ValueError(f'Unsupported metric {metric!r}.')
  metric_style = _METRIC_STYLES[metric]

  subset = df[
      (df[common.COL_LEVEL] == level) & (df[common.COL_METRIC] == metric)
  ]
  if subset.empty:
    raise ValueError(f'No rows for level {level!r}, metric {metric!r}.')

  # One list of per-replicate values per population bin, in label order. All
  # three sequences come from the same iteration so they cannot desynchronise.
  bin_labels = []
  data1 = []
  data2 = []
  for label, group in subset.groupby(common.COL_POPULATION_BIN, sort=False):
    bin_labels.append(str(label))
    data1.append(list(group[common.COL_BLUEDOWN_VALUE]))
    data2.append(list(group[common.COL_TOPDOWN_VALUE]))

  if legend_anchor is None:
    legend_anchor = (
        (0.95, 0.55)
        if level == common.LEVEL_BLOCK
        else metric_style.legend_anchor
    )

  fig, ax = boxplots.parallel_boxplots(
      data1=data1,
      data2=data2,
      group_names=bin_labels,
      condition_names=condition_names,
      showlegend=showlegend,
      y_label=metric_style.y_label,
      axhline=metric_style.axhline,
      rotation=0,
      legend_anchor=legend_anchor,
  )
  ax.set_xlabel(f'size of {common.LEVEL_DISPLAY_NAMES[level]}')
  return fig, ax
