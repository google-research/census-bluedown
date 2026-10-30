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

"""Boxplot generation routines for error metrics.

Supports plotting across geographic levels and queries.
"""

from collections.abc import Sequence
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from census_bluedown.analysis import common
from census_bluedown.analysis import style

sns.set_style('ticks')
sns.set_context('paper')
style.configure_matplotlib_style()


def parallel_boxplots(
    data1: Sequence[Sequence[float]],
    data2: Sequence[Sequence[float]],
    group_names: Sequence[str],
    condition_names: tuple[str, str] = style.DEFAULT_CONDITION_NAMES,
    showlegend: bool = True,
    opacityalpha: float = 0.7,
    y_label: str = 'normalized error',
    axhline: bool = False,
    rotation: int = 30,
    legend_anchor: tuple[float, float] = style.DEFAULT_LEGEND_ANCHOR,
    figsize: tuple[float, float] = (5, 3.5),
    dpi: int = style.DEFAULT_DPI,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Draws paired side-by-side boxplots with overlaid strip points.

  Args:
    data1: Nested sequence of numerical values for the first condition (e.g.
      BlueDown), one sequence of replicate errors per group.
    data2: Nested sequence of numerical values for the second condition (e.g.
      TopDown), one sequence of replicate errors per group.
    group_names: Label for each group (x-axis tick labels).
    condition_names: 2-tuple of labels for data1 and data2 respectively.
    showlegend: Whether to render a legend on the plot.
    opacityalpha: Opacity alpha for boxplot fills (0.0 to 1.0).
    y_label: Text label for the y-axis.
    axhline: Whether to draw a horizontal dotted red guide line at y = 0.
    rotation: Clockwise rotation angle for x-axis tick labels in degrees.
    legend_anchor: (x, y) coordinates for bounding box anchor of the legend.
    figsize: Figure dimensions (width, height) in inches.
    dpi: Figure resolution dots-per-inch.

  Returns:
    A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
  """
  style.configure_matplotlib_style()
  data_list = []
  for i, group_name in enumerate(group_names):
    for val in data1[i]:
      data_list.append({
          'Value': val,
          'Group': group_name,
          'Condition': condition_names[0],
      })
    for val in data2[i]:
      data_list.append({
          'Value': val,
          'Group': group_name,
          'Condition': condition_names[1],
      })

  df = pd.DataFrame(data_list)

  box_palette = {
      condition_names[0]: style.get_color_on_white(
          style.CB_PALETTE_COLORS[0], opacityalpha
      ),
      condition_names[1]: style.get_color_on_white(
          style.CB_PALETTE_COLORS[1], opacityalpha
      ),
  }
  dot_palette = {
      condition_names[0]: style.PURPLE,
      condition_names[1]: style.ORANGE,
  }

  fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

  if axhline:
    ax.axhline(0, color='red', linestyle=':')

  sns.boxplot(
      data=df,
      x='Group',
      y='Value',
      hue='Condition',
      palette=box_palette,
      ax=ax,
      fliersize=0,
      gap=0.15,
      linewidth=0.3,
      boxprops={'edgecolor': 'black', 'linewidth': 0},
      medianprops={'color': 'black', 'linewidth': 0.3},
      whiskerprops={'color': 'black', 'linewidth': 0.3},
      capprops={'color': 'black', 'linewidth': 0.3},
  )

  sns.stripplot(
      data=df,
      x='Group',
      y='Value',
      hue='Condition',
      palette=dot_palette,
      dodge=True,
      jitter=False,
      size=2.5,
      alpha=0.6,
      ax=ax,
      legend=False,
  )

  ax.set_ylabel(y_label)
  ax.set_xlabel('')

  ticks = ax.get_yticks()
  if len(ticks) > 0:
    ax.set_ylim(bottom=ticks[0])
  sns.despine(ax=ax, offset=5, trim=True)

  ax.tick_params(axis='x', which='major', labelsize=9)
  if rotation != 0:
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=rotation,
        ha='right',
        rotation_mode='anchor',
    )
  ax.tick_params(axis='y', which='major', labelsize=9)

  if showlegend:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=True,
        edgecolor='black',
        loc='upper right',
        bbox_to_anchor=legend_anchor,
        fontsize=9,
        title_fontsize=10,
    )
  else:
    legend = ax.get_legend()
    if legend is not None:
      legend.remove()

  fig.subplots_adjust(left=0.15, bottom=0.2, right=0.75, top=0.95)
  return fig, ax


def plot_level_queries(
    df: pd.DataFrame,
    level: str,
    showlegend: bool = True,
    rotation: int = 30,
    condition_names: tuple[str, str] = style.DEFAULT_CONDITION_NAMES,
    legend_anchor: tuple[float, float] = style.DEFAULT_LEGEND_ANCHOR,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Generates normalized parallel boxplots for standard paper queries.

  Args:
    df: Replicate summary DataFrame containing columns: * 'level' (str):
      Geographic level name (e.g. 'State', 'County', 'Tract'). * 'query' (str):
      Query name (matching common.STANDARD_LEVEL_QUERIES). * 'optimized'
      (list[float]): Error values across replicates for optimized. * 'baseline'
      (list[float]): Error values across replicates for baseline.
    level: The geographic level to filter and plot (e.g. 'County'). The TOTAL
      query is omitted for levels in common.LEVELS_WITH_TRIVIAL_TOTAL.
    showlegend: Whether to display the condition legend.
    rotation: Clockwise rotation angle for query labels on the x-axis.
    condition_names: 2-tuple of legend labels for optimized and baseline.
    legend_anchor: (x, y) coordinates for the legend box anchor.

  Returns:
    A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
  """
  queries, labels = common.level_queries(level)
  optimized, baseline = common.extract_query_errors(df, level, queries)
  data1, data2 = common.normalize_by_baseline_median(optimized, baseline)

  fig, ax = parallel_boxplots(
      data1=data1,
      data2=data2,
      group_names=labels,
      condition_names=condition_names,
      showlegend=showlegend,
      rotation=rotation,
      legend_anchor=legend_anchor,
  )

  ax.set_xlabel(f'{common.LEVEL_DISPLAY_NAMES[level]} queries')

  return fig, ax


def plot_detailed_queries(
    df: pd.DataFrame,
    spec: common.DetailedQuerySpec,
    showlegend: bool = True,
    rotation: int = 30,
    normalize: bool = True,
    condition_names: tuple[str, str] = style.DEFAULT_CONDITION_NAMES,
    legend_anchor: tuple[float, float] = style.DEFAULT_LEGEND_ANCHOR,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Generates parallel boxplot for a detailed summary query.

  Args:
    df: Replicate error summary DataFrame containing columns: * 'level' (str):
      Geographic level name. * 'query' (str): Query name. * 'optimized'
      (list[float]): Error values across replicates. * 'baseline' (list[float]):
      Error values across replicates.
    spec: common.DetailedQuerySpec defining the queries, labels, and level.
    showlegend: Whether to render the legend on the plot.
    rotation: Clockwise rotation angle for x-axis tick labels.
    normalize: Whether to normalize errors by median baseline error.
    condition_names: 2-tuple of labels for optimized and baseline conditions.
    legend_anchor: (x, y) coordinates for bounding box anchor of the legend.

  Returns:
    A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
  """
  data1, data2 = common.extract_query_errors(df, spec.level, spec.query_keys)

  if normalize:
    data1, data2 = common.normalize_by_baseline_median(data1, data2)
    y_label = 'normalized error'
  else:
    y_label = 'error'

  fig, ax = parallel_boxplots(
      data1=data1,
      data2=data2,
      group_names=spec.measurement_names,
      condition_names=condition_names,
      showlegend=showlegend,
      rotation=rotation,
      y_label=y_label,
      legend_anchor=legend_anchor,
  )

  ax.set_xlabel((spec.x_label or spec.name).lower())

  return fig, ax
