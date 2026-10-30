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

"""Dual-series ridgeline (stacked KDE distribution) plotting routines."""

from collections.abc import Sequence
from typing import Any
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from census_bluedown.analysis import common
from census_bluedown.analysis import style

style.configure_matplotlib_style()


def _weighted_kde(
    values: np.ndarray, weights: np.ndarray, grid: np.ndarray
) -> np.ndarray:
  """Evaluates a weighted Gaussian KDE of a histogram on a grid.

  Args:
    values: Distinct sample values (histogram bin centres).
    weights: Non-negative count of each value.
    grid: Points at which to evaluate the density.

  Returns:
    Density on the grid, or zeros if the histogram is empty or degenerate.
    The bandwidth is Scott's factor for an effective sample size of
    sum(weights).
  """
  total = float(np.sum(weights))
  if total <= 0:
    return np.zeros_like(grid)
  try:
    kde = stats.gaussian_kde(values, weights=weights, bw_method=total**-0.2)
  except (ValueError, np.linalg.LinAlgError):
    return np.zeros_like(grid)
  return kde(grid)


def plot_ridgeline_dual_series(
    level: str,
    values: Sequence[np.ndarray],
    weights_a: Sequence[np.ndarray],
    weights_b: Sequence[np.ndarray],
    group_labels: Sequence[str],
    series_labels: tuple[str, str] = style.DEFAULT_CONDITION_NAMES,
    series_colors: Sequence[tuple[Any, Any]] = style.DEFAULT_SERIES_COLORS,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = style.DEFAULT_DPI,
    hspace: float = 0.1,
    overlap_scale: float = 1.0,
    label_y_pos: float = 0.1,
    opacity_alpha: float = 0.6,
    x_min: float | None = None,
    x_max: float | None = None,
    showlegend: bool = True,
    legend_anchor: tuple[float, float] = style.DEFAULT_LEGEND_ANCHOR,
) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
  """Plots stacked, overlapping KDEs of two histogrammed series per group.

  Args:
    level: Geographic level name, used for the x-axis label.
    values: Per group, the distinct error values (shared by both series).
    weights_a: Per group, the count of each value for condition A.
    weights_b: Per group, the count of each value for condition B.
    group_labels: Label for each group row.
    series_labels: 2-tuple of legend labels for condition A and condition B.
    series_colors: (outline, fill) color pair for each of the two series.
    title: Optional super-title for the figure.
    figsize: Figure dimensions (width, height) in inches.
    dpi: Figure resolution dots-per-inch.
    hspace: Vertical space between subplots.
    overlap_scale: Scaling factor for KDE curve heights.
    label_y_pos: Vertical position in axes coordinates for group labels.
    opacity_alpha: Fill opacity alpha for KDE curves.
    x_min: Minimum x-axis limit; defaults to the smallest value minus half a
      standard deviation.
    x_max: Maximum x-axis limit; defaults symmetrically.
    showlegend: Whether to display the condition legend.
    legend_anchor: Bounding box coordinates for the legend anchor.

  Returns:
    A tuple of (matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]).

  Raises:
    ValueError: If the per-group sequences have different lengths.
  """
  style.configure_matplotlib_style()
  num_groups = len(values)
  if not (len(weights_a) == len(weights_b) == len(group_labels) == num_groups):
    raise ValueError('values, weights and group_labels must align per group.')

  fill_a = series_colors[0][1]
  fill_b = series_colors[1][1]
  overlap_color = style.get_balanced_overlap(fill_a, fill_b)

  all_values = (
      np.concatenate([np.asarray(v, float) for v in values])
      if values
      else np.empty(0)
  )
  if all_values.size == 0:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    return fig, [ax]
  half_sd = 0.5 * float(np.std(all_values))
  if x_min is None:
    x_min = float(all_values.min()) - half_sd
  if x_max is None:
    x_max = float(all_values.max()) + half_sd
  x_vals = np.linspace(x_min, x_max, max(int(x_max - x_min) + 1, 2))

  fig, axes_raw = plt.subplots(
      num_groups, 1, figsize=figsize, dpi=dpi, sharex=True
  )
  axes = [axes_raw] if num_groups == 1 else list(axes_raw)
  fig.subplots_adjust(hspace=hspace)

  for i, (vals, w_a, w_b, label) in enumerate(
      zip(values, weights_a, weights_b, group_labels)
  ):
    ax = axes[i]
    ax.set_facecolor('none')
    vals = np.asarray(vals, float)
    y_a = _weighted_kde(vals, np.asarray(w_a, float), x_vals) * overlap_scale
    y_b = _weighted_kde(vals, np.asarray(w_b, float), x_vals) * overlap_scale

    ax.fill_between(x_vals, y_a, color=fill_a, alpha=opacity_alpha, zorder=1)
    ax.fill_between(x_vals, y_b, color=fill_b, alpha=opacity_alpha, zorder=1)
    y_overlap = np.minimum(y_a, y_b)
    ax.fill_between(
        x_vals, y_overlap, color=overlap_color, alpha=opacity_alpha, zorder=1
    )

    ax.text(
        -0.02,
        label_y_pos,
        label,
        transform=ax.transAxes,
        fontsize=22,
        ha='right',
    )

    ax.set_yticks([])
    ax.set_ylim(0, None)
    for spine in ['left', 'right', 'top', 'bottom']:
      ax.spines[spine].set_visible(False)

    ax.axhline(0, color=style.DARK_GRAY, linewidth=1.0, alpha=0.6, zorder=2)

    if i == 0 and showlegend:
      handles = [
          plt.Rectangle((0, 0), 1, 1, color=c[1], ec=c[0], alpha=opacity_alpha)
          for c in series_colors
      ]
      ax.legend(
          handles,
          series_labels,
          loc='upper right',
          frameon=True,
          edgecolor='black',
          fontsize=20,
          bbox_to_anchor=legend_anchor,
          bbox_transform=fig.transFigure,
      )

  last_ax = axes[-1]
  last_ax.spines['bottom'].set_visible(True)
  last_ax.spines['bottom'].set_color(style.DARK_GRAY)
  last_ax.spines['bottom'].set_linewidth(1.0)
  last_ax.tick_params(axis='x', colors=style.DARK_GRAY, labelsize=22)
  last_ax.set_xlabel(
      f'Error distributions for {common.LEVEL_DISPLAY_NAMES[level]} queries',
      fontsize=24,
  )

  last_ax.set_xlim(x_min, x_max)
  if title:
    fig.suptitle(title, fontsize=26, y=0.98)

  return fig, axes


def plot_level_ridgeline(
    hist_df: pd.DataFrame,
    level: str,
    showlegend: bool | None = None,
    legend_anchor: tuple[float, float] = (0.89, 0.89),
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
  """Generates the canonical ridgeline plot for a geographic level.

  Args:
    hist_df: Tidy error histogram as returned by
      data_loader.histogram_query_errors or
      csv_utils.load_ridgeline_histograms_csv, with columns 'level', 'query',
      'error_value', 'bluedown_count', 'topdown_count'. Queries are plotted in
      the order they first appear.
    level: Geographic level name (one of common.RIDGELINE_LEVELS).
    showlegend: Whether to display the condition legend. If None, the legend is
      shown only for the Block_Group level.
    legend_anchor: Bounding box coordinates for legend anchor.
    x_min: Minimum x-axis limit; defaults to common.RIDGELINE_X_BOUNDS[level].
    x_max: Maximum x-axis limit; defaults to common.RIDGELINE_X_BOUNDS[level].

  Returns:
    A tuple of (matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]).
  """
  if showlegend is None:
    showlegend = level == common.LEVEL_BLOCK_GROUP
  default_min, default_max = common.RIDGELINE_X_BOUNDS[level]
  if x_min is None:
    x_min = default_min
  if x_max is None:
    x_max = default_max

  groups = [g for _, g in hist_df.groupby(common.COL_QUERY, sort=False)]
  queries = [str(g[common.COL_QUERY].iloc[0]) for g in groups]
  values = [g[common.COL_ERROR_VALUE].to_numpy(float) for g in groups]
  weights_a = [g[common.COL_BLUEDOWN_COUNT].to_numpy(float) for g in groups]
  weights_b = [g[common.COL_TOPDOWN_COUNT].to_numpy(float) for g in groups]

  return plot_ridgeline_dual_series(
      level=level,
      values=values,
      weights_a=weights_a,
      weights_b=weights_b,
      group_labels=[common.RIDGELINE_RELABEL.get(q, q) for q in queries],
      x_min=x_min,
      x_max=x_max,
      showlegend=showlegend,
      legend_anchor=legend_anchor,
  )
