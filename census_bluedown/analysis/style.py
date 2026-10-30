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

"""Style configurations, palettes, and color blending routines for plots."""

from collections.abc import Sequence
import colorsys
import functools
import os
from typing import Any
import matplotlib
from matplotlib import font_manager
import matplotlib.colors as mcolors


def darken_color(color: Any, factor: float = 0.6) -> tuple[float, float, float]:
  """Darkens a given color by scaling its value (brightness) in HSV space.

  Args:
    color: Any matplotlib-compatible color specification (e.g. hex string, RGB
      tuple).
    factor: Multiplicative factor applied to the HSV value (brightness)
      component. A factor < 1.0 darkens the color.

  Returns:
    Darkened RGB color as a 3-tuple of floats in [0.0, 1.0].
  """
  r, g, b = mcolors.to_rgb(color)
  h, s, v = mcolors.rgb_to_hsv((r, g, b))
  v_darkened = v * factor
  out = mcolors.hsv_to_rgb((h, s, v_darkened))
  return (float(out[0]), float(out[1]), float(out[2]))


def get_balanced_overlap(
    color1: Any, color2: Any, factor1: float = 1.0, factor2: float = 1.0
) -> str:
  """Blends two colors in HLS space to create a balanced overlap color.

  Args:
    color1: First matplotlib-compatible color.
    color2: Second matplotlib-compatible color.
    factor1: Scaling factor for the lightness of the first color.
    factor2: Scaling factor for the lightness of the second color.

  Returns:
    Hex color string of the resulting blended color.
  """
  rgb1 = mcolors.to_rgb(color1)
  rgb2 = mcolors.to_rgb(color2)
  avg_rgb = [(x + y) / 2 for x, y in zip(rgb1, rgb2)]
  hls1 = colorsys.rgb_to_hls(*rgb1)
  hls2 = colorsys.rgb_to_hls(*rgb2)
  avg_hls = colorsys.rgb_to_hls(*avg_rgb)
  target_lightness = factor1 * hls1[1] + factor2 * hls2[1]
  final_rgb = colorsys.hls_to_rgb(avg_hls[0], target_lightness, avg_hls[2])
  return mcolors.to_hex(final_rgb)


def get_color_on_white(
    rgb: Sequence[float], alpha: float
) -> tuple[float, float, float]:
  """Calculates effective RGB color resulting from RGBA on white background.

  Args:
    rgb: Sequence of 3 floats representing the RGB channels in [0.0, 1.0].
    alpha: Alpha opacity value in [0.0, 1.0].

  Returns:
    Simulated RGB color as a 3-tuple of floats in [0.0, 1.0].
  """
  r, g, b = (c * alpha + (1.0 - alpha) for c in rgb[:3])
  return (r, g, b)


# Color palette constants
_PASTEL_BLUE = mcolors.to_rgb('#0072B2')
_PASTEL_YELLOW = darken_color('#FBF719', factor=0.95)

ORANGE = get_balanced_overlap(
    '#D35400', _PASTEL_YELLOW, factor1=0.2, factor2=0.6
)
PURPLE = get_balanced_overlap(
    '#8E44AD', _PASTEL_BLUE, factor1=0.25, factor2=0.75
)

CB_PALETTE_COLORS = (_PASTEL_BLUE, _PASTEL_YELLOW)
DARK_GRAY = '#444444'

# (outline, fill) color pairs for the two series in ridgeline plots.
DEFAULT_SERIES_COLORS = (
    (darken_color(_PASTEL_BLUE, factor=0.36), _PASTEL_BLUE),
    (darken_color(_PASTEL_YELLOW, factor=0.6), _PASTEL_YELLOW),
)

DEFAULT_CONDITION_NAMES = ('BlueDown', 'TopDown')
DEFAULT_LEGEND_ANCHOR = (0.95, 0.85)

# Publication font and resolution defaults (Science figure guidelines).
FONT_FAMILIES = ('Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans')
DEFAULT_DPI = 300
_SYSTEM_FONT_DIRS = (
    os.path.expanduser('~/.local/share/fonts'),
    '/usr/local/share/fonts',
    '/usr/share/fonts/truetype/msttcorefonts',
    '/usr/share/fonts/truetype/liberation',
    '/usr/share/fonts/truetype',
)


@functools.lru_cache(maxsize=1)
def register_system_fonts() -> None:
  """Registers TrueType system fonts with Matplotlib's global FontManager."""
  for font_dir in _SYSTEM_FONT_DIRS:
    for font_file in font_manager.findSystemFonts(fontpaths=[font_dir]):
      try:
        font_manager.fontManager.addfont(font_file)
      except Exception:  # pylint: disable=broad-exception-caught
        continue


def configure_matplotlib_style() -> None:
  """Configures Matplotlib rcParams for publication-grade vector/raster plots."""
  register_system_fonts()
  matplotlib.rcParams['pdf.fonttype'] = 42
  matplotlib.rcParams['ps.fonttype'] = 42
  matplotlib.rcParams['font.family'] = 'sans-serif'
  matplotlib.rcParams['font.sans-serif'] = list(FONT_FAMILIES)
  matplotlib.rcParams['figure.dpi'] = DEFAULT_DPI
  matplotlib.rcParams['savefig.dpi'] = DEFAULT_DPI
  matplotlib.rcParams['axes.labelsize'] = 10
  matplotlib.rcParams['xtick.labelsize'] = 9
  matplotlib.rcParams['ytick.labelsize'] = 9
  matplotlib.rcParams['legend.fontsize'] = 9
  matplotlib.rcParams['legend.title_fontsize'] = 10
