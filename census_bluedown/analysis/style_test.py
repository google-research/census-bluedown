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

import matplotlib
from census_bluedown.analysis import style
from absl.testing import absltest


class StyleTest(absltest.TestCase):

  def test_darken_color(self):
    dark_blue = style.darken_color((0.0, 0.0, 1.0), factor=0.5)
    self.assertLen(dark_blue, 3)
    self.assertAlmostEqual(dark_blue[2], 0.5, places=2)

  def test_get_color_on_white(self):
    rgb = (0.2, 0.4, 0.6)
    blended = style.get_color_on_white(rgb, alpha=0.5)
    self.assertAlmostEqual(blended[0], 0.6)
    self.assertAlmostEqual(blended[1], 0.7)
    self.assertAlmostEqual(blended[2], 0.8)

  def test_get_balanced_overlap(self):
    overlap_hex = style.get_balanced_overlap('#0000FF', '#FFFF00')
    self.assertTrue(overlap_hex.startswith('#'))
    self.assertLen(overlap_hex, 7)

  def test_configure_matplotlib_style(self):
    style.configure_matplotlib_style()
    self.assertEqual(matplotlib.rcParams['pdf.fonttype'], 42)
    self.assertEqual(matplotlib.rcParams['ps.fonttype'], 42)
    self.assertEqual(
        list(matplotlib.rcParams['font.sans-serif']),
        list(style.FONT_FAMILIES),
    )
    self.assertEqual(matplotlib.rcParams['figure.dpi'], style.DEFAULT_DPI)


if __name__ == '__main__':
  absltest.main()
