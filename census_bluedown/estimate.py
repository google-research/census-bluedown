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

"""Data structure for vector-valued estimate and its covariance matrix."""

from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass
class Estimate:
  """Estimate vector with associated covariance matrix."""
  val: np.ndarray
  cov: np.ndarray

  def __add__(self, other: Estimate | int | float) -> Estimate:
    """Addition of independent estimate or constant shift."""
    if isinstance(other, Estimate):
      return Estimate(
          val=self.val + other.val,
          cov=self.cov + other.cov)
    elif isinstance(other, int) or isinstance(other, float):
      if other == 0:
        return self
      return Estimate(
          val=self.val + other,
          cov=self.cov)
    else:
      raise ValueError(f"Unsupported type for argument other: {type(other)}")

  def __radd__(self, other: Estimate | int | float) -> Estimate:
    """Addition of independent estimate or constant shift."""
    return self + other

  def __sub__(self, other: Estimate) -> Estimate:
    """Subtraction of independent estimates."""
    return Estimate(
        val=self.val - other.val,
        cov=self.cov + other.cov)

  def __mod__(self, other: Estimate) -> Estimate:
    """Subtraction of an estimate from a sum containing the same estimate.

    The other terms of the sum must be independent from the given estimate.
    Since the two estimates are not independent and have a very specific
    dependence, we subtract the covariances (as well as the values) instead of
    adding the covariances and subtracting the values.

    Arguments:
      other: The dependent estimate to be subtracted.

    Returns:
      The estimate obtained by subtracting the given other estimate from
      self, assuming that the self estimate represents a sum including the given
      other estimate.
    """
    return Estimate(
        val=self.val - other.val,
        cov=self.cov - other.cov)
