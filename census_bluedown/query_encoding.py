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

"""Census-specific query definitions, projections and design matrices.

A query aggregates the detailed histogram, of shape
constants.DETAILED_QUERY_SHAPE, feature by feature. For each feature, the query
shape entry determines how that feature is aggregated:

  - equal to the detailed size: the feature is sliced (kept as is);
  - 1: the feature is summed out;
  - len(constants.CENSUS_ASYMMETRIC_QUERY_PARTITION), first feature only: the
    feature values are grouped into contiguous bins whose sizes are given by
    the partition (the HHINSTLEVELS query groups the 8 housing types into bins
    of sizes 1, 4 and 3).

feature_groups is the single definition of this encoding; the projection and
design matrix functions below are derived from it.
"""

from collections.abc import Sequence

import numpy as np

from census_bluedown import constants
from census_bluedown import succinct


def feature_groups(query_shape: Sequence[int]) -> list[np.ndarray]:
  """Maps each value of each feature to its bin in a query.

  Args:
    query_shape: The shape of the query, with one entry per feature.

  Returns:
    For each feature i, an integer array of length DETAILED_QUERY_SHAPE[i]
    giving the index of the query bin containing each value of the feature.

  Raises:
    ValueError: If the query shape is not supported.
  """
  detailed_shape = constants.DETAILED_QUERY_SHAPE
  partition = constants.CENSUS_ASYMMETRIC_QUERY_PARTITION
  if len(query_shape) != len(detailed_shape):
    raise ValueError(f"Unsupported query shape {tuple(query_shape)}.")
  groups = []
  for axis, (query_size, detailed_size) in enumerate(
      zip(query_shape, detailed_shape)
  ):
    if query_size == detailed_size:
      groups.append(np.arange(detailed_size))
    elif query_size == 1:
      groups.append(np.zeros(detailed_size, dtype=np.int64))
    elif axis == 0 and query_size == len(partition):
      groups.append(np.repeat(np.arange(len(partition)), partition))
    else:
      raise ValueError(f"Unsupported query shape {tuple(query_shape)}.")
  return groups


def project_to_query_shape(
    values: np.ndarray | Sequence[float], query_shape: Sequence[int]
) -> np.ndarray:
  """Aggregates detailed histograms to the given query shape.

  Args:
    values: Array whose last axis has length constants.TOTAL_LEN, indexing the
      detailed histogram. Any leading axes are treated as batch dimensions, so
      this accepts a single histogram or a matrix with one histogram per row.
    query_shape: The shape of the query.

  Returns:
    Array with the same leading axes as values and last axis of length
    prod(query_shape).
  """
  values = np.asarray(values)
  batch_shape = values.shape[:-1]
  detailed_shape = constants.DETAILED_QUERY_SHAPE
  groups = feature_groups(query_shape)
  arr = values.reshape(batch_shape + detailed_shape)
  # Negative axes index the feature dimensions regardless of batch rank.
  num_features = len(detailed_shape)
  summed_axes = tuple(
      axis - num_features
      for axis, query_size in enumerate(query_shape)
      if query_size == 1
  )
  arr = arr.sum(axis=summed_axes, keepdims=True)
  for axis, (query_size, feature_group) in enumerate(zip(query_shape, groups)):
    if query_size not in (1, detailed_shape[axis]):
      bin_starts = np.flatnonzero(np.diff(feature_group, prepend=-1))
      arr = np.add.reduceat(arr, bin_starts, axis=axis - num_features)
  return arr.reshape(batch_shape + (-1,))


def query_cell_members(query_shape: Sequence[int]) -> list[np.ndarray]:
  """Lists the detailed histogram cells aggregated into each query cell.

  This is useful for aggregating values that numpy cannot sum efficiently,
  such as optimization variables.

  Args:
    query_shape: The shape of the query.

  Returns:
    A list with one entry per query cell, in flattened query order. Each entry
    is an array of the flat indices (in increasing order) of the detailed
    histogram cells that are summed into that query cell.
  """
  # cells[i] is the flat index of the query cell that detailed cell i sums into:
  # meshgrid broadcasts each feature's value-to-bin map over all detailed cells
  # (in detailed order), and ravel_multi_index combines the per-feature bins.
  cells = np.ravel_multi_index(
      np.meshgrid(*feature_groups(query_shape), indexing="ij"),
      tuple(query_shape),
  ).reshape(-1)
  # Sorting detailed cells by query cell (stably, so indices stay increasing
  # within a cell) and splitting at the cumulative cell sizes yields the
  # members of each query cell in order.
  order = np.argsort(cells, kind="stable")
  return np.split(order, np.cumsum(np.bincount(cells))[:-1])


def asymmetric_design_matrix(query_shape: tuple[int, ...]) -> np.ndarray:
  """Compute the asymmetric part of the design matrix for a query shape.

  This is specific to the Census application with shape (8, 2, 2, 63).
  The first three features are asymmetric (sizes 8, 2, 2). See the paper and
  Section 5.1 of https://arxiv.org/pdf/2204.08986v1 for details.

  Args:
    query_shape: The shape of the query, length 4.

  Returns:
    A numpy array of shape (m_j, 32) representing the asymmetric design matrix.
  """
  # One 0/1 matrix per feature, mapping feature values to query bins.
  a0, a1, a2 = (
      (np.arange(query_size)[:, None] == feature_group).astype(float)
      for query_size, feature_group in zip(
          query_shape[:3], feature_groups(query_shape)[:3]
      )
  )
  return np.kron(np.kron(a0, a1), a2)


def prepare_succinct_inputs(
    query_shapes: Sequence[tuple[int, ...]],
    query_variances: Sequence[float],
    query_values: Sequence[np.ndarray],
) -> tuple[
    succinct.SuccinctWorkload, succinct.SuccinctInputCovariance, np.ndarray
]:
  """Prepare succinct workload, input covariance, and concatenated observations.

  Queries either:
    - Slice over fourth feature (Kronecker term W_1 ⊗ I_d).
    - Aggregate over fourth feature (Kronecker term W_2 ⊗ 1_d^T).

  The resulting workload is W = [W_1 ⊗ I_d; W_2 ⊗ 1_d^T], input covariance is
  Σ = diag(Σ_1 ⊗ I_d, Σ_2), and observations are y = [y_1; y_2].

  Args:
    query_shapes: The shape of each query.
    query_variances: The variance of each query.
    query_values: The estimated values for each query.

  Returns:
    workload: SuccinctWorkload object representing W.
    cov: SuccinctInputCovariance object representing Σ.
    y: Concatenated observation vector [y_1; y_2].
  """
  w1_list = []
  w2_list = []
  diag1_list = []
  diag2_list = []
  y1_list = []
  y2_list = []

  d = succinct.SuccinctCovariance.projection_dim  # 63

  for shape, var, val in zip(query_shapes, query_variances, query_values):
    w_asym = asymmetric_design_matrix(shape)
    m_j = w_asym.shape[0]
    val_arr = np.array(val)

    if shape[3] == d:
      # Slice over fourth feature
      w1_list.append(w_asym)
      diag1_list.append(np.full(m_j, var))
      y1_list.append(val_arr.flatten())
    elif shape[3] == 1:
      # Aggregate over fourth feature
      w2_list.append(w_asym)
      diag2_list.append(np.full(m_j, var))
      y2_list.append(val_arr.flatten())
    else:
      raise ValueError(f"Unsupported symmetric query dimension: {shape[3]}")

  w1 = np.vstack(w1_list) if w1_list else np.zeros((0, 32))
  w2 = np.vstack(w2_list) if w2_list else np.zeros((0, 32))
  diag1 = np.concatenate(diag1_list) if diag1_list else np.zeros(0)
  diag2 = np.concatenate(diag2_list) if diag2_list else np.zeros(0)
  y1 = np.concatenate(y1_list) if y1_list else np.zeros(0)
  y2 = np.concatenate(y2_list) if y2_list else np.zeros(0)

  workload = succinct.SuccinctWorkload(w1, w2)
  cov = succinct.SuccinctInputCovariance(diag1, diag2)
  y = np.concatenate([y1, y2])

  return workload, cov, y
