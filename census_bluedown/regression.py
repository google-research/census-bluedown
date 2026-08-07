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

"""Linear regression postprocessing for histograms on blocks of estimates.
"""

from collections.abc import Iterator, Sequence
from typing import TypeAlias
import numpy as np
from census_bluedown import block
from census_bluedown import estimate


_FLOAT_ZERO = 1e-7
Estimate: TypeAlias = estimate.Estimate


def regression_combination(
    block_shape: block.BlockShape,
    query_shapes: Sequence[block.QueryShape],
    query_variances: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
  """Compute the GLS regression linear combination and covariance matrix.

  Computes the linear combination of measurements optimizing the generalized
  least squares (GLS) regression of the given queries. The covariance matrix
  resulting from this linear combination is also returned.

  Abstract GLS problem: For any unknown vector x in R^d, and a design matrix A
  in R^{m x d}, we observe y = Ax + e, where e is a zero-mean noise vector in
  R^m with covariance Q. Given A and Q, compute the linear transformation L such
  that Ly is the variance-minimizing linear estimate of x.

  The linear combination is given by L = (A^T P A)^{-1} A^T P, where P = Q^{-1}
  is the inverse of the covariance matrix (i.e. the precision matrix). The
  covariance matrix resulting from this linear combination is given by
  C = L Q L^T = (A^T P A)^{-1}. The GLS solution for x is Ly. (Aitken, 1935)

  The shape of L is the same as the shape of the transpose of the design matrix
  A, with a number of rows equal to block_shape.compressed_length and number of
  columns equal to the sum over queries of the product of the shapes of the
  asymmetric features sliced by the query, times two to the number of symmetric
  features sliced by the query. The shape of the covariance matrix C is
  block_shape.compressed_length x block_shape.compressed_length.

  Args:
    block_shape: The BlockShape object describing the block structure
    query_shapes: The shape of each query
    query_variances: The variance of each query

  Returns:
    linear_combination: The linear combination of measurements for the
      generalized least squares regression solution.
    covariance: The covariance matrix corresponding to this linear
      transformation of the measurements.
  """
  design = block_shape.design_matrix(query_shapes)
  original_covariance = block_shape.covariance_matrix(
      query_shapes,
      query_variances)
  precision_matrix = np.linalg.inv(original_covariance)

  # Perform generalized least squares regression
  updated_covariance = np.linalg.inv(design.T @ precision_matrix @ design)
  linear_combination = updated_covariance @ design.T @ precision_matrix

  # Note that the updated covariance is also given by the following expression:
  # linear_combination @ original_covariance @ linear_combination.T

  return linear_combination, updated_covariance


def query_contribution(
    block_shape: block.BlockShape,
    query_shape: block.QueryShape,
    query_value: np.ndarray,
    linear_combination_iterator: Iterator[np.ndarray]
) -> np.ndarray:
  """Compute contribution of specified query to block estimate.

  For each combination of values of the asymmetric feature values sliced by the
  query and for each partial rollup of the symmetric features sliced by the
  query, this function computes partial aggregates of the query corresponding to
  those asymmetric feature values and that partial rollup of symmetric features.
  It then multiplies these partial aggregates by the corresponding coefficients
  in the specified linear combination, broadcasting the result to shape
  block_shape.shape and summing over all such combinations to obtain the overall
  contribution of the query to the block estimate.

  Args:
    block_shape: The BlockShape object describing the block structure
    query_shape: The shape of the query
    query_value: The value of the query, as a numpy array shape either
      query_shape or a vector with the same number of elements as query_shape.
    linear_combination_iterator: The coefficients of the contribution to the
      block estimate from each estimate of the query, up to symmetry. Each
      element of the iterator is a vector of coefficients of length block_shape.
      asymmetric_length, corresponding to a column of the corresponding linear
      transformation.

  Returns:
    The value of the specified linear transformation of the given query
  """
  estimate_value = np.zeros(block_shape.shape)
  query_value = query_value.reshape(query_shape)

  for asymmetric_features in block_shape.asymmetric_feature_values(query_shape):
    for symmetric_rollup in block_shape.symmetric_partial_rollups(query_shape):
      aggregated_query = block_shape.query_aggregates(
          asymmetric_features,
          symmetric_rollup,
          query_value
      )
      linear_combination = next(linear_combination_iterator)
      coefficients = linear_combination.reshape(block_shape.asymmetric_shape)

      # Implicit broadcasting in update step:
      estimate_value += aggregated_query * coefficients

  return estimate_value


def apply_linear_transformation(
    block_shape: block.BlockShape,
    linear_transformation: np.ndarray,
    estimate_value: np.ndarray
) -> np.ndarray:
  """Apply linear transformation to initial estimate.

  This implementation ignores the portion of the linear transformation
  corresponding to partial aggregates of symmetric features, since these partial
  aggregates can be obtained from the result of the transformation on the
  unaggregated values.

  Args:
    block_shape: The BlockShape object describing the block structure
    linear_transformation: The linear transformation to apply to the initial
      estimate. The shape of this array should be
      block_shape.compressed_length x block_shape.compressed_length.
    estimate_value: The initial estimate to transform

  Returns:
    The new estimate obtained by applying the linear transformation.
  """
  linear_combination = np.array(
      linear_transformation[::block_shape.compressed_symmetric_length, :]
  )
  linear_combination_column_iterator = iter(linear_combination.T)
  return query_contribution(
      block_shape,
      block_shape.shape,
      estimate_value,
      linear_combination_column_iterator)


def apply_linear_transformation_alternate(
    block_shape: block.BlockShape,
    linear_transformation: np.ndarray,
    estimate_value: np.ndarray
) -> np.ndarray:
  """Apply linear transformation to initial estimate.

  This alternate implementation assumes that there is a single symmetric
  feature. In this case, it is equivalent to apply_linear_transformation.

  Args:
    block_shape: The BlockShape object describing the block structure
    linear_transformation: The linear transformation to apply to the initial
      estimate. The shape of this array should be
      block_shape.compressed_length x block_shape.compressed_length.
    estimate_value: The initial estimate to transform

  Returns:
    The new estimate obtained by applying the linear transformation.
  """
  if block_shape.num_features - block_shape.num_asymmetric_features != 1:
    raise ValueError('This implementation assumes a single symmetric feature')
  lt1 = linear_transformation[::2, ::2]
  lt2 = linear_transformation[::2, 1::2]
  x_matrix = estimate_value.reshape(
      (block_shape.asymmetric_length,
       block_shape.uncompressed_symmetric_length))
  partial_sums = np.sum(x_matrix, axis=1)

  y1 = lt1 @ x_matrix
  y2 = np.repeat(lt2 @ partial_sums, block_shape.uncompressed_symmetric_length)

  return y1 + np.reshape(y2, block_shape.shape)


def process_queries(
    block_shape: block.BlockShape,
    query_shapes: Sequence[block.QueryShape],
    query_variances: Sequence[float],
    query_values: Sequence[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
  """Compute best linear unbiased estimate of a block from the input queries.

  Coefficients for the generalized least squares regression problem are computed
  using the compressed representation and then applied to the query values to
  obtain the estimated values and covariance matrix for the block.

  Args:
    block_shape: The BlockShape object describing the block structure
    query_shapes: The shape of each query
    query_variances: The variance of each estimated value for each query
    query_values: The estimated values for each query, where each query's values
        are represented as a list whose length is the product of the shape of
        the query.

  Returns:
    A tuple consisting of two numpy arrays. The first array is the best linear
    unbiased estimate of the values of the block, and the second array is the
    associated covariance matrix for the estimate, in compressed form.
  """
  linear_transformation, covariance = (
      regression_combination(block_shape, query_shapes, query_variances)
  )

  # Only need the linear combination rows corresponding to unaggregated values
  # and not partial aggregates of symmetric features, since the latter can be
  # obtained from the result of the transformation on the former.
  linear_combination = np.array(
      linear_transformation[::block_shape.compressed_symmetric_length, :]
  )

  linear_combination_column_iterator = iter(linear_combination.T)
  estimate_value = np.zeros(block_shape.length)
  for query_shape, query_value in zip(query_shapes, query_values):
    estimate_value += query_contribution(
        block_shape,
        query_shape,
        np.array(query_value),
        linear_combination_column_iterator).flatten()

  return (estimate_value, covariance)


def combine_estimates(
    block_shape: block.BlockShape,
    estimate_1: Estimate,
    estimate_2: Estimate,
) -> Estimate:
  """Compute the minimum-error linear combination of two estimates.

  Given independent estimates X1 and X2 of the same vector z with covariance
  matrices C1 and C2, the minimum-error linear combination is given by
  A1 X1 + A2 X2, where A1 = C2 (C1 + C2)^+ and A2 = I - A1. This formula permits
  either or both of the input covariance matrices to be singular.

  Args:
    block_shape: The BlockShape object describing the block structure
    estimate_1: First estimate to combine
    estimate_2: Second estimate to combine

  Returns:
    The optimal linear combination of the two estimates.
  """
  covariance_shape = (block_shape.compressed_length,
                      block_shape.compressed_length)
  if (estimate_1.val.shape != (block_shape.length,) or
      estimate_2.val.shape != (block_shape.length,)):
    raise ValueError('Estimate value does not match block shape')
  if (estimate_1.cov.shape != covariance_shape or
      estimate_2.cov.shape != covariance_shape):
    raise ValueError('Estimate covariance does not match block shape')

  I = np.identity(block_shape.compressed_length)  # pylint: disable=invalid-name
  A = estimate_2.cov @ np.linalg.pinv(estimate_1.cov + estimate_2.cov)  # pylint: disable=invalid-name
  B = np.asarray(I - A)  # pylint: disable=invalid-name

  # Apply the linear transformations to the two uncompressed estimates, and
  # compute the covariance of the combined estimate.
  linear_combination_1 = apply_linear_transformation(
      block_shape, A, estimate_1.val)
  linear_combination_2 = apply_linear_transformation(
      block_shape, B, estimate_2.val)
  combined_val = linear_combination_1 + linear_combination_2
  combined_var = A @ estimate_1.cov @ A.T + B @ estimate_2.cov @ B.T

  # Ensure numeric symmetry
  combined_var = 0.5 * (combined_var + combined_var.T)

  return Estimate(combined_val.flatten(), combined_var)


def combine(
    block_shape: block.BlockShape,
    estimate_1: Estimate | float,
    estimate_2: Estimate | float,
) -> Estimate:
  """Compute the minimum-error linear combination of two estimates.

  All float inputs, including np.nan, are interpreted as missing data. This
  convention is made because np.nan is the value used by pandas.merge to
  indicate missing data.

  Args:
    block_shape: The BlockShape object describing the block structure
    estimate_1: First estimate, or np.nan
    estimate_2: Second estimate, or np.nan

  Returns:
    The optimal linear combination of the two estimates.
  """
  if not isinstance(estimate_1, Estimate):
    if not isinstance(estimate_2, Estimate):
      raise ValueError('Cannot combine two missing estimates')
    return estimate_2

  if not isinstance(estimate_2, Estimate):
    return estimate_1

  return combine_estimates(block_shape, estimate_1, estimate_2)
