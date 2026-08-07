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

"""Compressed data format for estimates and covariance with symmetric structure.
"""

from collections.abc import Sequence, Iterator
import dataclasses
import itertools
from typing import TypeAlias

import numpy as np
import scipy

from census_bluedown import estimate


Estimate: TypeAlias = estimate.Estimate
QueryShape: TypeAlias = tuple[int, ...]


@dataclasses.dataclass
class AsymmetricQuery:
  """A histogram query that groups some values of an asymmetric feature.

  This implementation allows only simple asymmetric queries, where a single
  asymmetric feature is sliced and all other features are aggregated.

  For example, the default values of the fields corresponds to a query that
  slices the first of four features, which in this instance has eight possible
  values. These eight values are grouped into three bins, where the shape field
  specifies the number of histogram bins each feature is split into. The
  grouping is specified by the partition field: the first feature value is in
  the first bin, the next four feature values are in the second bin, and the
  remaining three feature values are in the third bin.

  Attributes:
    num_features: The number of features in the query.
    sliced_feature: The index of the sliced feature.
    partition: A sequence specifying how many values of the sliced feature are
      included in each histogram bin, where the sum of the entries is equal to
      the number of possible values of the sliced feature.
    shape: A sequence specifying the number of histogram slices corresponding to
      each feature. All but one entry must equal 1, and the remaining entry
      equals len(partition).
  """
  num_features: int = 4
  sliced_feature: int = 0
  partition: Sequence[int] = (1, 4, 3)
  shape: QueryShape = dataclasses.field(init=False)

  def __post_init__(self):
    self.shape = tuple([1 if i != self.sliced_feature else len(self.partition)
                        for i in range(self.num_features)])


@dataclasses.dataclass
class BlockShape:
  """Structure of a block of estimates with covariance symmetries.

  The BlockShape class is used to define the structure of a tensor of linear
  estimates together with its covariance matrix. The tensor is indexed by a
  sequence of feature values, where some of the features are assumed to be
  symmetric in the sense that the covariance matrix of the estimates is
  invariant under reordering of the possible values of these features. This
  symmetry can be used to compress the covariance matrix, improving the space
  and time efficiency of processing of the estimates.

  The compression scheme is based on the following linear transformation of the
  estimates. For each asymmetric feature, the possible feature values are
  considered in uncompressed form. For each symmetric feature, instead of
  considering a separate estimate for each possible value of the feature, we
  consider two estimates, one for a particular (arbitrary) choice of value for
  the feature, and one for the sum of all possible values of the feature. Under
  our symmetry assumption, the covariance matrix of this linear transformation
  of the estimates is sufficient to reconstruct the covariance matrix of the
  original estimates.

  The implementation assumes that the features are ordered so that the
  possibly-asymmetric features come first, followed by the symmetric features.
  This property is ensured by the BlockShape constructor.

  In addition to the features, the BlockShape class also contains a dictionary
  of AsymmetricQuery objects indexed by the shape of the query. These are used
  to describe histogram queries that group some values of an asymmetric feature.
  It is assumed that the shapes of the asymmetric queries are unique.

  Attributes:
    shape: The number of possible values for each feature.
    num_asymmetric_features: The number of asymmetric features.
    asymmetric_features: The indices of the asymmetric features.
    symmetric_features: The indices of the symmetric features.
    num_features: The total number of features.
    length: The length of the estimate vector for the block.
    asymmetric_length: The length of the asymmetric features.
    asymmetric_shape: A tuple consisting of the number of possible values for
      each asymmetric feature, and a 1 for each symmetric feature.
    uncompressed_symmetric_length: The length of the symmetric features.
    compressed_symmetric_length: The length of the compressed symmetric
      features.
    compressed_length: The length of the compressed block.
    asymmetric_queries: a dictionary of AsymmetricQuery objects whose keys are
      the shapes of the queries.
    asymmetric_query_dict: a dictionary of AsymmetricQuery objects whose keys
      are the shapes of the queries.
  """
  shape: QueryShape = (8, 2, 2, 63)
  num_asymmetric_features: int = 3
  asymmetric_queries: tuple[AsymmetricQuery, ...] | tuple[()] = ()
  asymmetric_features: tuple[int, ...] = dataclasses.field(init=False)
  symmetric_features: tuple[int, ...] = dataclasses.field(init=False)
  num_features: int = dataclasses.field(init=False)
  length: int = dataclasses.field(init=False)
  asymmetric_length: int = dataclasses.field(init=False)
  asymmetric_shape: QueryShape = dataclasses.field(init=False)
  uncompressed_symmetric_length: int = dataclasses.field(init=False)
  compressed_symmetric_length: int = dataclasses.field(init=False)
  compressed_length: int = dataclasses.field(init=False)
  asymmetric_query_dict: dict[QueryShape, AsymmetricQuery] = dataclasses.field(
      init=False
  )

  def __post_init__(self):
    self.asymmetric_features = tuple(range(self.num_asymmetric_features))
    self.symmetric_features = tuple(
        range(self.num_asymmetric_features, len(self.shape)))
    self.num_features = len(self.shape)
    self.length = np.prod(self.shape)  # pyrefly: ignore[bad-assignment]
    self.asymmetric_length = np.prod(self.shape[:self.num_asymmetric_features])  # pyrefly: ignore[bad-assignment]
    self.asymmetric_shape = tuple([x if i in self.asymmetric_features else 1
                                   for i, x in enumerate(self.shape)])
    self.uncompressed_symmetric_length = self.length // self.asymmetric_length
    self.compressed_symmetric_length = int(2**(len(self.symmetric_features)))
    self.compressed_length = (self.asymmetric_length *
                              self.compressed_symmetric_length)
    self.asymmetric_query_dict = {query.shape: query
                                  for query in self.asymmetric_queries}
    if (self.num_asymmetric_features < 0 or
        self.num_asymmetric_features > len(self.shape)):
      raise ValueError(
          f'num_asymmetric_features {self.num_asymmetric_features} must be'
          f' between 0 and the number of features {len(self.shape)}'
      )
    for query in self.asymmetric_queries:
      if len(query.shape) != self.num_features:
        raise ValueError(
            f'Asymmetric query shape {query.shape} must have the same length as'
            f' the block shape {self.shape}'
        )
      if sum(query.partition) != self.shape[query.sliced_feature]:
        raise ValueError(
            f'Asymmetric query partition {query.partition} must sum to the'
            f' number of possible values of the sliced feature'
            f' {self.shape[query.sliced_feature]}'
        )
      if query.partition != self.asymmetric_query_dict[query.shape].partition:
        raise ValueError(
            f'The shape {query.shape} of each asymmetric query must be unique'
        )

  def symmetric_query_design_rows(self, query_shape: QueryShape) -> np.ndarray:
    """Obtain rows of design matrix for a given symmetric query shape.

    The input query_shape is a tuple of length self.num_features specifying the
    number of ways each feature is sliced in the query, where query_shape[i] is
    equal to self.shape[i] if the corresponding feature is sliced and 1
    otherwise. The returned design matrix enumerates all possible partial
    aggregates of the symmetric features and all possible feature values for the
    sliced asymmetric features. The number of columns in the returned design
    matrix is self.compressed_length, and the number of rows is the product of
    the number of feature values for each asymmetric feature sliced in the
    query, times two for each symmetric feature that is sliced in the query.

    The design matrix is generated iteratively using the Kronecker product. For
    each asymmetric feature, if the feature is aggregated in the query, we take
    the Kronecker product with the all-ones vector of length self.shape[feature]
    to represent the sum of all possible values of the feature; if the feature
    is sliced in the query, we take the Kronecker product with the identity
    matrix of size self.shape[feature] to represent self.shape[feature] separate
    queries each considering a different value of the feature.

    For symmetric features our encoding has two possible values, the first
    corresponding to a single (arbitrary) value of the feature and the second
    corresponding to the sum of all possible values of the feature. So for each
    symmetric feature, if the feature is aggregated in the query, we take the
    Kronecker product with the vector [0, 1] to indicate the coordinate that
    represents the sum of all possible values of the feature; if the feature is
    sliced in the query, we take the Kronecker product with the identity matrix
    of size 2 to represent two partial aggregates, one in which the feature is
    equal to the particular value it is sliced to, and one with the sum of all
    possible values of the feature.

    Args:
      query_shape: the shape of the query.

    Returns:
      A list of rows of the design matrix, each in the form of a list.
    """
    design = np.array([[1]])

    for feature in self.asymmetric_features:
      if query_shape[feature] == 1:
        # Aggregated features correspond to a single design row with all-1s
        # query.
        design = np.kron(design, np.ones((1, self.shape[feature])))
      else:
        # Sliced features correspond to multiple queries, each one querying a
        # different value of the feature.
        design = np.kron(design, np.eye(self.shape[feature]))

    for feature in self.symmetric_features:
      if query_shape[feature] == 1:
        # Aggregated features use the coordinate `1`.
        design = np.kron(design, np.array([0, 1]))
      else:
        # Sliced features require querying both individual and aggregated value.
        design = np.kron(design, np.eye(2))

    return design

  def asymmetric_query_design_rows(
      self,
      query: AsymmetricQuery,
  ) -> np.ndarray:
    """Obtain rows of design matrix for an asymmetric query.

    Args:
      query: The asymmetric query.

    Returns:
      A list of rows of the design matrix, each in the form of a list.
    """
    # Compute design rows for the symmetric query that slices the same feature
    # as this asymmetric query.
    symmetric_query_shape = list(query.shape)
    symmetric_query_shape[query.sliced_feature] = self.shape[
        query.sliced_feature]
    symmetric_design = self.symmetric_query_design_rows(
        tuple(symmetric_query_shape))

    # Aggregate the design rows from the symmetric query that correspond to the
    # asymmetric query's partition of feature values for the sliced feature.
    grouped_design = np.split(symmetric_design, np.cumsum(query.partition)[:-1])
    return np.array([sum(row) for row in grouped_design])

  def query_design_rows(self, query_shape: QueryShape) -> np.ndarray:
    """Obtain rows of design matrix for a given query shape.

    The query may be symmetric or asymmetric.

    Args:
      query_shape: the shape of the query

    Returns:
      A list of rows of the design matrix, each in the form of a list.
    """
    query_shape = tuple(query_shape)
    if query_shape in self.asymmetric_query_dict:
      return self.asymmetric_query_design_rows(
          self.asymmetric_query_dict[query_shape]
      )
    return self.symmetric_query_design_rows(query_shape)

  def covariance_block(
      self,
      query_shape: QueryShape,
      variance: float,
  ) -> np.ndarray:
    """Get covariance block for a query with given shape and variance.

    The compressed block representation involves representing a single semantic
    query as multiple design matrix rows corresponding to all possible rollups
    of the symmetric features. The covariance block is a matrix describing the
    covariance between the estimates corresponding to these design matrix rows.
    The covariance between two estimates is equal to the variance of the
    original query multiplied by the product of the number of possible values
    of each feature that is aggregated in both queries.

    This method returns the covariance block for a single combination of values
    for the asymmetric features; the covariance matrix for all design matrix
    rows corresponding to the query_shape is block diagonal where the number of
    blocks is given by the number of possible combinations of values of the
    asymmetric features.

    Args:
      query_shape: the shape of the query
      variance: the variance of each estimate

    Returns:
      A matrix describing the covariance between the estimates resulting from
      this query for each combination of values of the asymmetric features.
    """
    if tuple(query_shape) in self.asymmetric_query_dict:
      return np.array([[variance]])

    cov_block = np.array([[variance]])
    for feature in self.symmetric_features:
      if query_shape[feature] != 1:
        sub_block = np.array([[1, 1], [1, query_shape[feature]]])
        cov_block = np.kron(cov_block, sub_block)
    return cov_block

  def design_matrix(self, query_shapes: Sequence[QueryShape]) -> np.ndarray:
    """Compute design matrix for generalized least squares regression.

    The design matrix corresponds to the compressed representation described by
    the BlockShape instance. It is generated by concatenating the design matrix
    rows corresponding to each query.

    Args:
      query_shapes: The shape of each query

    Returns:
      The linear regression design matrix corresponding to the given queries.
    """
    return np.vstack([self.query_design_rows(q) for q in query_shapes])

  def covariance_matrix(
      self,
      query_shapes: Sequence[QueryShape],
      query_variances: Sequence[float],
  ) -> np.ndarray:
    """Compute covariance matrix for generalized least squares regression.

    The covariance matrix corresponds to the compressed representation described
    by the BlockShape instance. It is block diagonal, where each block
    corresponds to a single query and a single combination of values of the
    sliced asymmetric features.

    Args:
      query_shapes: The shape of each query
      query_variances: The variance of each query

    Returns:
      The covariance matrix of the linear regression measurements corresponding
      to the given queries.
    """
    cov_blocks = []
    for query_shape, query_variance in zip(query_shapes, query_variances):
      next_covariance_block = self.covariance_block(query_shape, query_variance)
      asymmetric_feature_combinations = int(
          np.prod([query_shape[i] for i in self.asymmetric_features])
      )
      for _ in range(asymmetric_feature_combinations):
        cov_blocks.append(next_covariance_block)

    return scipy.linalg.block_diag(*cov_blocks)

  def asymmetric_feature_values(
      self,
      query_shape: QueryShape,
  ) -> Iterator[tuple[int, ...]]:
    """Compute all combinations of asymmetric features sliced by query.

    For asymmetric features that are sliced in the query, the returned iterator
    enumerates all possible combinations of feature values. For asymmetric
    features that are not sliced in the query, enumerated tuples contain a
    a placeholder value equal to 0.

    Args:
      query_shape: The shape of the query.

    Returns:
      An iterator over all possible combinations of values for the asymmetric
      features sliced by the query.
    """
    return itertools.product(*[range(query_shape[i])
                               for i in self.asymmetric_features])

  def symmetric_partial_rollups(
      self,
      query_shape: QueryShape,
  ) -> Iterator[tuple[int, ...]]:
    """Compute partial rollup aggregates of symmetric portion of a query.

    The input query_shape is a tuple of length num_features specifying the
    number of ways each feature is sliced in the query, where query_shape[i] is
    equal to self.shape[i] if the corresponding feature is sliced and 1
    otherwise. The returned iterator enumerates all possible partial aggregates
    of the symmetric portion of the query, which are obtained by replacing by 1
    any subset of entries of query_shape corresponding to symmetric features.

    Args:
      query_shape: The shape of the query.

    Returns:
      An iterator over all possible partial aggregates of the symmetric portion
      of the given query shape.
    """
    iterator_set = []
    for i in self.symmetric_features:
      if query_shape[i] == 1:
        # Always aggregate over features that are not sliced in the query
        iterator_set.append([1])
      else:
        # Allow both slicing by and aggregating over features that are sliced in
        # the query
        iterator_set.append([query_shape[i], 1])
    return itertools.product(*iterator_set)

  def query_aggregates(
      self,
      asymmetric_feature_values: Sequence[int],
      symmetric_partial_rollup: Sequence[int],
      query_value: np.ndarray,
  ) -> np.ndarray:
    """Compute the selected aggregates of the query.

    Computes aggregated query values corresponding to the specified asymmetric
    features and the specified aggregations of a subset of the symmetric
    features. The shape of the returned array is given by
    symmetric_partial_rollup, with a leading dimension of length 1 for each
    asymmetric feature.

    Args:
      asymmetric_feature_values: The values of the asymmetric features
      symmetric_partial_rollup: A partial rollup specifying a subset of the
        symmetric features of the query to aggregate. The format is a sequence
        of length equal to the number of symmetric features, where each entry is
        either 1 if the corresponding symmetric feature is aggregated or the
        number of possible values of the feature if the feature is sliced.
      query_value: The value of the query

    Returns:
      The value of the selected aggregates of the query
    """
    sum_coords = np.flatnonzero(np.array(symmetric_partial_rollup) == 1)
    aggregated_query = np.sum(
        query_value[tuple(asymmetric_feature_values)],
        axis=tuple(sum_coords),
        keepdims=True,
    )

    # Return value will be broadcasted to block_shape.shape
    return np.expand_dims(aggregated_query, axis=self.asymmetric_features)
