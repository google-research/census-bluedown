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

import abc
from collections.abc import Sequence
import dataclasses
import math
from typing import ClassVar, overload

import numpy as np

from census_bluedown import constants


class Covariance(abc.ABC):
  """Abstract base class for structured covariance matrices."""

  @property
  @abc.abstractmethod
  def constraint_class(self) -> type[Constraint]:
    """Returns the compatible Constraint class for this covariance."""

  @abc.abstractmethod
  def __add__(self, other: Covariance) -> Covariance:
    """Adds another covariance of the exact same underlying type."""

  @abc.abstractmethod
  def __sub__(self, other: Covariance) -> Covariance:
    pass

  @abc.abstractmethod
  def __neg__(self) -> Covariance:
    pass

  @overload
  def __matmul__(self, other: Covariance) -> Covariance:
    pass

  @overload
  def __matmul__(self, other: np.ndarray) -> np.ndarray:
    pass

  @abc.abstractmethod
  def __matmul__(
      self, other: Covariance | np.ndarray
  ) -> Covariance | np.ndarray:
    pass

  @abc.abstractmethod
  def pinv(self) -> Covariance:
    pass


@dataclasses.dataclass
class DenseCovariance(Covariance):
  """Covariance matrix with dense underlying representation."""

  matrix: np.ndarray

  @property
  def constraint_class(self) -> type[Constraint]:
    return DenseConstraint

  def __add__(self, other: Covariance) -> DenseCovariance:
    if not isinstance(other, DenseCovariance):
      raise TypeError(f"Expected DenseCovariance, got {type(other).__name__}")
    return DenseCovariance(self.matrix + other.matrix)

  def __sub__(self, other: Covariance) -> DenseCovariance:
    if not isinstance(other, DenseCovariance):
      raise TypeError(f"Expected DenseCovariance, got {type(other).__name__}")
    return DenseCovariance(self.matrix - other.matrix)

  def __neg__(self) -> DenseCovariance:
    return DenseCovariance(-self.matrix)

  @overload
  def __matmul__(self, other: Covariance) -> DenseCovariance:
    pass

  @overload
  def __matmul__(self, other: np.ndarray) -> np.ndarray:
    pass

  def __matmul__(
      self, other: Covariance | np.ndarray
  ) -> DenseCovariance | np.ndarray:
    if isinstance(other, DenseCovariance):
      return DenseCovariance(self.matrix @ other.matrix)
    elif isinstance(other, np.ndarray):
      return self.matrix @ other
    else:
      raise TypeError(f"Unsupported type for @: {type(other).__name__}")

  def pinv(self) -> DenseCovariance:
    return DenseCovariance(np.linalg.pinv(self.matrix, hermitian=True))


class Constraint(abc.ABC):
  """Abstract base class for linear equality constraints R x = c."""

  @property
  @abc.abstractmethod
  def is_empty(self) -> bool:
    """Returns True if the constraint is empty (no equations)."""

  @abc.abstractmethod
  def apply(self, est: Estimate) -> Estimate:
    """Returns the estimate updated to satisfy the constraint R x = c.

    Implements the restricted least squares update of Lemma 2 and Algorithm 1 of
    the paper: the result is the best linear unbiased estimate of x given est
    and the constraint, together with its covariance. Writing (v, Σ) for the
    input and (v_new, Σ_new) for the output, the update satisfies:

    * Exactness: R v_new = c, and R Σ_new = 0, so the constrained combinations
      have zero variance under Σ_new.
    * Idempotence: applying the same constraint again leaves (v_new, Σ_new)
      unchanged.
    * Order invariance: for jointly consistent constraints, applying them one
      after another in any order gives the same result as applying them all at
      once as a single stacked constraint.

    Exactness and order invariance require the constraint to be consistent with
    est, i.e. R v - c lies in the column space of R Σ; this always holds when Σ
    is positive definite.

    Args:
      est: The estimate to constrain.

    Returns:
      The constrained estimate with updated covariance.
    """

  @classmethod
  @abc.abstractmethod
  def sum(cls, total: float) -> Constraint:
    """Create a constraint fixing the sum of all entries of x to total.

    This is the single linear equation 1^T x = total.

    Args:
      total: The required sum of all entries of the estimate vector.
    """

  @classmethod
  @abc.abstractmethod
  def zero_constraints(
      cls,
      first_feature_zero_indices: Sequence[bool],
      multifeature_zero_sequence: Sequence[int] | None = None,
  ) -> Constraint:
    """Create combined first-feature and multifeature zero constraints.

    View the estimate vector x as a tensor of shape feature_shapes (e.g.
    (8, 2, 2, 63)). The resulting constraint fixes to zero every entry
    x[i, ...] where first_feature_zero_indices[i] is True, together with every
    entry in the slice x[i_0, ..., i_{k-1}, ...] selected by the prefix
    multifeature_zero_sequence = (i_0, ..., i_{k-1}). Entries covered by both
    are constrained only once.

    Args:
      first_feature_zero_indices: Boolean mask over the first feature, of length
        feature_shapes[0]. Entries whose first feature value is marked True are
        fixed to zero.
      multifeature_zero_sequence: Optional prefix of feature values indexing the
        leading dimensions of the tensor. All entries in the corresponding slice
        are fixed to zero. If None, no multifeature constraint is added.
    """

  @classmethod
  def first_feature_zero_constraints(
      cls, zero_indices: Sequence[bool]
  ) -> Constraint:
    """Create first feature zero constraints."""
    return cls.zero_constraints(zero_indices)

  @classmethod
  def multifeature_zero_constraints(
      cls, feature_values: Sequence[int]
  ) -> Constraint:
    """Create multifeature zero constraints."""
    return cls.zero_constraints([], feature_values)


class DenseConstraint(Constraint):
  """Linear constraint with dense matrix representation R @ x = c."""

  # Configuration for the feature layout (Census defaults)
  feature_shapes: ClassVar[tuple[int, ...]] = constants.DETAILED_QUERY_SHAPE

  dim: ClassVar[int] = math.prod(feature_shapes)
  first_feature_stride: ClassVar[int] = dim // feature_shapes[0]

  # pylint: disable=invalid-name

  def __init__(self, R: np.ndarray, c: np.ndarray):
    """Initializes DenseConstraint.

    Args:
      R: Matrix of shape (K, N), one row per constraint, where N is the variable
        dimension and K is the number of constraints.
      c: Vector of shape (K,) containing constraint values.
    """
    self.R = R
    self.c = c

  @property
  def is_empty(self) -> bool:
    return self.R.shape[0] == 0

  @classmethod
  def empty(cls) -> DenseConstraint:
    return cls(np.zeros((0, cls.dim)), np.zeros(0))

  @classmethod
  def sum(cls, total: float) -> DenseConstraint:
    return cls(np.ones((1, cls.dim)), np.array([total]))

  @classmethod
  def zero_constraints(
      cls,
      first_feature_zero_indices: Sequence[bool],
      multifeature_zero_sequence: Sequence[int] | None = None,
  ) -> DenseConstraint:
    mask = np.zeros(cls.feature_shapes, dtype=bool)
    if any(first_feature_zero_indices):
      mask[np.asarray(first_feature_zero_indices, dtype=bool)] = True
    if multifeature_zero_sequence is not None:
      mask[tuple(multifeature_zero_sequence)] = True
    indices = np.flatnonzero(mask)
    return cls(np.eye(cls.dim)[indices], np.zeros(indices.size))

  def apply(self, est: Estimate) -> Estimate:
    """Applies the constraint R x = c via restricted least squares.

    Computes the update of Lemma 2 and Algorithm 1 of the paper:
      L = Σ R^T (R Σ R^T)^†
      v_new = v - L (R v - c)
      Σ_new = Σ - L R Σ
    See Constraint.apply for the properties of this update.

    Args:
      est: The estimate to constrain.

    Returns:
      The constrained estimate with updated covariance.
    """
    if self.is_empty:
      return est

    if isinstance(est.cov, np.ndarray):
      cov = est.cov
    elif isinstance(est.cov, DenseCovariance):
      cov = est.cov.matrix
    elif hasattr(est.cov, "materialize"):
      # Support materialization if available (e.g. for SuccinctCovariance in
      # tests).
      cov = est.cov.materialize()
    else:
      raise TypeError(
          f"Unsupported covariance type for DenseConstraint: {type(est.cov)}"
      )

    R, c = self.R, self.c
    L = cov @ R.T @ np.linalg.pinv(R @ cov @ R.T, hermitian=True)
    new_val = est.val - L @ (R @ est.val - c)
    new_cov = cov - L @ R @ cov

    if isinstance(est.cov, DenseCovariance):
      return Estimate(new_val, DenseCovariance(new_cov))
    else:
      return Estimate(new_val, new_cov)


@dataclasses.dataclass
class Estimate:
  """Estimate vector with associated covariance matrix."""
  val: np.ndarray
  cov: np.ndarray | Covariance

  def constrain(self, constraint: Constraint) -> Estimate:
    """Returns the estimate updated to satisfy the given linear constraint.

    Equivalent to constraint.apply(self); see Constraint.apply.

    Args:
      constraint: The linear equality constraint to impose.
    """
    return constraint.apply(self)

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
