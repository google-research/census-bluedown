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

"""Succinct representations of matrices using Kronecker products."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import ClassVar, overload

import numpy as np

from census_bluedown import constants
from census_bluedown import estimate


@dataclasses.dataclass
class SuccinctCovariance(estimate.Covariance):
  """Implicit covariance representing C_0 ⊗ P_0 + C_1 ⊗ P_1.

  The matrix acts on vectors of length coefficient_dim * projection_dim, viewed
  as (coefficient_dim, projection_dim) arrays. Since P_0 and P_1 are
  complementary orthogonal projections satisfying P_0 + P_1 = I and P_0 P_1 =
  P_1 P_0 = 0, sums, products, inverses, and Cholesky factors all act
  factor-wise on C_0 and C_1.

  Attributes:
    coefficient_0: Matrix C_0 of shape (coefficient_dim, coefficient_dim)
      multiplying P_0.
    coefficient_1: Matrix C_1 of shape (coefficient_dim, coefficient_dim)
      multiplying P_1.
    feature_shapes: Shape of the detailed query tensor. The last feature is the
      symmetric one on which P_0 and P_1 act.
    coefficient_dim: Size m of C_0 and C_1, the product of the asymmetric
      feature sizes (32 for the Census application).
    projection_dim: Size d of P_0 and P_1, the symmetric feature size (63).
    projection_0: P_0 = I_d - J_d / d, the projection onto zero-mean vectors.
    projection_1: P_1 = J_d / d, the projection onto constant vectors.
  """

  coefficient_0: np.ndarray
  coefficient_1: np.ndarray

  feature_shapes: ClassVar[tuple[int, ...]] = constants.DETAILED_QUERY_SHAPE
  coefficient_dim: ClassVar[int] = constants.COEFFICIENT_DIM
  projection_dim: ClassVar[int] = constants.PROJECTION_DIM

  projection_0: ClassVar[np.ndarray] = (
      np.eye(projection_dim)
      - np.ones((projection_dim, projection_dim)) / projection_dim
  )
  projection_1: ClassVar[np.ndarray] = (
      np.ones((projection_dim, projection_dim)) / projection_dim
  )

  @property
  def constraint_class(self) -> type[estimate.Constraint]:
    return SuccinctConstraint

  @classmethod
  def identity(cls) -> SuccinctCovariance:
    """Returns the identity matrix I_m ⊗ P_0 + I_m ⊗ P_1 = I."""
    return cls(np.eye(cls.coefficient_dim), np.eye(cls.coefficient_dim))

  def __add__(self, other: estimate.Covariance) -> SuccinctCovariance:
    """Adds another SuccinctCovariance factor-wise."""
    if not isinstance(other, SuccinctCovariance):
      raise TypeError(
          f"Expected SuccinctCovariance, got {type(other).__name__}"
      )
    return SuccinctCovariance(
        self.coefficient_0 + other.coefficient_0,
        self.coefficient_1 + other.coefficient_1,
    )

  def __sub__(self, other: estimate.Covariance) -> SuccinctCovariance:
    """Subtracts another SuccinctCovariance factor-wise."""
    if not isinstance(other, SuccinctCovariance):
      raise TypeError(
          f"Expected SuccinctCovariance, got {type(other).__name__}"
      )
    return SuccinctCovariance(
        self.coefficient_0 - other.coefficient_0,
        self.coefficient_1 - other.coefficient_1,
    )

  def __neg__(self) -> SuccinctCovariance:
    """Negates both coefficients."""
    return SuccinctCovariance(-self.coefficient_0, -self.coefficient_1)

  def __mul__(self, scalar: float) -> SuccinctCovariance:
    """Scales both coefficients by a scalar."""
    return SuccinctCovariance(
        self.coefficient_0 * scalar,
        self.coefficient_1 * scalar,
    )

  def __rmul__(self, scalar: float) -> SuccinctCovariance:
    """Scales both coefficients by a scalar."""
    return self * scalar

  @overload
  def __matmul__(self, other: estimate.Covariance) -> SuccinctCovariance:
    pass

  @overload
  def __matmul__(self, other: np.ndarray) -> np.ndarray:
    pass

  def __matmul__(
      self, other: estimate.Covariance | np.ndarray
  ) -> SuccinctCovariance | np.ndarray:
    """Computes the matrix product (with another SuccinctCovariance) or matrix-vector product.

    When other is an ndarray x of shape (2016,), computes Σ x using the
    vec-trick with orthogonal projection matrices P_0 and P_1.

    Args:
      other: Another SuccinctCovariance matrix or a 1D numpy array of shape
        (2016,).

    Returns:
      The product SuccinctCovariance or 1D numpy array.
    """
    if isinstance(other, SuccinctCovariance):
      return SuccinctCovariance(
          self.coefficient_0 @ other.coefficient_0,
          self.coefficient_1 @ other.coefficient_1,
      )
    elif isinstance(other, np.ndarray):
      other_matrix = other.reshape(self.coefficient_dim, self.projection_dim)
      # In terms of general projection matrices P_0 and P_1, this computes:
      #   self.coefficient_0 @ other_matrix @ self.projection_0.T
      #   + self.coefficient_1 @ other_matrix @ self.projection_1.T
      # Using P_1 = J/d and P_0 = I - J/d, other_matrix @ P_1.T equals the row
      # means broadcast across columns, avoiding dense (63, 63) matrix products.
      row_means = other_matrix.mean(axis=1, keepdims=True)
      result_matrix = (
          self.coefficient_0 @ (other_matrix - row_means)
          + self.coefficient_1 @ row_means
      )
      return result_matrix.flatten()
    else:
      raise TypeError(f"Unsupported type for @: {type(other).__name__}")

  @property
  def T(self) -> SuccinctCovariance:
    """Returns the transpose C_0^T ⊗ P_0 + C_1^T ⊗ P_1."""
    return SuccinctCovariance(self.coefficient_0.T, self.coefficient_1.T)

  def inv(self) -> SuccinctCovariance:
    """Computes the exact inverse C_0^{-1} ⊗ P_0 + C_1^{-1} ⊗ P_1."""
    return SuccinctCovariance(
        np.linalg.inv(self.coefficient_0),
        np.linalg.inv(self.coefficient_1),
    )

  def pinv(self) -> SuccinctCovariance:
    """Computes the pseudoinverse C_0^+ ⊗ P_0 + C_1^+ ⊗ P_1."""
    return SuccinctCovariance(
        np.linalg.pinv(self.coefficient_0, hermitian=True),
        np.linalg.pinv(self.coefficient_1, hermitian=True),
    )

  def cholesky(self) -> SuccinctCovariance:
    """Returns factor L = L_0 ⊗ P_0 + L_1 ⊗ P_1 satisfying L @ L.T = self."""
    return SuccinctCovariance(
        np.linalg.cholesky(self.coefficient_0),
        np.linalg.cholesky(self.coefficient_1),
    )

  def principal_submatrix(self, rows: Sequence[int]) -> SuccinctCovariance:
    """Returns the principal submatrix on the given coefficient rows."""
    index = np.ix_(rows, rows)
    return SuccinctCovariance(
        self.coefficient_0[index], self.coefficient_1[index]
    )

  def materialize(self) -> np.ndarray:
    """Materialize the dense covariance matrix."""
    return np.kron(self.coefficient_0, self.projection_0) + np.kron(
        self.coefficient_1, self.projection_1
    )

  def flatten(self) -> np.ndarray:
    """Flattens coefficient_0 and coefficient_1 into a single 1D array."""
    return np.concatenate(
        [self.coefficient_0.flatten(), self.coefficient_1.flatten()]
    )

  @classmethod
  def from_flattened(cls, arr: np.ndarray) -> SuccinctCovariance:
    """Reconstructs a SuccinctCovariance from a flattened 1D array."""
    m = cls.coefficient_dim
    c0, c1 = np.split(arr.flatten(), 2)
    return cls(c0.reshape((m, m)), c1.reshape((m, m)))


@dataclasses.dataclass
class SuccinctInputCovariance:
  """Implicit diagonal covariance representing diag(Σ_1 ⊗ I_d, Σ_2)."""

  diagonal_1: np.ndarray  # shape (m1,)
  diagonal_2: np.ndarray  # shape (m2,)
  projection_dim: ClassVar[int] = constants.PROJECTION_DIM

  def inv(self) -> SuccinctInputCovariance:
    """Computes Σ^{-1} = diag(Σ_1^{-1} ⊗ I_d, Σ_2^{-1}) by inverting the diagonals."""
    return SuccinctInputCovariance(1.0 / self.diagonal_1, 1.0 / self.diagonal_2)

  def multiply(self, y: np.ndarray) -> np.ndarray:
    """Computes the matrix-vector product Σ y without materializing Σ.

    Multiplies y_1 by diagonal Σ_1 (broadcasting across all d race categories)
    and y_2 by diagonal Σ_2.

    Args:
      y: 1D vector of shape (m1 * 63 + m2,) in the observation space.

    Returns:
      1D vector of shape (m1 * 63 + m2,) containing Σ y.
    """
    m1 = len(self.diagonal_1)
    m2 = len(self.diagonal_2)
    d = self.projection_dim
    if len(y) != m1 * d + m2:
      raise ValueError(f"y has wrong shape: {y.shape}, expected {m1 * d + m2}")
    y1 = y[: m1 * d].reshape(m1, d)
    y2 = y[m1 * d :]

    res1 = self.diagonal_1[:, np.newaxis] * y1
    res2 = self.diagonal_2 * y2
    return np.concatenate([res1.flatten(), res2])


@dataclasses.dataclass
class SuccinctWorkload:
  """Implicit workload matrix representing W = [W_1 ⊗ I_d; W_2 ⊗ 1_d^T]."""

  w1: np.ndarray  # shape (m1, coefficient_dim)
  w2: np.ndarray  # shape (m2, coefficient_dim)
  coefficient_dim: ClassVar[int] = constants.COEFFICIENT_DIM
  projection_dim: ClassVar[int] = constants.PROJECTION_DIM

  def multiply(self, x: np.ndarray) -> np.ndarray:
    """Computes the matrix-vector product W x without materializing W.

    Args:
      x: 1D vector of shape (2016,) in the block domain space.

    Returns:
      1D vector of shape (m1 * 63 + m2,) representing query measurements,
      where the first m1 * 63 entries correspond to individual queries (W_1 ⊗
      I_d)
      and the last m2 entries correspond to aggregate queries (W_2 ⊗ 1_d^T).
    """
    m = self.coefficient_dim
    d = self.projection_dim
    if len(x) != m * d:
      raise ValueError(f"x has wrong shape: {x.shape}, expected {m * d}")
    x_mat = x.reshape(m, d)
    res1 = self.w1 @ x_mat  # shape (m1, d)
    res2 = self.w2 @ np.sum(x_mat, axis=1)  # shape (m2,)
    return np.concatenate([res1.flatten(), res2])

  def multiply_transpose(self, y: np.ndarray) -> np.ndarray:
    """Computes the transposed matrix-vector product W^T y without materializing W.

    Args:
      y: 1D vector of shape (m1 * 63 + m2,) in the query observation space.

    Returns:
      1D vector of shape (2016,) in the block domain space.
    """
    m1 = len(self.w1)
    m2 = len(self.w2)
    d = self.projection_dim
    if len(y) != m1 * d + m2:
      raise ValueError(f"y has wrong shape: {y.shape}, expected {m1 * d + m2}")
    y1 = y[: m1 * d].reshape(m1, d)
    y2 = y[m1 * d :]

    res1 = self.w1.T @ y1  # shape (m, d)
    res2 = (self.w2.T @ y2)[
        :, np.newaxis
    ]  # shape (m, 1), will broadcast to (m, d)
    return (res1 + res2).flatten()


@dataclasses.dataclass
class SuccinctConstraint(estimate.Constraint):
  """Implicit constraint matrix representing R = [R_1 ⊗ I_d; R_2 ⊗ 1_d^T] and values c."""

  r1: np.ndarray  # shape (k1, coefficient_dim)
  r2: np.ndarray  # shape (k2, coefficient_dim)
  c1: np.ndarray  # shape (k1 * projection_dim,)
  c2: np.ndarray  # shape (k2,)

  feature_shapes: ClassVar[tuple[int, ...]] = constants.DETAILED_QUERY_SHAPE
  coefficient_dim: ClassVar[int] = constants.COEFFICIENT_DIM
  projection_dim: ClassVar[int] = constants.PROJECTION_DIM

  @classmethod
  def empty(cls) -> SuccinctConstraint:
    return cls(
        np.zeros((0, cls.coefficient_dim)),
        np.zeros((0, cls.coefficient_dim)),
        np.zeros(0),
        np.zeros(0),
    )

  @property
  def is_empty(self) -> bool:
    return self.r1.shape[0] == 0 and self.r2.shape[0] == 0

  @classmethod
  def combine(
      cls, constraints: Sequence[SuccinctConstraint]
  ) -> SuccinctConstraint:
    """Vertically stacks multiple SuccinctConstraint instances into a single constraint.

    The resulting constraint enforces R_total x = c_total, where R_1 and R_2
    blocks and target vectors c_1 and c_2 are concatenated.
    """
    r1_list = [c.r1 for c in constraints if c.r1.size > 0]
    r2_list = [c.r2 for c in constraints if c.r2.size > 0]
    c1_list = [c.c1 for c in constraints if c.c1.size > 0]
    c2_list = [c.c2 for c in constraints if c.c2.size > 0]

    r1 = np.vstack(r1_list) if r1_list else np.zeros((0, cls.coefficient_dim))
    r2 = np.vstack(r2_list) if r2_list else np.zeros((0, cls.coefficient_dim))
    c1 = np.concatenate(c1_list) if c1_list else np.zeros(0)
    c2 = np.concatenate(c2_list) if c2_list else np.zeros(0)
    return cls(r1, r2, c1, c2)

  def multiply(self, x: np.ndarray) -> np.ndarray:
    """Computes the matrix-vector product R x without materializing R.

    Args:
      x: 1D vector of shape (2016,) in the block domain space.

    Returns:
      1D vector of shape (k1 * 63 + k2,) representing constraint values R x.
    """
    m = self.coefficient_dim
    d = self.projection_dim
    if len(x) != m * d:
      raise ValueError(f"x has wrong shape: {x.shape}, expected {m * d}")
    x_mat = x.reshape(m, d)
    res1 = self.r1 @ x_mat  # shape (k1, d)
    res2 = self.r2 @ np.sum(x_mat, axis=1)  # shape (k2,)
    return np.concatenate([res1.flatten(), res2])

  def gram(self) -> SuccinctCovariance:
    """Returns R^T R as a SuccinctCovariance."""
    r1_gram = (
        self.r1.T @ self.r1
        if self.r1.size
        else np.zeros((self.coefficient_dim, self.coefficient_dim))
    )
    r2_gram = (
        self.r2.T @ self.r2
        if self.r2.size
        else np.zeros((self.coefficient_dim, self.coefficient_dim))
    )
    return SuccinctCovariance(
        r1_gram,
        r1_gram + self.projection_dim * r2_gram,
    )

  def apply(self, est: estimate.Estimate) -> estimate.Estimate:
    """Applies equality constraints R x = r via restricted least squares (Lemma 8).

    Computes the constrained estimate (x_new, Sigma_new) from (x, Sigma):
      L_A       = A R_1^T (R_1 A R_1^T)^+                         (Eq. S10)
      L_B       = B R_tilde^T (R_tilde B R_tilde^T)^+             (Eq. S11)
      I - LR    = (I - L_A R_1) ⊗ P_0 + (I - L_B R_tilde) ⊗ P_1   (Eqs. S12-S15)
      Sigma_new = (I - LR) Sigma                                  (Algorithm 1)
      x_new     = (I - LR) x + L r                                (Algorithm 1,
      Eq. S14)

    Args:
      est: The unconstrained Estimate to update.

    Returns:
      The constrained Estimate.
    """
    if not isinstance(est.cov, SuccinctCovariance):
      raise TypeError("Expected SuccinctCovariance")
    if self.is_empty:
      return est

    A, B = est.cov.coefficient_0, est.cov.coefficient_1  # pylint: disable=invalid-name
    R1, R2 = self.r1, self.r2  # pylint: disable=invalid-name
    R_tilde = np.vstack([R1, R2])  # pylint: disable=invalid-name

    # Gain matrices L_A (Eq. S10) and L_B (Eq. S11)
    L_A = (  # pylint: disable=invalid-name
        A @ R1.T @ np.linalg.pinv(R1 @ A @ R1.T, hermitian=True)
        if R1.size
        else np.zeros((self.coefficient_dim, 0))
    )
    L_B = B @ R_tilde.T @ np.linalg.pinv(R_tilde @ B @ R_tilde.T, hermitian=True)  # pylint: disable=invalid-name

    # Succinct projection operator I - LR (Eqs. S12, S13, S15)
    I_minus_LR = SuccinctCovariance.identity() - SuccinctCovariance(  # pylint: disable=invalid-name
        L_A @ R1 if R1.size else np.zeros_like(A),
        L_B @ R_tilde,
    )

    # Sigma_new = (I - LR) Sigma
    new_cov = I_minus_LR @ est.cov

    # Enforce numerical symmetry of the new covariance matrix.
    new_cov.coefficient_0 = 0.5 * (
        new_cov.coefficient_0 + new_cov.coefficient_0.T
    )
    new_cov.coefficient_1 = 0.5 * (
        new_cov.coefficient_1 + new_cov.coefficient_1.T
    )

    # x_new = (I - LR) x + L r (Eq. S14)
    new_val = I_minus_LR @ est.val
    if np.any(self.c1) or np.any(self.c2):
      d = self.projection_dim
      c1_mat = self.c1.reshape(-1, d)
      c1_bar = c1_mat.mean(axis=1, keepdims=True)
      c_tilde = np.vstack([c1_bar, self.c2.reshape(-1, 1) / d])
      Lr = L_A @ (c1_mat - c1_bar) + L_B @ c_tilde  # pylint: disable=invalid-name
      new_val = new_val + Lr.flatten()

    return estimate.Estimate(new_val, new_cov)

  @classmethod
  def sum(cls, total: float) -> SuccinctConstraint:
    """Returns a constraint fixing the sum of all block elements to total."""
    r1 = np.zeros((0, cls.coefficient_dim))  # pylint: disable=invalid-name
    r2 = np.ones((1, cls.coefficient_dim))  # pylint: disable=invalid-name
    c1 = np.zeros(0)
    c2 = np.array([total])
    return cls(r1, r2, c1, c2)

  @classmethod
  def zero_constraints(
      cls,
      first_feature_zero_indices: Sequence[bool],
      multifeature_zero_sequence: Sequence[int] | None = None,
  ) -> SuccinctConstraint:
    """Creates deduplicated zero constraints for first-feature and multifeature zeros."""
    asym_shape = cls.feature_shapes[:-1]
    mask = np.zeros(asym_shape, dtype=bool)
    if any(first_feature_zero_indices):
      mask[np.asarray(first_feature_zero_indices, dtype=bool), :, :] = True
    if multifeature_zero_sequence is not None:
      if len(multifeature_zero_sequence) > len(asym_shape):
        raise ValueError("Too many feature values")
      mask[tuple(multifeature_zero_sequence)] = True
    if not mask.any():
      return cls.empty()

    r1 = np.eye(cls.coefficient_dim)[mask.flatten()]
    return cls(
        r1=r1,
        r2=np.zeros((0, cls.coefficient_dim)),
        c1=np.zeros(r1.shape[0] * cls.projection_dim),
        c2=np.zeros(0),
    )

  @classmethod
  def first_feature_zero_constraints(
      cls, zero_indices: Sequence[bool]
  ) -> SuccinctConstraint:
    """Returns a constraint fixing specified values of the first feature to zero."""
    return cls.zero_constraints(zero_indices)

  @classmethod
  def multifeature_zero_constraints(
      cls, feature_values: Sequence[int]
  ) -> SuccinctConstraint:
    """Returns a constraint fixing a prefix combination of asymmetric features to zero."""
    return cls.zero_constraints([], feature_values)
