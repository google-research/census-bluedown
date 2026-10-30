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

from typing import TypeAlias
import numpy as np
from census_bluedown import estimate
from census_bluedown import succinct

Estimate: TypeAlias = estimate.Estimate


def compute_gls_estimate(
    workload: succinct.SuccinctWorkload,
    cov: succinct.SuccinctInputCovariance,
    y: np.ndarray,
) -> tuple[np.ndarray, succinct.SuccinctCovariance]:
  """Computes best linear unbiased estimate of a block from succinct inputs.

  Solves generalized least squares (GLS) for y = W x + ε with Cov(ε) = Σ
  (Lemma 1, using the succinct representation of Eq. S9):
    x_gls = (W^T Σ^{-1} W)^† W^T Σ^{-1} y
    Σ_gls = (W^T Σ^{-1} W)^†

  Using the Kronecker structure W = [W_1 ⊗ I_d; W_2 ⊗ 1_d^T] and
  Σ = diag(Σ_1 ⊗ I_d, Σ_2), W^T Σ^{-1} W decomposes over
  orthogonal projectors P_0 = I_d - (1/d) 1 1^T and P_1 = (1/d) 1 1^T:
    W^T Σ^{-1} W = (W_1^T Σ_1^{-1} W_1) ⊗ P_0 + (W_1^T Σ_1^{-1} W_1 + d W_2^T
    Σ_2^{-1} W_2) ⊗ P_1

  Thus Σ_gls = C_0 ⊗ P_0 + C_1 ⊗ P_1, where:
    C_0 = (W_1^T Σ_1^{-1} W_1)^†
    C_1 = (W_1^T Σ_1^{-1} W_1 + d W_2^T Σ_2^{-1} W_2)^†
    x' = W^T Σ^{-1} y
    x_gls = Σ_gls x'

  Args:
    workload: SuccinctWorkload object representing W.
    cov: SuccinctInputCovariance object representing Σ.
    y: Concatenated observation vector.

  Returns:
    x_gls: Best linear unbiased estimate of block values (length 2016).
    cov_gls: Associated covariance matrix in SuccinctCovariance form (Σ_gls).
  """
  prec = cov.inv()  # Precision matrix Σ^{-1}
  x_prime = workload.multiply_transpose(prec.multiply(y))

  # Contributions from detailed (W_1) and aggregate (W_2) queries:
  w1_term = (workload.w1.T * prec.diagonal_1) @ workload.w1
  w2_term = (workload.w2.T * prec.diagonal_2) @ workload.w2

  d = succinct.SuccinctCovariance.projection_dim
  cov_gls = succinct.SuccinctCovariance(
      coefficient_0=np.linalg.pinv(w1_term, hermitian=True),
      coefficient_1=np.linalg.pinv(w1_term + d * w2_term, hermitian=True),
  )

  x_gls = cov_gls @ x_prime
  return x_gls, cov_gls


def combine_estimates(
    estimate_1: Estimate,
    estimate_2: Estimate,
) -> Estimate:
  """Computes the minimum-error linear combination of two independent estimates.

  Given independent unbiased estimates with covariances Ω_1 and Ω_2, computes
  (Algorithm 2 and Lemma 5 of the paper):
    A = Ω_2 (Ω_1 + Ω_2)^†
    B = I - A
    v_combined = A v_1 + B v_2
    Ω_combined = B Ω_2

  This formula permits either or both of the input covariance matrices to be
  singular, provided v_1 - v_2 lies in the range of Ω_1 + Ω_2. All matrix
  operations are performed succinctly in the compressed Kronecker space.

  Args:
    estimate_1: First estimate to combine.
    estimate_2: Second estimate to combine.

  Returns:
    The optimal linear combination of the two estimates.
  """
  if not isinstance(
      estimate_1.cov, succinct.SuccinctCovariance
  ) or not isinstance(estimate_2.cov, succinct.SuccinctCovariance):
    raise TypeError("Expected SuccinctCovariance for both estimates")

  A = estimate_2.cov @ (estimate_1.cov + estimate_2.cov).pinv()  # pylint: disable=invalid-name
  I = succinct.SuccinctCovariance.identity()  # pylint: disable=invalid-name
  B = I - A  # pylint: disable=invalid-name

  combined_val = A @ estimate_1.val + B @ estimate_2.val
  combined_cov = B @ estimate_2.cov

  # Ensure numeric symmetry
  combined_cov.coefficient_0 = 0.5 * (
      combined_cov.coefficient_0 + combined_cov.coefficient_0.T
  )
  combined_cov.coefficient_1 = 0.5 * (
      combined_cov.coefficient_1 + combined_cov.coefficient_1.T
  )

  return Estimate(combined_val, combined_cov)


def combine(
    estimate_1: Estimate | float,
    estimate_2: Estimate | float,
) -> Estimate:
  """Compute the minimum-error linear combination of two estimates.

  All float inputs, including np.nan, are interpreted as missing data. This
  convention is made because np.nan is the value used by pandas.merge to
  indicate missing data.

  Args:
    estimate_1: First estimate, or np.nan.
    estimate_2: Second estimate, or np.nan.

  Returns:
    The optimal linear combination of the two estimates.
  """
  if not isinstance(estimate_1, Estimate):
    if not isinstance(estimate_2, Estimate):
      raise ValueError("Cannot combine two missing estimates")
    return estimate_2

  if not isinstance(estimate_2, Estimate):
    return estimate_1

  return combine_estimates(estimate_1, estimate_2)
