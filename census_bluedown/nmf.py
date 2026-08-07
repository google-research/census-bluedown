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

"""Generate nmf from aggregated ground truth."""

from collections.abc import Callable
import functools
import math

import numpy as np
import pandas as pd
from scipy import stats

from census_bluedown import constants
from census_bluedown import io

GEOCODE16 = 'geocode16'
VALUE = 'value'
VARIANCE = 'variance'


@functools.cache
def get_discrete_gaussian_generator(
    rng: np.random.Generator,
    sigma: float,
    tail_mass: float = 1e-40
) -> Callable[[float], np.ndarray]:
  """Returns a generator to samples from truncated Discrete Gaussian.

  N_Z(0, sigma^2) is the distribution supported over all integers given by the
  probability mass function p(x) proportional to exp(- 0.5 * (x/sigma)^2).

  This method returns a generator that samples from the truncated discrete
  Gaussian, which is the discrete Gaussian distribution conditioned on
  being in the interval [-tau, tau], where tau is chosen such that
  Pr_{X ~ N_Z(0, sigma^2)}[|X| >= tau] < tail_mass.

  Proposition 25 of Cannone et al. (https://arxiv.org/abs/2004.00010) shows
  Pr_{X ~ N_Z(0, sigma^2)}[X >= tau] <= Pr_{X ~ N(0, sigma^2)}[X >= tau-1]
  where N(0, sigma^2) is the standard normal distribution with scale sigma.
  Thus, tau can be chosen based on tails of standard Gaussian.

  Args:
    rng: The random number generator to use for sampling.
    sigma: The parameter sigma of the discrete Gaussian distribution.
    tail_mass: The mass of the tail that will be ignored for the sampling.
  Returns:
    A generator that samples from the truncated discrete Gaussian; this is a
    lambda function, that given a `size` parameter, returns a numpy array of
    samples with shape equal to `size`.
  """
  # Note, if we want to use a fixed `tail_mass`, we can also hard-code the value
  # of stats.norm.ppf(0.5 * tail_mass) in the expression below, saving the call
  # to `stats.norm.ppf`.
  tau = math.ceil(1 - stats.norm.ppf(0.5 * tail_mass) * sigma)

  # Generator will sample conditioned on [-{tau}, {tau}].
  support = np.arange(-tau, tau+1, dtype=int)
  pmf = np.exp(- 0.5 * np.power((support / sigma), 2))
  pmf = pmf / np.sum(pmf)
  return lambda size: rng.choice(support, size=size, p=pmf)  # pyrefly: ignore[no-matching-overload]


def add_noise_to_query(
    row: pd.Series,
    rng: np.random.Generator,
) -> list[int]:
  """Sample noise for a query."""
  scale = row[VARIANCE]**0.5
  noise_generator = get_discrete_gaussian_generator(rng=rng, sigma=scale)
  return list(np.array(row[VALUE], dtype=int)
              + noise_generator(len(row[VALUE])))


def generate_subtree_nmf(
    seed: str,
    subtree: str,
    ground_truth_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    out_constraint_io: io.AbstractBlockHierarchicalIO,
) -> None:
  """Generate nmf for a subtree folder (AIAN or non-AIAN region of state).

  Writes the nmf values and constraint data to the output IO objects.

  Args:
    seed: The seed for the random number generator, to be concatenated with
      the subtree code.
    subtree: The subtree code, as a length-3 string in constants.FOLDER_IDS.
    ground_truth_io: The input IO object for the ground truth data.
    constraint_io: The input IO object for the constraint data.
    out_io: The output IO object for the nmf data.
    out_constraint_io: The output IO object for the constraint data.
  """
  rng = np.random.default_rng(seed=int(seed + subtree))

  ground_truth_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
  out_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
  constraint_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree
  out_constraint_io.subtree_folder = constants.SUBTREE_FOLDER_PATTERN % subtree

  for level in constants.LEVELS:
    try:
      constraint_df = constraint_io.read(level, None)
      out_constraint_io.write(level, constants.CONSTRAINT_FNAME, constraint_df)
    except FileNotFoundError:
      pass

    try:
      df = ground_truth_io.read(level, None)
    except FileNotFoundError:
      continue
    df[VALUE] = df.apply(add_noise_to_query, args=(rng,), axis=1)
    out_io.write(level, constants.NMF_FNAME, df)


def generate_us_total_nmf(
    seed: str,
    ground_truth_io: io.AbstractBlockHierarchicalIO,
    constraint_io: io.AbstractBlockHierarchicalIO,
    out_io: io.AbstractBlockHierarchicalIO,
    out_constraint_io: io.AbstractBlockHierarchicalIO,
) -> None:
  """Write the US-level ground truth."""
  rng = np.random.default_rng(seed=int(seed))

  df = ground_truth_io.read(constants.ROOT_LEVEL, None)
  df[VALUE] = df.apply(add_noise_to_query, args=(rng,), axis=1)
  out_io.write(constants.ROOT_LEVEL, constants.NMF_FNAME, df)

  constraint_df = constraint_io.read(constants.ROOT_LEVEL, None)
  out_constraint_io.write(level=constants.ROOT_LEVEL,
                          filename=constants.CONSTRAINT_FNAME,
                          df=constraint_df)
