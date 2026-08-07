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

"""Constants used in the block hierarchical postprocessing algorithm."""
import enum

CONDITION_SCALING = 50
USE_ALTERNATE_PASSES = False
# Whether to use equality constraints for the bottom-up pass.
CONSTRAIN_BOTTOM_UP = True


# Nonlinear solver type
class SolverType(enum.Enum):
  GUROBI = 0
  SCIP = 1

SOLVER = SolverType.GUROBI

# Column names for Pandas DataFrames.
ID = 'geocode'
ID_PREFIX = 'geocode_prefix'
ESTIMATE = 'estimate'
QUERY_SHAPE = 'query_shape'
VARIANCE = 'variance'
VALUE = 'value'

# Column names and values for constraint Pandas DataFrames.
QUERY_NAME = 'query_name'
FIRST_FEATURE_UB_CONSTRAINT = 'hhgq_total_ub_con'
FIRST_FEATURE_LB_CONSTRAINT = 'hhgq_total_lb_con'

# Specification of the feature combination to constrain to zero in the
# multifeature zero constraint. Since the first two features are housing type
# and voting age, sequence (3, 0) corresponds to housing type value 0 (nursing
# facilities / skilled nursing facilities) and voting age value 0 (age < 18),
# corresponding to the constraint that no minors live in nursing facilities.
MULTIFEATURE_ZERO_SEQUENCE = (3, 0)

# Shape of the detailed query used for the census application.
DETAILED_QUERY_SHAPE = (8, 2, 2, 63)
TOTAL_QUERY_SHAPE = (1, 1, 1, 1)

CENSUS_ASYMMETRIC_QUERY_SHAPE = (3, 1, 1, 1)
CENSUS_ASYMMETRIC_QUERY_PARTITION = (1, 4, 3)
CENSUS_ROUNDER_SECOND_PASS_QUERY_SHAPES = (
    (1, 1, 1, 1),
    (8, 1, 1, 1),
    (8, 1, 2, 1),
    (8, 1, 2, 63),
    (8, 2, 2, 63),
)

# Tree structure
LEVELS = ('State', 'County', 'Tract', 'Block_Group', 'Block')
ROOT_LEVEL = 'US'
LEAF_LEVEL = LEVELS[-1]
MULTI_PASS_LEVELS = ('State', 'County', 'Tract', 'Block_Group')
SINGLE_PASS_LEVELS = ('US', 'Block')

# String formats for subtree folders with a given state ID.
SUBTREE_FOLDER_PATTERN = 'State=%s'
SUBTREE_FOLDER_PATTERNS = ('State=0%s', 'State=1%s')
SUBTREE_GEOCODE_PATTERNS = ('0%s', '1%s')

# State IDs for 2010 and 2020 census data. The lists of IDs are almost identical
# except that the states with AIAN regions differ slightly between the two
# datasets. In particular, the 2010 data includes ID 144 (AIAN region for Rhode
# Island) while the 2020 data does not, and the 2020 data includes IDs 118 and
# 147 (AIAN regions for Indiana and Tennessee) while the 2010 data does not.
# These are FIPS codes, with single-digit prefix 0=non-AIAN, 1=AIAN.
FOLDER_IDS_2010_DEMONSTRATION_DATA = (
    '001', '002', '004', '005', '006', '008', '009', '010', '011', '012', '013',
    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025',
    '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
    '037', '038', '039', '040', '041', '042', '044', '045', '046', '047', '048',
    '049', '050', '051', '053', '054', '055', '056', '101', '102', '104', '106',
    '108', '109', '112', '113', '115', '116', '119', '120', '122', '123', '125',
    '126', '127', '128', '130', '131', '132', '135', '136', '137', '138', '140',
    '141', '144', '145', '146', '148', '149', '151', '153', '155', '156'
)
FOLDER_IDS_2020 = (
    '001', '002', '004', '005', '006', '008', '009', '010', '011', '012', '013',
    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025',
    '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
    '037', '038', '039', '040', '041', '042', '044', '045', '046', '047', '048',
    '049', '050', '051', '053', '054', '055', '056', '101', '102', '104', '106',
    '108', '109', '112', '113', '115', '116', '118', '119', '120', '122', '123',
    '125', '126', '127', '128', '130', '131', '132', '135', '136', '137', '138',
    '140', '141', '145', '146', '147', '148', '149', '151', '153', '155', '156'
)
FOLDER_IDS = FOLDER_IDS_2020

# Normalization used for computing error metrics.
# Sources: https://arxiv.org/pdf/2204.08986 and
# https://www.census.gov/geographies/reference-files/time-series/geo/tallies.html
REGIONS_PER_LEVEL_2010 = {
    # arXiv paper used blocks with potential positive population: 6398202
    'Block': 11078297,
    'Block_Group': 217740,
    'Tract': 73057,
    'County': 3143,
    'State': 51,
}
REGIONS_PER_LEVEL_2020 = {
    # Blocks with potential positive population: 5892698
    'Block': 8132968,
    'Block_Group': 239780,
    'Tract': 84414,
    'County': 3143,
    'State': 51,
}
REGIONS_PER_LEVEL = REGIONS_PER_LEVEL_2020

# File names for Pandas DataFrames.
PICKLE_EXTENSION = '.pickle'
PARQUET_EXTENSION = '.parquet'
EXTENSION = PARQUET_EXTENSION
GROUND_TRUTH_FNAME = 'ground_truth' + PARQUET_EXTENSION
NMF_FNAME = 'nmf' + PARQUET_EXTENSION
CONSTRAINT_FNAME = 'constraints'
SUBTREE_TOTALS_FNAME = 'state_totals' + EXTENSION
BLOCK_EST_FNAME = 'block_estimate' + EXTENSION
LOWER_EST_FNAME = 'lower_estimate' + EXTENSION
UPPER_EST_FNAME = 'upper_estimate' + EXTENSION
COMBINED_EST_FNAME = 'combined_estimate' + EXTENSION
OPTIMIZED_EST_FNAME = 'optimized_estimate' + EXTENSION
BASELINE_EST_FNAME = 'baseline_estimate' + EXTENSION
ERRORS_FNAME = 'query_errors' + EXTENSION
ALTERNATE_ERRORS_FNAME = 'alternate_query_errors' + EXTENSION


if not CONSTRAIN_BOTTOM_UP:
  BLOCK_EST_FNAME = 'block_unconstrained' + EXTENSION
  LOWER_EST_FNAME = 'lower_unconstrained' + EXTENSION
  COMBINED_EST_FNAME = 'combined_unconstrained' + EXTENSION
  OPTIMIZED_EST_FNAME = 'optimized_unconstrained' + EXTENSION


NMF_PATH_SUFFIX = '.parquet/DPQuery/'
CONSTRAINT_PATH_SUFFIX = '.parquet/Constraint/'


# Objective function options
class ObjectiveFunctionType(enum.Enum):
  FULL = 0
  TOTAL_ONLY = 1
  FULL_ROUNDER = 2
  TOTAL_ONLY_ROUNDER = 3

L2_PASSES = (ObjectiveFunctionType.TOTAL_ONLY, ObjectiveFunctionType.FULL)
ROUNDER_PASSES = (ObjectiveFunctionType.TOTAL_ONLY_ROUNDER,
                  ObjectiveFunctionType.FULL_ROUNDER)
TOTAL_ONLY_PASSES = (ObjectiveFunctionType.TOTAL_ONLY,
                     ObjectiveFunctionType.TOTAL_ONLY_ROUNDER)
FULL_PASSES = (ObjectiveFunctionType.FULL,
               ObjectiveFunctionType.FULL_ROUNDER)
ALL_PASSES = (ObjectiveFunctionType.TOTAL_ONLY,
              ObjectiveFunctionType.FULL,
              ObjectiveFunctionType.TOTAL_ONLY_ROUNDER,
              ObjectiveFunctionType.FULL_ROUNDER)
ALTERNATE_PASSES = (
    ObjectiveFunctionType.TOTAL_ONLY,
    ObjectiveFunctionType.FULL,)
