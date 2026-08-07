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

r"""Generate synthetic data for the block hierarchical postprocessing.

Sample usage:
python3 census_bluedown/generate_synthetic_data.py \
  --pass=ground_truth \
  --state=02 \
  --nmf_dir=data/2020-pl94-nmf-parquets/US_Person_PL_PROD/ \
  --out_dir=output/synthetic-ground-truth/ \
  --zip_path=data/2020_ppmf_bystate.zip

python3 census_bluedown/generate_synthetic_data.py \
  --pass=nmf \
  --state=002 \
  --ground_truth_dir=output/synthetic-ground-truth/ \
  --out_dir=output/synthetic-nmf/
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import numpy as np

from census_bluedown import constants
from census_bluedown import ground_truth
from census_bluedown import io
from census_bluedown import nmf


_PASS = flags.DEFINE_enum(
    name='pass',
    default=None,
    enum_values=['ground_truth', 'nmf'],
    help='Pass to run.',
    required=True,
)
_STATE = flags.DEFINE_string(
    name='state', default=None, help='State ID.', required=False)
_NMF_DIR = flags.DEFINE_string(
    name='nmf_dir',
    default=None,
    help='CNS directory containing the NMF with the directory structure.',
)
_GROUND_TRUTH_DIR = flags.DEFINE_string(
    name='ground_truth_dir',
    default=None,
    help='CNS directory containing the ground truth.',
)
_OUT_DIR = flags.DEFINE_string(
    name='out_dir',
    default=None,
    help='Output directory.',
    required=True,
)
_ZIP_PATH = flags.DEFINE_string(
    name='zip_path',
    default=None,
    help='Path to the zip file containing the microdata format census data.',
)
_SEED = flags.DEFINE_string(name='seed', default='1234', help='Seed.')


def generate_ground_truth():
  """Generate ground truth for the census application."""
  if _NMF_DIR.value is None:
    raise ValueError('NMF directory is required for ground truth generation.')
  if _ZIP_PATH.value is None:
    raise ValueError('Zip path is required for ground truth generation.')
  nmf_io = io.InputFormatIO(
      path_prefix=str(_NMF_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.NMF_PATH_SUFFIX,
  )
  constraint_io = io.InputFormatIO(
      path_prefix=str(_NMF_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.CONSTRAINT_PATH_SUFFIX,
  )
  out_io = io.InputFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      path_suffix=constants.NMF_PATH_SUFFIX,
  )
  out_constraint_io = io.InputFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      path_suffix=constants.CONSTRAINT_PATH_SUFFIX,
  )

  if _STATE.value is None:
    # Load the previously-generated state-level totals.
    state_total_arrays = []
    for folder_id in constants.FOLDER_IDS:
      out_io.set_subtree_folder(constants.SUBTREE_FOLDER_PATTERN % folder_id)
      state_df = out_io.read(level='State', filename='ground_truth.parquet')
      state_total_arrays.append(
          state_df[state_df['query_name'] == 'detailed_dpq'].iloc[0]['value']
      )

    # Generate root level synthetic data
    us_total = np.sum(state_total_arrays, axis=0)
    ground_truth.generate_us_total_ground_truth(
        us_total, nmf_io, constraint_io, out_io, out_constraint_io
    )
  else:
    df = ground_truth.aggregate_microdata_format(
        ground_truth.read_state_microdata_format(
            _STATE.value, str(_ZIP_PATH.value)))
    ground_truth.generate_state_ground_truth(
        _STATE.value, df, nmf_io, constraint_io, out_io, out_constraint_io
    )


def generate_nmf():
  """Generate NMF for the census application."""
  if _GROUND_TRUTH_DIR.value is None:
    raise ValueError('Ground truth directory is required for NMF generation.')

  ground_truth_io = io.InputFormatIO(
      path_prefix=str(_GROUND_TRUTH_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.NMF_PATH_SUFFIX,
  )
  constraint_io = io.InputFormatIO(
      path_prefix=str(_GROUND_TRUTH_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.CONSTRAINT_PATH_SUFFIX,
  )
  out_io = io.InputFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      path_suffix=constants.NMF_PATH_SUFFIX,
  )
  out_constraint_io = io.InputFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      path_suffix=constants.CONSTRAINT_PATH_SUFFIX,
  )

  if _STATE.value is None:
    # Generate root level NMF.
    nmf.generate_us_total_nmf(
        _SEED.value, ground_truth_io, constraint_io, out_io, out_constraint_io
    )
  else:
    nmf.generate_subtree_nmf(
        seed=_SEED.value,
        subtree=_STATE.value,
        ground_truth_io=ground_truth_io,
        constraint_io=constraint_io,
        out_io=out_io,
        out_constraint_io=out_constraint_io)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  match _PASS.value:
    case 'ground_truth':
      generate_ground_truth()
    case 'nmf':
      generate_nmf()
    case _:
      raise ValueError(f'Unsupported pass: {_PASS.value}')

if __name__ == '__main__':
  app.run(main)
