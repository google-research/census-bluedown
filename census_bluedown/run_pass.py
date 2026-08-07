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

r"""Run passes of the block hierarchical postprocessing algorithm.

Sample usage:
python3 census_bluedown/run_pass.py \
  --pass=bottom_up \
  --state=125 \
  --nmf_dir=output/synthetic-nmf/ \
  --out_dir=output/synthetic-processed/

python3 census_bluedown/run_pass.py \
  --pass=bottom_up \
  --out_dir=output/synthetic-processed/ \
  --subtree_totals_dir=output/synthetic-processed/

python3 census_bluedown/run_pass.py \
  --pass=top_down \
  --in_dir=output/synthetic-processed/ \
  --out_dir=output/synthetic-processed/ \
  --subtree_totals_dir=output/synthetic-processed/

python3 census_bluedown/run_pass.py \
  --pass=top_down \
  --state=125 \
  --in_dir=output/synthetic-processed/ \
  --out_dir=output/synthetic-processed/
"""

from collections.abc import Sequence

from absl import app
from absl import flags

from census_bluedown import block
from census_bluedown import compute_errors
from census_bluedown import constants
from census_bluedown import io
from census_bluedown import main_passes
from census_bluedown import root


BLOCK_SHAPE = block.BlockShape(
    shape=(8, 2, 2, 63),
    num_asymmetric_features=3,
    asymmetric_queries=(
        block.AsymmetricQuery(
            num_features=4,
            sliced_feature=0,
            partition=(1, 4, 3)),
    )
)

_PASS = flags.DEFINE_enum(
    name='pass',
    default=None,
    enum_values=['bottom_up', 'top_down', 'baseline_census_topdown',
                 'compute_errors', 'compute_alternate_errors', 'test'],
    help='Pass to run.',
    required=True,
)
_STATE = flags.DEFINE_string(
    name='state', default=None, help='State ID.', required=False)
_NMF_DIR = flags.DEFINE_string(
    name='nmf_dir',
    default=None,
    help='CNS directory containing the NMF.',
)
_IN_DIR = flags.DEFINE_string(
    name='in_dir',
    default=None,
    help='Input directory.',
)
_OUT_DIR = flags.DEFINE_string(
    name='out_dir',
    default=None,
    help='Output directory.',
    required=True,
)
_SUBTREE_TOTALS_DIR = flags.DEFINE_string(
    name='subtree_totals_dir',
    default=None,
    help='Directory containing subtree totals.',
    required=False,
)


def bottom_up() -> None:
  """Run bottom-up pass."""
  nmf_io = io.InputFormatIO(
      path_prefix=str(_NMF_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.NMF_PATH_SUFFIX,
  )
  nmf_constraint_io = io.InputFormatIO(
      path_prefix=str(_NMF_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.CONSTRAINT_PATH_SUFFIX,
  )
  processing_io = io.ProcessingFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      split_estimates=True
  )
  constraint_out_io = io.ProcessingFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      split_estimates=False,
  )

  if _STATE.value is None:
    if _SUBTREE_TOTALS_DIR.value is None:
      raise ValueError('subtree_totals_dir must be specified for root pass.')
    subtree_totals_io = io.ProcessingFormatIO(
        path_prefix=str(_SUBTREE_TOTALS_DIR.value),
        subtree_folder='US',
        read_only=True,
    )
    root.root_bottom_up_passes(
        nmf_io=nmf_io,
        constraint_io=nmf_constraint_io,
        subtree_totals_io=subtree_totals_io,
        processed_io=processing_io,
        block_shape=BLOCK_SHAPE,
        constraint_out_io=constraint_out_io,
    )
  else:
    subtree_folder = constants.SUBTREE_FOLDER_PATTERN % _STATE.value
    nmf_io.set_subtree_folder(subtree_folder)
    nmf_constraint_io.set_subtree_folder(subtree_folder)
    processing_io.set_subtree_folder(subtree_folder)
    constraint_out_io.set_subtree_folder(subtree_folder)
    main_passes.input_pass(
        input_io=nmf_io,
        constraint_io=nmf_constraint_io,
        out_io=processing_io,
        block_shape=BLOCK_SHAPE,
        constraint_out_io=constraint_out_io,
    )
    main_passes.bottom_up_pass(
        in_io=processing_io,
        out_io=processing_io,
        block_shape=BLOCK_SHAPE,
    )


def top_down() -> None:
  """Run root topdown baseline pass."""
  in_io = io.ProcessingFormatIO(
      path_prefix=str(_IN_DIR.value),
      subtree_folder='',
      read_only=False,
      split_estimates=True
  )
  # Don't split output estimates since there is no explicit variance.
  out_io = io.ProcessingFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      split_estimates=False
  )
  constraint_io = io.ProcessingFormatIO(
      path_prefix=str(_IN_DIR.value),
      subtree_folder='',
      read_only=True,
      split_estimates=False,
  )
  if _STATE.value is None:
    if _SUBTREE_TOTALS_DIR.value is None:
      raise ValueError('subtree_totals_dir must be specified for root pass.')
    subtree_totals_io = io.ProcessingFormatIO(
        path_prefix=str(_SUBTREE_TOTALS_DIR.value),
        subtree_folder='US',
        read_only=True,
    )
    root.root_top_down_pass(
        in_io=in_io,
        out_io=out_io,
        constraint_io=constraint_io,
        subtree_totals_io=subtree_totals_io,
        block_shape=BLOCK_SHAPE
    )
  else:
    subtree_folder = constants.SUBTREE_FOLDER_PATTERN % _STATE.value
    in_io.set_subtree_folder(subtree_folder)
    out_io.set_subtree_folder(subtree_folder)
    constraint_io.set_subtree_folder(subtree_folder)

    main_passes.optimization_top_down_pass(
        in_io=in_io,
        out_io=out_io,
        constraint_io=constraint_io,
        block_shape=BLOCK_SHAPE
    )


def baseline_census_topdown() -> None:
  """Run root topdown baseline pass."""
  if _NMF_DIR.value is None:
    raise ValueError('nmf_dir must be specified for baseline topdown pass.')
  nmf_io = io.InputFormatIO(
      path_prefix=str(_NMF_DIR.value),
      subtree_folder='',
      read_only=True,
      path_suffix=constants.NMF_PATH_SUFFIX,
  )
  constraint_io = io.ProcessingFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=True,
      split_estimates=False,
  )
  out_io = io.ProcessingFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      split_estimates=False,
  )

  if _STATE.value is None:
    if _SUBTREE_TOTALS_DIR.value is None:
      raise ValueError('subtree_totals_dir must be specified for root pass.')
    subtree_totals_io = io.ProcessingFormatIO(
        path_prefix=str(_SUBTREE_TOTALS_DIR.value),
        subtree_folder='US',
        read_only=True,
    )
    root.root_baseline_census_topdown_pass(
        nmf_io=nmf_io,
        out_io=out_io,
        constraint_io=constraint_io,
        subtree_totals_io=subtree_totals_io,
        block_shape=BLOCK_SHAPE
    )
  else:
    subtree_folder = constants.SUBTREE_FOLDER_PATTERN % _STATE.value
    nmf_io.set_subtree_folder(subtree_folder)
    out_io.set_subtree_folder(subtree_folder)
    constraint_io.set_subtree_folder(subtree_folder)

    main_passes.baseline_census_top_down_pass(
        in_io=nmf_io,
        out_io=out_io,
        constraint_io=constraint_io,
        block_shape=BLOCK_SHAPE
    )


def compute_errs(alternate: bool = False) -> None:
  """Run compute errors pass."""
  processed_io = io.ProcessingFormatIO(
      path_prefix=str(_OUT_DIR.value),
      subtree_folder='',
      read_only=False,
      split_estimates=False,
  )
  if _STATE.value is None:
    queries = compute_errors.ALTERNATE_QUERIES if alternate else None
    file_name = constants.ALTERNATE_ERRORS_FNAME if alternate else None
    compute_errors.aggregate_errors(
        processed_io=processed_io,
        queries=queries,
        file_name=file_name,
    )
  else:
    ground_truth_io = io.InputFormatIO(
        path_prefix=str(_IN_DIR.value),
        subtree_folder='',
        read_only=True,
        path_suffix=constants.NMF_PATH_SUFFIX,
    )
    constraint_io = io.InputFormatIO(
        path_prefix=str(_IN_DIR.value),
        subtree_folder='',
        read_only=True,
        path_suffix=constants.CONSTRAINT_PATH_SUFFIX,
    )
    if alternate:
      compute_errors.compute_alternate_errors(
          folder_id=_STATE.value,
          ground_truth_io=ground_truth_io,
          processed_io=processed_io,
          constraint_io=constraint_io,
      )
    else:
      compute_errors.compute_errors(
          folder_id=_STATE.value,
          ground_truth_io=ground_truth_io,
          processed_io=processed_io,
          constraint_io=constraint_io,
      )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  match _PASS.value:
    case 'bottom_up':
      bottom_up()
    case 'top_down':
      top_down()
    case 'baseline_census_topdown':
      baseline_census_topdown()
    case 'compute_errors':
      compute_errs()
    case 'compute_alternate_errors':
      compute_errs(alternate=True)
    case _:
      raise ValueError(f'Unsupported pass: {_PASS.value}')


if __name__ == '__main__':
  app.run(main)
