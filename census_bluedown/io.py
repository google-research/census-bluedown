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

"""Input/output for block hierarchical aggregation."""

from __future__ import annotations

import abc
import dataclasses
import os

from absl import logging
from google.protobuf import text_format
import pandas as pd

from census_bluedown import constants
from census_bluedown import estimate


ROOT_LEVEL = constants.ROOT_LEVEL
ESTIMATE = constants.ESTIMATE
VARIANCE = constants.VARIANCE
VALUE = constants.VALUE

Estimate = estimate.Estimate


@dataclasses.dataclass
class AbstractBlockHierarchicalIO(abc.ABC):
  """Abstract class for input/output for block hierarchical aggregation."""
  path_prefix: str
  subtree_folder: str
  read_only: bool = True

  @abc.abstractmethod
  def read(self, level: str, filename: str | None) -> pd.DataFrame:
    pass

  @abc.abstractmethod
  def write(self, level: str, filename: str, df: pd.DataFrame) -> None:
    pass

  @abc.abstractmethod
  def exists(self, level: str, filename: str) -> bool:
    pass

  def set_subtree_folder(
      self,
      subtree_folder: str
  ) -> AbstractBlockHierarchicalIO:
    self.subtree_folder = subtree_folder
    return self


@dataclasses.dataclass
class InputFormatIO(AbstractBlockHierarchicalIO):
  """Input/output of raw or NMF data for block hierarchical aggregation.

  This is the format in which the census input data (e.g. Noisy Measurement
  Files) are stored. For example, if path_prefix='./data',
  subtree_folder='State=01', and path_suffix='.parquet/DPQuery/', then the data
  for level 'County' is expected to be in the unique file in directory
  './data/County.parquet/DPQuery/State=01/'. The name of this file
  is arbitrary.
  """
  path_suffix: str = ''

  def get_directory(self, level: str) -> str:
    """Get directory for level."""
    if level == ROOT_LEVEL:
      return os.path.join(self.path_prefix,
                          level + self.path_suffix)
    else:
      return os.path.join(self.path_prefix,
                          level + self.path_suffix,
                          self.subtree_folder)

  def read(self, level: str, filename: str | None = None) -> pd.DataFrame:
    """Read parquet file from input directory.

    The argument filename is ignored in this subclass of the abstract base
    class. The file to read is the unique file in the input directory. If there
    is not exactly one file in the input directory, an error is raised.

    Args:
      level: The level of the data to read.
      filename: This argument is ignored.

    Returns:
      A pandas DataFrame containing the data for the given level.

    Raises:
      FileNotFoundError: If no file is found in the input directory.
    """
    input_directory = self.get_directory(level)
    files = os.listdir(input_directory)
    if not files:
      raise FileNotFoundError(f'No file found in {input_directory}')
    if len(files) > 1:
      raise ValueError(f'Multiple files found in {input_directory}')
    file_path = os.path.join(input_directory, files[0])
    with open(file_path, 'rb') as in_f:
      logging.info('Reading input file %s', file_path)
      return pd.read_parquet(in_f, engine='pyarrow', use_threads=False)

  def write(self, level: str, filename: str, df: pd.DataFrame) -> None:
    """Write parquet file to directory.

    Args:
      level: The level of the data to write.
      filename: The name of the file to write.
      df: The data to write.
    """
    if self.read_only:
      raise ValueError('Cannot write to read-only InputFormatIO.')
    output_directory = self.get_directory(level)
    output_file = os.path.join(output_directory, filename)
    os.makedirs(os.path.dirname(output_file))
    logging.info('Writing file %s', output_file)
    df.to_parquet(output_file)

  def exists(self, level: str, filename: str = '') -> bool:
    """Check if any file exists in directory."""
    return bool(os.listdir(self.get_directory(level)))


@dataclasses.dataclass
class ProcessingFormatIO(AbstractBlockHierarchicalIO):
  """Input/output of processed data for block hierarchical aggregation.

  This is the format in which the processed census data files will generally
  be stored. For example, if path_prefix='./data',
  subtree_folder='State=01', then the data for level 'County' and filename
  'estimate.pickle' will be stored in the file
  './data/State=01/County/estimate.pickle'.

  The parameter expand_estimates controls the file and data format. If
  expand_estimates has value False then the data is stored in a pickle file as
  a single column of Estimate objects. If expand_estimates has value True then
  the data is stored in a parquet file with separate columns for value and
  variance. The former is faster to read and write, but the latter is more
  flexible.
  """
  split_estimates: bool = False
  pickle_compression: bool = False

  def get_filename(self, level: str, fname: str) -> str:
    """Get filename for level."""
    if level == ROOT_LEVEL:
      return os.path.join(self.path_prefix, level, fname)
    else:
      return os.path.join(self.path_prefix, self.subtree_folder, level, fname)

  def read(self, level: str, filename: str | None) -> pd.DataFrame:
    """Read parquet or pickle file from input directory.

    Args:
      level: The level of the data to read.
      filename: The name of the file to read.

    Returns:
      A pandas DataFrame containing the data for the given level.
    """
    input_filename = self.get_filename(level, filename)  # pyrefly: ignore[bad-argument-type]
    if self.split_estimates:
      logging.info('Reading parquet file %s', input_filename)
      return pack_estimates(pd.read_parquet(
          input_filename, engine='pyarrow', use_threads=False))
    elif self.pickle_compression:
      logging.info('Reading pickle file %s', input_filename)
      return pd.read_pickle(input_filename)  # pyrefly: ignore[bad-return]
    else:
      logging.info('Reading parquet file %s', input_filename)
      return pd.read_parquet(input_filename,
                             engine='pyarrow',
                             use_threads=False)

  def write(self, level: str, filename: str, df: pd.DataFrame) -> None:
    """Write parquet or pickle file to directory.

    Args:
      level: The level of the data to write.
      filename: The name of the file to write.
      df: The data to write.
    """
    if self.read_only:
      raise ValueError('Cannot write to read-only ProcessingFormatIO.')
    output_filename = self.get_filename(level, filename)
    os.makedirs(os.path.dirname(output_filename))
    if self.split_estimates:
      logging.info('Writing parquet file %s', output_filename)
      expand_estimates(df).to_parquet(output_filename)
    elif self.pickle_compression:
      logging.info('Writing pickle file %s', output_filename)
      df.to_pickle(output_filename)
    else:
      logging.info('Writing parquet file %s', output_filename)
      df.to_parquet(output_filename)

  def exists(self, level: str, filename: str) -> bool:
    """Check if given file exists in directory."""
    return os.path.exists(self.get_filename(level, filename))


def get_input_format_io(
    path_prefix: str,
    subtree_id: str,
    read_only: bool,
    io_for_constraints: bool = False,
) -> InputFormatIO:
  """Get InputFormatIO object for a given path and subtree."""
  subtree_folder = f'State={subtree_id}' if subtree_id else ''
  if io_for_constraints:
    path_suffix = '.parquet/Constraint/'
  else:
    path_suffix = '.parquet/DPQuery/'
  return InputFormatIO(
      path_prefix, subtree_folder, read_only, path_suffix
  )


def pack_estimates(df: pd.DataFrame) -> pd.DataFrame:
  """Pack estimates into a single column.

  Take a DataFrame with columns VALUE and VARIANCE consisting of np.ndarrays,
  and pack them into a single column ESTIMATE consisting of Estimate objects.

  Args:
    df: The DataFrame to pack.

  Returns:
    The packed DataFrame.
  """
  if VALUE not in df.columns:
    raise ValueError(f'Cannot pack nonexistent {VALUE} column into Estimate.')
  if VARIANCE not in df.columns:
    raise ValueError(
        f'Cannot pack nonexistent {VARIANCE} column into Estimate.'
    )
  var_shape = int(len(df[VARIANCE].iloc[0])**0.5)
  df[ESTIMATE] = df.apply(
      lambda x: Estimate(x[VALUE], x[VARIANCE].reshape((var_shape, -1))), axis=1
  )

  return df.drop(columns=[VALUE, VARIANCE])


def expand_estimates(df: pd.DataFrame) -> pd.DataFrame:
  """Unpack estimates into separate columns.

  Take a DataFrame with a column ESTIMATE consisting of Estimate objects,
  and expand it into two columns VALUE and VARIANCE consisting of np.ndarrays.

  Args:
    df: The DataFrame to expand.

  Returns:
    The expanded DataFrame.
  """
  if ESTIMATE not in df.columns:
    raise ValueError(f'Cannot expand nonexistent {ESTIMATE} column; columns: '
                     f'{df.columns}')
  df[VALUE] = df[ESTIMATE].apply(lambda x: x.val.flatten())
  df[VARIANCE] = df[ESTIMATE].apply(lambda x: x.cov.flatten())
  return df.drop(columns=ESTIMATE)


def write_model(filename, model_proto):
  """Write model to file."""
  os.makedirs(os.path.dirname(filename))
  with open(filename, 'w') as f:
    f.write(text_format.MessageToString(model_proto))
